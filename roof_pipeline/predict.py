"""Sliding-window roof prediction with optional georeferenced outputs.

Returns binary PNG + preview for any image. If GeoTIFF has a valid CRS, also
writes mask GeoTIFF and WGS84 polygon GeoJSON. No invented real-world area.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .dataset import normalize_rgb
from .models import load_model


def starts(n: int, block: int, stride: int) -> list[int]:
    if n <= block:
        return [0]
    out = list(range(0, n - block + 1, stride))
    if out[-1] != n - block:
        out.append(n - block)
    return out


@torch.no_grad()
def predict_array(
    model, rgb: np.ndarray, device, chip: int = 512, overlap: int = 64
) -> np.ndarray:
    """Probability raster at original resolution, tiled without resizing full scene."""
    if chip < 32 or chip % 32 or overlap < 0 or overlap >= chip:
        raise ValueError("chip >=32 and divisible by 32; overlap must be in [0, chip)")
    h, w = rgb.shape[:2]
    stride = chip - overlap
    total = np.zeros((h, w), dtype=np.float32)
    weights = np.zeros((h, w), dtype=np.float32)
    # Positive at all boundaries to avoid dividing by zero at image edges.
    window = 0.25 + 0.75 * np.hanning(chip).astype(np.float32)
    win2d = np.outer(window, window)
    for r in starts(h, chip, stride):
        for c in starts(w, chip, stride):
            part = rgb[r : min(r + chip, h), c : min(c + chip, w)]
            ph, pw = part.shape[:2]
            if ph != chip or pw != chip:
                part = np.pad(
                    part, ((0, chip - ph), (0, chip - pw), (0, 0)), mode="edge"
                )
            input_tensor = normalize_rgb(part).to(device)
            prob = model(input_tensor).sigmoid().float()[0, 0].cpu().numpy()[:ph, :pw]
            weight = win2d[:ph, :pw]
            total[r : r + ph, c : c + pw] += prob * weight
            weights[r : r + ph, c : c + pw] += weight
    return total / np.maximum(weights, 1e-8)


def _read_rgb(path: Path):
    """Return RGB uint8 array, rasterio metadata or None."""
    if path.suffix.lower() in (".tif", ".tiff"):
        import rasterio

        with rasterio.open(path) as src:
            if src.count < 3:
                raise ValueError("GeoTIFF needs >=3 RGB bands")
            rgb = np.moveaxis(src.read([1, 2, 3]), 0, -1)
            meta = {"crs": src.crs, "transform": src.transform, "profile": src.profile}
            if rgb.dtype != np.uint8:
                result = np.zeros_like(rgb, dtype=np.uint8)
                for i in range(3):
                    channel = rgb[:, :, i].astype(np.float32)
                    valid = channel[np.isfinite(channel)]
                    lo, hi = np.percentile(valid, [2, 98]) if valid.size else (0, 1)
                    result[:, :, i] = np.clip(
                        (channel - lo) * 255 / max(hi - lo, 1e-8), 0, 255
                    ).astype(np.uint8)
                rgb = result
            return rgb, meta
    with Image.open(path) as img:
        return np.asarray(img.convert("RGB")), None


def predict(
    checkpoint: Path,
    image: Path,
    out: Path,
    chip: int | None = None,
    overlap: int = 64,
    device: str = "cpu",
    threshold: float | None = None,
):
    model, cfg = load_model(checkpoint, device)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    t = float(ckpt["threshold"] if threshold is None else threshold)
    if not (0 < t < 1):
        raise ValueError("threshold must be in (0,1)")
    tile = int(chip or cfg["image_size"])
    rgb, meta = _read_rgb(image)
    prob = predict_array(model, rgb, torch.device(device), tile, overlap)
    mask = (prob >= t).astype(np.uint8) * 255
    out.mkdir(parents=True, exist_ok=True)
    out_mask = out / (image.stem + "_mask.png")
    out_prob = out / (image.stem + "_probability.png")
    Image.fromarray(mask).save(out_mask)
    Image.fromarray(np.rint(prob * 255).astype(np.uint8)).save(out_prob)
    overlay = Image.blend(
        Image.fromarray(rgb),
        Image.fromarray(
            np.where(mask[..., None] > 0, np.array([255, 40, 55], dtype=np.uint8), rgb)
        ),
        0.45,
    )
    overlay.save(out / (image.stem + "_overlay.png"))
    result = {
        "checkpoint": str(checkpoint),
        "image": str(image),
        "threshold": t,
        "roof_area_pixels": int((mask > 0).sum()),
        "geo_products": False,
        "mask_png": str(out_mask),
        "warning": "Pixel area is not a physical roof area without georeferencing.",
    }
    if meta and meta["crs"] is not None:
        import rasterio
        from rasterio.features import shapes
        from rasterio.warp import transform_geom
        from shapely.geometry import shape

        crs = meta["crs"]
        profile = meta["profile"].copy()
        profile.update(
            driver="GTiff",
            height=mask.shape[0],
            width=mask.shape[1],
            count=1,
            dtype="uint8",
            compress="deflate",
        )
        profile.pop("nodata", None)  # zero is a real background class
        profile.pop("photometric", None)  # a single-band mask is not RGB
        tif_path = out / (image.stem + "_mask.tif")
        with rasterio.open(tif_path, "w", **profile) as dst:
            dst.write(mask, 1)
        features = []
        total_area = None
        projected_meters = (
            crs.is_projected and abs(crs.linear_units_factor[1] - 1.0) < 1e-6
        )
        if projected_meters:
            total_area = 0.0
        for geom, v in shapes(mask, mask=mask > 0, transform=meta["transform"]):
            if int(v) == 0:
                continue
            props = {}
            if projected_meters:
                area = float(shape(geom).area)
                props["area_m2"] = area
                total_area += area
            features.append(
                {
                    "type": "Feature",
                    "geometry": transform_geom(crs, "EPSG:4326", geom),
                    "properties": props,
                }
            )
        geojson = {"type": "FeatureCollection", "features": features}
        gj_path = out / (image.stem + "_roofs.geojson")
        gj_path.write_text(json.dumps(geojson), encoding="utf-8")
        result.update(
            geo_products=True,
            crs=str(crs),
            mask_geotiff=str(tif_path),
            geojson=str(gj_path),
            building_components=len(features),
            area_m2=total_area,
        )
        result.pop("warning", None)
        if not projected_meters:
            result["warning"] = (
                "CRS is not projected in meters; physical area withheld. Reproject to an appropriate metric CRS."
            )
    (out / "prediction_manifest.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--tile-size", type=int, default=None)
    p.add_argument("--overlap", type=int, default=64)
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    a = p.parse_args()
    predict(a.checkpoint, a.image, a.out, a.tile_size, a.overlap, a.device, a.threshold)


if __name__ == "__main__":
    main()
