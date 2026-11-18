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
from .validity import read_rgb


def starts(n: int, block: int, stride: int) -> list[int]:
    if n <= block:
        return [0]
    out = list(range(0, n - block + 1, stride))
    if out[-1] != n - block:
        out.append(n - block)
    return out


@torch.no_grad()
def predict_array(
    model,
    rgb: np.ndarray,
    device,
    chip: int = 512,
    overlap: int = 64,
    *,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Predict original pixels; invalid slots are zero placeholders with a separate mask."""
    if chip < 32 or chip % 32 or overlap < 0 or overlap >= chip:
        raise ValueError("chip >=32 and divisible by 32; overlap must be in [0, chip)")
    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or min(rgb.shape[:2]) < 1:
        raise ValueError("Expected nonempty HWC RGB array")
    h, w = rgb.shape[:2]
    valid = (
        np.ones((h, w), dtype=bool)
        if valid_mask is None
        else np.asarray(valid_mask, dtype=bool)
    )
    if valid.shape != (h, w):
        raise ValueError("Validity mask must match the original image dimensions")
    if not valid.any():
        raise ValueError("Image contains no valid pixels")
    if not np.isfinite(rgb[valid]).all():
        raise ValueError("Valid RGB pixels must be finite")
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
            part_valid = valid[r : r + ph, c : c + pw]
            if not part_valid.any():
                continue
            model_valid = part_valid
            if ph != chip or pw != chip:
                part = np.pad(
                    part, ((0, chip - ph), (0, chip - pw), (0, 0)), mode="edge"
                )
                # Preserve legacy edge padding for valid RGB inputs. Padded
                # positions never contribute to source-pixel area or blending.
                model_valid = np.pad(
                    part_valid, ((0, chip - ph), (0, chip - pw)), mode="edge"
                )
            if not model_valid.all():
                # Remove NaN/sentinel values before normalization, then make the
                # actual normalized model input exactly zero at missing pixels.
                part = part.copy()
                part[~model_valid] = 0
            input_tensor = normalize_rgb(part).to(device)
            if not model_valid.all():
                input_tensor.masked_fill_(
                    ~torch.from_numpy(model_valid).to(device)[None, None], 0
                )
            prob = model(input_tensor).sigmoid().float()[0, 0].cpu().numpy()[:ph, :pw]
            weight = win2d[:ph, :pw] * part_valid
            total[r : r + ph, c : c + pw] += np.where(part_valid, prob, 0) * weight
            weights[r : r + ph, c : c + pw] += weight
    result = total / np.maximum(weights, 1e-8)
    result[~valid] = 0
    return result


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
    rgb, valid, meta = read_rgb(image)
    if not valid.any():
        raise ValueError("Image contains no valid pixels")
    prob = predict_array(
        model, rgb, torch.device(device), tile, overlap, valid_mask=valid
    )
    mask = ((prob >= t) & valid).astype(np.uint8) * 255
    out.mkdir(parents=True, exist_ok=True)
    out_mask = out / (image.stem + "_mask.png")
    out_prob = out / (image.stem + "_probability.png")
    out_valid = out / (image.stem + "_validity.png")
    Image.fromarray(mask).save(out_mask)
    Image.fromarray(np.rint(prob * 255).astype(np.uint8)).save(out_prob)
    Image.fromarray(valid.astype(np.uint8) * 255).save(out_valid)
    overlay = Image.blend(
        Image.fromarray(rgb),
        Image.fromarray(
            np.where(mask[..., None] > 0, np.array([255, 40, 55], dtype=np.uint8), rgb)
        ),
        0.45,
    )
    if not valid.all():
        overlay = overlay.convert("RGBA")
        overlay.putalpha(Image.fromarray(valid.astype(np.uint8) * 255))
    overlay.save(out / (image.stem + "_overlay.png"))
    result = {
        "checkpoint": str(checkpoint),
        "image": str(image),
        "threshold": t,
        "roof_area_pixels": int((mask > 0).sum()),
        "pixel_coverage": float((mask > 0).sum() / valid.sum()),
        "valid_pixel_count": int(valid.sum()),
        "invalid_pixel_count": int(valid.size - valid.sum()),
        "total_pixel_count": int(valid.size),
        "geo_products": False,
        "mask_png": str(out_mask),
        "probability_png": str(out_prob),
        "validity_png": str(out_valid),
        "validity_definition": "255 valid source pixel; 0 missing/transparent/nonfinite. Zero probability at invalid pixels is a placeholder, not background evidence.",
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
        with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True):
            with rasterio.open(tif_path, "w", **profile) as dst:
                dst.write(mask, 1)
                dst.write_mask(valid.astype(np.uint8) * 255)
        features = []
        total_area = None
        projected_meters = (
            crs.is_projected and abs(crs.linear_units_factor[1] - 1.0) < 1e-6
        )
        if projected_meters:
            total_area = 0.0
        for geom, v in shapes(
            mask, mask=(mask > 0) & valid, transform=meta["transform"]
        ):
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
