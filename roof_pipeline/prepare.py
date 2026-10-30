"""Prepare paired roof masks without creating artificial labels or split leakage.

Commands: whu, labelme, geopairs, audit. No network or external downloads.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .common import (
    IMAGE_SUFFIXES,
    SPLITS,
    dataset_audit,
    load_binary_mask,
    save_pair,
    write_manifest,
)


def _split_dir(root: Path, split: str) -> Path:
    aliases = {
        "train": ["train", "training"],
        "val": ["val", "valid", "validation", "eval"],
        "test": ["test", "testing"],
    }
    for alias in aliases[split]:
        direct = root / alias
        if direct.is_dir():
            return direct
    raise FileNotFoundError(
        f"Missing {split} partition under {root}. Keep official splits unchanged."
    )


def _subdir(root: Path, choices: list[str]) -> Path:
    for choice in choices:
        p = root / choice
        if p.is_dir():
            return p
    raise FileNotFoundError(f"No directory under {root} matching {choices}")


def import_whu(source: Path, out: Path, limit_per_split: int = 0) -> None:
    """Import WHU pairs, preserving supplied directories and recording their origin."""
    if limit_per_split < 0:
        raise ValueError("limit_per_split must be nonnegative; use 0 for all pairs")
    rows = []
    for split in SPLITS:
        part = _split_dir(source, split)
        image_dir = _subdir(part, ["Image", "Images", "image", "images", "img"])
        mask_dir = _subdir(
            part, ["Mask", "Masks", "mask", "masks", "label", "labels", "Label"]
        )
        images = sorted(
            p for p in image_dir.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES
        )
        if not images:
            raise ValueError(f"No imagery found in {image_dir}")
        if limit_per_split:
            # Deterministic spread over original filename order; for smoke/small experiments only.
            step = len(images) / min(len(images), limit_per_split)
            images = [
                images[min(len(images) - 1, int(i * step))]
                for i in range(min(len(images), limit_per_split))
            ]
        masks = {}
        for mp in mask_dir.rglob("*"):
            if mp.suffix.lower() in IMAGE_SUFFIXES:
                if mp.stem in masks:
                    raise ValueError(f"Ambiguous mask stem {mp.stem} under {mask_dir}")
                masks[mp.stem] = mp
        for idx, img_path in enumerate(images):
            mask_path = masks.get(img_path.stem)
            if mask_path is None:
                raise FileNotFoundError(f"Missing corresponding mask for {img_path}")
            with Image.open(img_path) as img:
                mask = load_binary_mask(mask_path, img.size)
                sample_id = f"whu_{split}_{idx:06d}"
                rows.append(
                    save_pair(
                        out,
                        img,
                        mask,
                        sample_id,
                        split,
                        group=f"whu_provided_{split}_partition",
                        source="WHU aerial buildings",
                    )
                )
            if (idx + 1) % 500 == 0 or idx + 1 == len(images):
                print(f"Prepared {split}: {idx + 1}/{len(images)} pairs", flush=True)
    write_manifest(
        out,
        rows,
        {
            "dataset": "WHU aerial buildings",
            "split_policy": "Supplied train/val/test directories preserved without reshuffling; geographic independence is not inferred from directory names",
            "license": "Check license of chosen official or third-party distribution",
            "labels": "binary building extent (not instance/roof-material classification)",
            "limited_subset": bool(limit_per_split),
            "source_provenance": json.loads(
                (source / "download_provenance.json").read_text(encoding="utf-8")
            )
            if (source / "download_provenance.json").is_file()
            else {"source_path": str(source), "origin": "user-supplied"},
        },
    )
    print(f"Imported {len(rows)} pairs -> {out / 'manifest.csv'}")


def _load_split_map(csv_path: Path) -> dict[str, tuple[str, str]]:
    with csv_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError("split map empty")
    out = {}
    for row in rows:
        name = row["stem"]
        if name in out:
            raise ValueError(f"Duplicate stem in split map: {name}")
        split, group = row["split"], row["group"]
        if split not in SPLITS or not group:
            raise ValueError(f"Invalid split/group for {name}")
        out[name] = (split, group)
    return out


def labelme_mask(
    annotation: dict, *, positive_labels: set[str] | None = None
) -> np.ndarray:
    positive = positive_labels or {"roof", "building"}
    width, height = int(annotation["imageWidth"]), int(annotation["imageHeight"])
    if width <= 0 or height <= 0:
        raise ValueError("Invalid annotation image dimensions")
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    for shape in annotation.get("shapes", []):
        label = str(shape.get("label", "")).strip().lower()
        if label not in positive and label not in {
            "background",
            "ignore",
            "__ignore__",
        }:
            raise ValueError(
                f"Unknown label {label!r}; ensure all intended roof polygons are annotated"
            )
        if label in {"ignore", "__ignore__"}:
            raise ValueError(
                "Ignore polygons are not supported in binary output; resolve annotation first"
            )
        pts = shape.get("points", [])
        shape_type = shape.get("shape_type", "polygon")
        if shape_type == "rectangle" and len(pts) == 2:
            (x0, y0), (x1, y1) = pts
            draw.rectangle(
                [round(x0), round(y0), round(x1), round(y1)],
                fill=255 if label in positive else 0,
            )
        elif shape_type == "polygon" and len(pts) >= 3:
            draw.polygon(
                [(round(x), round(y)) for x, y in pts],
                fill=255 if label in positive else 0,
            )
        else:
            raise ValueError(f"Unsupported shape {shape_type!r} with {len(pts)} points")
    return np.asarray(mask).copy()


def import_labelme(source: Path, out: Path, split_map: Path) -> None:
    """Convert manually verified Labelme polygon JSON files to binary masks."""
    mapping = _load_split_map(split_map)
    annotations = sorted(source.rglob("*.json"))
    rows = []
    seen = set()
    for path in annotations:
        annotation = json.loads(path.read_text(encoding="utf-8"))
        stem = path.stem
        if stem not in mapping:
            raise ValueError(
                f"No explicit split/group for annotation {stem}; add it to {split_map}"
            )
        if stem in seen:
            raise ValueError(f"Duplicate annotation stem {stem}")
        seen.add(stem)
        image_name = Path(annotation.get("imagePath", stem + ".png")).name
        img_path = path.parent / image_name
        if not img_path.is_file():
            raise FileNotFoundError(f"No image next to annotation {path}: {image_name}")
        with Image.open(img_path) as image:
            mask = labelme_mask(annotation)
            rows.append(
                save_pair(
                    out,
                    image,
                    mask,
                    f"manual_{stem}",
                    *mapping[stem],
                    source="manually labeled",
                )
            )
    missing = set(mapping) - seen
    if missing:
        raise ValueError(f"Missing Labelme annotations for: {sorted(missing)}")
    write_manifest(
        out,
        rows,
        {
            "dataset": "User-labeled imagery",
            "split_policy": "User-specified site/scene groups",
            "labels": "Manual Labelme roof/building polygon masks; annotations must be reviewed",
        },
    )
    print(f"Rasterized {len(rows)} manual annotations -> {out}")


def import_geopairs(pairs_csv: Path, out: Path, tile_size: int = 512) -> None:
    """GeoTIFF RGB + WGS84 GeoJSON polygon rasterizer; preserve scene split group.

    Pairs CSV columns: image,vector,split,group. Relative paths resolved from CSV parent.
    Chips are rendered as PNGs. Original CRS/transform retained in scene metadata.
    Border chips retain their actual dimensions, without padding or dropped pixels.
    """
    if tile_size < 1:
        raise ValueError("tile_size must be positive")
    try:
        import rasterio
        from rasterio.features import rasterize
        from rasterio.warp import transform_geom
    except ImportError as e:
        raise RuntimeError("GeoTIFF import requires rasterio") from e
    with pairs_csv.open(newline="", encoding="utf-8") as f:
        pairs = list(csv.DictReader(f))
    if not pairs or not set(["image", "vector", "split", "group"]).issubset(pairs[0]):
        raise ValueError("CSV must have image,vector,split,group columns")
    rows = []
    metadata = []
    for scene_idx, pair in enumerate(pairs):
        split, group = pair["split"], pair["group"]
        if split not in SPLITS or not group:
            raise ValueError("Split and group required for each scene")
        img_path = (pairs_csv.parent / pair["image"]).resolve()
        vec_path = (pairs_csv.parent / pair["vector"]).resolve()
        if not img_path.is_file() or not vec_path.is_file():
            raise FileNotFoundError(f"Geo pair missing: {img_path}, {vec_path}")
        data = json.loads(vec_path.read_text(encoding="utf-8"))
        feats = (
            data.get("features", [])
            if data.get("type") == "FeatureCollection"
            else [data]
        )
        geometries = [f.get("geometry", f) for f in feats]
        geometries = [
            g for g in geometries if g and g.get("type") in ("Polygon", "MultiPolygon")
        ]
        if not geometries:
            raise ValueError(f"No polygon geometry found in {vec_path}")
        with rasterio.open(img_path) as src:
            if src.crs is None:
                raise ValueError(f"GeoTIFF is missing CRS: {img_path}")
            if src.count < 3:
                raise ValueError(f"Expected RGB imagery with >=3 bands: {img_path}")
            shapes = [
                (transform_geom("EPSG:4326", src.crs.to_string(), g), 1)
                for g in geometries
            ]
            scene_mask = (
                rasterize(
                    shapes,
                    out_shape=(src.height, src.width),
                    transform=src.transform,
                    fill=0,
                    dtype="uint8",
                )
                * 255
            )
            if not scene_mask.any():
                raise ValueError(
                    f"Vector labels do not overlap raster: {img_path} {vec_path}; verify EPSG:4326 coordinates and source transform"
                )
            rgb = np.moveaxis(src.read([1, 2, 3]), 0, -1)
            if rgb.dtype != np.uint8:
                # Deterministic scene-level stretch, with clipping; never stretch each chip independently.
                r = np.zeros(rgb.shape, dtype=np.uint8)
                for band in range(3):
                    arr = rgb[:, :, band].astype(np.float32)
                    lo, hi = np.percentile(arr[np.isfinite(arr)], [2, 98])
                    r[:, :, band] = np.clip(
                        (arr - lo) * 255 / max(hi - lo, 1e-8), 0, 255
                    ).astype(np.uint8)
                rgb = r
            scene_id = f"geo_{scene_idx:04d}"
            metadata.append(
                {
                    "scene_id": scene_id,
                    "image": str(img_path),
                    "vector": str(vec_path),
                    "crs": src.crs.to_string(),
                    "transform": list(src.transform)[:6],
                    "width": src.width,
                    "height": src.height,
                    "split": split,
                    "group": group,
                }
            )
            for row0 in range(0, src.height, tile_size):
                for col0 in range(0, src.width, tile_size):
                    h = min(src.height - row0, tile_size)
                    w = min(src.width - col0, tile_size)
                    img_chip = rgb[row0 : row0 + h, col0 : col0 + w]
                    mask_chip = scene_mask[row0 : row0 + h, col0 : col0 + w]
                    ident = f"{scene_id}_{row0:05d}_{col0:05d}"
                    rows.append(
                        save_pair(
                            out,
                            img_chip,
                            mask_chip,
                            ident,
                            split,
                            group,
                            source="georeferenced polygon labels",
                        )
                    )
    write_manifest(
        out,
        rows,
        {
            "dataset": "GeoTIFF + GeoJSON",
            "split_policy": "Explicit per-scene spatial groups",
            "label_crs": "EPSG:4326 GeoJSON",
            "tile_size": tile_size,
            "border_policy": "retain_unpadded_partial_chips",
            "notes": "Chips are PNG; partial border chips retain actual dimensions. RoofDataset resizes each chip to the model input size, so metrics use that resized grid. Refer to scene_metadata.json for source georeferencing.",
        },
    )
    (out / "scene_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"Rasterized {len(rows)} geo-pairs -> {out}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser(
        "whu", help="Import official WHU or mirror train/Image,train/Mask folders"
    )
    w.add_argument("--source", type=Path, required=True)
    w.add_argument("--out", type=Path, required=True)
    w.add_argument(
        "--limit-per-split",
        type=int,
        default=0,
        help="Small smoke runs only; 0 for all",
    )
    labels = sub.add_parser(
        "labelme", help="Rasterize manually-reviewed Labelme polygons"
    )
    labels.add_argument("--source", type=Path, required=True)
    labels.add_argument("--out", type=Path, required=True)
    labels.add_argument("--split-map", type=Path, required=True)
    g = sub.add_parser(
        "geopairs", help="Rasterize WGS84 GeoJSON onto georeferenced imagery"
    )
    g.add_argument("--pairs-csv", type=Path, required=True)
    g.add_argument("--out", type=Path, required=True)
    g.add_argument("--tile-size", type=int, default=512)
    a = sub.add_parser(
        "audit", help="Validate pairs, groups, duplicates and produce mask overlays"
    )
    a.add_argument("--root", type=Path, required=True)
    a.add_argument("--no-hashes", action="store_true")
    args = p.parse_args()
    if args.cmd == "whu":
        import_whu(args.source, args.out, args.limit_per_split)
    elif args.cmd == "labelme":
        import_labelme(args.source, args.out, args.split_map)
    elif args.cmd == "geopairs":
        import_geopairs(args.pairs_csv, args.out, args.tile_size)
    elif args.cmd == "audit":
        print(
            json.dumps(
                dataset_audit(args.root, verify_hashes=not args.no_hashes), indent=2
            )
        )


if __name__ == "__main__":
    main()
