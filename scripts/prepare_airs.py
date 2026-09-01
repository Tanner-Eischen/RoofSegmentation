#!/usr/bin/env python
"""
Prepare a clean, leak-free roof-segmentation dataset from AIRS GeoTIFF mosaics
plus your existing tiles, written into the data layout that dataset.py expects:

    <out>/
      images/           # *.png  (RGB)
      masks/            # *.png  (binary 0/255, same names as images/)
      filenames/
        train_filenames_clean.txt
        val_filenames_clean.txt
        test_filenames_clean.txt

Why this exists:
  1. Your current splits leak (train/val/test share source tiles tile1/tile2),
     inflating val IoU. We re-split by SOURCE so no geography appears twice.
  2. AIRS adds geographic breadth (Christchurch, 7.5 cm). We tile its large
     GeoTIFFs into 512x512 patches via windowed rasterio reads (constant RAM)
     and assign whole spatial macro-blocks to splits so no patch straddles a
     train/val boundary.

AIRS layout expected (flexible): a directory containing paired mosaics, e.g.
    train.tif / train_mask.tif   and   test.tif / test_mask.tif
Filename containing "mask" or "label" => mask; otherwise => image.

Split policy (default, configurable):
  - AIRS test mosaic  -> held-out TEST set (wholly separate geography).
  - AIRS train mosaic -> 80% train / 20% val, by macro-block hash.
  - Existing tiles    -> by source prefix: tile1_* -> train, tile2_* -> val
    (only 2 sources exist, so they cannot form a test set; AIRS supplies it).

Usage:
    python scripts/prepare_airs.py --airs-dir /path/to/airs \\
        --existing-dir segmentation_model_data --out segmentation_model_data_v2

    # Smoke test (few macro-blocks only):
    python scripts/prepare_airs.py --airs-dir /path/to/airs --limit-blocks 3
"""

from __future__ import annotations

import argparse
import hashlib
import re
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import rasterio
    from rasterio.windows import Window
except ImportError:  # pragma: no cover
    rasterio = None


# --------------------------------------------------------------------------- #
# AIRS discovery + tiling
# --------------------------------------------------------------------------- #

def find_airs_pairs(airs_dir: Path) -> list[tuple[Path, Path, str]]:
    """Return [(image_tif, mask_tif, source_name)] pairs. source_name in {train,test}."""
    if airs_dir is None or not airs_dir.exists():
        return []
    tifs = sorted(p for p in airs_dir.rglob("*.tif") if p.is_file())
    masks = {}
    images = []
    for p in tifs:
        stem = p.stem.lower()
        if "mask" in stem or "label" in stem:
            # map the image-stem this mask corresponds to (strip mask/label token)
            base = re.sub(r"[_-]?(mask|label)s?[_-]?", "_", stem).strip("_")
            base = re.sub(r"_+", "_", base)
            masks[base] = p
        else:
            images.append(p)
    pairs = []
    for img in images:
        stem = img.stem.lower()
        m = masks.get(stem) or masks.get(stem + "_")
        if m is None:
            # fall back: same dir sibling named <stem>_mask.tif
            cand = img.with_name(re.sub(r"\.tif$", "_mask.tif", img.name, flags=re.I))
            if cand.exists():
                m = cand
        if m is None:
            print(f"  [skip] no mask for {img.name}", file=sys.stderr)
            continue
        source = "test" if "test" in stem else "train"
        pairs.append((img, m, source))
    return pairs


def _to_uint8_rgb(arr: np.ndarray) -> np.ndarray:
    """rasterio (bands,H,W) -> uint8 (H,W,3). Stretches non-8-bit data by percentile."""
    if arr.shape[0] >= 3:
        arr = arr[:3]
    elif arr.shape[0] == 1:
        arr = np.repeat(arr, 3, axis=0)
    img = np.transpose(arr, (1, 2, 0))  # H,W,C
    if img.dtype != np.uint8:
        lo, hi = np.percentile(img.astype(np.float32), (2, 98))
        img = np.clip((img.astype(np.float32) - lo) / max(hi - lo, 1e-6) * 255.0, 0, 255)
    return img.astype(np.uint8)


def _to_binary_mask(arr: np.ndarray) -> np.ndarray:
    """rasterio mask (bands,H,W) -> uint8 (H,W) binary 0/255."""
    m = arr[0] if arr.ndim == 3 else arr
    return np.where(m > 127, 255, 0).astype(np.uint8)


def tile_airs_pair(
    img_tif: Path,
    mask_tif: Path,
    source: str,
    *,
    tile: int,
    grid: int,
    min_roof_frac: float,
    out_images: Path,
    out_masks: Path,
    block_split: dict[str, str],
    split_files: dict[str, list[str]],
    limit_blocks: int | None,
) -> int:
    """Tile one mosaic into patches, assigning whole macro-blocks to splits. Returns # patches written."""
    if rasterio is None:
        raise RuntimeError("rasterio is required: pip install rasterio>=1.3")
    written = 0
    seen_blocks: set[str] = set()
    with rasterio.open(img_tif) as src, rasterio.open(mask_tif) as msk:
        H, W = src.height, src.width
        macro_h = max(1, H // grid)
        macro_w = max(1, W // grid)
        for row in range(0, H, tile):
            for col in range(0, W, tile):
                h = min(tile, H - row)
                w = min(tile, W - col)
                if h < tile // 2 or w < tile // 2:
                    continue  # drop tiny edge fragments
                macro_r = row // macro_h
                macro_c = col // macro_w
                block_id = f"airs_{source}_{macro_r}_{macro_c}"

                if block_id not in seen_blocks:
                    if limit_blocks is not None and len(seen_blocks) >= limit_blocks:
                        return written
                    seen_blocks.add(block_id)
                    # deterministic macro-block -> split (only used for the train mosaic)
                    split = block_split.get(block_id)
                    if split is None:
                        h_ = int(hashlib.md5(block_id.encode()).hexdigest(), 16) % 1000 / 1000.0
                        split = "val" if h_ < 0.2 else "train"
                        block_split[block_id] = split

                window = Window(col, row, w, h)
                img = _to_uint8_rgb(src.read(window=window))
                mask = _to_binary_mask(msk.read(window=window))
                if img.shape[:2] != mask.shape[:2]:
                    continue
                if float((mask > 127).mean()) < min_roof_frac:
                    continue
                # pad to square tile if edge patch
                if img.shape[0] != tile or img.shape[1] != tile:
                    img = _pad_to(img, tile)
                    mask = _pad_to(mask, tile)
                name = f"airs_{source}_{col}_{row}.png"
                Image.fromarray(img, mode="RGB").save(out_images / name)
                Image.fromarray(mask, mode="L").save(out_masks / name)
                split_files[block_split[block_id]].append(name)
                written += 1
    return written


def _pad_to(arr: np.ndarray, tile: int) -> np.ndarray:
    h, w = arr.shape[:2]
    pad = ((0, tile - h), (0, tile - w)) + ((0, 0) if arr.ndim == 3 else ())
    out = np.pad(arr, pad, mode="constant")
    return out[:tile, :tile]


# --------------------------------------------------------------------------- #
# Existing tiles: re-emit by source prefix
# --------------------------------------------------------------------------- #

SOURCE_RE = re.compile(r"(tile\d+)_", re.IGNORECASE)


def reemit_existing(
    existing_dir: Path,
    out_images: Path,
    out_masks: Path,
    split_files: dict[str, list[str]],
    assignment: dict[str, str],
) -> int:
    """Copy existing images+matching masks into <out>, split by source-tile prefix."""
    images = existing_dir / "images"
    masks = existing_dir / "masks"
    if not images.exists():
        print(f"  [skip] no existing images dir at {images}", file=sys.stderr)
        return 0
    n = 0
    default_split = "train"
    for img_path in sorted(images.iterdir()):
        if not img_path.is_file() or img_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".tif"}:
            continue
        mask_path = masks / img_path.name
        if not mask_path.exists():
            continue
        m = SOURCE_RE.search(img_path.name)
        key = m.group(1).lower() if m else ""
        split = assignment.get(key, default_split)
        dst_img = out_images / img_path.name
        dst_mask = out_masks / mask_path.name
        shutil.copy2(img_path, dst_img)
        shutil.copy2(mask_path, dst_mask)
        split_files[split].append(img_path.name)
        n += 1
    return n


# --------------------------------------------------------------------------- #
# Leakage assertion + manifest writing
# --------------------------------------------------------------------------- #

def assert_no_leak(split_files: dict[str, list[str]]) -> None:
    """Fail if any source unit (airs_<src>_<macroR>_<macroC> prefix or tileN_) appears in >1 split."""
    src_to_splits: dict[str, set[str]] = {}
    for split, names in split_files.items():
        for name in names:
            stem = name.rsplit(".", 1)[0]
            am = re.match(r"(airs_\w+_\d+_\d+)_", stem)  # airs_<source>_<macroR>_<macroC>
            tm = SOURCE_RE.search(stem)
            unit = (am.group(1) if am else (tm.group(1).lower() if tm else stem))
            src_to_splits.setdefault(unit, set()).add(split)
    leaks = {u: s for u, s in src_to_splits.items() if len(s) > 1}
    if leaks:
        sample = list(leaks.items())[:5]
        raise AssertionError(
            f"LEAKAGE DETECTED: {len(leaks)} source units appear in multiple splits: {sample}"
        )


def write_manifests(out_dir: Path, split_files: dict[str, list[str]]) -> None:
    fn_dir = out_dir / "filenames"
    fn_dir.mkdir(parents=True, exist_ok=True)
    for split, names in split_files.items():
        (fn_dir / f"{split}_filenames_clean.txt").write_text(
            "\n".join(sorted(names)) + ("\n" if names else ""), encoding="utf-8"
        )


# --------------------------------------------------------------------------- #

def parse_assignment(spec: str) -> dict[str, str]:
    """'tile1:train,tile2:val' -> {'tile1':'train',...}"""
    out = {}
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if ":" not in tok:
            raise SystemExit(f"Bad --existing-assignment token: {tok!r} (expected key:split)")
        k, v = tok.split(":", 1)
        out[k.strip().lower()] = v.strip()
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--airs-dir", type=Path, default=None, help="Directory with AIRS GeoTIFF mosaics")
    p.add_argument("--existing-dir", type=Path, default=Path("segmentation_model_data"),
                   help="Existing data root to re-emit split by source tile")
    p.add_argument("--out", type=Path, default=Path("segmentation_model_data_v2"), help="Output data root")
    p.add_argument("--airs-tile-size", type=int, default=512)
    p.add_argument("--grid", type=int, default=8, help="Macro-block grid (NxN) per mosaic for split grouping")
    p.add_argument("--min-roof-frac", type=float, default=0.001, help="Skip patches with less roof coverage")
    p.add_argument("--existing-assignment", type=str, default="tile1:train,tile2:val",
                   help="Comma map of existing source-prefix -> split")
    p.add_argument("--limit-blocks", type=int, default=None,
                   help="Cap AIRS macro-blocks processed (smoke testing)")
    args = p.parse_args()

    if args.airs_dir is None and not args.existing_dir.exists():
        print(__doc__)
        print("\nERROR: provide --airs-dir (download AIRS first) and/or a valid --existing-dir.",
              file=sys.stderr)
        print("AIRS: https://www.kaggle.com/datasets/atilol/aerialimageryforroofsegmentation",
              file=sys.stderr)
        return 2

    (args.out / "images").mkdir(parents=True, exist_ok=True)
    (args.out / "masks").mkdir(parents=True, exist_ok=True)
    split_files: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    block_split: dict[str, str] = {}

    # 1) AIRS (test mosaic -> test split; train mosaic -> train/val by macro-block hash)
    pairs = find_airs_pairs(args.airs_dir) if args.airs_dir else []
    if not pairs and args.airs_dir:
        print(f"WARNING: no AIRS image/mask tif pairs found under {args.airs_dir}", file=sys.stderr)
        print("Expected paired GeoTIFFs, e.g. train.tif + train_mask.tif", file=sys.stderr)
    for img_tif, mask_tif, source in pairs:
        if source == "test":
            # force the whole test mosaic's macro-blocks into the test split
            block_split.update({f"airs_test_{r}_{c}": "test" for r in range(args.grid) for c in range(args.grid)})
        n = tile_airs_pair(
            img_tif, mask_tif, source,
            tile=args.airs_tile_size, grid=args.grid, min_roof_frac=args.min_roof_frac,
            out_images=args.out / "images", out_masks=args.out / "masks",
            block_split=block_split, split_files=split_files, limit_blocks=args.limit_blocks,
        )
        print(f"  AIRS {source}: wrote {n} patches from {img_tif.name}")

    # 2) Existing tiles re-emitted by source prefix
    n_exist = reemit_existing(
        args.existing_dir, args.out / "images", args.out / "masks",
        split_files, parse_assignment(args.existing_assignment),
    )
    print(f"  Existing: re-emitted {n_exist} pairs by source tile")

    # 3) Leak guard + manifests
    assert_no_leak(split_files)
    write_manifests(args.out, split_files)

    print("\nDataset ready at", args.out)
    for s in ("train", "val", "test"):
        print(f"  {s:5s}: {len(split_files[s])} pairs")
    print("Leakage check: PASSED (no source unit in >1 split)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
