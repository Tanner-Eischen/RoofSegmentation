#!/usr/bin/env python
"""
Prepare the Christchurch/AIRS pixel-mask dataset into the flat images+masks+
manifests layout dataset.py expects, written to <out> (v3).

Input is a zip (or extracted dir) of pre-split GeoTIFF tiles:
    train/{image,label}/*.tif   val/{image,label}/*.tif   test/{image,label}/*.tif
    train.txt / val.txt           (tile-name lists defining the split)
Each tile is ~10000x10000 (7.5 cm). Labels are 0/1 (a *_vis.tif 0/255 twin exists
too); we normalize to 0/255 so dataset.py's >127 threshold works.

- Reads tiles directly from the zip via rasterio MemoryFile (no 18 GiB extraction).
- Tiles each into <tile-size> patches, drops near-empty patches (< min-roof-frac).
- Preserves the dataset's own train/val/test split (each tile's patches stay in
  that tile's split -> no cross-split leak).
- Optionally caps per-split patches (--max-patches) for practical laptop training.
- Merges your existing true-outline masks into train (--existing-assignment).

Usage:
    python scripts/prepare_airs_tiled.py \\
        --zip "C:/Users/tanne/Downloads/archive.zip" \\
        --out segmentation_model_data_v3 \\
        --existing-dir segmentation_model_data \\
        --existing-assignment tile1:train,tile2:train
"""

from __future__ import annotations

import argparse
import random
import re
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image
from rasterio.io import MemoryFile
from rasterio.windows import Window
from tqdm import tqdm

SOURCE_RE = re.compile(r"(tile\d+)_", re.IGNORECASE)
TILE_RE = re.compile(r"(christchurch_\d+)")


def list_tiles(zf: zipfile.ZipFile, split: str) -> list[str]:
    """Stems for a split: from <split>.txt if present, else the image/ dir."""
    txt = f"{split}.txt"
    if txt in zf.namelist():
        names = [line.strip() for line in zf.read(txt).decode("utf-8", "replace").splitlines() if line.strip()]
        return [n.replace(".tif", "").replace("\\", "/").split("/")[-1] for n in names]
    prefix = f"{split}/image/"
    return [n[len(prefix):-4] for n in zf.namelist() if n.startswith(prefix) and n.endswith(".tif")]


def process_tile(zf, split, stem, tile_size, min_roof_frac, out_images, out_masks) -> list[str]:
    """Tile one image+label into kept patches. Returns output filenames."""
    img_name, lbl_name = f"{split}/image/{stem}.tif", f"{split}/label/{stem}.tif"
    if img_name not in zf.namelist() or lbl_name not in zf.namelist():
        return []
    img_bytes, lbl_bytes = zf.read(img_name), zf.read(lbl_name)
    out = []
    with MemoryFile(img_bytes) as mfi, mfi.open() as src, MemoryFile(lbl_bytes) as mfl, mfl.open() as msk:
        H, W = src.height, src.width
        for r in range(0, H, tile_size):
            for c in range(0, W, tile_size):
                h, w = min(tile_size, H - r), min(tile_size, W - c)
                if h != tile_size or w != tile_size:
                    continue  # keep only full patches; skip partial edges
                mw = msk.read(1, window=Window(c, r, w, h))
                if float((mw > 0).mean()) < min_roof_frac:
                    continue
                iw = src.read(window=Window(c, r, w, h))  # (bands,h,w)
                img = np.transpose(iw[:3], (1, 2, 0))
                if img.dtype != np.uint8:
                    img = np.clip(img, 0, 255).astype(np.uint8)
                mask = np.where(mw > 0, 255, 0).astype(np.uint8)
                stem_name = f"{split}_{stem}_{r}_{c}"
                Image.fromarray(img).save(out_images / f"{stem_name}.jpg", format="JPEG", quality=95)
                Image.fromarray(mask).save(out_masks / f"{stem_name}.png")
                out.append(f"{stem_name}.jpg")
    return out


def reemit_existing(existing_dir, out_images, out_masks, split_files, assignment) -> int:
    images, masks = existing_dir / "images", existing_dir / "masks"
    if not images.exists():
        return 0
    n = 0
    for img_path in sorted(images.iterdir()):
        if img_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".tif"}:
            continue
        mp = masks / img_path.name
        if not mp.exists():
            continue
        m = SOURCE_RE.search(img_path.name)
        split = assignment.get(m.group(1).lower() if m else "", "train")
        shutil.copy2(img_path, out_images / img_path.name)
        shutil.copy2(mp, out_masks / img_path.name)
        split_files[split].append(img_path.name)
        n += 1
    return n


def assert_no_leak(split_files) -> None:
    unit_splits: dict[str, set[str]] = {}
    for split, names in split_files.items():
        for name in names:
            stem = name.rsplit(".", 1)[0]
            cm = TILE_RE.search(stem)
            tm = SOURCE_RE.search(stem)
            unit = cm.group(1) if cm else (tm.group(1).lower() if tm else stem)
            unit_splits.setdefault(unit, set()).add(split)
    leaks = {u: s for u, s in unit_splits.items() if len(s) > 1}
    if leaks:
        raise AssertionError(f"LEAKAGE DETECTED: {list(leaks.items())[:5]}")


def write_manifests(out_dir, split_files) -> None:
    d = out_dir / "filenames"
    d.mkdir(parents=True, exist_ok=True)
    for split, names in split_files.items():
        (d / f"{split}_filenames_clean.txt").write_text(
            "\n".join(sorted(names)) + ("\n" if names else ""), encoding="utf-8")


def parse_assignment(spec: str) -> dict[str, str]:
    out = {}
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if ":" not in tok:
            raise SystemExit(f"bad --existing-assignment token: {tok!r}")
        k, v = tok.split(":", 1)
        out[k.strip().lower()] = v.strip()
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--zip", type=Path, default=Path(r"C:\Users\tanne\Downloads\archive.zip"))
    p.add_argument("--out", type=Path, default=Path("segmentation_model_data_v3"))
    p.add_argument("--tile-size", type=int, default=512)
    p.add_argument("--min-roof-frac", type=float, default=0.001, help="Drop patches below this roof coverage")
    p.add_argument("--max-patches", type=int, default=8000, help="TRAIN cap (0 = no cap). ~120k train available.")
    p.add_argument("--max-eval-patches", type=int, default=2000, help="VAL/TEST cap (fast per-epoch eval). 0 = no cap.")
    p.add_argument("--existing-dir", type=Path, default=Path("segmentation_model_data"))
    p.add_argument("--existing-assignment", type=str, default="tile1:train,tile2:train")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--splits", type=str, default="train,val,test")
    args = p.parse_args()

    if not args.zip.exists():
        print(f"ERROR: zip not found: {args.zip}", file=sys.stderr)
        return 2
    random.seed(args.seed)
    zf = zipfile.ZipFile(args.zip)
    (args.out / "images").mkdir(parents=True, exist_ok=True)
    (args.out / "masks").mkdir(parents=True, exist_ok=True)
    split_files = {"train": [], "val": [], "test": []}

    for split in [s.strip() for s in args.splits.split(",")]:
        internal = "val" if split in ("val", "valid") else split
        stems = list_tiles(zf, internal)
        print(f"[{split}] {len(stems)} tiles in zip")
        kept: list[str] = []
        for stem in tqdm(stems, desc=f"tile {split}"):
            kept += process_tile(zf, internal, stem, args.tile_size, args.min_roof_frac,
                                 args.out / "images", args.out / "masks")
        pre = len(kept)
        cap = args.max_patches if split == "train" else args.max_eval_patches
        if cap and pre > cap:
            kept = random.sample(kept, cap)
            print(f"[{split}] capped {pre} -> {cap} patches")
        split_files[split] = kept
        print(f"[{split}] kept {len(kept)} patches")

    n_exist = reemit_existing(args.existing_dir, args.out / "images", args.out / "masks",
                              split_files, parse_assignment(args.existing_assignment))
    print(f"existing: merged {n_exist} true-outline pairs into train")

    assert_no_leak(split_files)
    write_manifests(args.out, split_files)
    print("\nDataset ready at", args.out)
    for s in ("train", "val", "test"):
        print(f"  {s:5s}: {len(split_files[s])} pairs")
    print("Leakage check: PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
