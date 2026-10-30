"""Shared on-disk format and numerical metrics for roof binary segmentation."""

from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

FIELDS = ("sample_id", "split", "image", "mask", "group", "source")
SPLITS = ("train", "val", "test")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"}


def read_manifest(
    root: Path, *, require_all_splits: bool = True
) -> list[dict[str, str]]:
    path = root / "manifest.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing manifest: {path}")
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = set(FIELDS) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing columns in manifest: {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError("Manifest is empty")
    groups: dict[str, str] = {}
    ids: set[str] = set()
    counts = Counter()
    for row in rows:
        ident = row["sample_id"]
        split = row["split"]
        group = row["group"]
        if not ident or ident in ids:
            raise ValueError(f"Duplicate or blank sample_id: {ident!r}")
        ids.add(ident)
        if split not in SPLITS:
            raise ValueError(f"Invalid split {split!r} for {ident}")
        if not group:
            raise ValueError(f"No group for {ident}; supply a spatial/source group")
        previous = groups.setdefault(group, split)
        if previous != split:
            raise ValueError(f"LEAKAGE: group {group!r} in both {previous} and {split}")
        for field in ("image", "mask"):
            rel = Path(row[field])
            if rel.is_absolute() or ".." in rel.parts:
                raise ValueError(f"Unsafe {field} path for {ident}")
            if not (root / rel).is_file():
                raise FileNotFoundError(f"Missing {field} for {ident}: {root / rel}")
        counts[split] += 1
    if require_all_splits:
        empty = [s for s in SPLITS if not counts[s]]
        if empty:
            raise ValueError(f"Missing required splits: {empty}")
    return rows


def write_manifest(
    root: Path, rows: list[dict[str, str]], meta: dict | None = None
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / "manifest.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(
            sorted(rows, key=lambda x: (SPLITS.index(x["split"]), x["sample_id"]))
        )
    if meta:
        (root / "dataset_meta.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )
    read_manifest(root)
    return path


def load_binary_mask(
    path: Path, expected_size: tuple[int, int] | None = None
) -> np.ndarray:
    with Image.open(path) as img:
        if img.mode not in ("L", "1", "P"):
            raise ValueError(f"Mask must be single-channel: {path} (found {img.mode})")
        # Palette indices are class IDs; palette colors are only for display.
        arr = np.asarray(img if img.mode == "P" else img.convert("L"))
        size = img.size
    if expected_size is not None and size != expected_size:
        raise ValueError(f"Mask/image size mismatch: {path}, {size} vs {expected_size}")
    values = set(int(x) for x in np.unique(arr))
    if not values.issubset({0, 1, 255}):
        raise ValueError(
            f"Mask {path} contains non-binary values {sorted(values)[:20]}. Do not interpolate labels."
        )
    return np.where(arr > 0, 255, 0).astype(np.uint8)


def save_pair(
    root: Path,
    image: Image.Image | np.ndarray,
    mask: np.ndarray,
    sample_id: str,
    split: str,
    group: str,
    source: str,
) -> dict[str, str]:
    img = (
        Image.fromarray(image).convert("RGB")
        if isinstance(image, np.ndarray)
        else image.convert("RGB")
    )
    if mask.shape != (img.height, img.width):
        raise ValueError(f"Mask and image shape differ for {sample_id}")
    if not set(np.unique(mask)).issubset({0, 1, 255}):
        raise ValueError(f"Non-binary mask for {sample_id}")
    image_rel = f"images/{sample_id}.png"
    mask_rel = f"masks/{sample_id}.png"
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "masks").mkdir(parents=True, exist_ok=True)
    img.save(root / image_rel, compress_level=5)
    Image.fromarray(np.where(mask > 0, 255, 0).astype(np.uint8)).save(root / mask_rel)
    return dict(
        sample_id=sample_id,
        split=split,
        image=image_rel,
        mask=mask_rel,
        group=group,
        source=source,
    )


def dataset_audit(
    root: Path, preview_count: int = 4, verify_hashes: bool = True
) -> dict:
    rows = read_manifest(root)
    summary: dict = {
        "counts": dict(Counter(r["split"] for r in rows)),
        "by_source": {},
        "empty_masks_by_split": {},
        "roof_fraction_by_split": {},
        "duplicate_images": 0,
        "groups_by_split": {},
        "total": len(rows),
    }
    for split in SPLITS:
        sub = [r for r in rows if r["split"] == split]
        summary["groups_by_split"][split] = len(set(r["group"] for r in sub))
        fractions = []
        empty = 0
        for r in sub:
            with Image.open(root / r["image"]) as im:
                size = im.size
                if im.mode not in ("RGB", "RGBA", "L"):
                    raise ValueError(
                        f"Unexpected image mode for {r['sample_id']}: {im.mode}"
                    )
            mask = load_binary_mask(root / r["mask"], size)
            fraction = float((mask > 0).mean())
            fractions.append(fraction)
            empty += fraction == 0
        summary["empty_masks_by_split"][split] = empty
        summary["roof_fraction_by_split"][split] = {
            "mean": float(np.mean(fractions)),
            "median": float(np.median(fractions)),
            "min": float(np.min(fractions)),
            "max": float(np.max(fractions)),
        }
    summary["by_source"] = dict(Counter(r["source"] for r in rows))
    if verify_hashes:
        seen = {}
        duplicate = []
        for r in rows:
            # Decoded pixels catch lossless re-encoding, not lossy JPEG copies.
            with Image.open(root / r["image"]) as im:
                rgb = np.asarray(im.convert("RGB"))
                h = hashlib.sha256(rgb.tobytes()).hexdigest()
            if h in seen:
                other = seen[h]
                if other["split"] != r["split"]:
                    raise ValueError(
                        f"LEAKAGE: identical image across splits: {other['sample_id']}, {r['sample_id']}"
                    )
                duplicate.append((other["sample_id"], r["sample_id"]))
            seen[h] = r
        summary["duplicate_images"] = len(duplicate)
    (root / "audit.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if preview_count > 0:
        rng = random.Random(41)
        for split in SPLITS:
            sub = [r for r in rows if r["split"] == split]
            samples = rng.sample(sub, min(preview_count, len(sub)))
            if not samples:
                continue
            out = Image.new("RGB", (512 * 2, 256 * len(samples)), (22, 24, 30))
            draw = ImageDraw.Draw(out)
            for i, r in enumerate(samples):
                with Image.open(root / r["image"]) as im:
                    img = im.convert("RGB").resize((256, 256))
                mask = Image.fromarray(
                    load_binary_mask(root / r["mask"]).astype(np.uint8)
                ).resize((256, 256), Image.Resampling.NEAREST)
                overlay = Image.new("RGB", img.size, (255, 40, 60))
                overlay = Image.composite(overlay, img, mask)
                overlay = Image.blend(img, overlay, alpha=0.45)
                out.paste(img, (0, i * 256))
                out.paste(overlay, (256, i * 256))
                draw.text((515, i * 256 + 8), r["sample_id"], fill=(250, 250, 250))
            # crop to useful width, only first two columns needed
            out.crop((0, 0, 512, 256 * len(samples))).save(
                root / f"preview_{split}.png"
            )
    return summary


def dataset_fingerprint(root: Path) -> str:
    """Bind a run to the manifest and the bytes of every paired image and mask."""
    rows = read_manifest(root)
    digest = hashlib.sha256()
    digest.update((root / "manifest.csv").read_bytes())
    metadata = root / "dataset_meta.json"
    if metadata.is_file():
        digest.update(metadata.read_bytes())
    for row in sorted(rows, key=lambda item: item["sample_id"]):
        for field in ("image", "mask"):
            relative = row[field]
            file_digest = hashlib.sha256()
            with (root / relative).open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    file_digest.update(chunk)
            digest.update(
                json.dumps(
                    [row["sample_id"], field, relative], ensure_ascii=True
                ).encode("utf-8")
            )
            digest.update(file_digest.digest())
    return digest.hexdigest()
