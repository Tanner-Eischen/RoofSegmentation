"""Reproducible train/validation annotation review packets; never changes labels."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .common import load_binary_mask, read_manifest

STRATA = ("empty", "positive_lt_1pct", "1_to_10pct", "at_least_10pct")
REVIEW_FIELDS = (
    "sample_id",
    "split",
    "stratum",
    "shifted_outline",
    "omitted_building",
    "greenhouse",
    "container_trailer",
    "notes",
    "reviewer",
    "review_status",
)


def coverage_stratum(fraction: float) -> str:
    if fraction == 0:
        return STRATA[0]
    if fraction < 0.01:
        return STRATA[1]
    if fraction < 0.1:
        return STRATA[2]
    return STRATA[3]


def mask_outline(mask: np.ndarray) -> np.ndarray:
    """One-pixel interior contour, including shapes cropped at tile edges."""
    foreground = mask > 0
    padded = np.pad(foreground, 1, constant_values=False)
    interior = foreground.copy()
    for dy, dx in ((0, 1), (2, 1), (1, 0), (1, 2)):
        interior &= padded[dy : dy + mask.shape[0], dx : dx + mask.shape[1]]
    return foreground & ~interior


def select_samples(root: Path, per_stratum: int = 4, seed: int = 20261029):
    if per_stratum < 1:
        raise ValueError("per_stratum must be positive")
    buckets = defaultdict(list)
    for row in read_manifest(root):
        if row["split"] not in {"train", "val"}:
            continue
        fraction = float((load_binary_mask(root / row["mask"]) > 0).mean())
        stratum = coverage_stratum(fraction)
        buckets[(row["split"], stratum)].append(
            dict(row, coverage=fraction, stratum=stratum)
        )
    rng = random.Random(seed)
    selected, population = [], []
    for split in ("train", "val"):
        for stratum in STRATA:
            rows = sorted(buckets[(split, stratum)], key=lambda row: row["sample_id"])
            count = min(per_stratum, len(rows))
            population.append(
                dict(split=split, stratum=stratum, population=len(rows), sampled=count)
            )
            for row in sorted(
                rng.sample(rows, count), key=lambda row: row["sample_id"]
            ):
                selected.append(dict(row, inclusion_probability=count / len(rows)))
    return selected, population


def make_packet(root: Path, out: Path, per_stratum: int = 4, seed: int = 20261029):
    root, out = Path(root), Path(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to replace annotation review packet: {out}")
    selected, population = select_samples(root, per_stratum, seed)
    (out / "panels").mkdir(parents=True, exist_ok=True)
    groups = defaultdict(list)
    for row in selected:
        with Image.open(root / row["image"]) as source:
            image = source.convert("RGB")
        mask = load_binary_mask(root / row["mask"], image.size)
        rgb = np.asarray(image).copy()
        rgb[mask_outline(mask)] = (255, 0, 255)
        panel = Image.new("RGB", (image.width * 2, image.height + 28), (25, 25, 25))
        ImageDraw.Draw(panel).text(
            (6, 6),
            f"{row['sample_id']} | {row['stratum']} | RGB / magenta GT contour",
            fill="white",
        )
        panel.paste(image, (0, 28))
        panel.paste(Image.fromarray(rgb), (image.width, 28))
        relative = f"panels/{row['sample_id']}.png"
        panel.save(out / relative)
        row["panel"] = relative
        for field in ("image", "mask"):
            row[f"{field}_sha256"] = hashlib.sha256(
                (root / row[field]).read_bytes()
            ).hexdigest()
        groups[(row["split"], row["stratum"])].append(panel)
    for (split, stratum), panels in groups.items():
        width = max(panel.width for panel in panels)
        sheet = Image.new("RGB", (width, sum(panel.height for panel in panels)))
        y = 0
        for panel in panels:
            sheet.paste(panel, (0, y))
            y += panel.height
        sheet.save(out / f"{split}_{stratum}.png")
    packet = dict(
        seed=seed,
        per_stratum=per_stratum,
        dataset_root=str(root.resolve()),
        manifest_sha256=hashlib.sha256(
            (root / "manifest.csv").read_bytes()
        ).hexdigest(),
        design="Equal-allocation stratified random sample of train/val by GT tile coverage; no test imagery reviewed.",
        limitations="Coverage is not object type. Unweighted review counts are not population prevalence. Aerial object identity may remain uncertain.",
        population=population,
        samples=selected,
    )
    (out / "packet.json").write_text(json.dumps(packet, indent=2), encoding="utf-8")
    with (out / "review.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=REVIEW_FIELDS)
        writer.writeheader()
        for row in selected:
            writer.writerow(
                {
                    **{key: row[key] for key in ("sample_id", "split", "stratum")},
                    "review_status": "pending",
                }
            )
    return packet


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--per-stratum", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261029)
    args = parser.parse_args()
    packet = make_packet(args.data, args.out, args.per_stratum, args.seed)
    print(
        json.dumps(
            {"sampled": len(packet["samples"]), "population": packet["population"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
