"""Tiny synthetic image+mask fixture for checking pipeline wiring ONLY.
Not suitable for any scientific IoU claim or portfolio model-performance figure.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from roof_pipeline.common import save_pair, write_manifest


def make_fixture(
    out: Path,
    count_train: int = 12,
    count_val: int = 4,
    count_test: int = 4,
    size: int = 96,
):
    rng = np.random.default_rng(14)
    rows = []
    for split, count in (
        ("train", count_train),
        ("val", count_val),
        ("test", count_test),
    ):
        for i in range(count):
            # Different seeds per split, distinct backgrounds and spatial group IDs.
            base = np.full((size, size, 3), [81, 99, 75], dtype=np.uint8)
            noise = rng.normal(0, 9, size=(size, size, 3))
            arr = np.clip(base.astype(float) + noise, 0, 255).astype(np.uint8)
            image = Image.fromarray(arr)
            draw = ImageDraw.Draw(image)
            mask = Image.new("L", (size, size), 0)
            mdraw = ImageDraw.Draw(mask)
            # A few rectilinear synthetic roofs.
            for roof in range(2 + (i % 3)):
                width = int(rng.integers(12, 22))
                height = int(rng.integers(9, 20))
                x = int(rng.integers(5, size - width - 5))
                y = int(rng.integers(5, size - height - 5))
                box = [x, y, x + width, y + height]
                shade = int(rng.integers(126, 210))
                draw.rectangle(box, fill=(shade, shade - 14, shade - 18))
                mdraw.rectangle(box, fill=255)
            sid = f"synthetic_{split}_{i:03d}"
            rows.append(
                save_pair(
                    out,
                    image,
                    np.array(mask),
                    sid,
                    split,
                    group=f"synthetic_site_{split}_{i // 4}",
                    source="artificial_smoke_only",
                )
            )
    write_manifest(
        out,
        rows,
        {
            "dataset": "Synthetic 96px roof-like fixtures",
            "warning": "Only for testing code paths; never report model accuracy from this data",
        },
    )
    print(f"Wrote {len(rows)} synthetic image/mask pairs in {out}")
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("smoke_data"))
    p.add_argument("--train", type=int, default=12)
    p.add_argument("--val", type=int, default=4)
    p.add_argument("--test", type=int, default=4)
    p.add_argument("--size", type=int, default=96)
    a = p.parse_args()
    make_fixture(a.out, a.train, a.val, a.test, a.size)
