"""Held-out test evaluation using the frozen validation-selected threshold.

Do not use test results to choose the model, augmentation, or threshold.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .common import dataset_fingerprint
from .dataset import RoofDataset
from .metrics import evaluate_loader
from .models import load_model


def evaluate(
    checkpoint: Path,
    data: Path,
    out: Path,
    split: str = "test",
    batch_size: int = 4,
    device: str = "cpu",
):
    model, config = load_model(checkpoint, device)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    manifest_sha = hashlib.sha256((data / "manifest.csv").read_bytes()).hexdigest()
    if manifest_sha != config["dataset_manifest_sha256"]:
        raise ValueError(
            "Dataset manifest SHA differs from training. Explicitly version dataset before evaluating."
        )
    if config.get("dataset_content_sha256") != dataset_fingerprint(data):
        raise ValueError(
            "Dataset content SHA differs from training or is missing. Images/masks must be frozen with the checkpoint."
        )
    ds = RoofDataset(data, split, size=int(config["image_size"]), augment=False)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    threshold = float(state["threshold"])
    aggregate, detail = evaluate_loader(
        model, loader, torch.device(device), threshold=threshold, with_rows=True
    )
    for row in detail:
        item = ds.rows[row["index"]]
        row["sample_id"] = item["sample_id"]
        row["group"] = item["group"]
    out.mkdir(parents=True, exist_ok=True)
    result = {
        "split": split,
        "checkpoint": str(checkpoint),
        "samples": len(ds),
        "threshold_frozen_from_validation": threshold,
        "aggregate": aggregate,
        "dataset_content_sha256": config["dataset_content_sha256"],
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "note": "No threshold sweep or checkpoint selection was performed on test",
    }
    (out / f"{split}_metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    with (out / f"{split}_per_image.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(detail[0]))
        writer.writeheader()
        writer.writerows(detail)
    print(json.dumps(result, indent=2))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--split", choices=["train", "val", "test"], default="test")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = p.parse_args()
    evaluate(
        args.checkpoint, args.data, args.out, args.split, args.batch_size, args.device
    )


if __name__ == "__main__":
    main()
