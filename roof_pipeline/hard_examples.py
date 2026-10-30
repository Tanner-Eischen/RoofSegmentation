"""Mine bounded sampling probabilities using frozen predictions on TRAIN only."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .common import dataset_fingerprint
from .dataset import RoofDataset
from .metrics import evaluate_loader
from .models import load_model

POLICY = {
    "name": "train_error_fraction_uniform_mix_v1",
    "uniform_fraction": 0.5,
    "max_relative_probability": 5.0,
    "threshold": 0.5,
}


def sampling_probabilities(error_fractions: list[float]) -> list[float]:
    """Mix uniform/error-proportional draws, then redistribute mass above 5/N."""
    errors = np.asarray(error_fractions, dtype=np.float64)
    if errors.ndim != 1 or not errors.size:
        raise ValueError("At least one training error fraction is required")
    if not np.all(np.isfinite(errors)) or np.any(errors < 0) or np.any(errors > 1):
        raise ValueError("Error fractions must be finite and in [0, 1]")
    count = errors.size
    total = float(errors.sum())
    if total == 0:
        return np.full(count, 1 / count).tolist()
    probabilities = 0.5 / count + 0.5 * errors / total
    cap = 5.0 / count
    for _ in range(count):
        excess_entries = probabilities > cap
        if not excess_entries.any():
            break
        excess = float((probabilities[excess_entries] - cap).sum())
        probabilities[excess_entries] = cap
        remaining = probabilities < cap
        if not remaining.any():
            raise ValueError("Sampling probability cap has no remaining capacity")
        probabilities[remaining] += excess / int(remaining.sum())
    else:
        raise RuntimeError("Sampling probability redistribution did not converge")
    probabilities /= probabilities.sum()
    return probabilities.tolist()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _training_ids(rows: list[dict]) -> list[str]:
    if not rows or any(
        not isinstance(row, dict) or row.get("split") != "train" for row in rows
    ):
        raise ValueError("Sampling rows must contain training samples only")
    identities = [row.get("sample_id") for row in rows]
    if any(not isinstance(ident, str) or not ident for ident in identities):
        raise ValueError("Training sample IDs must be nonempty strings")
    if len(set(identities)) != len(identities):
        raise ValueError("Duplicate training sample IDs")
    return identities


def load_sampling_weights(
    report_path: Path, rows: list[dict], dataset_sha: str
) -> list[float]:
    """Validate a train-only report and order its probabilities to dataset rows.

    The caller must also match the report's image_size to its training input size.
    The returned probabilities are directly usable as WeightedRandomSampler weights.
    """
    identities = _training_ids(rows)
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise ValueError("Hard-example report must be a JSON object")
    if report.get("split") != "train":
        raise ValueError("Hard-example report must be train-only")
    if report.get("dataset_content_sha256") != dataset_sha:
        raise ValueError("Hard-example dataset fingerprint differs from training")
    if report.get("policy") != POLICY:
        raise ValueError("Unknown hard-example sampling policy")
    samples = report.get("samples")
    if not isinstance(samples, list):
        raise ValueError("Hard-example report is missing its samples")
    report_ids = _training_ids(samples)
    if len(report_ids) != len(identities) or set(report_ids) != set(identities):
        raise ValueError("Hard-example sample IDs differ from training rows")
    probabilities = []
    errors = []
    for sample in samples:
        probability = sample.get("probability")
        error = sample.get("error_fraction")
        if (
            isinstance(probability, bool)
            or not isinstance(probability, (float, int))
            or not math.isfinite(probability)
            or probability <= 0
        ):
            raise ValueError("Sampling probabilities must be finite and positive")
        if isinstance(error, bool) or not isinstance(error, (float, int)):
            raise ValueError("Error fractions must be numeric")
        probabilities.append(float(probability))
        errors.append(float(error))
    if not math.isclose(math.fsum(probabilities), 1.0, rel_tol=0, abs_tol=1e-10):
        raise ValueError("Sampling probabilities must sum to one")
    lower, upper = 0.5 / len(samples), 5.0 / len(samples)
    if any(p < lower - 1e-12 or p > upper + 1e-12 for p in probabilities):
        raise ValueError("Sampling probabilities violate uniform floor or cap")
    expected = sampling_probabilities(errors)
    if not np.allclose(probabilities, expected, rtol=1e-9, atol=1e-12):
        raise ValueError("Sampling probabilities do not match their declared policy")
    by_id = dict(zip(report_ids, probabilities))
    return [by_id[ident] for ident in identities]


def mine_hard_examples(
    data: Path,
    checkpoint: Path,
    out: Path,
    device: str = "cpu",
    batch_size: int = 4,
    workers: int = 0,
) -> dict:
    """Score training tiles only; val/test files are hashed for frozen-data integrity."""
    data, checkpoint, out = Path(data), Path(checkpoint), Path(out)
    if batch_size < 1 or workers < 0:
        raise ValueError("Batch size must be positive and workers nonnegative")
    if out.suffix.lower() == ".csv":
        raise ValueError("JSON report path must differ from its companion CSV path")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config = state["config"]
    dataset_sha = dataset_fingerprint(data)
    if config.get("dataset_content_sha256") != dataset_sha:
        raise ValueError("Checkpoint dataset fingerprint differs from the mining data")
    image_size = int(config["image_size"])
    dataset = RoofDataset(data, "train", size=image_size, augment=False)
    _training_ids(dataset.rows)
    model, loaded_config = load_model(checkpoint, device)
    if loaded_config["image_size"] != image_size:
        raise ValueError("Checkpoint image size changed during loading")
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.device(device).type == "cuda",
    )
    print(f"Mining {len(dataset)} TRAIN tiles at {image_size}px", flush=True)
    aggregate, detail = evaluate_loader(
        model,
        loader,
        torch.device(device),
        threshold=POLICY["threshold"],
        with_rows=True,
    )
    errors = []
    for item in detail:
        total = sum(item[key] for key in ("tp", "fp", "fn", "tn"))
        if total <= 0:
            raise ValueError("A training tile has no scored pixels")
        errors.append((item["fp"] + item["fn"]) / total)
    probabilities = sampling_probabilities(errors)
    if len(detail) != len(dataset):
        raise ValueError("Mining did not score every training tile exactly once")
    samples = []
    for index, (item, error, probability) in enumerate(
        zip(detail, errors, probabilities)
    ):
        if item["index"] != index:
            raise ValueError("Mining loader returned an unexpected sample order")
        samples.append(
            {
                "sample_id": dataset.rows[index]["sample_id"],
                "split": "train",
                "error_fraction": error,
                "probability": probability,
                **{key: item[key] for key in ("tp", "fp", "fn", "tn")},
            }
        )
    report = {
        "split": "train",
        "dataset_content_sha256": dataset_sha,
        "checkpoint_sha256": _sha256(checkpoint),
        "image_size": image_size,
        "policy": dict(POLICY),
        "aggregate_train_metrics": aggregate,
        "samples": samples,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    load_sampling_weights(out, dataset.rows, dataset_sha)
    csv_path = out.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(samples[0]))
        writer.writeheader()
        writer.writerows(samples)
    print(f"Saved train-only sampling report: {out}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    mine_hard_examples(
        args.data, args.checkpoint, args.out, args.device, args.batch_size, args.workers
    )


if __name__ == "__main__":
    main()
