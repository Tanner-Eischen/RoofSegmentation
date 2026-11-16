"""Explicit post-study test inference for the three validation-best checkpoints.

The existing training/validation artifacts stay immutable. This reused WHU test
split already informed project development; results are descriptive follow-up,
not new model selection or independent geographic confirmation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

COUNTS = ("tp", "fp", "fn", "tn")
METRICS = ("iou", "dice", "precision", "recall", "f1", "pixel_accuracy")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def packet_checksums(packet):
    write_json(
        packet / "checksums.json",
        {
            path.name: sha256(path)
            for path in sorted(packet.iterdir())
            if path.is_file() and path.name != "checksums.json"
        },
    )


def render_comparison(comparison, packet):
    """Plot paired validation/test endpoints, not model-selection evidence."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from scripts.analyze_architecture_comparison import LABELS

    models = comparison["models"]
    with plt.rc_context({"font.size": 10, "svg.fonttype": "none"}):
        figure, axis = plt.subplots(figsize=(8.5, 5), constrained_layout=True)
        for index, row in enumerate(models):
            axis.plot(
                [index, index],
                [row["validation_iou"], row["test"]["iou"]],
                color="#999999",
                linewidth=1.5,
            )
        for label, color, values in (
            ("Validation", "#1775a8", [row["validation_iou"] for row in models]),
            ("Reused test", "#bd530a", [row["test"]["iou"] for row in models]),
        ):
            axis.scatter(range(3), values, color=color, s=65, label=label, zorder=3)
            for index, value in enumerate(values):
                axis.annotate(
                    f"{value:.6f}",
                    (index, value),
                    xytext=(8, 0),
                    textcoords="offset points",
                    fontsize=9,
                )
        axis.set_xticks(range(3), LABELS)
        axis.set_xlim(-0.4, 2.65)
        axis.set_ylabel("Global building IoU at threshold 0.50")
        axis.grid(axis="y", alpha=0.25)
        axis.legend(frameon=False, loc="lower left")
        axis.set_title("WHU validation-best checkpoints: separate test follow-up")
        figure.suptitle(
            "One seed (42) per model; 40 epochs; training batch 16\n"
            "Test previously inspected during development; descriptive results",
            fontsize=10,
        )
        for extension in ("png", "svg"):
            path = packet / f"comparison.{extension}"
            figure.savefig(path, dpi=250)
            if extension == "svg":
                path.write_text(
                    "\n".join(
                        line.rstrip()
                        for line in path.read_text(encoding="utf-8").splitlines()
                    )
                    + "\n",
                    encoding="utf-8",
                )
        plt.close(figure)


def verify_test_result(result, rows, expected_ids, checkpoint_sha, dataset_sha):
    """Reconcile exact coverage and global metrics against per-image counts."""
    from roof_pipeline.metrics import metrics_from_counts

    ids = [row["sample_id"] for row in rows]
    if (
        len(ids) != len(expected_ids)
        or len(set(ids)) != len(ids)
        or set(ids) != set(expected_ids)
    ):
        raise ValueError("Test inference must cover the exact registered test IDs once")
    if (
        result.get("split") != "test"
        or result.get("samples") != len(expected_ids)
        or result.get("threshold_frozen_from_validation") != 0.5
        or result.get("checkpoint_sha256") != checkpoint_sha
        or result.get("dataset_content_sha256") != dataset_sha
    ):
        raise ValueError(
            "Test checkpoint, split, threshold or dataset identity differs"
        )
    totals = dict.fromkeys(COUNTS, 0)
    for row in rows:
        values = {}
        for key in COUNTS:
            value = float(row[key])
            if not math.isfinite(value) or value < 0 or not value.is_integer():
                raise ValueError(
                    "Per-image confusion counts must be nonnegative integers"
                )
            values[key] = int(value)
        if sum(values.values()) != 512 * 512:
            raise ValueError("Per-image confusion counts do not cover 512x512 pixels")
        for key in COUNTS:
            totals[key] += values[key]
    recomputed = metrics_from_counts(*(totals[key] for key in COUNTS))
    for key in (*COUNTS, *METRICS):
        value = result["aggregate"][key]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not math.isclose(value, recomputed[key], abs_tol=1e-12, rel_tol=0)
        ):
            raise ValueError(f"Aggregate {key} differs from the per-image counts")
    return recomputed


def checkpoint_contract(path, row, dataset):
    import torch

    if sha256(path) != row["checkpoint_sha256"]:
        raise ValueError("Checkpoint bytes differ from the validation-best assessment")
    state = torch.load(path, map_location="cpu", weights_only=True)
    config = state["config"]
    expected = {
        "arch": row["architecture"],
        "seed": 42,
        "epoch": row["best_epoch"],
        "image_size": 512,
        "normalization": "imagenet_rgb_v1",
        "dataset_content_sha256": dataset["content_sha256"],
        "dataset_manifest_sha256": dataset["manifest_sha256"],
    }
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError("Checkpoint selection/configuration differs from validation")
    # evaluate() uses this stored threshold. Never alter weights or tune on test.
    if state.get("threshold") != 0.5 or row["secondary_validation_threshold"] != 0.5:
        raise ValueError("This follow-up requires the already frozen threshold 0.50")


def original_files(results):
    """Include all completed training/validation artifacts, including checkpoints."""
    return {
        path.relative_to(results).as_posix(): sha256(path)
        for path in sorted(results.rglob("*"))
        if path.is_file()
    }


def run(results, data, out, packet, *, device="cuda", batch_size=4):
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    import torch

    from roof_pipeline.common import dataset_fingerprint, read_manifest
    from roof_pipeline.evaluate import evaluate
    from scripts.analyze_architecture_comparison import LABELS, load_results

    results, data, out, packet = (
        Path(p).resolve() for p in (results, data, out, packet)
    )
    if out.exists() or packet.exists():
        raise FileExistsError("Use fresh inference and published-packet directories")
    if any(path == results or path.is_relative_to(results) for path in (out, packet)):
        raise ValueError("Test follow-up must not overwrite original study artifacts")
    if any(path == data or path.is_relative_to(data) for path in (out, packet)):
        raise ValueError("Test follow-up must not write into the registered dataset")
    if batch_size < 1 or device not in ("cuda", "cpu"):
        raise ValueError("Use a positive inference batch and an explicit device")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; no silent CPU fallback")
    summary, _ = load_results(results)
    frozen_data = summary["dataset"]
    if (
        sha256(data / "manifest.csv") != frozen_data["manifest_sha256"]
        or dataset_fingerprint(data) != frozen_data["content_sha256"]
    ):
        raise ValueError("WHU imagery, labels or geographic assignments changed")
    test_ids = [
        row["sample_id"] for row in read_manifest(data) if row["split"] == "test"
    ]
    if len(test_ids) != 2416 or len(set(test_ids)) != 2416:
        raise ValueError("Expected the complete 2,416-tile publisher test split")
    protected = original_files(results)
    for row in summary["rows"]:
        checkpoint_contract(
            results / "runs" / f"{row['architecture']}_seed42" / "best.pt",
            row,
            frozen_data,
        )
    out.mkdir(parents=True)
    packet.mkdir(parents=True)
    registration = {
        "schema_version": 1,
        "registered_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Explicitly authorized descriptive post-study test comparison",
        "checkpoint_selection": "Original validation-best at 0.50, earliest tie; unchanged",
        "validation_selected_architecture": max(
            summary["rows"], key=lambda row: row["primary_validation_iou_0p5"]
        )["architecture"],
        "split": "test",
        "samples": 2416,
        "test_ids_sha256": hashlib.sha256(
            "\n".join(sorted(test_ids)).encode()
        ).hexdigest(),
        "dataset": {k: v for k, v in frozen_data.items() if k != "path"},
        "threshold": 0.5,
        "inference_batch_size": batch_size,
        "training_batch_size": 16,
        "training_seed": 42,
        "epochs_each": 40,
        "device": device,
        "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
        "precision": "float32, no autocast",
        "environment": {
            "python": platform.python_version(),
            **{
                name: importlib.metadata.version(name)
                for name in ("torch", "numpy", "Pillow", "segmentation-models-pytorch")
            },
        },
        "training_comparison_sha256": summary["source_json_sha256"]["comparison.json"],
        "test_use_note": "The official WHU test has already informed project development. This is a reused-test descriptive follow-up, not pristine holdout evidence or independent geographic confirmation.",
        "test_selection_used": False,
        "registered_checkpoints": [
            {
                k: row[k]
                for k in (
                    "architecture",
                    "seed",
                    "best_epoch",
                    "checkpoint_sha256",
                    "primary_validation_iou_0p5",
                )
            }
            for row in summary["rows"]
        ],
    }
    write_json(out / "registration.json", registration)
    write_json(out / "original_artifact_hashes.json", protected)
    write_json(packet / "registration.json", registration)
    rows = []
    for row in summary["rows"]:
        arch = row["architecture"]
        checkpoint = results / "runs" / f"{arch}_seed42" / "best.pt"
        destination = out / arch
        print(
            f"Evaluating {arch}: validation-best epoch {row['best_epoch']}, test 2416 tiles, threshold 0.50",
            flush=True,
        )
        started = time.monotonic()
        result = evaluate(checkpoint, data, destination, "test", batch_size, device)
        elapsed = time.monotonic() - started
        with (destination / "test_per_image.csv").open(
            newline="", encoding="utf-8"
        ) as stream:
            details = list(csv.DictReader(stream))
        aggregate = verify_test_result(
            result,
            details,
            test_ids,
            row["checkpoint_sha256"],
            frozen_data["content_sha256"],
        )
        if sha256(checkpoint) != row["checkpoint_sha256"]:
            raise ValueError("Inference modified its selected checkpoint")
        public = dict(
            result,
            architecture=arch,
            seed=42,
            best_epoch=row["best_epoch"],
            elapsed_seconds=elapsed,
        )
        public["checkpoint"] = f"registered-validation-best:{row['checkpoint_sha256']}"
        write_json(packet / f"{arch}_metrics.json", public)
        (packet / f"{arch}_per_image.csv").write_bytes(
            (destination / "test_per_image.csv").read_bytes()
        )
        rows.append(
            {
                "architecture": arch,
                "seed": 42,
                "best_epoch": row["best_epoch"],
                "checkpoint_sha256": row["checkpoint_sha256"],
                "samples": 2416,
                "threshold": 0.5,
                "validation_iou": row["primary_validation_iou_0p5"],
                "test": aggregate,
                "test_minus_validation_iou_pp": 100
                * (aggregate["iou"] - row["primary_validation_iou_0p5"]),
                "inference_seconds": elapsed,
            }
        )
        write_json(
            out / "progress.json",
            {"completed_models": len(rows), "models": rows, "phase": "evaluating"},
        )
        if device == "cuda":
            torch.cuda.empty_cache()
    if original_files(results) != protected:
        raise ValueError(
            "Original training or validation artifacts changed during test follow-up"
        )
    if dataset_fingerprint(data) != frozen_data["content_sha256"]:
        raise ValueError("Dataset changed during inference")
    comparison = {
        **registration,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "models": rows,
        "original_artifacts_unchanged": True,
        "per_image_counts_verified": True,
    }
    write_json(out / "comparison.json", comparison)
    write_json(packet / "comparison.json", comparison)
    with (packet / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "architecture",
                "best_epoch",
                "threshold",
                "validation_iou",
                *METRICS,
                *COUNTS,
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "architecture": row["architecture"],
                    "best_epoch": row["best_epoch"],
                    "threshold": 0.5,
                    "validation_iou": row["validation_iou"],
                    **row["test"],
                }
            )
    lines = [
        "# WHU architecture test follow-up",
        "",
        registration["test_use_note"],
        "",
        "All three original validation-best checkpoints; 2,416 test tiles each; threshold 0.50. No retraining, checkpoint reselection or threshold sweep.",
        "",
        "| Model | Epoch | Validation IoU | Test IoU | Test Dice | Precision | Recall |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row, label in zip(rows, LABELS):
        a = row["test"]
        lines.append(
            f"| {label} | {row['best_epoch']} | {row['validation_iou']:.6f} | {a['iou']:.6f} | {a['dice']:.6f} | {a['precision']:.6f} | {a['recall']:.6f} |"
        )
    lines += [
        "",
        "Original training/validation artifacts and checkpoints verified unchanged. Per-image test counts reconcile with all aggregate metrics.",
        "",
        "Inference used " + registration["gpu"]
        if device == "cuda"
        else "Inference used CPU float32.",
        "",
        "CSV/JSON scores are fractions (0–1). The original validation selection remains the selection criterion; these test scores do not change it.",
        "",
        "[Validation/test comparison figure](comparison.png) | [SVG](comparison.svg).",
        "",
        "The per-model `inference_seconds` fields record evaluator wall time, including checkpoint loading, full-dataset integrity checks, data loading and scoring. They are not pure-GPU inference latency or a controlled throughput benchmark.",
        "",
    ]
    (packet / "README.md").write_text("\n".join(lines), encoding="utf-8")
    render_comparison(comparison, packet)
    packet_checksums(packet)
    write_json(
        out / "progress.json",
        {"completed_models": 3, "phase": "complete", "test_selection_used": False},
    )
    print(
        json.dumps(
            {
                "completed_models": 3,
                "packet": str(packet),
                "original_artifacts_unchanged": True,
            }
        ),
        flush=True,
    )
    return comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    run(
        args.results,
        args.data,
        args.out,
        args.packet,
        device=args.device,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
