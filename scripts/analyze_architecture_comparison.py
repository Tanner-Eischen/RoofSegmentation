#!/usr/bin/env python3
"""Summarize the completed three-model, seed-42 WHU batch-16 pilot.

Reads JSON only: no checkpoint loading, dataset access, GPU work or inference.
Matplotlib (Agg backend) is required only when rendering figures.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

ARCHITECTURES = ("unet-resnet34", "unetplusplus-resnet34", "segformer-b0")
LABELS = ("U-Net / ResNet-34", "U-Net++ / ResNet-34", "SegFormer-B0")
COLORS = ("#1775a8", "#bd530a", "#488c35")
OFFICIAL_COUNTS = {"train": 4736, "val": 1036, "test": 2416}
TIMING_NOTE = (
    "Recorded epoch times include training and per-epoch validation and exclude "
    "startup audits, final calibration and checkpoint reassessment. Early U-Net "
    "training overlapped a subsequently cancelled batch-32 probe, so actual epoch "
    "times are not a clean speed comparison. Use the separate batch-16 runtime "
    "profile for throughput; short-probe rates are not end-to-end training rates."
)
LIMITATIONS = (
    "One seed per architecture (n=1, seed 42): seed variability and confidence intervals are not estimable.",
    "Validation-selected exploratory results on existing WHU geography; no new test inference or independent geographic confirmation.",
    "The primary score is global validation IoU at threshold 0.50; calibration is a secondary result on the same selected checkpoint and validation data.",
    "Architecture packages differ in encoder family, capacity and initialization; these comparisons do not isolate attention or any individual architectural mechanism.",
    "WHU masks describe binary building extent, not roof material, damage, pitch or individual roof instances.",
    "This utility verifies JSON consistency only; checkpoint tensors, dataset contents and remote execution must be verified separately by the training pipeline.",
)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def number(value, label, upper=None):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
        or (upper is not None and value > upper)
    ):
        raise ValueError(f"Invalid {label}: {value!r}")
    return float(value)


def equal(actual, expected, label, tolerance=1e-10):
    actual = number(actual, label)
    if not math.isclose(actual, expected, abs_tol=tolerance, rel_tol=0):
        raise ValueError(f"{label} differs from history/assessment")


def load_results(results: Path):
    """Consume only the exact run identities listed in comparison.json."""
    results = results.resolve()
    plan = read_json(results / "plan.json")
    comparison = read_json(results / "comparison.json")
    fixed = plan["settings"]
    if (
        set(plan["architectures"]) != set(ARCHITECTURES)
        or len(plan["architectures"]) != 3
        or plan["seeds"] != [42]
        or fixed["epochs"] != 40
        or fixed["batch_size"] != 16
        or fixed["size"] != 512
        or fixed["primary_threshold"] != 0.5
        or fixed["min_delta"] != 0
        or plan["dataset"]["split_counts"] != OFFICIAL_COUNTS
        or plan.get("test_inference") is not False
    ):
        raise ValueError(
            "Expected the three-model, seed-42, 40-epoch batch-16 WHU pilot"
        )
    plan_sha = sha256(results / "plan.json")
    if (
        comparison.get("plan_sha256") != plan_sha
        or comparison.get("test_inference") is not False
    ):
        raise ValueError("Comparison is not bound to this validation-only plan")
    records = comparison["runs"]
    identities = [(item["architecture"], item["seed"]) for item in records]
    if len(identities) != 3 or set(identities) != {
        (arch, 42) for arch in ARCHITECTURES
    }:
        raise ValueError("Comparison must contain exactly the three registered runs")
    profile = read_json(results / "runtime_estimate.json")
    if (
        profile.get("plan_sha256") != plan_sha
        or profile["probe"]["batch_size"] != 16
        or profile["probe"]["size"] != 512
        or set(profile["architectures"]) != set(ARCHITECTURES)
    ):
        raise ValueError("Runtime profile differs from batch-16 plan")
    sources = {
        name: sha256(results / name)
        for name in ("plan.json", "comparison.json", "runtime_estimate.json")
    }
    rows, histories = [], {}
    by_arch = {item["architecture"]: item for item in records}
    for arch in ARCHITECTURES:
        record = by_arch[arch]
        relative = Path("runs") / f"{arch}_seed42"
        run = results / relative
        names = (
            "history.json",
            "training_summary.json",
            "experiment.json",
            "validation_best.json",
        )
        history, summary, experiment, assessment = [
            read_json(run / name) for name in names
        ]
        sources.update(
            {(relative / name).as_posix(): sha256(run / name) for name in names}
        )
        if assessment != record:
            raise ValueError(f"{arch}: comparison and validation assessment differ")
        expected_args = {
            key: fixed[key]
            for key in (
                "epochs",
                "size",
                "batch_size",
                "workers",
                "device",
                "lr",
                "weight_decay",
                "pos_weight",
                "patience",
                "min_delta",
                "photometric",
                "boundary_weight",
            )
        }
        expected_args.update(
            {
                "arch": arch,
                "seed": 42,
                "pretrained": True,
                "no_amp": False,
                "init_checkpoint": None,
                "sampling_report": None,
                "uniform_replacement": False,
                "training_review": None,
                "expected_encoder_sha256": plan["initializations"][arch][
                    "encoder_sha256"
                ],
                "data": plan["dataset"]["path"],
            }
        )
        if any(
            experiment["arguments"].get(key) != value
            for key, value in expected_args.items()
        ):
            raise ValueError(f"{arch}: training arguments differ from common plan")
        expected_config = {
            "arch": arch,
            "seed": 42,
            "image_size": 512,
            "normalization": "imagenet_rgb_v1",
            "dataset_content_sha256": plan["dataset"]["content_sha256"],
            "dataset_manifest_sha256": plan["dataset"]["manifest_sha256"],
            "effective_training_content_sha256": plan["dataset"]["content_sha256"],
            "pretrained_encoder": True,
            "encoder_initialization_sha256": plan["initializations"][arch][
                "encoder_sha256"
            ],
            "sampling_report_sha256": None,
            "uniform_replacement": False,
            "photometric_profile": "standard",
            "boundary_weight": 0.0,
            "boundary_supervision": False,
            "training_review_content_sha256": None,
        }
        for config in (experiment["config"], summary["config"]):
            if (
                any(config.get(key) != value for key, value in expected_config.items())
                or config.get("initial_checkpoint_sha256") is not None
            ):
                raise ValueError(f"{arch}: checkpoint/data/training identity differs")
        if [entry.get("epoch") for entry in history] != list(range(1, 41)):
            raise ValueError(f"{arch}: incomplete or unordered 40-epoch history")
        scores = [
            number(entry["val_iou_0p5"], "validation IoU", 1) for entry in history
        ]
        for entry in history:
            number(entry["train_loss"], "training loss")
            number(entry["seconds"], "epoch timing")
            if entry["train_draws"] != 4736 or entry["unique_train_samples"] != 4736:
                raise ValueError(f"{arch}: incomplete uniform training coverage")
        best_index = max(range(40), key=scores.__getitem__)
        best_epoch = best_index + 1
        if (
            summary["best_epoch"] != best_epoch
            or summary["config"]["epoch"] != best_epoch
            or record["best_epoch"] != best_epoch
            or summary["final_epoch"] != 40
            or summary.get("test_evaluated") is not False
            or summary["train_samples"] != 4736
            or summary["val_samples"] != 1036
            or record["samples"] != 1036
            or record["split"] != "val"
            or record.get("test_inference") is not False
        ):
            raise ValueError(
                f"{arch}: selected epoch, sample coverage or evaluation policy differs"
            )
        for value in (summary["val_iou_at_0p5"], record["primary_validation_iou_0p5"]):
            equal(value, scores[best_index], f"{arch} selected IoU")
        equal(summary["final_val_iou_at_0p5"], scores[-1], f"{arch} final IoU")
        if record["primary_0p5"]["threshold"] != 0.5:
            raise ValueError(f"{arch}: primary threshold differs")
        equal(
            record["primary_0p5"]["aggregate"]["iou"],
            scores[best_index],
            f"{arch} reassessed primary IoU",
            1e-6,
        )
        threshold = record["secondary_validation_threshold"]
        if (
            not any(
                math.isclose(
                    number(threshold, "calibrated threshold", 1),
                    0.15 + i * 0.05,
                    abs_tol=1e-12,
                )
                for i in range(15)
            )
            or threshold != summary["val_tuned_threshold"]
            or threshold != record["secondary_calibrated"]["threshold"]
        ):
            raise ValueError(f"{arch}: secondary threshold differs")
        secondary = number(
            record["secondary_calibrated_validation_iou"], "calibrated IoU", 1
        )
        equal(summary["val_metrics_tuned"]["iou"], secondary, f"{arch} calibrated IoU")
        equal(
            record["secondary_calibrated"]["aggregate"]["iou"],
            secondary,
            f"{arch} reassessed calibrated IoU",
            1e-6,
        )
        if (
            not isinstance(record["checkpoint_sha256"], str)
            or len(record["checkpoint_sha256"]) != 64
        ):
            raise ValueError(f"{arch}: missing checkpoint identity")
        epoch_minutes = sum(entry["seconds"] for entry in history) / 60
        equal(
            record["epoch_minutes"],
            epoch_minutes,
            f"{arch} recorded epoch minutes",
            1e-6,
        )
        timing = profile["architectures"][arch]
        train_step = number(
            timing["train_step_seconds"]["median"], "profile train step"
        )
        val_step = number(
            timing["validation_step_seconds"]["median"], "profile validation step"
        )
        if train_step == 0 or val_step == 0:
            raise ValueError("Profile step durations must be positive")
        if timing["encoder_sha256"] != plan["initializations"][arch]["encoder_sha256"]:
            raise ValueError(f"{arch}: profile encoder differs")
        rows.append(
            {
                "architecture": arch,
                "seed": 42,
                "best_epoch": best_epoch,
                "primary_validation_iou_0p5": scores[best_index],
                "secondary_validation_threshold": threshold,
                "secondary_calibrated_validation_iou": secondary,
                "recorded_epoch_minutes": epoch_minutes,
                "profile_median_train_step_seconds": train_step,
                "profile_train_images_per_second": 16 / train_step,
                "profile_median_validation_step_seconds": val_step,
                "profile_validation_images_per_second": 16 / val_step,
                "checkpoint_sha256": record["checkpoint_sha256"],
            }
        )
        histories[arch] = history
    baseline = rows[0]["primary_validation_iou_0p5"]
    for row in rows:
        row["primary_delta_vs_unet_iou_percentage_points"] = 100 * (
            row["primary_validation_iou_0p5"] - baseline
        )
    return {
        "schema_version": 1,
        "study": "WHU three-architecture exploratory pilot",
        "results_root": str(results),
        "seeds_per_architecture": 1,
        "seed": 42,
        "epochs": 40,
        "batch_size": 16,
        "image_size": 512,
        "test_inference": False,
        "dataset": plan["dataset"],
        "source_json_sha256": sources,
        "rows": rows,
        "profile_device": profile.get("device"),
        "timing_note": TIMING_NOTE,
        "limitations": list(LIMITATIONS),
    }, histories


def report_text(summary):
    lines = [
        "# WHU architecture comparison: exploratory pilot (n=1 per model)",
        "",
        "Seed 42; 40 epochs; common batch size 16; 512px inputs; unchanged publisher geographic partitions.",
        "Primary: validation-best checkpoint, global pixel IoU at threshold 0.50 (earliest tied epoch).",
        "Secondary: validation-only threshold calibration of that same checkpoint.",
        "",
        "| Architecture | Selected epoch | Primary IoU @ 0.50 | Delta vs U-Net (IoU pp) | Secondary threshold | Secondary calibrated IoU |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["rows"]:
        lines.append(
            f"| {row['architecture']} | {row['best_epoch']} | {row['primary_validation_iou_0p5']:.6f} | {row['primary_delta_vs_unet_iou_percentage_points']:+.3f} | {row['secondary_validation_threshold']:.2f} | {row['secondary_calibrated_validation_iou']:.6f} |"
        )
    lines += [
        "",
        "## Separate batch-16 profiling",
        "",
        TIMING_NOTE,
        "",
        f"Profile device: {summary['profile_device']}. Rates derive from median batch time, without confidence intervals.",
        "",
        "| Architecture | Median train batch (s) | Train images/s | Median validation batch (s) | Validation images/s |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["rows"]:
        lines.append(
            f"| {row['architecture']} | {row['profile_median_train_step_seconds']:.4f} | {row['profile_train_images_per_second']:.1f} | {row['profile_median_validation_step_seconds']:.4f} | {row['profile_validation_images_per_second']:.1f} |"
        )
    lines += [
        "",
        "## Interpretation and verification limits",
        "",
        *[f"- {note}" for note in LIMITATIONS],
        "",
        "Curves: `learning_curves.png` (300 dpi) and `learning_curves.svg`. Circles mark the validation-selected epoch on both panels; they do not select the minimum training loss.",
        "",
        "`summary.json` records the exact input JSON hashes, primary/secondary endpoints and observed epoch timing separately from the runtime profile.",
        "",
    ]
    return "\n".join(lines)


def render_curves(summary, histories, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    with plt.rc_context(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
        }
    ):
        figure, axes = plt.subplots(1, 2, figsize=(12, 5.3), constrained_layout=True)
        for arch, label, color in zip(ARCHITECTURES, LABELS, COLORS):
            history = histories[arch]
            row = next(item for item in summary["rows"] if item["architecture"] == arch)
            epoch = row["best_epoch"]
            for axis, key in zip(axes, ("train_loss", "val_iou_0p5")):
                axis.plot(
                    [item["epoch"] for item in history],
                    [item[key] for item in history],
                    color=color,
                    linewidth=1.7,
                    label=f"{label} (selected epoch {epoch})",
                )
                axis.scatter(
                    [epoch],
                    [history[epoch - 1][key]],
                    color=color,
                    edgecolor="white",
                    linewidth=0.8,
                    s=65,
                    zorder=4,
                )
        for axis in axes:
            axis.set_xlabel("Epoch")
            axis.set_xlim(1, 40)
            axis.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))
            axis.grid(axis="y", alpha=0.22)
        axes[0].set_ylabel("Training BCE + soft Dice loss")
        axes[1].set_ylabel("Global validation IoU at threshold 0.50")
        axes[0].set_title("Training objective")
        axes[1].set_title("Primary checkpoint-selection metric")
        axes[1].legend(loc="lower right", frameon=False, fontsize=8)
        figure.suptitle(
            "WHU architecture pilot: one seed per model (42), batch 16, 40 epochs",
            fontsize=12,
        )
        for extension in ("png", "svg"):
            figure.savefig(
                out / f"learning_curves.{extension}",
                dpi=300,
                metadata={"Creator": "RoofSegmentation architecture analysis"},
            )
            if extension == "svg":
                path = out / "learning_curves.svg"
                path.write_text(
                    "\n".join(line.rstrip() for line in path.read_text().splitlines())
                    + "\n",
                    encoding="utf-8",
                )
        plt.close(figure)


def generate(results, out):
    summary, histories = load_results(results)
    out = out.resolve()
    # Never overwrite training artifacts when a caller accidentally passes a run root.
    if out == results.resolve() or any(
        (out / name).exists()
        for name in ("plan.json", "comparison.json", "history.json")
    ):
        raise ValueError("Choose a separate analysis output directory")
    out.mkdir(parents=True, exist_ok=True)
    render_curves(summary, histories, out)
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (out / "report.md").write_text(report_text(summary), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        required=True,
        help="Extracted root containing plan.json and complete comparison.json",
    )
    parser.add_argument(
        "--out", type=Path, required=True, help="Analysis output directory"
    )
    args = parser.parse_args()
    generate(args.results, args.out)
    print(
        f"Wrote verified JSON summary, report and PNG/SVG curves to {args.out.resolve()}"
    )


if __name__ == "__main__":
    main()
