#!/usr/bin/env python3
"""Analyze complete RoofSegmentation 2x2 factorial arms (standard library only).

Per-run global pixel metrics are observations. Never treat individual pixels or
adjacent tiles as independent experimental replicates. No p-values are reported.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

ARMS = {
    "a0b0_uniform_standard": (0, 0),
    "a1b0_hard_standard": (1, 0),
    "a0b1_uniform_robust": (0, 1),
    "a1b1_hard_robust": (1, 1),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def finite_number(value, label: str, minimum=0.0, maximum=None) -> float:
    number = float(value)
    if (
        not math.isfinite(number)
        or number < minimum
        or (maximum is not None and number > maximum)
    ):
        raise ValueError(f"Invalid {label}: {value}")
    return number


def effects(values: dict[tuple[int, int], float]) -> dict[str, float]:
    """Factorial main effects and interaction in IoU fraction (not percentage points)."""
    c = values[0, 0]
    s = values[1, 0]
    a = values[0, 1]
    both = values[1, 1]
    return {
        "sampling": ((s - c) + (both - a)) / 2,
        "augmentation": ((a - c) + (both - s)) / 2,
        "interaction": both - s - a + c,
        "combined_vs_control": both - c,
    }


def read_results(out: Path):
    registration_file = out / "registration.json"
    if not registration_file.is_file():
        raise ValueError("Missing registration.json")
    reg = json.loads(registration_file.read_text(encoding="utf-8"))
    if reg.get("phase") != "exploratory_validation_only":
        raise ValueError("Unexpected study registration phase")
    seeds = reg.get("seeds", [])
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Registration must contain distinct seeds")
    mining_path = out / "hard_examples.json"
    if not mining_path.is_file():
        raise ValueError("Missing frozen mining report")
    mining_sha = sha256(mining_path)
    mining = json.loads(mining_path.read_text(encoding="utf-8"))
    if (
        mining.get("split") != "train"
        or mining.get("checkpoint_sha256") != reg["checkpoint_sha256"]
        or mining.get("dataset_content_sha256") != reg["dataset_content_sha256"]
        or mining.get("image_size") != reg["image_size"]
    ):
        raise ValueError("Mining report differs from registration")
    frozen = json.loads((out / "mining_fingerprint.json").read_text(encoding="utf-8"))
    if frozen != {"sha256": mining_sha}:
        raise ValueError("Mining report changed after freezing")
    rows = [read_run(out, reg, seed, arm) for seed in seeds for arm in ARMS]
    common_ids = None
    for row in rows:
        run = out / "runs" / f"{row['arm']}_seed{row['seed']}"
        with (run / "eval_final_val/val_per_image.csv").open(
            newline="", encoding="utf-8"
        ) as stream:
            identities = {item["sample_id"] for item in csv.DictReader(stream)}
        if common_ids is None:
            common_ids = identities
        elif identities != common_ids:
            raise ValueError(f"Different validation tiles across study arms: {run}")
    return reg, rows


def read_run(out: Path, reg: dict, seed: int, arm: str, registration_sha=None):
    """Verify one complete run, also used by the runner before skipping on resume."""
    out = out.resolve()
    factors = ARMS[arm]
    registration_sha = registration_sha or sha256(out / "registration.json")
    mining_path = out / "hard_examples.json"
    mining_sha = sha256(mining_path)
    frozen = json.loads((out / "mining_fingerprint.json").read_text(encoding="utf-8"))
    if frozen != {"sha256": mining_sha}:
        raise ValueError("Mining report changed after freezing")
    run = out / "runs" / f"{arm}_seed{seed}"
    required = [
        run / name
        for name in (
            "registration.json",
            "training_summary.json",
            "history.json",
            "experiment.json",
            "final.pt",
            "best.pt",
            "eval_final_val/val_metrics.json",
            "eval_final_val/val_per_image.csv",
        )
    ]
    if not all(path.is_file() for path in required):
        raise ValueError(f"Incomplete study arm: {run}")
    armreg = json.loads((run / "registration.json").read_text(encoding="utf-8"))
    expected_armreg = {
        "arm": arm,
        "seed": seed,
        "hard": bool(factors[0]),
        "photometric": "robust" if factors[1] else "standard",
        "study_registration_sha256": registration_sha,
        "epochs": reg["epochs"],
    }
    if armreg != expected_armreg:
        raise ValueError(f"Arm preregistration mismatch: {run}")
    a = json.loads((run / "training_summary.json").read_text(encoding="utf-8"))
    b = json.loads(
        (run / "eval_final_val/val_metrics.json").read_text(encoding="utf-8")
    )
    experiment = json.loads((run / "experiment.json").read_text(encoding="utf-8"))
    config = a["config"]
    expected_config = {
        "arch": reg["arch"],
        "image_size": reg["image_size"],
        "seed": seed,
        "normalization": "imagenet_rgb_v1",
        "dataset_manifest_sha256": reg["data_manifest_sha256"],
        "dataset_content_sha256": reg["dataset_content_sha256"],
        "effective_training_content_sha256": reg["dataset_content_sha256"],
        "initial_checkpoint_sha256": reg["checkpoint_sha256"],
        "photometric_profile": expected_armreg["photometric"],
        "uniform_replacement": not bool(factors[0]),
        "sampling_report_sha256": mining_sha if factors[0] else None,
        "boundary_supervision": False,
        "boundary_weight": 0.0,
        "training_review_content_sha256": None,
        "optimizer_restarted": True,
    }
    if any(config.get(key) != value for key, value in expected_config.items()):
        raise ValueError(
            f"Training config differs from registered factors/inputs: {run}"
        )
    if any(
        experiment["config"].get(key) != value for key, value in expected_config.items()
    ):
        raise ValueError(f"Experiment config differs from training summary: {run}")
    expected_args = {
        "arch": reg["arch"],
        "size": reg["image_size"],
        "seed": seed,
        "epochs": reg["epochs"],
        "patience": reg["epochs"] + 1,
        "batch_size": reg["batch_size"],
        "workers": reg["workers"],
        "device": reg["device"],
        "lr": reg["learning_rate"],
        "weight_decay": reg["weight_decay"],
        "pos_weight": reg["positive_weight"],
        "min_delta": reg["min_delta"],
        "boundary_weight": 0.0,
        "photometric": expected_armreg["photometric"],
        "uniform_replacement": not bool(factors[0]),
        "training_review": None,
        "no_amp": False,
        "pretrained": False,
        "init_checkpoint": reg["checkpoint"],
        "data": reg["data"],
        "sampling_report": str(mining_path.resolve()) if factors[0] else None,
    }
    if any(
        experiment["arguments"].get(key) != value
        for key, value in expected_args.items()
    ):
        raise ValueError(f"Training arguments differ from registration: {run}")
    history = json.loads((run / "history.json").read_text(encoding="utf-8"))
    if [h.get("epoch") for h in history] != list(range(1, reg["epochs"] + 1)):
        raise ValueError(f"Incomplete fixed-epoch history: {run}")
    train_minutes = (
        sum(finite_number(h["seconds"], "epoch seconds") for h in history) / 60
    )
    actual = finite_number(b["aggregate"]["iou"], "validation IoU", maximum=1)
    recorded = finite_number(a["final_val_iou_at_0p5"], "training IoU", maximum=1)
    endpoint = finite_number(history[-1]["val_iou_0p5"], "endpoint IoU", maximum=1)
    if abs(actual - recorded) > 0.000001 or abs(recorded - endpoint) > 1e-10:
        raise ValueError(f"IoU mismatch between final checkpoint and training: {run}")
    if (
        b.get("split") != "val"
        or b.get("threshold_frozen_from_validation") != 0.5
        or a.get("final_threshold") != 0.5
        or a.get("test_evaluated") is not False
    ):
        raise ValueError(f"Non-frozen final validation evaluation: {run}")
    if b.get("checkpoint_sha256") != sha256(run / "final.pt"):
        raise ValueError(f"Evaluated checkpoint is not current final.pt: {run}")
    if b["dataset_content_sha256"] != reg["dataset_content_sha256"]:
        raise ValueError(f"Dataset differs from registration: {run}")
    if a["final_epoch"] != reg["epochs"]:
        raise ValueError(f"Full fixed training budget not completed: {run}")
    with (run / "eval_final_val/val_per_image.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        detail = list(csv.DictReader(stream))
    identities = [r["sample_id"] for r in detail]
    if (
        not identities
        or len(set(identities)) != len(identities)
        or len(detail) != b["samples"]
        or b["samples"] != a["val_samples"]
        or b["samples"] != reg["split_counts"]["val"]
    ):
        raise ValueError(f"Incomplete or duplicate validation tiles: {run}")
    for key in ("tp", "fp", "fn", "tn"):
        total = sum(finite_number(r[key], key) for r in detail)
        if total != finite_number(b["aggregate"][key], key):
            raise ValueError(f"Per-tile counts disagree with aggregate {key}: {run}")
    counts = b["aggregate"]
    denominator = counts["tp"] + counts["fp"] + counts["fn"]
    counts_iou = counts["tp"] / denominator if denominator else 0.0
    if abs(counts_iou - actual) > 1e-10:
        raise ValueError(f"IoU disagrees with pixel counts: {run}")
    return {
        "seed": seed,
        "arm": arm,
        "hard_sampling": factors[0],
        "robust_augmentation": factors[1],
        "val_iou": actual,
        "val_dice": finite_number(b["aggregate"]["dice"], "Dice", maximum=1),
        "val_precision": finite_number(
            b["aggregate"]["precision"], "precision", maximum=1
        ),
        "val_recall": finite_number(b["aggregate"]["recall"], "recall", maximum=1),
        "train_minutes": train_minutes,
        "final_epoch": a["final_epoch"],
        "checkpoint": str(run / "final.pt"),
    }


def summaries(rows):
    grouped = defaultdict(list)
    for r in rows:
        grouped[r["arm"]].append(r["val_iou"])
    arm_summary = [
        {
            "arm": arm,
            "mean_val_iou": statistics.mean(grouped[arm]),
            "min_val_iou": min(grouped[arm]),
            "max_val_iou": max(grouped[arm]),
            "seed_sd": statistics.stdev(grouped[arm]) if len(grouped[arm]) > 1 else 0.0,
            "mean_train_minutes": statistics.mean(
                r["train_minutes"] for r in rows if r["arm"] == arm
            ),
        }
        for arm in ARMS
    ]
    by_seed = defaultdict(dict)
    for r in rows:
        by_seed[r["seed"]][ARMS[r["arm"]]] = r["val_iou"]
    contrasts = [{"seed": seed, **effects(v)} for seed, v in sorted(by_seed.items())]
    contrast_summary = [
        {
            "contrast": metric,
            "mean_effect_iou": statistics.mean(row[metric] for row in contrasts),
            "min_effect_iou": min(row[metric] for row in contrasts),
            "max_effect_iou": max(row[metric] for row in contrasts),
            "seed_sd": statistics.stdev(row[metric] for row in contrasts)
            if len(contrasts) > 1
            else 0.0,
        }
        for metric in ("sampling", "augmentation", "interaction", "combined_vs_control")
    ]
    return arm_summary, contrasts, contrast_summary


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def report_text(reg, arm_summary, contrast_summary, rows):
    text = [
        "# 2x2 roof/building segmentation ablation: exploratory validation report",
        "",
        "**Status:** Aggregation of actual completed model runs. NOT independent test confirmation.",
        "",
        f"Model: {reg['arch']} | Dataset: {reg['data']} | Seeds: {', '.join(map(str, reg['seeds']))}",
        f"Fixed endpoint: {reg['epochs']} training epochs, 512px, threshold 0.50.",
        "",
        "## Results by arm",
        "",
        "| Arm | Mean validation IoU | Min | Max | SD across seeds | Mean train minutes |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in arm_summary:
        text.append(
            f"| {r['arm']} | {r['mean_val_iou']:.5f} | {r['min_val_iou']:.5f} | {r['max_val_iou']:.5f} | {r['seed_sd']:.5f} | {r['mean_train_minutes']:.1f} |"
        )
    text += [
        "",
        "## Factorial effects (percentage points of IoU)",
        "",
        "| Contrast | Mean | Minimum seed | Maximum seed | SD across seeds |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for r in contrast_summary:
        text.append(
            f"| {r['contrast']} | {r['mean_effect_iou'] * 100:+.3f} | {r['min_effect_iou'] * 100:+.3f} | {r['max_effect_iou'] * 100:+.3f} | {r['seed_sd'] * 100:.3f} |"
        )
    text += [
        "",
        "Training minutes include epoch fitting and per-epoch validation; exclude audits, mining, final evaluation, and threshold calibration.",
        "",
        "## Interpretation restrictions",
        "",
        "- Each seed is one replicate. This is a small exploratory experiment, not a hypothesis-significance claim.",
        "- WHU's prepared `group` covers the whole validation partition. Adjacent tiles cannot be treated as independent spatial replicates.",
        "- The original WHU test was previously inspected. These validation contrasts do not establish out-of-domain generalization.",
        "- The primary endpoint uses the FINAL trained checkpoint at threshold 0.50, not the highest selected checkpoint.",
        "- Confirm on a genuinely new, labeled geographic/sensor holdout with matching label ontology before broad claims.",
        "- Do not describe WHU building-extent masks as material type, roof damage, roof pitch, or independent roof instances.",
        "",
        "## Data files",
        "",
        "- `runs.csv`",
        "- `contrasts_by_seed.csv`",
        "- `contrasts_summary.csv`",
        "- `arms_summary.csv`",
        "",
    ]
    return "\n".join(text)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True, help="Existing study output root")
    args = p.parse_args()
    out = args.out.resolve()
    reg, rows = read_results(out)
    arm_summary, contrasts, contrast_summary = summaries(rows)
    dest = out / "analysis"
    dest.mkdir(parents=True, exist_ok=True)
    write_csv(dest / "runs.csv", rows)
    write_csv(dest / "arms_summary.csv", arm_summary)
    write_csv(dest / "contrasts_by_seed.csv", contrasts)
    write_csv(dest / "contrasts_summary.csv", contrast_summary)
    (dest / "report.md").write_text(
        report_text(reg, arm_summary, contrast_summary, rows) + "\n", encoding="utf-8"
    )
    print(f"Wrote {dest / 'report.md'} from {len(rows)} completed validation arms")


if __name__ == "__main__":
    main()
