"""JSON-only scientific guards and reproducible figures for the one-seed pilot."""

from __future__ import annotations

import copy
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts/analyze_architecture_comparison.py"
)
spec = importlib.util.spec_from_file_location("architecture_analysis", SCRIPT)
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def results(tmp_path):
    root = tmp_path / "results"
    fixed = {
        "epochs": 40,
        "size": 512,
        "batch_size": 16,
        "workers": 4,
        "device": "cuda",
        "lr": 0.0003,
        "weight_decay": 0.0001,
        "pos_weight": 1.0,
        "patience": 41,
        "min_delta": 0.0,
        "photometric": "standard",
        "boundary_weight": 0.0,
        "primary_threshold": 0.5,
    }
    plan = {
        "architectures": list(analysis.ARCHITECTURES),
        "seeds": [42],
        "settings": fixed,
        "dataset": {
            "path": "/fixture/data",
            "content_sha256": "data",
            "manifest_sha256": "manifest",
            "split_counts": analysis.OFFICIAL_COUNTS,
        },
        "initializations": {
            arch: {"encoder_sha256": f"encoder-{arch}"}
            for arch in analysis.ARCHITECTURES
        },
        "test_inference": False,
    }
    save(root / "plan.json", plan)
    records = []
    for index, arch in enumerate(analysis.ARCHITECTURES):
        history = []
        for epoch in range(1, 41):
            # Epochs 30 and 31 tie; selection must retain 30.
            score = 0.5 + min(epoch, 30) * 0.01 + index * 0.005
            if epoch > 31:
                score -= 0.01
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": 0.5 / epoch,
                    "val_iou_0p5": score,
                    "seconds": 60.0,
                    "train_draws": 4736,
                    "unique_train_samples": 4736,
                }
            )
        config = {
            "arch": arch,
            "seed": 42,
            "image_size": 512,
            "normalization": "imagenet_rgb_v1",
            "dataset_content_sha256": "data",
            "dataset_manifest_sha256": "manifest",
            "effective_training_content_sha256": "data",
            "pretrained_encoder": True,
            "encoder_initialization_sha256": f"encoder-{arch}",
            "sampling_report_sha256": None,
            "uniform_replacement": False,
            "photometric_profile": "standard",
            "boundary_weight": 0.0,
            "boundary_supervision": False,
            "training_review_content_sha256": None,
        }
        arguments = {
            key: value for key, value in fixed.items() if key != "primary_threshold"
        }
        arguments.update(
            {
                "arch": arch,
                "seed": 42,
                "pretrained": True,
                "no_amp": False,
                "init_checkpoint": None,
                "sampling_report": None,
                "uniform_replacement": False,
                "training_review": None,
                "data": "/fixture/data",
                "expected_encoder_sha256": f"encoder-{arch}",
            }
        )
        primary = history[29]["val_iou_0p5"]
        secondary = primary + 0.001
        record = {
            "architecture": arch,
            "seed": 42,
            "best_epoch": 30,
            "primary_validation_iou_0p5": primary,
            "secondary_validation_threshold": 0.45,
            "secondary_calibrated_validation_iou": secondary,
            "epoch_minutes": 40.0,
            "checkpoint_sha256": str(index) * 64,
            "samples": 1036,
            "split": "val",
            "test_inference": False,
            "primary_0p5": {"threshold": 0.5, "aggregate": {"iou": primary}},
            "secondary_calibrated": {
                "threshold": 0.45,
                "aggregate": {"iou": secondary},
            },
        }
        summary = {
            "config": dict(config, epoch=30),
            "best_epoch": 30,
            "final_epoch": 40,
            "test_evaluated": False,
            "train_samples": 4736,
            "val_samples": 1036,
            "val_iou_at_0p5": primary,
            "final_val_iou_at_0p5": history[-1]["val_iou_0p5"],
            "val_tuned_threshold": 0.45,
            "val_metrics_tuned": {"iou": secondary},
        }
        run = root / "runs" / f"{arch}_seed42"
        for name, value in (
            ("history.json", history),
            ("training_summary.json", summary),
            ("experiment.json", {"arguments": arguments, "config": config}),
            ("validation_best.json", record),
        ):
            save(run / name, value)
        records.append(record)
    save(
        root / "comparison.json",
        {
            "plan_sha256": analysis.sha256(root / "plan.json"),
            "test_inference": False,
            "runs": records,
        },
    )
    save(
        root / "runtime_estimate.json",
        {
            "plan_sha256": analysis.sha256(root / "plan.json"),
            "device": "fixture GPU",
            "probe": {"batch_size": 16, "size": 512},
            "architectures": {
                arch: {
                    "encoder_sha256": f"encoder-{arch}",
                    "train_step_seconds": {"median": 0.2},
                    "validation_step_seconds": {"median": 0.1},
                }
                for arch in analysis.ARCHITECTURES
            },
        },
    )
    return root


def test_exact_registered_runs_and_distinct_endpoints(results):
    # A supplemental run must never enter the common-batch comparison.
    save(results / "supplemental_batch4_unet/history.json", {"not": "a run"})
    summary, histories = analysis.load_results(results)
    assert len(summary["rows"]) == len(histories) == 3
    assert [row["best_epoch"] for row in summary["rows"]] == [30] * 3
    assert summary["rows"][0]["primary_delta_vs_unet_iou_percentage_points"] == 0
    assert summary["rows"][1][
        "primary_delta_vs_unet_iou_percentage_points"
    ] == pytest.approx(0.5)
    assert summary["rows"][0]["profile_train_images_per_second"] == pytest.approx(80)
    assert (
        summary["rows"][0]["secondary_calibrated_validation_iou"]
        > summary["rows"][0]["primary_validation_iou_0p5"]
    )
    report = analysis.report_text(summary)
    assert "n=1" in report and "cancelled batch-32 probe" in report
    assert "do not isolate attention" in report


@pytest.mark.parametrize(
    "mutation",
    [
        "seed",
        "batch",
        "incomplete",
        "selection",
        "dataset",
        "primary_threshold",
        "nonfinite",
    ],
)
def test_rejects_noncomparable_or_partial_runs(results, mutation):
    run = results / "runs/unet-resnet34_seed42"
    if mutation in {"batch", "seed"}:
        path = run / "experiment.json"
        value = analysis.read_json(path)
        value["arguments"]["batch_size" if mutation == "batch" else "seed"] = (
            4 if mutation == "batch" else 43
        )
    elif mutation in {"incomplete", "nonfinite"}:
        path = run / "history.json"
        value = analysis.read_json(path)
        if mutation == "incomplete":
            value.pop()
        else:
            value[0]["train_loss"] = float("nan")
    else:
        path = run / "training_summary.json"
        value = analysis.read_json(path)
        if mutation == "selection":
            value["best_epoch"] = 31  # Later tied maximum is disallowed.
        elif mutation == "dataset":
            value["config"]["dataset_content_sha256"] = "changed"
        else:
            path = run / "validation_best.json"
            value = analysis.read_json(path)
            value["primary_0p5"]["threshold"] = 0.6
            comparison = analysis.read_json(results / "comparison.json")
            comparison["runs"][0] = copy.deepcopy(value)
            save(results / "comparison.json", comparison)
    save(path, value)
    with pytest.raises(ValueError):
        analysis.load_results(results)


def test_rejects_duplicate_comparison_and_wrong_profile(results):
    path = results / "comparison.json"
    comparison = analysis.read_json(path)
    original = copy.deepcopy(comparison)
    comparison["runs"][1] = comparison["runs"][0]
    save(path, comparison)
    with pytest.raises(ValueError, match="exactly the three"):
        analysis.load_results(results)
    save(path, original)
    path = results / "runtime_estimate.json"
    value = analysis.read_json(path)
    value["probe"]["batch_size"] = 4
    save(path, value)
    with pytest.raises(ValueError, match="Runtime profile"):
        analysis.load_results(results)


def test_cli_renders_cpu_only_artifacts(results, tmp_path):
    pytest.importorskip("matplotlib")
    output = tmp_path / "analysis"
    subprocess.run(
        [sys.executable, str(SCRIPT), "--results", str(results), "--out", str(output)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert (output / "learning_curves.png").read_bytes().startswith(b"\x89PNG")
    assert "<svg" in (output / "learning_curves.svg").read_text(encoding="utf-8")
    assert "selected epoch 30" in (output / "learning_curves.svg").read_text(
        encoding="utf-8"
    )
    assert len(analysis.read_json(output / "summary.json")["rows"]) == 3
    assert (output / "report.md").is_file()
