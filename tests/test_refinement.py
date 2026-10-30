"""Exercise experiment selection and subprocess boundaries without training."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import pytest
import torch

from roof_pipeline import refine
from roof_pipeline.common import dataset_fingerprint
from roof_pipeline.metrics import metrics_from_counts
from roof_pipeline.refine import choose_candidate, run_refinement, valid_cached_test
from scripts.make_smoke_dataset import make_fixture


def test_selection_uses_validation_instead_of_test_results():
    candidates = [
        {"name": "baseline", "validation_iou": 0.9, "test_iou": 0.999},
        {"name": "targeted", "validation_iou": 0.92, "test_iou": 0.5},
        {"name": "control", "validation_iou": 0.91, "test_iou": 0.99},
    ]
    assert choose_candidate(candidates)["name"] == "targeted"


def test_selection_compares_calibrated_scores_when_best_thresholds_differ():
    candidates = [
        {"name": "baseline", "val_iou_at_0p5": 0.9, "validation_iou": 0.91},
        {"name": "targeted", "val_iou_at_0p5": 0.899, "validation_iou": 0.93},
        {"name": "control", "val_iou_at_0p5": 0.92, "validation_iou": 0.92},
    ]
    assert choose_candidate(candidates)["name"] == "targeted"


@pytest.mark.parametrize(
    "targeted,control,winner",
    [
        (0.9, 0.9, "baseline"),
        (0.9001, 0.899, "baseline"),
        (0.92, 0.92, "control"),
        (0.91, 0.92, "control"),
        (0.921, 0.92, "targeted"),
    ],
)
def test_selection_requires_improvement_and_uses_conservative_ties(
    targeted, control, winner
):
    candidates = [
        {"name": "baseline", "validation_iou": 0.9},
        {"name": "targeted", "validation_iou": targeted},
        {"name": "control", "validation_iou": control},
    ]
    assert choose_candidate(candidates)["name"] == winner


@pytest.mark.parametrize("score", [float("nan"), float("inf"), -0.1, 1.1])
def test_selection_rejects_invalid_validation_score(score):
    with pytest.raises(ValueError):
        choose_candidate([{"name": "baseline", "validation_iou": score}])


def _metrics(checkpoint_sha="checkpoint", dataset_sha="dataset", samples=2, iou=0.95):
    return {
        "split": "test",
        "samples": samples,
        "checkpoint_sha256": checkpoint_sha,
        "dataset_content_sha256": dataset_sha,
        "threshold_frozen_from_validation": 0.5,
        "aggregate": metrics_from_counts(iou * 100, (1 - iou) * 100, 0, 100),
    }


@pytest.mark.parametrize(
    "key,value",
    [
        ("split", "val"),
        ("samples", 3),
        ("samples", True),
        ("checkpoint_sha256", "other"),
        ("dataset_content_sha256", "other"),
        ("threshold_frozen_from_validation", 0.6),
        ("threshold_frozen_from_validation", float("nan")),
    ],
)
def test_cache_rejects_nonmatching_frozen_inputs(key, value):
    result = _metrics()
    result[key] = value
    assert not valid_cached_test(
        result,
        checkpoint_sha="checkpoint",
        dataset_sha="dataset",
        threshold=0.5,
        test_samples=2,
    )


def test_cache_rejects_inconsistent_metrics_and_accepts_valid_record():
    result = _metrics()
    assert valid_cached_test(
        result,
        checkpoint_sha="checkpoint",
        dataset_sha="dataset",
        threshold=0.5,
        test_samples=2,
    )
    result["aggregate"]["iou"] = 0.99
    assert not valid_cached_test(
        result,
        checkpoint_sha="checkpoint",
        dataset_sha="dataset",
        threshold=0.5,
        test_samples=2,
    )


def _save_candidate(
    checkpoint, dataset_sha, manifest_sha, score, epoch, initial_sha=None
):
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    config = {
        "arch": "tiny",
        "image_size": 64,
        "normalization": "imagenet_rgb_v1",
        "dataset_content_sha256": dataset_sha,
        "dataset_manifest_sha256": manifest_sha,
        "epoch": epoch,
    }
    if initial_sha is not None:
        config["initial_checkpoint_sha256"] = initial_sha
    torch.save(
        {"config": config, "val_iou_at_0p5": score, "threshold": 0.5}, checkpoint
    )
    (checkpoint.parent / "training_summary.json").write_text(
        json.dumps(
            {
                "config": config,
                "best_epoch": epoch,
                "val_iou_at_0p5": score,
                "val_tuned_threshold": 0.5,
                "val_metrics_tuned": {"iou": score},
            }
        )
    )


def _setup(tmp_path):
    data = make_fixture(tmp_path / "data", 2, 1, 2, 64)
    dataset_sha = dataset_fingerprint(data)
    manifest_sha = hashlib.sha256((data / "manifest.csv").read_bytes()).hexdigest()
    baseline = tmp_path / "baseline/best.pt"
    _save_candidate(baseline, dataset_sha, manifest_sha, 0.9, 32)
    baseline_sha = hashlib.sha256(baseline.read_bytes()).hexdigest()
    (baseline.parent / "test").mkdir()
    baseline_test = _metrics(baseline_sha, dataset_sha, iou=0.89)
    (baseline.parent / "test/test_metrics.json").write_text(json.dumps(baseline_test))
    args = argparse.Namespace(
        data=data,
        baseline=baseline,
        out=tmp_path / "experiment",
        epochs=2,
        workers=0,
        batch_size=2,
        device="cpu",
        seed=42,
        lr=0.0001,
    )
    return args, dataset_sha, manifest_sha, baseline_sha


def _fake_jobs(
    monkeypatch,
    args,
    dataset_sha,
    manifest_sha,
    baseline_sha,
    targeted_score=0.92,
    control_score=0.91,
    test_iou=0.95,
    fail_phase=None,
):
    calls = []

    def fake_run(command, **kwargs):
        module = command[3]
        flags = command[4:]

        def get(flag):
            return flags[flags.index(flag) + 1]

        calls.append(command)
        assert (args.out / "experiment.json").exists(), (
            "Experiment must be registered before subprocesses"
        )
        assert (args.out / "source_snapshot.zip").exists()
        if fail_phase == module:
            raise subprocess.CalledProcessError(1, command)
        if module == "roof_pipeline.hard_examples":
            Path(get("--out")).write_text(json.dumps({"split": "train"}))
        elif module == "roof_pipeline.train":
            run = Path(get("--run"))
            score = targeted_score if run.name == "targeted" else control_score
            _save_candidate(
                run / "best.pt", dataset_sha, manifest_sha, score, 1, baseline_sha
            )
        elif module == "roof_pipeline.evaluate":
            # Both training arms must exist before any test scoring.
            assert (args.out / "targeted/training_summary.json").exists()
            assert (args.out / "control/training_summary.json").exists()
            assert (args.out / "selection.json").exists()
            checkpoint = Path(get("--checkpoint"))
            result = _metrics(
                hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                dataset_sha,
                iou=test_iou,
            )
            output = Path(get("--out"))
            output.mkdir(parents=True)
            (output / "test_metrics.json").write_text(json.dumps(result))
        else:
            pytest.fail(f"Unexpected subprocess {module}")

    monkeypatch.setattr(refine.subprocess, "run", fake_run)
    return calls


@pytest.mark.parametrize("test_iou,reached", [(0.95, False), (0.96, True)])
def test_orchestration_runs_targeted_then_matched_control_then_selected_test(
    tmp_path, monkeypatch, test_iou, reached
):
    args, dataset_sha, manifest_sha, baseline_sha = _setup(tmp_path)
    original_baseline = args.baseline.read_bytes()
    calls = _fake_jobs(
        monkeypatch, args, dataset_sha, manifest_sha, baseline_sha, test_iou=test_iou
    )
    comparison = run_refinement(args)
    assert [call[3] for call in calls] == [
        "roof_pipeline.hard_examples",
        "roof_pipeline.train",
        "roof_pipeline.train",
        "roof_pipeline.evaluate",
    ]
    targeted, control = calls[1], calls[2]
    assert "--sampling-report" in targeted and "robust" in targeted
    assert "--sampling-report" not in control and "standard" in control
    for command in (targeted, control):
        assert command[command.index("--init-checkpoint") + 1] == str(
            args.baseline.resolve()
        )
        assert command[command.index("--epochs") + 1] == "2"
        assert "--no-pretrained" in command
    assert comparison["selected"]["name"] == "targeted"
    assert comparison["target_reached"] is reached
    assert comparison["selection_used_test_metrics"] is False
    assert comparison["baseline_test_iou"] == pytest.approx(0.89)
    status = json.loads((args.out / "workflow_status.json").read_text())
    assert status["phase"] == "complete" and status["test_evaluated"] is True
    assert "DEVELOPMENT benchmark" in (args.out / "report.md").read_text()
    assert args.baseline.read_bytes() == original_baseline


def test_no_validation_improvement_reuses_matching_baseline_test_without_new_scoring(
    tmp_path, monkeypatch
):
    args, dataset_sha, manifest_sha, baseline_sha = _setup(tmp_path)
    calls = _fake_jobs(
        monkeypatch,
        args,
        dataset_sha,
        manifest_sha,
        baseline_sha,
        targeted_score=0.9,
        control_score=0.9,
    )
    comparison = run_refinement(args)
    assert comparison["selected"]["name"] == "baseline"
    assert all(call[3] != "roof_pipeline.evaluate" for call in calls)
    assert comparison["test_iou_change_vs_baseline"] == 0


def test_invalid_baseline_cache_is_recomputed_only_after_baseline_selected(
    tmp_path, monkeypatch
):
    args, dataset_sha, manifest_sha, baseline_sha = _setup(tmp_path)
    cache = args.baseline.parent / "test/test_metrics.json"
    result = json.loads(cache.read_text())
    result["checkpoint_sha256"] = "wrong"
    cache.write_text(json.dumps(result))
    calls = _fake_jobs(
        monkeypatch,
        args,
        dataset_sha,
        manifest_sha,
        baseline_sha,
        targeted_score=0.9,
        control_score=0.9,
        test_iou=0.89,
    )
    comparison = run_refinement(args)
    evaluations = [call for call in calls if call[3] == "roof_pipeline.evaluate"]
    assert len(evaluations) == 1
    assert evaluations[0][evaluations[0].index("--checkpoint") + 1] == str(
        args.baseline.resolve()
    )
    assert comparison["baseline_test_iou"] == pytest.approx(0.89)


@pytest.mark.parametrize("change", ["data", "summary_score", "summary_arch"])
def test_invalid_baseline_fails_before_any_new_fitting(tmp_path, monkeypatch, change):
    args, dataset_sha, manifest_sha, baseline_sha = _setup(tmp_path)
    calls = _fake_jobs(monkeypatch, args, dataset_sha, manifest_sha, baseline_sha)
    if change == "data":
        (args.data / "dataset_meta.json").write_text('{"changed": true}')
    else:
        summary_path = args.baseline.parent / "training_summary.json"
        summary = json.loads(summary_path.read_text())
        if change == "summary_score":
            summary["val_iou_at_0p5"] = 0.99
        else:
            summary["config"]["arch"] = "unet-resnet34"
        summary_path.write_text(json.dumps(summary))
    with pytest.raises(ValueError):
        run_refinement(args)
    assert not calls
    status = json.loads((args.out / "workflow_status.json").read_text())
    assert status["phase"] == "failed" and status["failed_phase"] == "validating"


def test_subprocess_failure_is_recorded_and_prevents_later_test_evaluation(
    tmp_path, monkeypatch
):
    args, dataset_sha, manifest_sha, baseline_sha = _setup(tmp_path)
    calls = _fake_jobs(
        monkeypatch,
        args,
        dataset_sha,
        manifest_sha,
        baseline_sha,
        fail_phase="roof_pipeline.train",
    )
    with pytest.raises(subprocess.CalledProcessError):
        run_refinement(args)
    assert len(calls) == 2
    status = json.loads((args.out / "workflow_status.json").read_text())
    assert status["phase"] == "failed" and status["failed_phase"] == "targeted"
    assert status["test_evaluated"] is False


def test_existing_experiment_is_not_overwritten(tmp_path):
    args, _, _, _ = _setup(tmp_path)
    args.out.mkdir()
    status = args.out / "workflow_status.json"
    status.write_text('{"phase":"running"}')
    with pytest.raises(FileExistsError):
        run_refinement(args)
    assert status.read_text() == '{"phase":"running"}'
