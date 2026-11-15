"""Scientific controls, validation-best selection, and explicit execution gates."""

import argparse
import copy
import subprocess
import sys
import types
import zipfile
from pathlib import Path

import pytest

from roof_pipeline import architecture_comparison as comparison


def make_plan(tmp_path, epochs=2):
    repo = tmp_path / "repo"
    source = repo / "roof_pipeline/train.py"
    source.parent.mkdir(parents=True)
    source.write_text("# frozen trainer fixture\n")
    return {
        "schema_version": 1,
        "repository_path": str(repo),
        "dataset": {
            "path": str(tmp_path / "data"),
            "manifest_sha256": "manifest",
            "content_sha256": "data",
            "split_counts": dict(comparison.OFFICIAL_COUNTS),
        },
        "architectures": list(comparison.ARCHITECTURES),
        "seeds": [42],
        "settings": comparison.settings(epochs=epochs, device="cpu"),
        "initializations": {
            arch: {"encoder_sha256": "mit" if arch == "segformer-b0" else "resnet"}
            for arch in comparison.ARCHITECTURES
        },
        "source_sha256": {"roof_pipeline/train.py": comparison.sha256(source)},
        "environment": {"fixture": True},
    }


def save_prepared(tmp_path, plan):
    out = tmp_path / "study"
    out.mkdir()
    with zipfile.ZipFile(out / "source_snapshot.zip", "w") as archive:
        for relative in plan["source_sha256"]:
            archive.write(Path(plan["repository_path"]) / relative, relative)
    plan["source_snapshot_sha256"] = comparison.sha256(out / "source_snapshot.zip")
    comparison.write_json(out / "plan.json", plan)
    estimate = {
        "schema_version": 1,
        "plan_sha256": comparison.sha256(out / "plan.json"),
        "architectures": {
            arch: {"run_seconds": {"lower": 10, "upper": 20}}
            for arch in plan["architectures"]
        },
        "total_seconds": {"lower": 30, "upper": 60},
    }
    comparison.write_json(out / "runtime_estimate.json", estimate)
    return out, estimate


def save_run(run, plan, arch, seed, scores=None):
    fixed = plan["settings"]
    scores = scores or [0.8, 0.7]
    best = max(range(len(scores)), key=scores.__getitem__)
    config = {
        "arch": arch,
        "seed": seed,
        "image_size": 512,
        "normalization": "imagenet_rgb_v1",
        "dataset_content_sha256": "data",
        "dataset_manifest_sha256": "manifest",
        "effective_training_content_sha256": "data",
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
    arguments = {
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
    arguments.update(
        {
            "arch": arch,
            "seed": seed,
            "pretrained": True,
            "no_amp": False,
            "init_checkpoint": None,
            "sampling_report": None,
            "uniform_replacement": False,
            "training_review": None,
            "expected_encoder_sha256": config["encoder_initialization_sha256"],
            "data": plan["dataset"]["path"],
            "run": str(run.resolve()),
        }
    )
    summary = {
        "config": dict(config, epoch=best + 1),
        "best_epoch": best + 1,
        "val_iou_at_0p5": scores[best],
        "val_tuned_threshold": 0.5,
        "val_metrics_tuned": {"iou": scores[best]},
        "train_samples": 4736,
        "val_samples": 1036,
        "test_evaluated": False,
        "final_epoch": len(scores),
    }
    history = [
        {
            "epoch": index + 1,
            "val_iou_0p5": value,
            "seconds": 5.0,
            "train_draws": 4736,
            "unique_train_samples": 4736,
        }
        for index, value in enumerate(scores)
    ]
    comparison.write_json(run / "training_summary.json", summary)
    comparison.write_json(
        run / "experiment.json", {"config": config, "arguments": arguments}
    )
    comparison.write_json(run / "history.json", history)
    comparison.write_json(
        run / "best.pt",
        {
            "config": dict(config, epoch=best + 1),
            "val_iou_at_0p5": scores[best],
            "threshold": 0.5,
        },
    )


def test_dry_run_has_three_matched_fresh_commands_and_no_files(tmp_path):
    out = tmp_path / "absent"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "roof_pipeline.architecture_comparison",
            "--data",
            str(tmp_path / "missing"),
            "--out",
            str(out),
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    commands = result.stdout.splitlines()[:3]
    assert len(commands) == 3 and not out.exists()
    for arch, command in zip(comparison.ARCHITECTURES, commands):
        assert f"--arch {arch}" in command
        assert "--pretrained" in command and "--epochs 40" in command
        assert "--min-delta 0" in command and "--patience 41" in command
        assert "--init-checkpoint" not in command and "--sampling-report" not in command
        assert "--uniform-replacement" not in command
    assert "no files, data access, downloads or fitting" in result.stdout


def test_preparation_freezes_sources_and_initialization_without_fitting(
    tmp_path, monkeypatch
):
    plan = make_plan(tmp_path)
    out = tmp_path / "prepared"
    args = argparse.Namespace(
        out=out,
        data=Path(plan["dataset"]["path"]),
        epochs=None,
        batch_size=None,
        workers=None,
        device=None,
        seeds=None,
    )
    monkeypatch.setattr(comparison, "dataset_identity", lambda _: plan["dataset"])
    monkeypatch.setattr(
        comparison, "encoder_initializations", lambda: plan["initializations"]
    )
    monkeypatch.setattr(comparison, "runtime_environment", lambda: plan["environment"])
    source = Path(comparison.__file__)
    monkeypatch.setattr(comparison, "source_files", lambda _: [source])
    monkeypatch.setattr(
        comparison, "execute_command", lambda *args: pytest.fail("prepare must not fit")
    )
    saved = comparison.prepare(args)
    assert saved["settings"]["epochs"] == 40 and saved["seeds"] == [42]
    assert (out / "source_snapshot.zip").is_file() and not (out / "runs").exists()
    assert (
        comparison.read_json(out / "status.json")["phase"]
        == "prepared_awaiting_runtime_estimate"
    )
    with pytest.raises(FileExistsError):
        comparison.prepare(args)


@pytest.mark.parametrize("change", ["plan", "architecture", "bounds"])
def test_runtime_estimate_must_match_complete_plan(tmp_path, change):
    plan = make_plan(tmp_path)
    out, estimate = save_prepared(tmp_path, plan)
    if change == "plan":
        estimate["plan_sha256"] = "other"
    if change == "architecture":
        estimate["architectures"].pop("segformer-b0")
    if change == "bounds":
        estimate["total_seconds"]["upper"] = float("nan")
    with pytest.raises(ValueError):
        comparison.validate_estimate(
            estimate, plan, comparison.sha256(out / "plan.json")
        )


def test_missing_runtime_estimate_prevents_any_training(tmp_path, monkeypatch):
    plan = make_plan(tmp_path)
    out, _ = save_prepared(tmp_path, plan)
    (out / "runtime_estimate.json").unlink()
    monkeypatch.setattr(
        comparison, "execute_command", lambda *args: pytest.fail("must not fit")
    )
    with pytest.raises(FileNotFoundError):
        comparison.execute_plan(out)
    assert not (out / "runs").exists()


@pytest.mark.parametrize("scores", [[0.8, 0.7], [0.8, 0.8]])
def test_selected_checkpoint_is_validation_best_and_earliest_tie(
    tmp_path, monkeypatch, scores
):
    plan = make_plan(tmp_path)
    run = tmp_path / "run"
    run.mkdir()
    save_run(run, plan, comparison.ARCHITECTURES[0], 42, scores)
    monkeypatch.setattr(comparison, "load_checkpoint", comparison.read_json)
    result = comparison.verify_run(plan, run, comparison.ARCHITECTURES[0], 42)
    assert result["best_epoch"] == 1
    assert result["primary_validation_iou_0p5"] == 0.8


@pytest.mark.parametrize(
    "change",
    [
        "last_instead_of_best",
        "different_encoder",
        "sampling",
        "missing_epoch",
        "wrong_lr",
        "test_inference",
    ],
)
def test_invalid_completed_run_is_rejected(tmp_path, monkeypatch, change):
    plan = make_plan(tmp_path)
    run = tmp_path / "run"
    run.mkdir()
    arch = comparison.ARCHITECTURES[0]
    save_run(run, plan, arch, 42)
    if change == "last_instead_of_best":
        item = comparison.read_json(run / "best.pt")
        item["config"]["epoch"] = 2
        comparison.write_json(run / "best.pt", item)
    elif change in ("different_encoder", "sampling", "test_inference"):
        item = comparison.read_json(run / "training_summary.json")
        if change == "different_encoder":
            item["config"]["encoder_initialization_sha256"] = "other"
        if change == "sampling":
            item["config"]["uniform_replacement"] = True
        if change == "test_inference":
            item["test_evaluated"] = True
        comparison.write_json(run / "training_summary.json", item)
    elif change == "missing_epoch":
        comparison.write_json(
            run / "history.json", comparison.read_json(run / "history.json")[:1]
        )
    else:
        item = comparison.read_json(run / "experiment.json")
        item["arguments"]["lr"] = 1e-3
        comparison.write_json(run / "experiment.json", item)
    monkeypatch.setattr(comparison, "load_checkpoint", comparison.read_json)
    with pytest.raises(ValueError):
        comparison.verify_run(plan, run, arch, 42)


def test_explicit_execution_trains_all_architectures_from_snapshot_then_reports(
    tmp_path, monkeypatch
):
    plan = make_plan(tmp_path)
    out, _ = save_prepared(tmp_path, plan)
    monkeypatch.setattr(comparison, "dataset_identity", lambda _: plan["dataset"])
    monkeypatch.setattr(comparison, "runtime_environment", lambda: plan["environment"])
    monkeypatch.setattr(comparison, "load_checkpoint", comparison.read_json)
    calls = []

    def execute(command, cwd, log):
        assert cwd == out / "frozen_source"
        assert (
            cwd / "roof_pipeline/train.py"
        ).read_text() == "# frozen trainer fixture\n"
        calls.append(command)
        seed = int(command[command.index("--seed") + 1])
        if "roof_pipeline.train" in command:
            arch = command[command.index("--arch") + 1]
            save_run(log.parent, plan, arch, seed)
        else:
            assert "--assess-run" in command
            arch = command[command.index("--architecture") + 1]
            result = comparison.verify_run(plan, log.parent, arch, seed)
            comparison.write_json(
                log.parent / "validation_best.json",
                {**result, "split": "val", "test_inference": False, "samples": 1036},
            )

    monkeypatch.setattr(comparison, "execute_command", execute)
    results = comparison.execute_plan(out)
    assert len(results) == 3 and len(calls) == 6
    assert all(row["best_epoch"] == 1 for row in results)
    assert comparison.read_json(out / "status.json")["phase"] == "complete"
    assert (out / "report.md").is_file()
    assert "not estimable (one seed)" in (out / "report.md").read_text()
    with pytest.raises(FileExistsError):
        comparison.execute_plan(out)


def test_interrupted_run_does_not_produce_comparison_report(tmp_path, monkeypatch):
    plan = make_plan(tmp_path)
    out, _ = save_prepared(tmp_path, plan)
    monkeypatch.setattr(comparison, "dataset_identity", lambda _: plan["dataset"])
    monkeypatch.setattr(comparison, "runtime_environment", lambda: plan["environment"])

    def fail(*args):
        raise RuntimeError("simulated training interruption")

    monkeypatch.setattr(comparison, "execute_command", fail)
    with pytest.raises(RuntimeError, match="interruption"):
        comparison.execute_plan(out)
    assert comparison.read_json(out / "status.json")["phase"] == "failed"
    assert not (out / "comparison.json").exists() and not (out / "report.md").exists()


def test_source_or_dataset_drift_rejected_before_fitting(tmp_path, monkeypatch):
    plan = make_plan(tmp_path)
    out, _ = save_prepared(tmp_path, plan)
    changed = copy.deepcopy(plan["dataset"])
    changed["content_sha256"] = "changed"
    monkeypatch.setattr(comparison, "dataset_identity", lambda _: changed)
    with pytest.raises(ValueError, match="data changed"):
        comparison.execute_plan(out)
    monkeypatch.setattr(comparison, "dataset_identity", lambda _: plan["dataset"])
    monkeypatch.setattr(comparison, "runtime_environment", lambda: plan["environment"])
    (Path(plan["repository_path"]) / "roof_pipeline/train.py").write_text("changed")
    with pytest.raises(ValueError, match="source changed"):
        comparison.execute_plan(out)


def test_validation_assessment_cannot_use_live_source(tmp_path):
    plan = make_plan(tmp_path)
    out, _ = save_prepared(tmp_path, plan)
    with pytest.raises(ValueError, match="registered source snapshot"):
        comparison.assess_run(out, comparison.ARCHITECTURES[0], 42)


def test_runtime_estimate_gpu_must_match_execution_hardware(monkeypatch):
    fake = types.SimpleNamespace(
        cuda=types.SimpleNamespace(
            is_available=lambda: True,
            get_device_name=lambda: "GPU A",
            get_device_properties=lambda _: types.SimpleNamespace(total_memory=8),
        )
    )
    monkeypatch.setitem(sys.modules, "torch", fake)
    comparison.validate_hardware({"device": "GPU A", "device_total_bytes": 8}, "cuda")
    with pytest.raises(ValueError, match="different GPU"):
        comparison.validate_hardware(
            {"device": "GPU B", "device_total_bytes": 8}, "cuda"
        )


@pytest.mark.parametrize("argument", ["--epochs", "--batch-size"])
def test_zero_budget_is_not_silently_replaced_by_default(tmp_path, argument):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "roof_pipeline.architecture_comparison",
            "--data",
            str(tmp_path / "missing"),
            "--out",
            str(tmp_path / "out"),
            argument,
            "0",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0 and "Invalid fixed training budget" in result.stderr
