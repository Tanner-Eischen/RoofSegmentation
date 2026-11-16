"""Post-study inference must retain validation selection and exact test coverage."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest
import torch

from roof_pipeline.metrics import metrics_from_counts

spec = importlib.util.spec_from_file_location(
    "architecture_test_followup",
    Path(__file__).parents[1] / "scripts/evaluate_architecture_test.py",
)
followup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(followup)


def evaluation():
    counts = [
        dict(tp=20000, fp=100, fn=200, tn=241844),
        dict(tp=10000, fp=50, fn=100, tn=251994),
    ]
    rows = [
        {"sample_id": name, **values}
        for name, values in zip(("test-a", "test-b"), counts)
    ]
    totals = [sum(item[key] for item in counts) for key in followup.COUNTS]
    result = {
        "split": "test",
        "samples": 2,
        "threshold_frozen_from_validation": 0.5,
        "checkpoint_sha256": "selected-best",
        "dataset_content_sha256": "data",
        "aggregate": metrics_from_counts(*totals),
    }
    return result, rows


def test_exact_coverage_and_recomputed_metrics():
    result, rows = evaluation()
    verified = followup.verify_test_result(
        result, rows, ["test-b", "test-a"], "selected-best", "data"
    )
    assert verified["tp"] == 30000
    assert verified["iou"] == pytest.approx(30000 / 30450)


@pytest.mark.parametrize(
    "field,value",
    [
        ("split", "val"),
        ("samples", 1),
        ("threshold_frozen_from_validation", 0.45),
        ("checkpoint_sha256", "final-checkpoint"),
        ("dataset_content_sha256", "other-data"),
    ],
)
def test_refuses_different_split_threshold_or_checkpoint(field, value):
    result, rows = evaluation()
    result[field] = value
    with pytest.raises(ValueError, match="identity differs"):
        followup.verify_test_result(
            result, rows, ["test-a", "test-b"], "selected-best", "data"
        )


def test_duplicate_missing_and_extra_tile_cannot_look_complete():
    result, rows = evaluation()
    for bad in ([rows[0], rows[0]], rows[:1], rows + [rows[0]]):
        with pytest.raises(ValueError, match="test IDs once"):
            followup.verify_test_result(
                result, bad, ["test-a", "test-b"], "selected-best", "data"
            )


@pytest.mark.parametrize("value", [-1, 0.5, float("nan")])
def test_invalid_pixel_counts_rejected(value):
    result, rows = evaluation()
    rows[0]["fp"] = value
    with pytest.raises(ValueError, match="nonnegative integers"):
        followup.verify_test_result(
            result, rows, ["test-a", "test-b"], "selected-best", "data"
        )


def test_wrong_resolution_and_tampered_aggregate_rejected():
    result, rows = evaluation()
    wrong = copy.deepcopy(rows)
    wrong[0]["tn"] -= 1
    with pytest.raises(ValueError, match="512x512"):
        followup.verify_test_result(
            result, wrong, ["test-a", "test-b"], "selected-best", "data"
        )
    result["aggregate"]["iou"] += 0.01
    with pytest.raises(ValueError, match="Aggregate iou"):
        followup.verify_test_result(
            result, rows, ["test-a", "test-b"], "selected-best", "data"
        )


def test_checkpoint_contract_uses_selected_epoch_hash_and_frozen_threshold(tmp_path):
    checkpoint = tmp_path / "best.pt"
    data = {"content_sha256": "data", "manifest_sha256": "manifest"}
    config = {
        "arch": "unet-resnet34",
        "seed": 42,
        "epoch": 34,
        "image_size": 512,
        "normalization": "imagenet_rgb_v1",
        "dataset_content_sha256": "data",
        "dataset_manifest_sha256": "manifest",
    }
    state = {"config": config, "threshold": 0.5, "state_dict": {}}
    torch.save(state, checkpoint)
    before = checkpoint.read_bytes()
    row = {
        "architecture": "unet-resnet34",
        "seed": 42,
        "best_epoch": 34,
        "checkpoint_sha256": followup.sha256(checkpoint),
        "secondary_validation_threshold": 0.5,
    }
    followup.checkpoint_contract(checkpoint, row, data)
    assert checkpoint.read_bytes() == before
    row["best_epoch"] = 40
    with pytest.raises(ValueError, match="selection/configuration"):
        followup.checkpoint_contract(checkpoint, row, data)
    row["best_epoch"] = 34
    state["threshold"] = 0.45
    torch.save(state, checkpoint)
    row["checkpoint_sha256"] = followup.sha256(checkpoint)
    with pytest.raises(ValueError, match="frozen threshold"):
        followup.checkpoint_contract(checkpoint, row, data)
    row["checkpoint_sha256"] = "changed"
    with pytest.raises(ValueError, match="Checkpoint bytes"):
        followup.checkpoint_contract(checkpoint, row, data)


def test_registration_serialization_rejects_nonfinite_values(tmp_path):
    with pytest.raises(ValueError):
        followup.write_json(tmp_path / "registration.json", {"threshold": float("nan")})
    followup.write_json(tmp_path / "registration.json", {"threshold": 0.5})
    assert json.loads((tmp_path / "registration.json").read_text())["threshold"] == 0.5


def test_packet_manifest_binds_bytes_without_self_reference(tmp_path):
    evidence = tmp_path / "test.csv"
    evidence.write_bytes(b"sample_id,tp\r\ntest-a,12\r\n")
    followup.packet_checksums(tmp_path)
    manifest = tmp_path / "checksums.json"
    first = manifest.read_bytes()
    recorded = json.loads(first)
    assert set(recorded) == {"test.csv"}
    assert recorded["test.csv"] == followup.sha256(evidence)
    followup.packet_checksums(tmp_path)
    assert manifest.read_bytes() == first
    evidence.write_bytes(b"tampered")
    assert recorded["test.csv"] != followup.sha256(evidence)


def test_no_silent_cpu_fallback_or_original_artifact_overwrite(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="no silent CPU fallback"):
        followup.run(
            tmp_path / "results",
            tmp_path / "data",
            tmp_path / "out",
            tmp_path / "packet",
        )
    original = tmp_path / "results"
    original.mkdir()
    evidence = original / "history.json"
    evidence.write_bytes(b"original training record")
    with pytest.raises(ValueError, match="overwrite original"):
        followup.run(original, tmp_path / "data", original / "out", tmp_path / "packet")
    assert evidence.read_bytes() == b"original training record"
