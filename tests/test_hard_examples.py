"""Check bounded train-only mining and strict report reuse across dataset rows."""

import copy
import csv
import json

import numpy as np
import pytest
import torch

from roof_pipeline import hard_examples
from roof_pipeline.common import dataset_fingerprint, read_manifest
from roof_pipeline.hard_examples import (
    POLICY,
    load_sampling_weights,
    mine_hard_examples,
    sampling_probabilities,
)
from roof_pipeline.models import create_model
from scripts.make_smoke_dataset import make_fixture


def test_extreme_hardness_retains_floor_and_caps_outlier():
    errors = [1.0] + [0.0] * 99
    probabilities = np.array(sampling_probabilities(errors))
    assert probabilities.sum() == pytest.approx(1)
    assert probabilities.min() >= 0.5 / 100
    assert probabilities.max() <= 5 / 100 + 1e-12
    assert probabilities[0] == pytest.approx(0.05)
    np.testing.assert_allclose(probabilities[1:], 0.95 / 99)


def test_redistribution_preserves_more_than_one_hardness_level():
    errors = [1.0, 0.5] + [0.0001] * 98
    probabilities = sampling_probabilities(errors)
    assert sum(probabilities) == pytest.approx(1)
    assert max(probabilities) <= 0.05 + 1e-12
    assert min(probabilities) >= 0.005
    assert probabilities[0] == pytest.approx(probabilities[1])


@pytest.mark.parametrize("errors", [[0.0], [0.0] * 8, [0.25] * 8])
def test_equal_or_zero_errors_sample_uniformly(errors):
    np.testing.assert_allclose(sampling_probabilities(errors), 1 / len(errors))


@pytest.mark.parametrize("errors", [[], [float("nan")], [float("inf")], [-0.1], [1.1]])
def test_invalid_error_fractions_rejected(errors):
    with pytest.raises(ValueError):
        sampling_probabilities(errors)


def _report(tmp_path):
    rows = [{"sample_id": f"train_{i}", "split": "train"} for i in range(4)]
    errors = [0.0, 0.1, 0.2, 0.3]
    report = {
        "split": "train",
        "dataset_content_sha256": "frozen",
        "image_size": 64,
        "policy": copy.deepcopy(POLICY),
        "samples": [
            {**row, "error_fraction": error, "probability": probability}
            for row, error, probability in zip(
                rows, errors, sampling_probabilities(errors)
            )
        ],
    }
    path = tmp_path / "hardness.json"
    path.write_text(json.dumps(report))
    return rows, report, path


def test_report_reorders_to_training_manifest(tmp_path):
    rows, report, path = _report(tmp_path)
    weights = load_sampling_weights(path, list(reversed(rows)), "frozen")
    assert weights == [sample["probability"] for sample in reversed(report["samples"])]


@pytest.mark.parametrize("split", ["val", "test"])
def test_loader_rejects_nontraining_input_rows(tmp_path, split):
    rows, _, path = _report(tmp_path)
    rows[0]["split"] = split
    with pytest.raises(ValueError, match="training samples only"):
        load_sampling_weights(path, rows, "frozen")


@pytest.mark.parametrize(
    "mutation",
    [
        "report_split",
        "sample_split",
        "wrong_sha",
        "duplicate_id",
        "missing_id",
        "unknown_id",
        "policy",
    ],
)
def test_loader_rejects_mismatched_report(tmp_path, mutation):
    rows, report, path = _report(tmp_path)
    if mutation == "report_split":
        report["split"] = "test"
    elif mutation == "sample_split":
        report["samples"][0]["split"] = "val"
    elif mutation == "wrong_sha":
        report["dataset_content_sha256"] = "changed"
    elif mutation == "duplicate_id":
        report["samples"][0]["sample_id"] = report["samples"][1]["sample_id"]
    elif mutation == "missing_id":
        report["samples"].pop()
    elif mutation == "unknown_id":
        report["samples"][0]["sample_id"] = "test_0"
    elif mutation == "policy":
        report["policy"]["uniform_fraction"] = 0.9
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        load_sampling_weights(path, rows, "frozen")


@pytest.mark.parametrize("probability", [0.0, -0.1, float("nan"), float("inf"), True])
def test_loader_rejects_invalid_probabilities(tmp_path, probability):
    rows, report, path = _report(tmp_path)
    report["samples"][0]["probability"] = probability
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="finite and positive"):
        load_sampling_weights(path, rows, "frozen")


def test_loader_rejects_mass_change_or_weights_outside_policy(tmp_path):
    rows, report, path = _report(tmp_path)
    report["samples"][0]["probability"] += 0.1
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="sum to one"):
        load_sampling_weights(path, rows, "frozen")
    report["samples"][0]["probability"] -= 0.1
    report["samples"][0]["probability"] += 0.01
    report["samples"][1]["probability"] -= 0.01
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="declared policy"):
        load_sampling_weights(path, rows, "frozen")


def test_mining_scores_only_training_tiles_and_writes_reloadable_report(
    tmp_path, monkeypatch
):
    data = make_fixture(tmp_path / "data", 3, 2, 2, 64)
    fingerprint = dataset_fingerprint(data)
    checkpoint = tmp_path / "baseline.pt"
    config = {"arch": "tiny", "image_size": 64, "dataset_content_sha256": fingerprint}
    torch.save(
        {"state_dict": create_model("tiny").state_dict(), "config": config}, checkpoint
    )

    class CountingModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.samples_seen = 0

        def forward(self, x):
            assert self.training is False
            assert not torch.is_grad_enabled()
            self.samples_seen += x.shape[0]
            return torch.zeros((x.shape[0], 1, x.shape[2], x.shape[3]))

    model = CountingModel()
    monkeypatch.setattr(hard_examples, "load_model", lambda *args: (model, config))
    out = tmp_path / "mined.json"
    report = mine_hard_examples(data, checkpoint, out, batch_size=2, workers=0)
    train_rows = [row for row in read_manifest(data) if row["split"] == "train"]
    assert model.samples_seen == len(train_rows) == 3
    assert [row["sample_id"] for row in report["samples"]] == [
        row["sample_id"] for row in train_rows
    ]
    assert len(load_sampling_weights(out, train_rows, fingerprint)) == 3
    with out.with_suffix(".csv").open(newline="") as stream:
        assert len(list(csv.DictReader(stream))) == 3
    assert len(report["checkpoint_sha256"]) == 64


def test_mining_rejects_changed_checkpoint_data_before_model_inference(
    tmp_path, monkeypatch
):
    data = make_fixture(tmp_path / "data", 1, 1, 1, 64)
    checkpoint = tmp_path / "baseline.pt"
    torch.save(
        {"config": {"image_size": 64, "dataset_content_sha256": "different"}},
        checkpoint,
    )
    monkeypatch.setattr(
        hard_examples,
        "load_model",
        lambda *args: pytest.fail("Model must not be loaded"),
    )
    with pytest.raises(ValueError, match="fingerprint differs"):
        mine_hard_examples(data, checkpoint, tmp_path / "report.json")
