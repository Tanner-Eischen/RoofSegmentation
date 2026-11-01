"""Matched replacement controls and real final-epoch training artifacts."""

import argparse
import hashlib
import json

import pytest
import torch
from torch.utils.data import WeightedRandomSampler

from roof_pipeline.common import dataset_fingerprint
from roof_pipeline.dataset import RoofDataset
from roof_pipeline.hard_examples import POLICY, sampling_probabilities
from roof_pipeline.train import ObservedWeightedRandomSampler, train, training_sampler
from scripts.make_smoke_dataset import make_fixture


def _report(data, dataset, path):
    errors = [index / len(dataset) for index in range(len(dataset))]
    probabilities = sampling_probabilities(errors)
    path.write_text(
        json.dumps(
            {
                "split": "train",
                "image_size": 64,
                "dataset_content_sha256": dataset_fingerprint(data),
                "policy": POLICY,
                "samples": [
                    dict(row, error_fraction=error, probability=probability)
                    for row, error, probability in zip(
                        dataset.rows, errors, probabilities
                    )
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


@pytest.mark.parametrize("weights", [[1.0] * 100, [0.1] * 90 + [1.0] * 10])
def test_observation_preserves_weighted_rng_and_records_real_coverage(weights):
    observed = ObservedWeightedRandomSampler(
        weights, 100, replacement=True, generator=torch.Generator().manual_seed(42)
    )
    reference = WeightedRandomSampler(
        weights, 100, replacement=True, generator=torch.Generator().manual_seed(42)
    )
    for _ in range(2):
        indices = list(observed)
        assert indices == list(reference)
        assert len(indices) == 100
        assert observed.unique_samples == len(set(indices)) < 100


def test_matched_sampler_policies_and_legacy_default(tmp_path):
    data = make_fixture(tmp_path / "data", 8, 1, 1, 64)
    ds = RoofDataset(data, "train", 64)
    sha = dataset_fingerprint(data)
    report = _report(data, ds, tmp_path / "mining.json")
    args = argparse.Namespace(size=64, uniform_replacement=True, sampling_report=None)
    uniform = training_sampler(args, ds, sha, torch.Generator().manual_seed(42))
    args.uniform_replacement = False
    args.sampling_report = report
    hard = training_sampler(args, ds, sha, torch.Generator().manual_seed(42))
    assert uniform.replacement is hard.replacement is True
    assert uniform.num_samples == hard.num_samples == len(ds)
    assert torch.equal(uniform.weights, torch.ones(len(ds), dtype=torch.double))
    assert len(torch.unique(hard.weights)) > 1
    args.uniform_replacement = True
    with pytest.raises(ValueError, match="cannot be combined"):
        training_sampler(args, ds, sha, torch.Generator())
    args.uniform_replacement = False
    args.sampling_report = None
    assert training_sampler(args, ds, sha, torch.Generator()) is None
    # Older callers do not need to supply the opt-in argument.
    del args.uniform_replacement
    assert training_sampler(args, ds, sha, torch.Generator()) is None
    args.sampling_report = report
    args.size = 96
    with pytest.raises(ValueError, match="image size differs"):
        training_sampler(args, ds, sha, torch.Generator())


@pytest.mark.parametrize("policy", ["legacy", "uniform", "hard"])
def test_tiny_training_retains_fixed_endpoint_and_logs_epoch_draws(tmp_path, policy):
    torch.set_num_threads(2)
    data = make_fixture(tmp_path / "data", 8, 2, 2, 64)
    ds = RoofDataset(data, "train", 64)
    report = _report(data, ds, tmp_path / "mining.json") if policy == "hard" else None
    args = argparse.Namespace(
        data=data,
        run=tmp_path / "run",
        arch="tiny",
        pretrained=False,
        epochs=1,
        batch_size=2,
        size=64,
        workers=0,
        lr=0.0001,
        weight_decay=0.0001,
        pos_weight=1.0,
        device="cpu",
        seed=42,
        patience=2,
        min_delta=0.0001,
        no_amp=True,
        init_checkpoint=None,
        sampling_report=report,
        uniform_replacement=policy == "uniform",
        photometric="standard",
    )
    result = train(args)
    history = json.loads((args.run / "history.json").read_text())
    final = torch.load(args.run / "final.pt", map_location="cpu", weights_only=True)
    assert history[0]["train_draws"] == len(ds)
    assert 1 <= history[0]["unique_train_samples"] <= len(ds)
    if policy == "legacy":
        assert history[0]["unique_train_samples"] == len(ds)
    assert final["threshold"] == result["final_threshold"] == 0.5
    assert final["config"]["epoch"] == result["final_epoch"] == 1
    assert final["val_iou_at_0p5"] == result["final_val_iou_at_0p5"]
    assert final["config"]["uniform_replacement"] is (policy == "uniform")
    assert final["config"]["sampling_report_sha256"] == (
        hashlib.sha256(report.read_bytes()).hexdigest() if report else None
    )
    assert result["test_evaluated"] is False
