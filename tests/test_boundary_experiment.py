"""Validation-only candidate selection and native-pixel outline diagnostics."""

import argparse
import json

import pytest
import torch

from roof_pipeline.boundary_experiment import (
    boundary_counts,
    boundary_scores,
    run_experiment,
    select_candidate,
)
from roof_pipeline.models import load_model
from roof_pipeline.train import train
from scripts.make_smoke_dataset import make_fixture


def candidates(incumbent=0.9, control=0.9, boundary=0.9):
    return [
        {"name": name, "validation_iou": value}
        for name, value in zip(
            ("incumbent", "control", "boundary"), (incumbent, control, boundary)
        )
    ]


def test_selection_is_validation_only_and_conservative():
    assert select_candidate(candidates())["name"] == "incumbent"
    assert select_candidate(candidates(boundary=0.90005))["name"] == "incumbent"
    assert (
        select_candidate(candidates(control=0.91, boundary=0.91))["name"] == "control"
    )
    values = candidates(control=0.91, boundary=0.92)
    values[0]["test_iou"] = 0.99
    values[2]["test_iou"] = 0.5
    assert select_candidate(values)["name"] == "boundary"


@pytest.mark.parametrize("score", [float("nan"), float("inf"), -0.1, 1.1, True])
def test_selection_rejects_invalid_scores(score):
    with pytest.raises(ValueError):
        select_candidate(candidates(boundary=score))


def test_boundary_diagnostics_measure_alignment_and_misses():
    torch.set_num_threads(2)
    target = torch.zeros(1, 1, 32, 32)
    target[:, :, 5:20, 7:23] = 1
    assert boundary_scores(boundary_counts(target, target))["f1"] == 1
    shifted = target.roll(5, -1)
    assert boundary_scores(boundary_counts(shifted, target))["f1"] < 1
    assert boundary_scores(boundary_counts(torch.zeros_like(target), target))["f1"] == 0
    assert boundary_counts(torch.ones_like(target), torch.ones_like(target)) == (
        0,
        0,
        0,
        0,
    )
    assert boundary_scores(boundary_counts(target.roll(1, -1), target))["f1"] == 1


def test_complete_matched_workflow_keeps_incumbent_and_never_infers_test(
    tmp_path, monkeypatch
):
    torch.set_num_threads(2)
    data = make_fixture(tmp_path / "data", 4, 2, 2, 64)
    initial = tmp_path / "initial"
    train(
        argparse.Namespace(
            data=data,
            run=initial,
            arch="tiny",
            pretrained=False,
            epochs=1,
            batch_size=2,
            size=64,
            workers=0,
            lr=0.001,
            weight_decay=0.0001,
            pos_weight=1.0,
            device="cpu",
            seed=42,
            patience=1,
            min_delta=0.0001,
            no_amp=True,
            init_checkpoint=None,
            sampling_report=None,
            photometric="standard",
        )
    )
    # Child training has no test loader by design; guard the in-process diagnostics.
    from roof_pipeline import boundary_experiment as module

    original_dataset = module.RoofDataset

    def validation_only(root, split, **kwargs):
        assert split == "val"
        return original_dataset(root, split, **kwargs)

    monkeypatch.setattr(module, "RoofDataset", validation_only)
    checkpoint = initial / "best.pt"
    frozen = checkpoint.read_bytes()
    out = tmp_path / "pilot"
    report = run_experiment(
        argparse.Namespace(
            data=data,
            incumbent=checkpoint,
            out=out,
            epochs=1,
            batch_size=2,
            workers=0,
            device="cpu",
            seed=42,
            lr=0.0001,
            boundary_weight=0.2,
            boundary_radius=1,
        )
    )
    assert checkpoint.read_bytes() == frozen
    assert report["test_evaluated"] is False
    assert report["frozen_incumbent_unchanged"] is True
    assert set(report["validation"]) == {"incumbent", "control", "boundary"}
    assert all(result["samples"] == 2 for result in report["validation"].values())
    assert not (out / "test").exists()
    assert json.loads((out / "workflow_status.json").read_text())["phase"] == "complete"
    model, config = load_model(out / "boundary/best.pt")
    assert config["boundary_supervision"] is True
    assert model(torch.zeros(1, 3, 64, 64)).shape == (1, 1, 64, 64)
    with pytest.raises(FileExistsError):
        run_experiment(
            argparse.Namespace(
                data=data,
                incumbent=checkpoint,
                out=out,
                epochs=1,
                batch_size=2,
                workers=0,
                device="cpu",
                seed=42,
                lr=0.0001,
                boundary_weight=0.2,
                boundary_radius=1,
            )
        )
