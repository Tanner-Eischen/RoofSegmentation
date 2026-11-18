"""Unknown pixels cannot influence optimization, selection or reported counts."""

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from roof_pipeline.boundary import boundary_target
from roof_pipeline.metrics import choose_threshold, evaluate_loader
from roof_pipeline.train import bce_dice_loss


def test_masked_loss_matches_valid_pixels_and_has_zero_invalid_gradient():
    logits = torch.tensor([[[[0.3, -0.4, float("nan")]]]], requires_grad=True)
    labels = torch.tensor([[[[1.0, 0.0, -1.0]]]])
    loss = bce_dice_loss(logits, labels, 2)
    expected = bce_dice_loss(logits[..., :2], labels[..., :2], 2)
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert logits.grad[..., 2].item() == 0
    assert torch.isfinite(logits.grad).all()


def test_all_invalid_loss_zero_and_mixed_batch_excludes_empty_dice():
    x = torch.full((1, 1, 2, 2), float("nan"), requires_grad=True)
    y = torch.full_like(x, -1)
    loss = bce_dice_loss(x, y)
    assert loss.item() == 0
    loss.backward()
    assert (x.grad == 0).all()
    good = torch.zeros_like(x)
    torch.testing.assert_close(
        bce_dice_loss(torch.cat((good, x)), torch.cat((good, y))),
        bce_dice_loss(good, good),
    )


def test_all_valid_loss_exact_legacy_arithmetic():
    torch.manual_seed(3)
    logits = torch.randn(2, 1, 5, 5)
    labels = (torch.rand_like(logits) > 0.5).float()
    probs = logits.sigmoid()
    dice = (2 * (probs * labels).sum((1, 2, 3)) + 1) / (
        (probs + labels).sum((1, 2, 3)) + 1
    )
    expected = F.binary_cross_entropy_with_logits(logits, labels) + (1 - dice).mean()
    assert torch.equal(bce_dice_loss(logits, labels), expected)


class Logits(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.seen = 0

    def forward(self, x):
        self.seen += len(x)
        return x


def test_metrics_and_calibration_exclude_nodata_and_skip_empty_samples():
    # The very confident invalid positive must contribute neither FP nor TN.
    x = torch.tensor([[[[2.0, -2.0, 20.0]]], [[[20.0, 20.0, 20.0]]]])
    y = torch.tensor([[[[1.0, 0.0, -1.0]]], [[[-1.0, -1.0, -1.0]]]])
    loader = DataLoader(TensorDataset(x, y), batch_size=2)
    model = Logits()
    scores, rows = evaluate_loader(model, loader, "cpu", with_rows=True)
    assert model.seen == 1
    assert (scores["tp"], scores["fp"], scores["fn"], scores["tn"]) == (1, 0, 0, 1)
    assert scores["valid_pixel_count"] == 2
    assert scores["invalid_pixel_count"] == 4
    assert rows[0]["roof_fraction"] == 0.5
    assert rows[1]["valid_pixel_count"] == 0
    assert rows[1]["tn"] == 0
    threshold, curve = choose_threshold(model, loader, "cpu", edges=[0.5, 0.9])
    assert threshold == 0.5
    assert curve == {"0.5": 1.0, "0.9": 0.0}


def test_empty_evaluation_and_calibration_fail_without_forward():
    loader = DataLoader(TensorDataset(torch.zeros(1, 1, 2, 2), -torch.ones(1, 1, 2, 2)))
    model = Logits()
    with pytest.raises(ValueError, match="no valid pixels"):
        evaluate_loader(model, loader, "cpu")
    with pytest.raises(ValueError, match="no valid pixels"):
        choose_threshold(model, loader, "cpu")
    assert model.seen == 0


@pytest.mark.parametrize("dtype", [torch.int64, torch.bool, torch.float32])
def test_metrics_probabilities_do_not_inherit_target_dtype(dtype):
    x = torch.tensor([[[[2.0, -2.0]]]])
    y = torch.tensor([[[[1, 0]]]], dtype=dtype)
    scores, _ = evaluate_loader(Logits(), DataLoader(TensorDataset(x, y)), "cpu")
    assert scores["tp"] == scores["tn"] == 1
    assert scores["fp"] == scores["fn"] == 0


@pytest.mark.parametrize("value", [float("nan"), 255.0, -2.0])
def test_reject_undeclared_target_codes(value):
    with pytest.raises(ValueError, match="Targets"):
        bce_dice_loss(torch.zeros(1, 1, 1, 1), torch.tensor([[[[value]]]]))


def test_boundary_target_ignores_unknown_neighborhood_without_false_edges():
    labels = torch.ones(1, 1, 7, 7)
    labels[..., 3, 3] = -1
    target = boundary_target(labels)
    expected = np.zeros((7, 7))
    expected[2:5, 2:5] = -1
    np.testing.assert_array_equal(target[0, 0], expected)


@pytest.mark.parametrize("ignored", [1, 4])
def test_training_skips_all_invalid_samples_before_forward(
    tmp_path, monkeypatch, ignored
):
    import argparse
    import json

    from PIL import Image

    from roof_pipeline import train as training
    from roof_pipeline.common import read_manifest, write_manifest
    from scripts.make_smoke_dataset import make_fixture

    torch.set_num_threads(2)
    data = make_fixture(tmp_path / "data", 4, 2, 2, 64)
    rows = read_manifest(data)
    for row in [r for r in rows if r["split"] == "train"][:ignored]:
        relative = f"valid_masks/{row['sample_id']}.png"
        (data / "valid_masks").mkdir(exist_ok=True)
        Image.fromarray(np.zeros((64, 64), np.uint8)).save(data / relative)
        row["valid_mask"] = relative
    write_manifest(data, rows, json.loads((data / "dataset_meta.json").read_text()))
    seen = []
    original = training.initialize_model

    def initialize(*args, **kwargs):
        model, info = original(*args, **kwargs)

        def observe(module, values):
            if module.training:
                assert values[0].flatten(1).any(1).all()
                seen.append(len(values[0]))

        model.register_forward_pre_hook(observe)
        return model, info

    monkeypatch.setattr(training, "initialize_model", initialize)
    args = argparse.Namespace(
        data=data,
        run=tmp_path / "run",
        arch="tiny",
        pretrained=False,
        epochs=1,
        batch_size=2,
        size=64,
        workers=0,
        lr=0.001,
        weight_decay=0.0,
        pos_weight=1.0,
        device="cpu",
        seed=42,
        patience=2,
        min_delta=0.0001,
        no_amp=True,
    )
    if ignored == 4:
        with pytest.raises(ValueError, match="Training epoch has no valid pixels"):
            training.train(args)
        assert seen == []
    else:
        training.train(args)
        history = json.loads((args.run / "history.json").read_text())[0]
        assert sum(seen) == history["train_scored_samples"] == 3
        assert history["train_draws"] == 4 and history["train_ignored_samples"] == 1


def test_boundary_diagnostics_do_not_manufacture_edges_at_nodata():
    from roof_pipeline.boundary_experiment import boundary_counts

    truth = torch.ones(1, 1, 12, 12, dtype=torch.bool)
    valid = truth.clone()
    valid[..., 5:7, 5:7] = False
    prediction = truth & valid
    assert boundary_counts(prediction, truth, valid=valid) == (0, 0, 0, 0)
