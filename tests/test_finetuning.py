"""Verify warm-start safety and synchronized geometry under robust photometry."""

import argparse
import hashlib
import random

import numpy as np
import pytest
import torch
from PIL import Image

from roof_pipeline.common import dataset_fingerprint
from roof_pipeline.dataset import MEAN, STD, RoofDataset
from roof_pipeline.models import create_model
from roof_pipeline.train import initialize_model, train
from scripts.make_smoke_dataset import make_fixture


def initial_checkpoint(data, checkpoint):
    model = create_model("tiny")
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.head.bias.fill_(10)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": {
                "arch": "tiny",
                "image_size": 64,
                "normalization": "imagenet_rgb_v1",
                "dataset_content_sha256": dataset_fingerprint(data),
                "epoch": 32,
            },
            "threshold": 0.5,
        },
        checkpoint,
    )
    return checkpoint


def test_warm_start_retains_initial_candidate_and_original_checkpoint(tmp_path):
    torch.set_num_threads(2)
    data = make_fixture(tmp_path / "data", 4, 2, 2, 64)
    initial = initial_checkpoint(data, tmp_path / "initial.pt")
    original_sha = hashlib.sha256(initial.read_bytes()).hexdigest()
    args = argparse.Namespace(
        data=data,
        run=tmp_path / "fine",
        arch="tiny",
        pretrained=False,
        epochs=1,
        batch_size=2,
        size=64,
        workers=0,
        lr=0.0,
        weight_decay=0.0,
        pos_weight=1.0,
        device="cpu",
        seed=42,
        patience=2,
        min_delta=0.0001,
        no_amp=True,
        init_checkpoint=initial,
        sampling_report=None,
        photometric="robust",
    )
    summary = train(args)
    assert summary["best_epoch"] == 0
    assert summary["initial_val_metrics_at_0p5"]["iou"] == summary["val_iou_at_0p5"]
    assert summary["config"]["initial_checkpoint_sha256"] == original_sha
    assert summary["config"]["optimizer_restarted"] is True
    assert summary["config"]["photometric_profile"] == "robust"
    assert hashlib.sha256(initial.read_bytes()).hexdigest() == original_sha


@pytest.mark.parametrize("change", ["fingerprint", "size", "arch"])
def test_initialization_rejects_incompatible_run(tmp_path, change):
    data = make_fixture(tmp_path / "data", 1, 1, 1, 64)
    initial = initial_checkpoint(data, tmp_path / "initial.pt")
    args = argparse.Namespace(init_checkpoint=initial, arch="tiny", size=64)
    digest = dataset_fingerprint(data)
    if change == "fingerprint":
        digest = "0" * 64
    elif change == "size":
        args.size = 96
    else:
        args.arch = "unet-resnet34"
    with pytest.raises(ValueError, match="differs"):
        initialize_model(args, "cpu", digest)


def test_robust_photometry_keeps_image_and_mask_geometry_aligned(tmp_path):
    data = make_fixture(tmp_path / "data", 1, 1, 1, 64)
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[8:27, 11:40] = 255
    image = np.repeat(mask[..., None], 3, axis=2)
    Image.fromarray(image).save(data / "images/synthetic_train_000.png")
    Image.fromarray(mask).save(data / "masks/synthetic_train_000.png")
    dataset = RoofDataset(data, "train", size=64, augment=True, photometric="robust")
    for seed in range(20):
        random.seed(seed)
        x, y = dataset[0]
        red = x[0].numpy() * STD[0, 0, 0] + MEAN[0, 0, 0]
        midpoint = (red.max() + red.min()) / 2
        np.testing.assert_array_equal(red > midpoint, y[0].numpy() > 0)
        assert set(np.unique(y.numpy())).issubset({0, 1})


def test_validation_photometry_is_deterministic_and_augmentation_is_rejected(tmp_path):
    data = make_fixture(tmp_path / "data", 1, 1, 1, 64)
    standard = RoofDataset(data, "val", 64, photometric="standard")
    robust = RoofDataset(data, "val", 64, photometric="robust")
    for first, second in zip(standard[0], robust[0]):
        assert torch.equal(first, second)
    with pytest.raises(ValueError, match="restricted"):
        RoofDataset(data, "val", 64, augment=True)
