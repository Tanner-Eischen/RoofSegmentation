"""Auxiliary outlines must preserve binary inference and frozen warm starts."""

import argparse
import hashlib
import json
import math

import pytest
import torch

from roof_pipeline.boundary import BOUNDARY_TARGET, boundary_target
from roof_pipeline.common import dataset_fingerprint
from roof_pipeline.models import BoundaryUNet, create_model, load_model
from roof_pipeline.train import bce_dice_loss, initialize_model, train
from scripts.make_smoke_dataset import make_fixture


@pytest.fixture(autouse=True)
def small_thread_pool():
    torch.set_num_threads(2)


@pytest.mark.parametrize("value", [0.0, 1.0])
@pytest.mark.parametrize("radius", [1, 2])
def test_uniform_mask_has_no_artificial_tile_perimeter(value, radius):
    labels = torch.full((2, 1, 8, 8), value)
    assert torch.count_nonzero(boundary_target(labels, radius)) == 0


def test_edge_band_is_on_both_sides_of_visible_contour_only():
    labels = torch.zeros(1, 1, 8, 8)
    labels[..., :4] = 1
    expected = torch.zeros_like(labels)
    expected[..., 3:5] = 1
    torch.testing.assert_close(boundary_target(labels), expected)
    assert boundary_target(labels).dtype == labels.dtype


@pytest.mark.parametrize("radius", [0, -1, 1.5, True])
def test_invalid_target_radius_is_rejected(radius):
    with pytest.raises(ValueError, match="radius"):
        boundary_target(torch.zeros(1, 1, 8, 8), radius)


def test_dual_head_has_shared_gradients_and_binary_inference():
    model = create_model("tiny", boundary_supervision=True)
    x = torch.randn(2, 3, 32, 32)
    y = torch.zeros(2, 1, 32, 32)
    y[..., 8:24, 8:24] = 1
    logits, edges = model.forward_with_boundary(x)
    assert logits.shape == edges.shape == y.shape
    bce_dice_loss(edges, boundary_target(y)).backward()
    assert torch.count_nonzero(model.base.a.block[0].weight.grad) > 0
    assert torch.count_nonzero(model.boundary_head.weight.grad) > 0
    assert model.base.head.weight.grad is None
    model.zero_grad(set_to_none=True)
    logits, edges = model.forward_with_boundary(x)
    (
        bce_dice_loss(logits, y) + 0.2 * bce_dice_loss(edges, boundary_target(y))
    ).backward()
    for parameter in (
        model.base.a.block[0].weight,
        model.base.head.weight,
        model.boundary_head.weight,
    ):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert torch.count_nonzero(parameter.grad) > 0
    model.eval()
    with torch.no_grad():
        logits, _ = model.forward_with_boundary(x)
        torch.testing.assert_close(model(x), logits, rtol=0, atol=0)


def test_auxiliary_head_does_not_change_binary_weights_or_global_rng():
    torch.manual_seed(42)
    binary = create_model("tiny")
    binary_rng = torch.random.get_rng_state().clone()
    torch.manual_seed(42)
    auxiliary = create_model("tiny", boundary_supervision=True)
    assert torch.equal(torch.random.get_rng_state(), binary_rng)
    for key, tensor in binary.state_dict().items():
        torch.testing.assert_close(
            auxiliary.base.state_dict()[key], tensor, rtol=0, atol=0
        )


def write_checkpoint(path, *, auxiliary=False, digest="frozen", radius=1):
    model = create_model("tiny", boundary_supervision=auxiliary).eval()
    config = {
        "arch": "tiny",
        "image_size": 64,
        "normalization": "imagenet_rgb_v1",
        "dataset_content_sha256": digest,
        "epoch": 3,
    }
    if auxiliary:
        config.update(
            boundary_supervision=True,
            boundary_weight=0.2,
            boundary_radius=radius,
            boundary_target=BOUNDARY_TARGET,
        )
    torch.save(
        {"config": config, "state_dict": model.state_dict(), "threshold": 0.5}, path
    )
    return model


def test_binary_warm_start_only_initializes_edge_head(tmp_path):
    path = tmp_path / "binary.pt"
    original = write_checkpoint(path)
    original_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    args = argparse.Namespace(
        arch="tiny",
        size=64,
        init_checkpoint=path,
        boundary_weight=0.2,
        boundary_radius=1,
    )
    model, initialization = initialize_model(args, "cpu", "frozen")
    assert isinstance(model, BoundaryUNet)
    assert initialization["initial_checkpoint_sha256"] == original_sha
    image = torch.randn(1, 3, 64, 64)
    model.eval()
    with torch.no_grad():
        torch.testing.assert_close(model(image), original(image), rtol=0, atol=0)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == original_sha


def test_auxiliary_checkpoint_round_trip_preserves_both_heads(tmp_path):
    path = tmp_path / "auxiliary.pt"
    original = write_checkpoint(path, auxiliary=True)
    restored, config = load_model(path)
    image = torch.randn(1, 3, 64, 64)
    assert config["boundary_target"] == BOUNDARY_TARGET
    with torch.no_grad():
        for before, after in zip(
            original.forward_with_boundary(image), restored.forward_with_boundary(image)
        ):
            torch.testing.assert_close(before, after, rtol=0, atol=0)


@pytest.mark.parametrize(
    "weight,radius,message", [(0, 1, "disable"), (0.2, 2, "radius")]
)
def test_auxiliary_warm_start_rejects_incompatible_target(
    tmp_path, weight, radius, message
):
    path = tmp_path / "auxiliary.pt"
    write_checkpoint(path, auxiliary=True)
    args = argparse.Namespace(
        arch="tiny",
        size=64,
        init_checkpoint=path,
        boundary_weight=weight,
        boundary_radius=radius,
    )
    with pytest.raises(ValueError, match=message):
        initialize_model(args, "cpu", "frozen")


@pytest.mark.parametrize("weight", [-0.2, math.inf, math.nan])
def test_invalid_weight_is_rejected_before_model_creation(weight):
    args = argparse.Namespace(arch="tiny", pretrained=False, boundary_weight=weight)
    with pytest.raises(ValueError, match="weight"):
        initialize_model(args, "cpu", "frozen")


def test_auxiliary_training_retains_initial_candidate_and_records_loss(tmp_path):
    data = make_fixture(tmp_path / "data", 4, 2, 2, 64)
    initial = tmp_path / "binary.pt"
    write_checkpoint(initial, digest=dataset_fingerprint(data))
    args = argparse.Namespace(
        data=data,
        run=tmp_path / "run",
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
        min_delta=1.0,
        no_amp=True,
        init_checkpoint=initial,
        boundary_weight=0.2,
        boundary_radius=1,
    )
    summary = train(args)
    assert summary["best_epoch"] == 0
    assert summary["config"]["boundary_supervision"] is True
    assert summary["config"]["boundary_weight"] == 0.2
    assert summary["config"]["boundary_radius"] == 1
    assert summary["test_evaluated"] is False
    history = json.loads((args.run / "history.json").read_text())
    assert history[0]["boundary_loss"] > 0
    assert history[0]["segmentation_loss"] > 0
    assert history[0]["train_loss"] == pytest.approx(
        history[0]["segmentation_loss"] + 0.2 * history[0]["boundary_loss"]
    )
    restored, _ = load_model(args.run / "best.pt")
    assert isinstance(restored, BoundaryUNet)


def test_real_smp_unet_uses_shared_decoder_features_and_round_trips(tmp_path):
    pytest.importorskip("segmentation_models_pytorch")
    model = create_model("unet-resnet34", boundary_supervision=True).eval()
    image = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        masks, edges = model.forward_with_boundary(image)
        torch.testing.assert_close(model(image), masks, rtol=0, atol=0)
    assert masks.shape == edges.shape == (1, 1, 64, 64)
    path = tmp_path / "smp.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": {
                "arch": "unet-resnet34",
                "boundary_supervision": True,
            },
        },
        path,
    )
    restored, _ = load_model(path)
    with torch.no_grad():
        torch.testing.assert_close(restored(image), masks, rtol=0, atol=0)


@pytest.mark.parametrize(
    "arch", ["unetplusplus-efficientnet-b0", "deeplabv3plus-resnet34"]
)
def test_unsupported_architecture_is_rejected_without_loading_smp(arch):
    with pytest.raises(ValueError, match="supports only"):
        create_model(arch, boundary_supervision=True)
