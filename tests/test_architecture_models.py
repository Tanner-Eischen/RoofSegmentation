"""Binary architecture contracts for the matched WHU comparison."""

import argparse
import copy
import sys
from types import ModuleType

import pytest
import torch
from torch import nn

from roof_pipeline import models
from roof_pipeline.boundary import validate_boundary_options
from roof_pipeline.train import parse_args

ARCHITECTURES = (
    ("unet-resnet34", "Unet", "resnet34"),
    ("unetplusplus-resnet34", "UnetPlusPlus", "resnet34"),
    ("segformer-b0", "Segformer", "mit_b0"),
)


def test_encoder_fingerprint_tracks_tensor_content_and_ignores_decoder():
    model = nn.Module()
    model.encoder = nn.Sequential(nn.Conv2d(3, 2, 1), nn.BatchNorm2d(2))
    model.decoder = nn.Conv2d(2, 1, 1)
    before = models.encoder_fingerprint(model)
    assert len(before) == 64
    assert models.encoder_fingerprint(copy.deepcopy(model)) == before
    with torch.no_grad():
        model.decoder.weight.add_(1)
    assert models.encoder_fingerprint(model) == before
    with torch.no_grad():
        model.encoder[1].num_batches_tracked.add_(1)
    assert models.encoder_fingerprint(model) != before
    with pytest.raises(ValueError, match="does not expose an encoder"):
        models.encoder_fingerprint(nn.Conv2d(3, 1, 1))


def test_expected_encoder_sha_rejects_drift_before_optimization(tmp_path, monkeypatch):
    from roof_pipeline import train as training
    from scripts.make_smoke_dataset import make_fixture

    torch.set_num_threads(2)
    data = make_fixture(tmp_path / "data", 2, 1, 1, 64)
    model = nn.Module()
    model.encoder = nn.Conv2d(3, 1, 1)
    monkeypatch.setattr(training, "initialize_model", lambda *args: (model, {}))
    args = argparse.Namespace(
        data=data,
        run=tmp_path / "run",
        arch="unet-resnet34",
        pretrained=True,
        epochs=1,
        batch_size=2,
        size=64,
        workers=0,
        device="cpu",
        seed=42,
        expected_encoder_sha256="0" * 64,
    )
    with pytest.raises(ValueError, match="initialization SHA differs"):
        training.train(args)
    assert not (args.run / "experiment.json").exists()
    assert not (args.run / "best.pt").exists()


def test_matching_encoder_sha_is_saved_without_creating_epoch_zero_candidate(
    tmp_path, recording_smp
):
    from roof_pipeline import train as training
    from scripts.make_smoke_dataset import make_fixture

    torch.set_num_threads(2)
    data = make_fixture(tmp_path / "data", 2, 1, 1, 64)
    training.seed_everything(42)
    initial = models.create_model("unetplusplus-resnet34", pretrained=True)
    expected = models.encoder_fingerprint(initial)
    args = argparse.Namespace(
        data=data,
        run=tmp_path / "run",
        arch="unetplusplus-resnet34",
        pretrained=True,
        epochs=1,
        batch_size=2,
        size=64,
        workers=0,
        device="cpu",
        seed=42,
        expected_encoder_sha256=expected,
        lr=0.001,
        weight_decay=0.0001,
        no_amp=True,
        pos_weight=1,
        patience=2,
        min_delta=0.0001,
    )
    summary = training.train(args)
    state = torch.load(args.run / "best.pt", map_location="cpu", weights_only=True)
    assert summary["initial_val_metrics_at_0p5"] is None
    assert summary["best_epoch"] == 1
    assert summary["config"]["encoder_initialization_sha256"] == expected
    assert state["config"]["encoder_initialization_sha256"] == expected


@pytest.fixture
def recording_smp(monkeypatch):
    module = ModuleType("segmentation_models_pytorch")
    calls = []

    class EncoderStub(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Conv2d(3, 4, 1)
            self.head = nn.Conv2d(4, 1, 1)

        def forward(self, image):
            return self.head(self.encoder(image).relu())

    def constructor(name):
        def construct(**kwargs):
            calls.append((name, kwargs))
            return EncoderStub()

        return construct

    for name in ("Unet", "UnetPlusPlus", "DeepLabV3Plus", "Segformer"):
        setattr(module, name, constructor(name))
    monkeypatch.setitem(sys.modules, "segmentation_models_pytorch", module)
    return module, calls


@pytest.mark.parametrize("arch,constructor,encoder", ARCHITECTURES)
@pytest.mark.parametrize("pretrained", [False, True])
def test_architecture_initialization_has_one_rgb_logit_channel(
    recording_smp, arch, constructor, encoder, pretrained
):
    _, calls = recording_smp
    models.create_model(arch, pretrained=pretrained)
    name, options = calls.pop()
    assert name == constructor
    assert options["encoder_name"] == encoder
    assert options["encoder_weights"] == ("imagenet" if pretrained else None)
    assert options["in_channels"] == 3
    assert options["classes"] == 1
    assert options["activation"] is None
    if arch == "segformer-b0":
        assert options["decoder_segmentation_channels"] == 256
        assert options["upsampling"] == 4


def test_optional_segformer_support_does_not_break_older_architectures(recording_smp):
    module, calls = recording_smp
    del module.Segformer
    models.create_model("unet-resnet34")
    models.create_model("unetplusplus-resnet34")
    assert len(calls) == 2
    with pytest.raises(ImportError, match="Segformer"):
        models.create_model("segformer-b0")


@pytest.mark.parametrize("arch", ["unetplusplus-resnet34", "segformer-b0"])
def test_new_architectures_reject_unrelated_supervision_and_dilations(arch):
    with pytest.raises(ValueError, match="ASPP rates are only supported"):
        models.create_model(arch, aspp_rates=[6, 12, 18])
    with pytest.raises(ValueError, match="Boundary supervision supports only"):
        models.create_model(arch, boundary_supervision=True)
    with pytest.raises(ValueError, match="Boundary supervision supports only"):
        validate_boundary_options(arch, 0.2, 1)


@pytest.mark.parametrize("arch", ["unetplusplus-resnet34", "segformer-b0"])
def test_new_architectures_are_training_cli_choices(monkeypatch, tmp_path, arch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train",
            "--data",
            str(tmp_path),
            "--run",
            str(tmp_path / "run"),
            "--arch",
            arch,
        ],
    )
    assert parse_args().arch == arch


@pytest.mark.parametrize("arch", ["unetplusplus-resnet34", "segformer-b0"])
def test_real_cpu_outputs_roundtrip_exactly_without_downloading_weights(
    tmp_path, monkeypatch, arch
):
    smp = pytest.importorskip("segmentation_models_pytorch")
    if arch == "segformer-b0" and not hasattr(smp, "Segformer"):
        pytest.skip("Installed optional SMP version predates Segformer")
    from roof_pipeline.dataset import MEAN, STD

    encoder = "mit_b0" if arch == "segformer-b0" else "resnet34"
    preprocessing = smp.encoders.get_preprocessing_params(encoder, "imagenet")
    assert preprocessing["input_space"] == "RGB"
    assert preprocessing["input_range"] == [0, 1]
    torch.testing.assert_close(
        torch.tensor(preprocessing["mean"], dtype=torch.float32),
        torch.from_numpy(MEAN.reshape(-1)),
    )
    torch.testing.assert_close(
        torch.tensor(preprocessing["std"], dtype=torch.float32),
        torch.from_numpy(STD.reshape(-1)),
    )
    torch.set_num_threads(2)
    original = models.create_model(arch, pretrained=False).eval()
    image = torch.randn(1, 3, 64, 96)
    with torch.inference_mode():
        expected = original(image)
    assert expected.shape == (1, 1, 64, 96)
    assert torch.isfinite(expected).all()
    config = {
        "arch": arch,
        "image_size": 64,
        "normalization": "imagenet_rgb_v1",
        "pretrained_encoder": True,
        "epoch": 2,
    }
    checkpoint = tmp_path / "binary.pt"
    state = {"state_dict": original.state_dict(), "config": config, "threshold": 0.5}
    torch.save(state, checkpoint)
    calls = []
    create = models.create_model

    def recorded_create(*args, **kwargs):
        calls.append(kwargs)
        return create(*args, **kwargs)

    monkeypatch.setattr(models, "create_model", recorded_create)
    restored, restored_config = models.load_model(checkpoint, "cpu")
    assert calls[-1]["pretrained"] is False
    assert restored_config == config
    with torch.inference_mode():
        torch.testing.assert_close(restored(image), expected, rtol=0, atol=0)
    # A real missing tensor must fail strict restoration, rather than quietly
    # initializing part of a supposedly frozen evaluated checkpoint.
    state["state_dict"].pop(next(iter(state["state_dict"])))
    torch.save(state, checkpoint)
    with pytest.raises(RuntimeError, match="Missing key"):
        models.load_model(checkpoint, "cpu")
