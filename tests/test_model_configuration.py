"""Decoder dilation must survive serialization even when weight shapes match."""

import argparse
import sys
from types import ModuleType

import pytest
import torch
from torch import nn

from roof_pipeline.models import DEFAULT_ASPP_RATES, create_model, load_model
from roof_pipeline.train import initialize_model, train
from scripts.make_smoke_dataset import make_fixture

DEEP = "deeplabv3plus-resnet34"


class DilationStub(nn.Module):
    def __init__(self, *, decoder_atrous_rates=DEFAULT_ASPP_RATES, **kwargs):
        super().__init__()
        self.rates = tuple(decoder_atrous_rates)
        self.constructor_options = kwargs
        self.branches = nn.ModuleList(
            [
                nn.Conv2d(3, 1, 3, dilation=rate, padding=rate, bias=False)
                for rate in self.rates
            ]
        )

    def forward(self, x):
        return sum(branch(x) for branch in self.branches)


@pytest.fixture
def stub_smp(monkeypatch):
    module = ModuleType("segmentation_models_pytorch")
    module.DeepLabV3Plus = DilationStub
    module.Unet = DilationStub
    module.UnetPlusPlus = DilationStub
    monkeypatch.setitem(sys.modules, "segmentation_models_pytorch", module)


def _checkpoint(path, rates, *, include_rates=True):
    model = create_model(DEEP, aspp_rates=rates)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(1)
    config = {
        "arch": DEEP,
        "image_size": 64,
        "normalization": "imagenet_rgb_v1",
        "dataset_content_sha256": "frozen",
        "pretrained_encoder": True,
        "epoch": 32,
    }
    if include_rates:
        config["aspp_rates"] = list(rates) if rates is not None else None
    torch.save(
        {"state_dict": model.state_dict(), "config": config, "threshold": 0.5}, path
    )
    return model


def test_custom_dilation_round_trip_preserves_predictions_despite_identical_weight_shapes(
    tmp_path, stub_smp
):
    torch.set_num_threads(2)
    path = tmp_path / "custom.pt"
    original = _checkpoint(path, [6, 12, 18])
    image = torch.zeros(1, 3, 80, 80)
    image[:, :, 40, 40] = 1
    restored, config = load_model(path)
    assert config["aspp_rates"] == [6, 12, 18]
    assert restored.rates == (6, 12, 18)
    torch.testing.assert_close(restored(image), original(image), rtol=0, atol=0)
    legacy = create_model(DEEP)
    legacy.load_state_dict(original.state_dict(), strict=True)
    assert {k: v.shape for k, v in legacy.state_dict().items()} == {
        k: v.shape for k, v in original.state_dict().items()
    }
    assert not torch.equal(legacy(image), original(image))


@pytest.mark.parametrize("include_rates", [False, True])
def test_legacy_missing_or_null_rates_keep_original_defaults(
    tmp_path, stub_smp, include_rates
):
    path = tmp_path / "legacy.pt"
    original = _checkpoint(path, None, include_rates=include_rates)
    restored, _ = load_model(path)
    assert original.rates == restored.rates == (12, 24, 36)


@pytest.mark.parametrize("include_rates", [False, True])
def test_legacy_restoration_is_independent_of_future_constructor_defaults(
    tmp_path, stub_smp, monkeypatch, include_rates
):
    checkpoint = tmp_path / "legacy.pt"
    original = _checkpoint(checkpoint, None, include_rates=include_rates)

    class ChangedDefaultStub(DilationStub):
        def __init__(self, *, decoder_atrous_rates=(99, 99, 99), **kwargs):
            super().__init__(decoder_atrous_rates=decoder_atrous_rates, **kwargs)

    monkeypatch.setattr(
        sys.modules["segmentation_models_pytorch"], "DeepLabV3Plus", ChangedDefaultStub
    )
    assert create_model(DEEP).rates == (99, 99, 99)
    restored, _ = load_model(checkpoint)
    assert restored.rates == DEFAULT_ASPP_RATES
    image = torch.zeros(1, 3, 80, 80)
    image[:, :, 40, 40] = 1
    torch.testing.assert_close(restored(image), original(image), rtol=0, atol=0)


@pytest.mark.parametrize(
    "rates",
    [
        [],
        [6, 12],
        [6, 12, 18, 24],
        [6, 0, 18],
        [6, -1, 18],
        [6, True, 18],
        [6, 12.0, 18],
        [None, 12, 18],
        "6,12,18",
    ],
)
def test_invalid_rates_are_rejected_without_loading_smp(rates):
    with pytest.raises(ValueError, match="rates"):
        create_model(DEEP, aspp_rates=rates)


@pytest.mark.parametrize(
    "arch", ["tiny", "unet-resnet34", "unetplusplus-efficientnet-b0"]
)
def test_non_deeplab_architectures_reject_aspp_configuration(arch):
    with pytest.raises(ValueError, match="only supported"):
        create_model(arch, aspp_rates=[6, 12, 18])


def test_new_initialization_passes_rates_and_imagenet_flag(stub_smp):
    args = argparse.Namespace(arch=DEEP, pretrained=True, aspp_rates=[6, 12, 18])
    model, initialization = initialize_model(args, "cpu", "frozen")
    assert model.rates == (6, 12, 18)
    assert model.constructor_options["encoder_weights"] == "imagenet"
    assert initialization == {}


@pytest.mark.parametrize(
    "rates,requested,include_rates,actual",
    [
        ([6, 12, 18], None, True, [6, 12, 18]),
        ([6, 12, 18], [6, 12, 18], True, [6, 12, 18]),
        (None, None, False, [12, 24, 36]),
        (None, [12, 24, 36], False, [12, 24, 36]),
    ],
)
def test_warm_start_preserves_actual_checkpoint_rates(
    tmp_path, stub_smp, rates, requested, include_rates, actual
):
    checkpoint = tmp_path / "initial.pt"
    _checkpoint(checkpoint, rates, include_rates=include_rates)
    original_bytes = checkpoint.read_bytes()
    args = argparse.Namespace(
        arch=DEEP, size=64, init_checkpoint=checkpoint, aspp_rates=requested
    )
    model, initialization = initialize_model(args, "cpu", "frozen")
    assert list(model.rates) == initialization["aspp_rates"] == actual
    assert initialization["pretrained_encoder"] is True
    assert checkpoint.read_bytes() == original_bytes


@pytest.mark.parametrize(
    "rates,requested,include_rates",
    [
        ([6, 12, 18], [12, 24, 36], True),
        (None, [6, 12, 18], False),
    ],
)
def test_warm_start_rejects_explicit_dilation_change(
    tmp_path, stub_smp, rates, requested, include_rates
):
    checkpoint = tmp_path / "initial.pt"
    _checkpoint(checkpoint, rates, include_rates=include_rates)
    args = argparse.Namespace(
        arch=DEEP, size=64, init_checkpoint=checkpoint, aspp_rates=requested
    )
    with pytest.raises(ValueError, match="ASPP rates differ"):
        initialize_model(args, "cpu", "frozen")


def test_training_checkpoint_records_primitive_rates_and_reloads_them(
    tmp_path, stub_smp
):
    torch.set_num_threads(2)
    data = make_fixture(tmp_path / "data", 2, 1, 1, 64)
    args = argparse.Namespace(
        data=data,
        run=tmp_path / "run",
        arch=DEEP,
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
        patience=2,
        min_delta=0.0001,
        no_amp=True,
        aspp_rates=[6, 12, 18],
    )
    summary = train(args)
    state = torch.load(args.run / "best.pt", map_location="cpu", weights_only=True)
    assert (
        summary["config"]["aspp_rates"] == state["config"]["aspp_rates"] == [6, 12, 18]
    )
    restored, _ = load_model(args.run / "best.pt")
    assert restored.rates == (6, 12, 18)
