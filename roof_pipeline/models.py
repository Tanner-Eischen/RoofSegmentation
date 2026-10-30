"""Simple smoke model plus configurable segmentation_models_pytorch architectures."""

from __future__ import annotations

import inspect

import torch
import torch.nn.functional as F
from torch import nn

DEFAULT_ASPP_RATES = (12, 24, 36)


def validate_aspp_rates(arch: str, aspp_rates) -> tuple[int, int, int] | None:
    """Keep decoder dilation an explicit, primitive-safe checkpoint contract."""
    if aspp_rates is None:
        return None
    if arch != "deeplabv3plus-resnet34":
        raise ValueError("ASPP rates are only supported for deeplabv3plus-resnet34")
    if (
        not isinstance(aspp_rates, (tuple, list))
        or len(aspp_rates) != 3
        or any(type(rate) is not int or rate <= 0 for rate in aspp_rates)
    ):
        raise ValueError("ASPP rates must contain exactly three positive integers")
    return tuple(aspp_rates)


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class TinyUNet(nn.Module):
    def __init__(self, width: int = 8):
        super().__init__()
        self.a = ConvBlock(3, width)
        self.b = ConvBlock(width, width * 2)
        self.c = ConvBlock(width * 2, width * 4)
        self.d = ConvBlock(width * 4, width * 8)
        self.up_c = ConvBlock(width * 12, width * 4)
        self.up_b = ConvBlock(width * 6, width * 2)
        self.up_a = ConvBlock(width * 3, width)
        self.head = nn.Conv2d(width, 1, 1)

    def forward_features(self, x):
        a = self.a(x)
        b = self.b(F.max_pool2d(a, 2))
        c = self.c(F.max_pool2d(b, 2))
        d = self.d(F.max_pool2d(c, 2))
        c2 = self.up_c(
            torch.cat(
                [
                    F.interpolate(
                        d, size=c.shape[-2:], mode="bilinear", align_corners=False
                    ),
                    c,
                ],
                dim=1,
            )
        )
        b2 = self.up_b(
            torch.cat(
                [
                    F.interpolate(
                        c2, size=b.shape[-2:], mode="bilinear", align_corners=False
                    ),
                    b,
                ],
                dim=1,
            )
        )
        a2 = self.up_a(
            torch.cat(
                [
                    F.interpolate(
                        b2, size=a.shape[-2:], mode="bilinear", align_corners=False
                    ),
                    a,
                ],
                dim=1,
            )
        )
        return a2

    def forward(self, x):
        return self.head(self.forward_features(x))


class BoundaryUNet(nn.Module):
    """One binary inference output, with an independent head on shared features."""

    def __init__(self, base: nn.Module):
        super().__init__()
        self.base = base
        if isinstance(base, TinyUNet):
            channels = base.head.in_channels
            self.decoder_variadic = False
        else:
            channels = base.segmentation_head[0].in_channels
            self.decoder_variadic = any(
                parameter.kind is inspect.Parameter.VAR_POSITIONAL
                for parameter in inspect.signature(
                    base.decoder.forward
                ).parameters.values()
            )
        # A fresh head must not perturb the training RNG stream relative to a
        # binary-only control. The head is constructed on CPU before .to(device).
        with torch.random.fork_rng(devices=[]):
            self.boundary_head = nn.Conv2d(channels, 1, kernel_size=1)

    def forward(self, x):
        return self.base(x)

    def forward_with_boundary(self, x):
        if isinstance(self.base, TinyUNet):
            features = self.base.forward_features(x)
            masks = self.base.head(features)
        else:
            self.base.check_input_shape(x)
            encoded = self.base.encoder(x)
            features = (
                self.base.decoder(*encoded)
                if self.decoder_variadic
                else self.base.decoder(encoded)
            )
            masks = self.base.segmentation_head(features)
        return masks, self.boundary_head(features)


def create_model(
    arch: str, *, pretrained: bool = False, aspp_rates=None, boundary_supervision=False
) -> nn.Module:
    rates = validate_aspp_rates(arch, aspp_rates)
    if boundary_supervision and arch not in {"tiny", "unet-resnet34"}:
        raise ValueError("Boundary supervision supports only tiny and unet-resnet34")
    if arch == "tiny":
        model = TinyUNet()
        return BoundaryUNet(model) if boundary_supervision else model
    try:
        import segmentation_models_pytorch as smp
    except ImportError as exc:
        raise ImportError(
            "Install requirements-roof-pipeline.txt for pretrained U-Net architectures. 'tiny' works without SMP."
        ) from exc
    weights = "imagenet" if pretrained else None
    configs = {
        "unet-resnet34": (smp.Unet, "resnet34"),
        "unetplusplus-efficientnet-b0": (smp.UnetPlusPlus, "efficientnet-b0"),
        "deeplabv3plus-resnet34": (smp.DeepLabV3Plus, "resnet34"),
    }
    if arch not in configs:
        raise ValueError(f"Unknown model {arch!r}: choose tiny or {list(configs)}")
    cls, encoder = configs[arch]
    decoder_options = {"decoder_atrous_rates": rates} if rates is not None else {}
    model = cls(
        encoder_name=encoder,
        encoder_weights=weights,
        in_channels=3,
        classes=1,
        **decoder_options,
    )
    return BoundaryUNet(model) if boundary_supervision else model


def load_model(checkpoint, device="cpu"):
    from pathlib import Path

    state = torch.load(Path(checkpoint), map_location="cpu", weights_only=True)
    cfg = state["config"]
    rates = cfg.get("aspp_rates")
    if cfg["arch"] == "deeplabv3plus-resnet34" and rates is None:
        # Legacy weights used these dilations; tensor shapes cannot detect a change.
        rates = DEFAULT_ASPP_RATES
    model = create_model(
        cfg["arch"],
        pretrained=False,
        aspp_rates=rates,
        boundary_supervision=cfg.get("boundary_supervision", False),
    )
    model.load_state_dict(state["state_dict"], strict=True)
    model = model.to(device).eval()
    return model, cfg
