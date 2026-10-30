"""Mask-derived auxiliary targets; binary building labels remain the primary task."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

BOUNDARY_TARGET = "morphological_band_replicate_v1"


def validate_boundary_options(arch: str, weight: float, radius: int):
    if not math.isfinite(weight) or weight < 0:
        raise ValueError("Boundary weight must be finite and nonnegative")
    if type(radius) is not int or radius < 1:
        raise ValueError("Boundary radius must be a positive integer")
    if weight > 0 and arch not in {"tiny", "unet-resnet34"}:
        raise ValueError("Boundary supervision supports only tiny and unet-resnet34")


def boundary_target(labels: torch.Tensor, radius: int = 1) -> torch.Tensor:
    """Dilated minus eroded binary mask, without assuming background beyond a tile.

    Call on the resized/augmented batch, so the edge and segmentation targets
    share exactly the same pixel grid. An all-positive tile has no visible edge.
    """
    if type(radius) is not int or radius < 1:
        raise ValueError("Boundary radius must be a positive integer")
    if labels.ndim != 4 or labels.shape[1] != 1:
        raise ValueError("Boundary targets require N x 1 x H x W masks")
    mask = (labels > 0.5).to(dtype=labels.dtype)
    padded = F.pad(mask, (radius,) * 4, mode="replicate")
    kernel = 2 * radius + 1
    dilated = F.max_pool2d(padded, kernel, stride=1)
    eroded = -F.max_pool2d(-padded, kernel, stride=1)
    return dilated - eroded
