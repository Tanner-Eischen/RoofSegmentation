"""Paired image/mask dataset with synchronized geometric augmentations."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset

from .common import read_manifest, read_paired_sample
from .validity import IGNORE_LABEL, resize_rgb_valid

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)


class RoofDataset(Dataset):
    def __init__(
        self,
        root: Path,
        split: str,
        size: int = 512,
        augment: bool = False,
        *,
        photometric: str = "standard",
        mask_overrides: dict[str, Path] | None = None,
    ):
        if photometric not in {"standard", "robust"}:
            raise ValueError("photometric must be standard or robust")
        if augment and split != "train":
            raise ValueError("Augmentation is restricted to the training split")
        self.root = Path(root)
        self.rows = [r for r in read_manifest(self.root) if r["split"] == split]
        if not self.rows:
            raise ValueError(f"Empty {split} split")
        if size < 32 or size % 32:
            raise ValueError("size must be >=32 and divisible by 32")
        self.size = int(size)
        self.augment = bool(augment)
        self.photometric = photometric
        self.mask_overrides = {
            ident: Path(path).resolve()
            for ident, path in (mask_overrides or {}).items()
        }
        if self.mask_overrides and split != "train":
            raise ValueError("Mask overrides are restricted to the training split")
        if set(self.mask_overrides) - {row["sample_id"] for row in self.rows}:
            raise ValueError("Mask overrides contain unknown training sample IDs")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        mask_path = self.mask_overrides.get(row["sample_id"], self.root / row["mask"])
        rgb, labels, valid = read_paired_sample(
            self.root,
            row,
            mask_path if row["sample_id"] in self.mask_overrides else None,
        )
        image = Image.fromarray(rgb)
        mask = Image.fromarray(labels)
        if image.size != (self.size, self.size):
            rgb, valid = resize_rgb_valid(rgb, valid, (self.size, self.size))
            image = Image.fromarray(rgb)
            mask = mask.resize((self.size, self.size), Image.Resampling.NEAREST)
        validity = Image.fromarray(valid.astype(np.uint8) * 255)
        if self.augment:
            if random.random() < 0.5:
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                validity = validity.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            if random.random() < 0.5:
                image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
                mask = mask.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
                validity = validity.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            rotations = random.randrange(4)
            if rotations:
                image = image.rotate(90 * rotations, expand=False)
                mask = mask.rotate(90 * rotations, expand=False)
                validity = validity.rotate(90 * rotations, expand=False)
            bounds = (0.7, 1.3) if self.photometric == "robust" else (0.85, 1.15)
            if random.random() < 0.65:
                image = ImageEnhance.Brightness(image).enhance(random.uniform(*bounds))
            if random.random() < 0.65:
                contrast_bounds = (
                    (0.75, 1.25) if self.photometric == "robust" else (0.85, 1.15)
                )
                factor = random.uniform(*contrast_bounds)
                keep = np.asarray(validity) > 0
                if keep.all():
                    image = ImageEnhance.Contrast(image).enhance(factor)
                else:
                    gray = np.asarray(image.convert("L"))
                    mean = int(gray[keep].mean() + 0.5) if keep.any() else 0
                    image = Image.blend(
                        Image.new("RGB", image.size, (mean,) * 3), image, factor
                    )
            if self.photometric == "robust" and random.random() < 0.5:
                gamma = random.uniform(0.7, 1.4)
                table = [round(255 * (value / 255) ** gamma) for value in range(256)]
                image = image.point(table * 3)
        x = np.asarray(image, dtype=np.float32) / 255.0
        x = (x - MEAN) / STD
        y = (np.asarray(mask, dtype=np.uint8) > 0).astype(np.float32)
        valid = np.asarray(validity) > 0
        x[~valid] = 0
        y[~valid] = IGNORE_LABEL
        return torch.from_numpy(
            np.ascontiguousarray(np.moveaxis(x, -1, 0))
        ), torch.from_numpy(y.copy()[None, ...])


def normalize_rgb(rgb: np.ndarray) -> torch.Tensor:
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError("Expected HWC RGB array")
    normalized = (rgb.astype(np.float32) / 255.0 - MEAN) / STD
    return torch.from_numpy(
        np.ascontiguousarray(normalized.transpose(2, 0, 1))
    ).unsqueeze(0)
