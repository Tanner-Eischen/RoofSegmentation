"""Paired image/mask dataset with synchronized geometric augmentations."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset

from .common import load_binary_mask, read_manifest

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
        with Image.open(self.root / row["image"]) as raw:
            image = raw.convert("RGB")
        mask_path = self.mask_overrides.get(row["sample_id"], self.root / row["mask"])
        mask = Image.fromarray(load_binary_mask(mask_path, image.size))
        if image.size != (self.size, self.size):
            image = image.resize((self.size, self.size), Image.Resampling.BILINEAR)
            mask = mask.resize((self.size, self.size), Image.Resampling.NEAREST)
        if self.augment:
            if random.random() < 0.5:
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            if random.random() < 0.5:
                image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
                mask = mask.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            rotations = random.randrange(4)
            if rotations:
                image = image.rotate(90 * rotations, expand=False)
                mask = mask.rotate(90 * rotations, expand=False)
            bounds = (0.7, 1.3) if self.photometric == "robust" else (0.85, 1.15)
            if random.random() < 0.65:
                image = ImageEnhance.Brightness(image).enhance(random.uniform(*bounds))
            if random.random() < 0.65:
                contrast_bounds = (
                    (0.75, 1.25) if self.photometric == "robust" else (0.85, 1.15)
                )
                image = ImageEnhance.Contrast(image).enhance(
                    random.uniform(*contrast_bounds)
                )
            if self.photometric == "robust" and random.random() < 0.5:
                gamma = random.uniform(0.7, 1.4)
                table = [round(255 * (value / 255) ** gamma) for value in range(256)]
                image = image.point(table * 3)
        x = np.asarray(image, dtype=np.float32) / 255.0
        x = (x - MEAN) / STD
        y = (np.asarray(mask, dtype=np.uint8) > 0).astype(np.float32)
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
