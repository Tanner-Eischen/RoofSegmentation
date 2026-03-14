"""
Rooftop segmentation dataset: images + masks (excluding mask files with 'label' in the name).
Uses pre-defined train/val/test filename lists.
"""

import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def _read_filenames(path: Path) -> list[str]:
    """Read newline-separated filenames, strip whitespace, skip empty."""
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    return [ln.strip() for ln in lines if ln.strip()]


def build_image_mask_pairs(
    images_dir: Path,
    masks_dir: Path,
    filename_list: list[str],
    exclude_label_in_mask_name: bool = True,
    exclude_empty_masks: bool = False,
    empty_threshold: float = 0.001,
) -> list[tuple[Path, Path]]:
    """
    Build (image_path, mask_path) pairs from a list of image filenames.
    Only includes pairs where the mask exists and (if exclude_label_in_mask_name)
    the mask filename does not contain the word 'label'.

    Args:
        exclude_empty_masks: If True, exclude masks with < empty_threshold roof coverage
        empty_threshold: Minimum fraction of roof pixels to include (default 0.1%)
    """
    pairs = []
    for name in filename_list:
        image_path = images_dir / name
        mask_path = masks_dir / name
        if not image_path.exists():
            continue
        if not mask_path.exists():
            continue
        if exclude_label_in_mask_name and "label" in os.path.basename(mask_path).lower():
            continue

        # Check if mask has roof pixels (load only a thumbnail for efficiency)
        if exclude_empty_masks:
            with Image.open(mask_path) as mask_img:
                # Load at reduced size for faster coverage check
                thumb = mask_img.resize((64, 64), Image.Resampling.NEAREST)
                if thumb.mode != 'L':
                    thumb = thumb.convert('L')
                thumb_arr = np.array(thumb)
                roof_fraction = np.mean(thumb_arr > 127)
                if roof_fraction < empty_threshold:
                    continue

        pairs.append((image_path, mask_path))
    return pairs


class RoofSegmentationDataset(Dataset):
    """Dataset of (image, mask) for rooftop segmentation. Masks are binary (0/1 or 0/255)."""

    def __init__(
        self,
        pairs: list[tuple[Path, Path]],
        transform=None,
        mask_binary_threshold: float = 127,
    ):
        self.pairs = pairs
        self.transform = transform
        self.mask_binary_threshold = mask_binary_threshold

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path, mask_path = self.pairs[idx]
        image = np.array(Image.open(image_path).convert("RGB"))
        mask = np.array(Image.open(mask_path))
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        mask = (mask > self.mask_binary_threshold).astype(np.float32)

        if self.transform:
            out = self.transform(image=image, mask=mask)
            image = out["image"]
            mask = out["mask"]
            if mask.dim() == 2:
                mask = mask.unsqueeze(0)
            mask = mask.float()
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            mask = torch.from_numpy(mask).unsqueeze(0).float()

        return image, mask


def get_train_val_test_pairs(
    data_root: Path,
    filenames_dir: str = "filenames",
    train_file: str = "train_filenames_roof_outline_v1.txt",
    val_file: str = "val_filenames_roof_outline_v1.txt",
    test_file: str = "test_filenames_roof_outline_v1.txt",
    exclude_label_in_mask_name: bool = True,
    exclude_empty_masks: bool = False,
    empty_threshold: float = 0.001,
    images_dir: Optional[Path] = None,
    masks_dir: Optional[Path] = None,
) -> tuple[list[tuple[Path, Path]], list[tuple[Path, Path]], list[tuple[Path, Path]]]:
    """Load train/val/test (image, mask) pairs. Filenames come from data_root/filenames_dir.
    If images_dir/masks_dir are provided, use them instead of data_root/images and data_root/masks.

    Args:
        exclude_empty_masks: If True, exclude samples with no roof pixels
        empty_threshold: Minimum fraction of roof pixels to include
    """
    data_root = Path(data_root)
    images_dir = Path(images_dir) if images_dir is not None else data_root / "images"
    masks_dir = Path(masks_dir) if masks_dir is not None else data_root / "masks"
    fn_dir = data_root / filenames_dir

    train_names = _read_filenames(fn_dir / train_file)
    val_names = _read_filenames(fn_dir / val_file)
    test_names = _read_filenames(fn_dir / test_file)

    train_pairs = build_image_mask_pairs(
        images_dir, masks_dir, train_names, exclude_label_in_mask_name,
        exclude_empty_masks, empty_threshold
    )
    val_pairs = build_image_mask_pairs(
        images_dir, masks_dir, val_names, exclude_label_in_mask_name,
        exclude_empty_masks, empty_threshold
    )
    test_pairs = build_image_mask_pairs(
        images_dir, masks_dir, test_names, exclude_label_in_mask_name,
        exclude_empty_masks, empty_threshold
    )
    return train_pairs, val_pairs, test_pairs
