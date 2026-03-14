"""
Train U-Net (with optional pre-trained encoder) for rooftop segmentation.
Uses images and masks from segmentation_model_data; only masks without 'label' in the filename.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import segmentation_models_pytorch as smp

from dataset import (
    RoofSegmentationDataset,
    get_train_val_test_pairs,
)


def get_args():
    p = argparse.ArgumentParser(description="Train rooftop segmentation U-Net")
    default_data = os.environ.get("SM_CHANNEL_TRAIN") or os.environ.get("DATA_ROOT") or "segmentation_model_data"
    p.add_argument(
        "--data-root",
        type=Path,
        default=Path(default_data),
        help="Root dir for filenames/ (and images/masks if --images-dir/--masks-dir not set)",
    )
    p.add_argument(
        "--images-dir",
        type=str,
        default=None,
        help="Images directory or S3 URI (e.g. s3://bucket/data/images/). Overrides data-root/images.",
    )
    p.add_argument(
        "--masks-dir",
        type=str,
        default=None,
        help="Masks directory or S3 URI (e.g. s3://bucket/data/masks/). Overrides data-root/masks.",
    )
    p.add_argument(
        "--sync-s3",
        action="store_true",
        help="If --images-dir/--masks-dir are s3:// URIs, sync them to local cache before training",
    )
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--image-size", type=int, default=400, help="Train crop/size (square)")
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--exclude-empty-masks",
        action="store_true",
        help="Exclude training samples with no roof pixels",
    )
    p.add_argument(
        "--pos-weight",
        type=float,
        default=20.0,
        help="Weight for positive class in BCE loss (for class imbalance)",
    )
    p.add_argument(
        "--use-focal",
        action="store_true",
        help="Use Focal loss instead of BCE",
    )
    p.add_argument(
        "--use-cosine",
        action="store_true",
        help="Use cosine annealing learning rate scheduler",
    )
    p.add_argument(
        "--encoder",
        type=str,
        default="resnet34",
        help="Encoder for U-Net (e.g. resnet34, efficientnet-b0)",
    )
    p.add_argument(
        "--encoder-weights",
        type=str,
        default="imagenet",
        help="Pre-trained encoder weights (imagenet or None)",
    )
    out_default = os.environ.get("SM_MODEL_DIR") or os.environ.get("OUTPUT_DIR") or "outputs"
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path(out_default),
        help="Where to save checkpoints and logs",
    )
    p.add_argument(
        "--save-every",
        type=int,
        default=5,
        help="Save checkpoint every N epochs",
    )
    p.add_argument(
        "--no-label-filter",
        action="store_true",
        help="Do NOT exclude mask files with 'label' in filename",
    )
    return p.parse_args()


def _is_s3_uri(path: str) -> bool:
    return path.strip().lower().startswith("s3://")


def _sync_s3_to_local(s3_uri: str, local_dir: Path) -> Path:
    """Sync s3_uri to local_dir using aws s3 sync. Returns local_dir."""
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    s3_uri = s3_uri.rstrip("/")
    subprocess.run(
        ["aws", "s3", "sync", s3_uri, str(local_dir), "--quiet"],
        check=True,
    )
    return local_dir


def _resolve_images_masks_dirs(args) -> tuple[Path, Path]:
    """Resolve effective images_dir and masks_dir (local paths). If S3 URIs, sync to local when --sync-s3."""
    data_root = Path(args.data_root)
    images_dir = args.images_dir or str(data_root / "images")
    masks_dir = args.masks_dir or str(data_root / "masks")
    cache_base = args.output_dir / "s3_data"
    if _is_s3_uri(images_dir):
        if not getattr(args, "sync_s3", False):
            print("Error: --images-dir is an S3 URI. Add --sync-s3 to download before training.", file=sys.stderr)
            sys.exit(1)
        images_dir = _sync_s3_to_local(images_dir, cache_base / "images")
        print(f"Synced images from S3 to {images_dir}")
    else:
        images_dir = Path(images_dir)
    if _is_s3_uri(masks_dir):
        if not getattr(args, "sync_s3", False):
            print("Error: --masks-dir is an S3 URI. Add --sync-s3 to download before training.", file=sys.stderr)
            sys.exit(1)
        masks_dir = _sync_s3_to_local(masks_dir, cache_base / "masks")
        print(f"Synced masks from S3 to {masks_dir}")
    else:
        masks_dir = Path(masks_dir)
    return images_dir, masks_dir


def dice_loss(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    pred = pred.sigmoid()
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    dice = (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    return 1.0 - dice


def focal_loss(pred: torch.Tensor, target: torch.Tensor, alpha: float = 0.25, gamma: float = 2.0) -> torch.Tensor:
    """Focal loss for handling class imbalance. Focuses on hard examples."""
    pred_prob = pred.sigmoid()
    bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
    p_t = target * pred_prob + (1 - target) * (1 - pred_prob)
    focal_weight = alpha * (1 - p_t) ** gamma
    return (focal_weight * bce).mean()


def combined_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    criterion_bce: nn.Module,
    use_focal: bool = False,
    dice_weight: float = 1.0,
    bce_weight: float = 1.0,
) -> torch.Tensor:
    """Combined BCE/Dice or Focal/Dice loss."""
    if use_focal:
        loss_pixel = focal_loss(pred, target)
    else:
        loss_pixel = criterion_bce(pred, target)
    loss_dice = dice_loss(pred, target)
    return bce_weight * loss_pixel + dice_weight * loss_dice


def iou_binary(pred: torch.Tensor, target: torch.Tensor, thresh: float = 0.5, smooth: float = 1e-6) -> float:
    pred = (pred.sigmoid() > thresh).float()
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection + smooth) / (union + smooth)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion_bce: nn.Module,
    device: torch.device,
    use_focal: bool = False,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_iou = 0.0
    n = 0
    pbar = tqdm(loader, desc="Train", leave=False)
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = combined_loss(logits, masks, criterion_bce, use_focal)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            iou = iou_binary(logits, masks)
        total_loss += loss.item()
        total_iou += iou.item()
        n += images.size(0)
        pbar.set_postfix(loss=f"{loss.item():.4f}", iou=f"{iou.item():.4f}")
    return total_loss / max(n, 1), total_iou / max(n, 1)


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion_bce: nn.Module,
    device: torch.device,
    use_focal: bool = False,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_iou = 0.0
    n = 0
    for images, masks in tqdm(loader, desc="Val", leave=False):
        images = images.to(device)
        masks = masks.to(device)
        logits = model(images)
        loss = combined_loss(logits, masks, criterion_bce, use_focal)
        iou = iou_binary(logits, masks)
        total_loss += loss.item()
        total_iou += iou.item()
        n += images.size(0)
    return total_loss / max(n, 1), total_iou / max(n, 1)


def main():
    args = get_args()
    args.output_dir = Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    images_dir, masks_dir = _resolve_images_masks_dirs(args)
    train_pairs, val_pairs, test_pairs = get_train_val_test_pairs(
        args.data_root,
        exclude_label_in_mask_name=not args.no_label_filter,
        exclude_empty_masks=args.exclude_empty_masks,
        images_dir=images_dir,
        masks_dir=masks_dir,
    )
    if not train_pairs:
        print("No training pairs found. Check --data-root and that images/masks exist and mask names don't contain 'label' (unless --no-label-filter).", file=sys.stderr)
        sys.exit(1)
    print(f"Train pairs: {len(train_pairs)}, Val: {len(val_pairs)}, Test: {len(test_pairs)}")

    try:
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
        train_tf = A.Compose([
            A.Resize(args.image_size, args.image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
        val_tf = A.Compose([
            A.Resize(args.image_size, args.image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    except ImportError:
        train_tf = val_tf = None

    train_ds = RoofSegmentationDataset(train_pairs, transform=train_tf)
    val_ds = RoofSegmentationDataset(val_pairs, transform=val_tf) if val_pairs else None
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=(args.device == "cuda"),
    )
    val_loader = None
    if val_ds:
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=(args.device == "cuda"),
        )

    model = smp.Unet(
        encoder_name=args.encoder,
        encoder_weights=args.encoder_weights or None,
        in_channels=3,
        classes=1,
    )
    device = torch.device(args.device)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # Use pos_weight to handle class imbalance (roof pixels are rare)
    pos_weight = torch.tensor([args.pos_weight]).to(device)
    criterion_bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Optional cosine annealing scheduler
    scheduler = None
    if args.use_cosine:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    use_focal = args.use_focal
    best_val_iou = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss, train_iou = train_one_epoch(
            model, train_loader, optimizer, criterion_bce, device, use_focal
        )
        log = {"epoch": epoch, "train_loss": train_loss, "train_iou": train_iou}
        if val_loader:
            val_loss, val_iou = validate(model, val_loader, criterion_bce, device, use_focal)
            log["val_loss"] = val_loss
            log["val_iou"] = val_iou
            if val_iou > best_val_iou:
                best_val_iou = val_iou
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_iou": val_iou,
                        "args": vars(args),
                    },
                    args.output_dir / "best.pt",
                )
        history.append(log)

        lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch}: train_loss={train_loss:.4f} train_iou={train_iou:.4f} lr={lr:.2e}", end="")
        if val_loader:
            print(f" val_loss={val_loss:.4f} val_iou={val_iou:.4f}")
        else:
            print()

        if scheduler:
            scheduler.step()

        if epoch % args.save_every == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "args": vars(args),
                },
                args.output_dir / f"checkpoint_epoch_{epoch}.pt",
            )
    (args.output_dir / "history.json").write_text(json.dumps(history, indent=2))
    print(f"Done. Best val IoU: {best_val_iou:.4f}. Checkpoints in {args.output_dir}")


if __name__ == "__main__":
    main()
