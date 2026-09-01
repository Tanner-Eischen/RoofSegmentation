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
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--image-size", type=int, default=256, help="Train crop/size (square)")
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
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
    p.add_argument(
        "--train-file",
        type=str,
        default=None,
        help="Optional manifest filename under data-root/filenames/ for training images",
    )
    p.add_argument(
        "--val-file",
        type=str,
        default=None,
        help="Optional manifest filename under data-root/filenames/ for validation images",
    )
    p.add_argument(
        "--test-file",
        type=str,
        default=None,
        help="Optional manifest filename under data-root/filenames/ for test images",
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


def iou_binary(pred: torch.Tensor, target: torch.Tensor, thresh: float = 0.5, smooth: float = 1e-6) -> float:
    pred = (pred.sigmoid() > thresh).float()
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection + smooth) / (union + smooth)


def intersection_union_binary(pred: torch.Tensor, target: torch.Tensor, thresh: float = 0.5) -> tuple[torch.Tensor, torch.Tensor]:
    pred = (pred.sigmoid() > thresh).float()
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return intersection, union


def _resolve_manifest_name(data_root: Path, explicit_name: str | None, clean_name: str, default_name: str) -> str:
    if explicit_name:
        return explicit_name
    filenames_dir = Path(data_root) / "filenames"
    if (filenames_dir / clean_name).exists():
        return clean_name
    return default_name


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion_bce: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_iou = 0.0
    num_batches = 0
    pbar = tqdm(loader, desc="Train", leave=False)
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss_bce = criterion_bce(logits, masks)
        loss_dice = dice_loss(logits, masks)
        loss = loss_bce + loss_dice
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            iou = iou_binary(logits, masks)
        total_loss += loss.item()
        total_iou += iou.item()
        num_batches += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}", iou=f"{iou.item():.4f}")
    return total_loss / max(num_batches, 1), total_iou / max(num_batches, 1)


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion_bce: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_iou = 0.0
    total_intersection = 0.0
    total_union = 0.0
    smooth = 1e-6
    num_batches = 0
    for images, masks in tqdm(loader, desc="Val", leave=False):
        images = images.to(device)
        masks = masks.to(device)
        logits = model(images)
        loss_bce = criterion_bce(logits, masks)
        loss_dice = dice_loss(logits, masks)
        loss = loss_bce + loss_dice
        iou = iou_binary(logits, masks)
        intersection, union = intersection_union_binary(logits, masks)
        total_loss += loss.item()
        total_iou += iou.item()
        total_intersection += intersection.item()
        total_union += union.item()
        num_batches += 1
    val_loss = total_loss / max(num_batches, 1)
    val_iou_batch_mean = total_iou / max(num_batches, 1)
    val_iou_global = (total_intersection + smooth) / (total_union + smooth)
    return val_loss, val_iou_batch_mean, val_iou_global


def main():
    args = get_args()
    args.output_dir = Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    images_dir, masks_dir = _resolve_images_masks_dirs(args)
    train_file = _resolve_manifest_name(
        args.data_root,
        args.train_file,
        "train_filenames_roof_outline_v1_clean.txt",
        "train_filenames_roof_outline_v1.txt",
    )
    val_file = _resolve_manifest_name(
        args.data_root,
        args.val_file,
        "val_filenames_roof_outline_v1_clean.txt",
        "val_filenames_roof_outline_v1.txt",
    )
    test_file = _resolve_manifest_name(
        args.data_root,
        args.test_file,
        "test_filenames_roof_outline_v1_clean.txt",
        "test_filenames_roof_outline_v1.txt",
    )

    train_pairs, val_pairs, test_pairs = get_train_val_test_pairs(
        args.data_root,
        train_file=train_file,
        val_file=val_file,
        test_file=test_file,
        exclude_label_in_mask_name=not args.no_label_filter,
        images_dir=images_dir,
        masks_dir=masks_dir,
    )
    if not train_pairs:
        print("No training pairs found. Check --data-root and that images/masks exist and mask names don't contain 'label' (unless --no-label-filter).", file=sys.stderr)
        sys.exit(1)
    print(
        f"Train pairs: {len(train_pairs)}, Val: {len(val_pairs)}, Test: {len(test_pairs)} "
        f"| manifests: train={train_file}, val={val_file}, test={test_file}"
    )

    try:
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
        train_tf = A.Compose([
            A.Resize(args.image_size, args.image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion_bce = nn.BCEWithLogitsLoss()

    best_val_iou = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss, train_iou = train_one_epoch(model, train_loader, optimizer, criterion_bce, device)
        log = {"epoch": epoch, "train_loss": train_loss, "train_iou": train_iou}
        if val_loader:
            val_loss, val_iou_batch_mean, val_iou_global = validate(model, val_loader, criterion_bce, device)
            log["val_loss"] = val_loss
            log["val_iou"] = val_iou_global
            log["val_iou_global"] = val_iou_global
            log["val_iou_batch_mean"] = val_iou_batch_mean
            if val_iou_global > best_val_iou:
                best_val_iou = val_iou_global
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_iou": val_iou_global,
                        "val_iou_global": val_iou_global,
                        "val_iou_batch_mean": val_iou_batch_mean,
                        "args": vars(args),
                    },
                    args.output_dir / "best.pt",
                )
        history.append(log)
        print(f"Epoch {epoch}: train_loss={train_loss:.4f} train_iou={train_iou:.4f}", end="")
        if val_loader:
            print(
                f" val_loss={val_loss:.4f}"
                f" val_iou_global={val_iou_global:.4f}"
                f" val_iou_batch_mean={val_iou_batch_mean:.4f}"
            )
        else:
            print()
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
    print(f"Done. Best global val IoU: {best_val_iou:.4f}. Checkpoints in {args.output_dir}")


if __name__ == "__main__":
    main()
