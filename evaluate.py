"""
evaluate.py — evaluate a trained checkpoint on a split with honest metrics.

Loads best.pt (any arch trained by train_v2.py, or the original train.py U-Net),
runs IoU / Dice / pixel-accuracy / precision / recall / F1 at the tuned
threshold, with optional test-time augmentation (TTA) and threshold sweep.

Reuses dataset.py for pairs and train_v2.py for metrics/transforms/arch build.

Examples:
    python evaluate.py --checkpoint outputs_v2_unetpp_seco/best.pt \\
        --data-root segmentation_model_data_v2 --split test

    python evaluate.py --checkpoint best.pt --data-root ... --split val --tta --sweep
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


from dataset import RoofSegmentationDataset, get_train_val_test_pairs
from train_v2 import (
    ARCHES, build_transforms, evaluate_split, threshold_sweep,
)

SPLIT_FILES = {
    "train": "train_filenames_clean.txt",
    "val": "val_filenames_clean.txt",
    "test": "test_filenames_clean.txt",
}


class TTAWrapper(nn.Module):
    """Average logits over identity + horizontal flip + vertical flip."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        hflip = torch.flip(self.model(torch.flip(x, dims=[3])), dims=[3])
        vflip = torch.flip(self.model(torch.flip(x, dims=[2])), dims=[2])
        return (out + hflip + vflip) / 3.0


def load_checkpoint(checkpoint: Path, device: torch.device):
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    args = ckpt.get("args", {})
    arch = args.get("arch", "unet")
    encoder = args.get("encoder", "resnet34")
    image_size = int(args.get("image_size", 256))
    threshold = float(ckpt.get("threshold", args.get("threshold", 0.5)))
    if arch not in ARCHES:
        print(f"[load] arch '{arch}' not in {list(ARCHES)}; defaulting to 'unet'.", file=sys.stderr)
        arch = "unet"
    model = ARCHES[arch](encoder_name=encoder, encoder_weights=None, in_channels=3, classes=1)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model = model.to(device).eval()
    return model, {"arch": arch, "encoder": encoder, "image_size": image_size,
                   "threshold": threshold, "epoch": ckpt.get("epoch")}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--images-dir", type=str, default=None)
    p.add_argument("--masks-dir", type=str, default=None)
    p.add_argument("--split", choices=list(SPLIT_FILES), default="test")
    p.add_argument("--file", type=str, default=None, help="Override manifest filename")
    p.add_argument("--threshold", type=float, default=None,
                   help="Override threshold (default: checkpoint's tuned threshold)")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--tta", action="store_true", help="Test-time augmentation (h/v flip avg)")
    p.add_argument("--sweep", action="store_true", help="Sweep thresholds and report best IoU")
    p.add_argument("--no-label-filter", action="store_true")
    p.add_argument("--output", type=Path, default=None, help="Write metrics JSON here")
    args = p.parse_args()

    if not args.checkpoint.exists():
        print(f"ERROR: checkpoint not found: {args.checkpoint}", file=sys.stderr)
        return 2
    device = torch.device(args.device)
    use_amp = device.type == "cuda"

    model, info = load_checkpoint(args.checkpoint, device)
    threshold = args.threshold if args.threshold is not None else info["threshold"]
    print(f"Checkpoint: {args.checkpoint} | arch={info['arch']} encoder={info['encoder']} "
          f"image_size={info['image_size']} epoch={info['epoch']} threshold={threshold:.3f} tta={args.tta}")

    images_dir = args.images_dir or str(args.data_root / "images")
    masks_dir = args.masks_dir or str(args.data_root / "masks")
    split_file = args.file or SPLIT_FILES[args.split]
    # Pull only the requested split's pairs.
    train_pairs, val_pairs, test_pairs = get_train_val_test_pairs(
        args.data_root,
        train_file="train_filenames_clean.txt", val_file="val_filenames_clean.txt",
        test_file="test_filenames_clean.txt",
        exclude_label_in_mask_name=not args.no_label_filter,
        images_dir=images_dir, masks_dir=masks_dir,
    )
    pairs = {"train": train_pairs, "val": val_pairs, "test": test_pairs}[args.split]
    if not pairs:
        print(f"No '{args.split}' pairs resolved ({split_file}). Check --data-root / manifests.",
              file=sys.stderr)
        return 2
    print(f"{args.split} pairs: {len(pairs)}")

    _, val_tf = build_transforms(info["image_size"])
    loader = DataLoader(RoofSegmentationDataset(pairs, transform=val_tf),
                        batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=(device.type == "cuda"))

    fwd = TTAWrapper(model) if args.tta else model

    result = {"checkpoint": str(args.checkpoint), "split": args.split,
              "n_pairs": len(pairs), "tta": args.tta, **info}

    if args.sweep:
        best_thr, best_iou, sweep = threshold_sweep(fwd, loader, device, use_amp)
        result["sweep_best_threshold"] = best_thr
        result["sweep_best_iou"] = best_iou
        print(f"Sweep: best threshold={best_thr:.3f} -> IoU={best_iou:.4f}")
        metrics = evaluate_split(fwd, loader, device, use_amp, best_thr)
    else:
        metrics = evaluate_split(fwd, loader, device, use_amp, threshold)

    result["metrics"] = metrics
    print("\nMetrics at threshold {:.3f}{}:".format(metrics["threshold"], " (TTA)" if args.tta else ""))
    for k in ("iou", "dice", "accuracy", "precision", "recall", "f1"):
        print(f"  {k:10s}: {metrics[k]:.4f}")

    out = args.output or (args.checkpoint.parent / f"eval_{args.split}.json")
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
