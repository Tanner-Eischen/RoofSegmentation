"""Leakage-conscious reproducible training for building/roof binary segmentation.

Example:
 python -m roof_pipeline.train --data data/whu --run runs/whu_r34 --arch unet-resnet34 --epochs 40

The test partition is NEVER read for selection/threshold tuning. Call evaluate.py once after finalizing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from .boundary import BOUNDARY_TARGET, boundary_target, validate_boundary_options
from .common import dataset_audit, dataset_fingerprint
from .correction_overlay import load_training_overlay, source_training_metadata
from .dataset import RoofDataset
from .metrics import choose_threshold, evaluate_loader
from .models import (
    DEFAULT_ASPP_RATES,
    BoundaryUNet,
    create_model,
    load_model,
    validate_aspp_rates,
)


def initialize_model(args, device, dataset_sha):
    """Warm-start full weights only when data, architecture and preprocessing match."""
    requested_rates = validate_aspp_rates(args.arch, getattr(args, "aspp_rates", None))
    boundary_weight = getattr(args, "boundary_weight", 0.0)
    boundary_radius = getattr(args, "boundary_radius", 1)
    validate_boundary_options(args.arch, boundary_weight, boundary_radius)
    checkpoint = getattr(args, "init_checkpoint", None)
    if checkpoint is None:
        return create_model(
            args.arch,
            pretrained=args.pretrained,
            aspp_rates=requested_rates,
            boundary_supervision=boundary_weight > 0,
        ).to(device), {}
    model, config = load_model(checkpoint, device)
    if config.get("dataset_content_sha256") != dataset_sha:
        raise ValueError("Initialization checkpoint dataset content SHA differs")
    if (
        config.get("arch") != args.arch
        or config.get("image_size") != args.size
        or config.get("normalization") != "imagenet_rgb_v1"
    ):
        raise ValueError(
            "Initialization checkpoint architecture/size/normalization differs"
        )
    actual_rates = validate_aspp_rates(config["arch"], config.get("aspp_rates"))
    if config["arch"] == "deeplabv3plus-resnet34":
        actual_rates = actual_rates or DEFAULT_ASPP_RATES
        if requested_rates is not None and requested_rates != actual_rates:
            raise ValueError(
                "Initialization checkpoint ASPP rates differ from requested rates"
            )
    if config.get("boundary_supervision", False):
        if boundary_weight == 0:
            raise ValueError(
                "Cannot disable boundary supervision on an auxiliary checkpoint"
            )
        if (
            config.get("boundary_radius") != boundary_radius
            or config.get("boundary_target") != BOUNDARY_TARGET
        ):
            raise ValueError("Initialization checkpoint boundary target/radius differs")
    elif boundary_weight > 0:
        # The checkpoint loader has strictly restored every original tensor.
        # Wrapping it preserves those weights and initializes only the edge head.
        model = BoundaryUNet(model).to(device)
    return model, {
        "initial_checkpoint": str(checkpoint),
        "initial_checkpoint_sha256": hashlib.sha256(
            Path(checkpoint).read_bytes()
        ).hexdigest(),
        "initial_checkpoint_epoch": config.get("epoch"),
        "pretrained_encoder": bool(config.get("pretrained_encoder", False)),
        "optimizer_restarted": True,
        "aspp_rates": list(actual_rates) if actual_rates is not None else None,
    }


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Warn rather than claim bitwise determinism across different GPU/CUDA versions.
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except AttributeError:
        pass


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def bce_dice_loss(logits, labels, pos_weight: float = 1.0):
    weight = torch.tensor(pos_weight, device=logits.device, dtype=logits.dtype)
    bce = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=weight)
    probs = torch.sigmoid(logits)
    dims = (1, 2, 3)
    dice = (2 * (probs * labels).sum(dim=dims) + 1) / (
        (probs + labels).sum(dim=dims) + 1
    )
    return bce + (1 - dice).mean()


def train(args):
    boundary_weight = getattr(args, "boundary_weight", 0.0)
    boundary_radius = getattr(args, "boundary_radius", 1)
    validate_boundary_options(args.arch, boundary_weight, boundary_radius)
    # Fail fast on broken masks, blank/missing splits and exact duplicate imagery.
    # This is intentionally a full pass; do not optimize away scientific validation.
    print(f"Auditing paired dataset: {args.data}", flush=True)
    dataset_audit(args.data, preview_count=4, verify_hashes=True)
    if args.batch_size < 1 or args.epochs < 1:
        raise ValueError("Batch size and epochs must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    if (args.run / "best.pt").exists():
        raise FileExistsError(
            f"Run already contains a checkpoint: {args.run}. Choose a new run directory."
        )
    seed_everything(args.seed)
    args.run.mkdir(parents=True, exist_ok=True)
    dev = torch.device(args.device)
    dataset_sha = dataset_fingerprint(args.data)
    training_review = getattr(args, "training_review", None)
    allow_provisional_review = getattr(args, "allow_provisional_review", False)
    overrides = {}
    overlay_metadata = source_training_metadata(dataset_sha)
    if training_review is not None:
        overrides, overlay_metadata = load_training_overlay(
            training_review,
            args.data,
            dataset_sha,
            getattr(args, "init_checkpoint", None),
            allow_provisional=allow_provisional_review,
        )
    profile = getattr(args, "photometric", "standard")
    train_ds = RoofDataset(
        args.data,
        "train",
        size=args.size,
        augment=True,
        photometric=profile,
        mask_overrides=overrides,
    )
    val_ds = RoofDataset(args.data, "val", size=args.size, augment=False)
    generator = torch.Generator().manual_seed(args.seed)
    common = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": dev.type == "cuda",
        "worker_init_fn": seed_worker,
        "generator": generator,
        "persistent_workers": args.workers > 0,
    }
    sampling_report = getattr(args, "sampling_report", None)
    sampler = None
    if sampling_report is not None:
        from .hard_examples import load_sampling_weights

        weights = load_sampling_weights(sampling_report, train_ds.rows, dataset_sha)
        if (
            json.loads(Path(sampling_report).read_text(encoding="utf-8"))["image_size"]
            != args.size
        ):
            raise ValueError("Sampling report image size differs from training")
        sampler = WeightedRandomSampler(
            weights, len(train_ds), replacement=True, generator=generator
        )
    train_loader = DataLoader(
        train_ds, shuffle=sampler is None, sampler=sampler, **common
    )
    val_loader = DataLoader(val_ds, shuffle=False, **common)
    model, initialization = initialize_model(args, dev, dataset_sha)
    opt = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(1, args.epochs), eta_min=args.lr * 0.02
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=dev.type == "cuda" and not args.no_amp
    )
    config = {
        "arch": args.arch,
        "image_size": args.size,
        "normalization": "imagenet_rgb_v1",
        "seed": args.seed,
        "pretrained_encoder": args.pretrained,
        "aspp_rates": list(args.aspp_rates)
        if getattr(args, "aspp_rates", None) is not None
        else None,
        "dataset_path": str(args.data),
        "dataset_manifest_sha256": hashlib.sha256(
            (args.data / "manifest.csv").read_bytes()
        ).hexdigest(),
        "dataset_content_sha256": dataset_sha,
        "photometric_profile": profile,
        "boundary_supervision": boundary_weight > 0,
        "boundary_weight": boundary_weight,
        "boundary_radius": boundary_radius,
        "boundary_target": BOUNDARY_TARGET if boundary_weight > 0 else None,
        "sampling_report_sha256": hashlib.sha256(
            Path(sampling_report).read_bytes()
        ).hexdigest()
        if sampling_report is not None
        else None,
        **initialization,
        **overlay_metadata,
    }
    experiment = {
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "config": config,
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "numpy": np.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": torch.cuda.get_device_name(dev) if dev.type == "cuda" else "cpu",
    }
    (args.run / "experiment.json").write_text(
        json.dumps(experiment, indent=2), encoding="utf-8"
    )
    best = -1.0
    stalled = 0
    history = []
    initial_val = None
    if initialization:
        initial_val, _ = evaluate_loader(model, val_loader, dev, threshold=0.5)
        best = initial_val["iou"]
        torch.save(
            {
                "state_dict": model.state_dict(),
                "config": dict(config, epoch=0),
                "val_iou_at_0p5": best,
                "threshold": 0.5,
            },
            args.run / "best.pt",
        )
        print(
            f"Initial checkpoint VAL IoU@0.50={best:.4f}; retained unless fine-tuning improves it",
            flush=True,
        )
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.monotonic()
        model.train()
        total = 0.0
        segmentation_total = 0.0
        boundary_total = 0.0
        n = 0
        for batch_index, (x, y) in enumerate(train_loader, start=1):
            x = x.to(dev, non_blocking=True)
            y = y.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=dev.type, enabled=scaler.is_enabled()):
                if boundary_weight > 0:
                    logits, edges = model.forward_with_boundary(x)
                    edge_loss = bce_dice_loss(
                        edges, boundary_target(y, boundary_radius)
                    )
                else:
                    logits = model(x)
                    edge_loss = logits.new_zeros(())
                segmentation_loss = bce_dice_loss(logits, y, args.pos_weight)
                loss = segmentation_loss + boundary_weight * edge_loss
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Nonfinite loss at epoch {epoch}, batch {batch_index}"
                )
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(opt)
            scaler.update()
            total += loss.item() * x.size(0)
            segmentation_total += segmentation_loss.item() * x.size(0)
            boundary_total += edge_loss.item() * x.size(0)
            n += x.size(0)
            if (
                batch_index == 1
                or batch_index % 100 == 0
                or batch_index == len(train_loader)
            ):
                progress = {
                    "phase": "training",
                    "epoch": epoch,
                    "epochs": args.epochs,
                    "batch": batch_index,
                    "batches": len(train_loader),
                    "samples_seen": n,
                    "train_loss": total / n,
                    "segmentation_loss": segmentation_total / n,
                    "boundary_loss": boundary_total / n,
                    "elapsed_seconds": time.monotonic() - epoch_start,
                }
                (args.run / "progress.json").write_text(
                    json.dumps(progress, indent=2), encoding="utf-8"
                )
                print(
                    f"epoch={epoch}/{args.epochs} batch={batch_index}/{len(train_loader)} "
                    f"train_loss={total / n:.4f}",
                    flush=True,
                )
        val_metrics, _ = evaluate_loader(model, val_loader, dev, threshold=0.5)
        current = val_metrics["iou"]
        entry = {
            "epoch": epoch,
            "train_loss": total / max(1, n),
            "segmentation_loss": segmentation_total / max(1, n),
            "boundary_loss": boundary_total / max(1, n),
            "val_iou_0p5": current,
            "val_dice_0p5": val_metrics["dice"],
            "lr": opt.param_groups[0]["lr"],
            "seconds": time.monotonic() - epoch_start,
        }
        history.append(entry)
        (args.run / "history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )
        print(
            f"epoch={epoch} train_loss={entry['train_loss']:.4f} val_IoU@0.50={current:.4f} val_Dice={val_metrics['dice']:.4f}",
            flush=True,
        )
        if current > best + args.min_delta:
            best = current
            stalled = 0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "config": dict(config, epoch=epoch),
                    "val_iou_at_0p5": best,
                    "threshold": 0.5,
                },
                args.run / "best.pt",
            )
        else:
            stalled += 1
        scheduler.step()
        if stalled >= args.patience:
            print(f"Early stop: no improvement for {args.patience} epochs")
            break
    (args.run / "history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    if training_review is not None:
        frozen_overrides, frozen_metadata = load_training_overlay(
            training_review,
            args.data,
            dataset_sha,
            getattr(args, "init_checkpoint", None),
            allow_provisional=allow_provisional_review,
        )
        if frozen_overrides != overrides or frozen_metadata != overlay_metadata:
            raise ValueError("Training review changed during fitting")
    # Preserve the trained endpoint even when validation correctly retains epoch zero.
    # This endpoint uses the preregistered fixed 0.50 threshold, without calibration.
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": dict(config, epoch=history[-1]["epoch"]),
            "val_iou_at_0p5": history[-1]["val_iou_0p5"],
            "threshold": 0.5,
            "selection_role": "final_trained_endpoint_not_selected_or_calibrated",
        },
        args.run / "final.pt",
    )
    best_model, _ = load_model(args.run / "best.pt", dev)
    threshold, sweep = choose_threshold(best_model, val_loader, dev)
    metrics, _ = evaluate_loader(best_model, val_loader, dev, threshold=threshold)
    state = torch.load(args.run / "best.pt", map_location="cpu", weights_only=True)
    state["threshold"] = threshold
    torch.save(state, args.run / "best.pt")
    summary = {
        "best_epoch": state["config"]["epoch"],
        "val_iou_at_0p5": state["val_iou_at_0p5"],
        "val_tuned_threshold": threshold,
        "val_metrics_tuned": metrics,
        "val_threshold_sweep": sweep,
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "test_evaluated": False,
        "initial_val_metrics_at_0p5": initial_val,
        "config": state["config"],
        "final_epoch": history[-1]["epoch"],
        "final_val_iou_at_0p5": history[-1]["val_iou_0p5"],
        "final_threshold": 0.5,
    }
    (args.run / "training_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (args.run / "progress.json").write_text(
        json.dumps(
            {
                "phase": "complete",
                "epochs_completed": len(history),
                "best_epoch": summary["best_epoch"],
                "test_evaluated": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        f"Best epoch {summary['best_epoch']}, VAL IoU@tuned={metrics['iou']:.4f}, threshold={threshold:.2f}; TEST NOT USED FOR MODEL SELECTION"
    )
    return summary


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Prepared dataset directory containing manifest.csv",
    )
    p.add_argument(
        "--run",
        type=Path,
        required=True,
        help="Output directory for model and training history",
    )
    p.add_argument(
        "--arch",
        default="unet-resnet34",
        choices=[
            "tiny",
            "unet-resnet34",
            "unetplusplus-efficientnet-b0",
            "deeplabv3plus-resnet34",
        ],
    )
    p.add_argument(
        "--pretrained",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Imagenet encoder; downloads on first use",
    )
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--workers", type=int, default=0, help="0 recommended on Windows")
    p.add_argument("--lr", type=float, default=0.0003)
    p.add_argument("--weight-decay", type=float, default=0.0001)
    p.add_argument("--pos-weight", type=float, default=1.0)
    p.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--min-delta", type=float, default=0.0001)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument(
        "--init-checkpoint",
        type=Path,
        help="Warm-start matching full model weights; restart optimizer",
    )
    p.add_argument(
        "--sampling-report",
        type=Path,
        help="Train-only hard-example probabilities from roof_pipeline.hard_examples",
    )
    p.add_argument("--photometric", choices=["standard", "robust"], default="standard")
    p.add_argument(
        "--training-review",
        type=Path,
        help="Sealed reviewed training-mask corrections; physical parent data remains unchanged",
    )
    p.add_argument(
        "--allow-provisional-review",
        action="store_true",
        help="Explicitly permit AI-provisional training-review corrections",
    )
    p.add_argument(
        "--boundary-weight",
        type=float,
        default=0.0,
        help="Auxiliary edge BCE+Dice weight; 0 preserves binary-only training",
    )
    p.add_argument(
        "--boundary-radius",
        type=int,
        default=1,
        help="Morphological boundary-band radius in training-grid pixels",
    )
    p.add_argument(
        "--aspp-rates",
        type=int,
        nargs=3,
        metavar=("R1", "R2", "R3"),
        default=None,
        help="DeepLabV3+ decoder dilations; omission preserves SMP defaults or warm-start checkpoint rates",
    )
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
