"""
train_v2.py — improved rooftop segmentation trainer.

Drop-in upgrades over train.py while preserving the checkpoint contract so
api/inference.py (patched) keeps loading best.pt for the app:
  - Architecture choice: U-Net / U-Net++ / DeepLabV3+  (--arch, default unetplusplus)
  - Remote-sensing self-supervised encoder init  (--encoder-init seco|seco100k|imagenet|none|<path>)
  - Cosine/OneCycle LR scheduler, AMP + grad clipping
  - Stronger augmentation (RandomResizedCrop + color jitter + noise)
  - BCE + Dice (+ optional Focal) loss for class imbalance
  - Threshold tuning on validation, then honest TEST-set metrics (IoU/Dice/Acc/Prec/Recall/F1)

Reuses dataset.py (RoofSegmentationDataset, get_train_val_test_pairs) unchanged.

Example:
    python train_v2.py --data-root segmentation_model_data_v2 --epochs 30 \\
        --arch unetplusplus --encoder resnet50 --encoder-init seco \\
        --image-size 512 --batch-size 8 --scheduler cosine --loss bce_dice_focal \\
        --output-dir outputs_v2_unetpp_seco
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import segmentation_models_pytorch as smp

from dataset import RoofSegmentationDataset, get_train_val_test_pairs

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
ARCHES = {"unet": smp.Unet, "unetplusplus": smp.UnetPlusPlus, "deeplabv3plus": smp.DeepLabV3Plus}

# Where fetch_remote_weights.py stores SeCo checkpoints (project root / pretrained).
PRETRAINED_DIR = Path(__file__).resolve().parent / "pretrained"
SECO_PATHS = {
    "seco": PRETRAINED_DIR / "seco_resnet50_1m.ckpt",
    "seco100k": PRETRAINED_DIR / "seco_resnet50_100k.ckpt",
}

# Prefixes a SeCo/SSL checkpoint may wrap its ResNet backbone under.
_STRIP_PREFIXES = ("backbone.", "encoder.", "module.", "feature_extractor.", "features.", "net.")

# SeCo (MoCo) stores ResNet-50 as nn.Sequential under 'encoder_q.': idx -> name.
_SECO_SEQ = {"0": "conv1", "1": "bn1", "4": "layer1", "5": "layer2", "6": "layer3", "7": "layer4"}


def _load_lenient_ckpt(path) -> dict:
    """torch.load a checkpoint that may reference pytorch_lightning classes
    (e.g. SeCo .ckpt files) without requiring lightning installed. Lightning
    objects are stubbed; the plain-tensor state_dict survives intact.
    """
    import importlib.abc
    import importlib.machinery
    import types

    class _Stub:
        def __init__(self, *a, **k):
            pass

        def __setstate__(self, s):
            if isinstance(s, dict):
                self.__dict__.update(s)

        def __reduce__(self):
            return (_Stub, ())

    class _Loader(importlib.abc.Loader):
        def create_module(self, spec):
            m = types.ModuleType(spec.name)
            m.__getattr__ = lambda name: _Stub
            return m

        def exec_module(self, module):
            pass

    class _Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname.split(".")[0] in ("pytorch_lightning", "lightning", "lightning_fabric", "pl"):
                return importlib.machinery.ModuleSpec(fullname, _Loader())
            return None

    finder = _Finder()
    sys.meta_path.insert(0, finder)
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    finally:
        sys.meta_path.remove(finder)


def _remap_to_encoder(src_sd: dict, enc_keys: set[str]) -> dict:
    """Map a backbone state_dict onto SMP encoder key names. Handles:
      - bare torchvision names (conv1.weight, layer1.0.conv1.weight, ...)
      - common prefixes (backbone. / encoder. / module. / ...)
      - SeCo/MoCo 'encoder_q.<idx>.' sequential layout
    """
    remapped = {}
    for k, v in src_sd.items():
        if k in enc_keys:
            remapped[k] = v
            continue
        cand = k
        for _ in range(6):  # iteratively strip known prefixes
            stripped = False
            for pre in _STRIP_PREFIXES:
                if cand.startswith(pre):
                    cand = cand[len(pre):]
                    stripped = True
                    break
            if not stripped:
                break
        if cand in enc_keys:
            remapped[cand] = v
            continue
        if k.startswith("encoder_q."):  # SeCo/MoCo sequential encoder
            rest = k[len("encoder_q."):]
            parts = rest.split(".", 1)
            name = _SECO_SEQ.get(parts[0])
            if name:
                tail = parts[1] if len(parts) > 1 else ""
                cand2 = f"{name}.{tail}" if tail else name
                if cand2 in enc_keys:
                    remapped[cand2] = v
    return remapped


# --------------------------------------------------------------------------- #
# Args
# --------------------------------------------------------------------------- #

def get_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", type=Path, default=Path("segmentation_model_data_v2"))
    p.add_argument("--images-dir", type=str, default=None)
    p.add_argument("--masks-dir", type=str, default=None)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--image-size", type=int, default=512)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--arch", choices=list(ARCHES), default="unetplusplus")
    p.add_argument("--encoder", type=str, default="resnet50")
    p.add_argument("--encoder-init", type=str, default="seco",
                   help="imagenet|seco|seco100k|none|<path/to/ckpt>")
    p.add_argument("--scheduler", choices=["none", "cosine", "onecycle"], default="cosine")
    p.add_argument("--loss", choices=["bce_dice", "bce_dice_focal"], default="bce_dice_focal")
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--no-amp", action="store_true", help="Disable mixed precision")
    p.add_argument("--compile", action="store_true", help="torch.compile the model (~1.3-2x faster, +warmup)")
    p.add_argument("--output-dir", type=Path,
                   default=Path("outputs_v2_unetpp_seco"))
    p.add_argument("--save-every", type=int, default=5)
    p.add_argument("--train-file", type=str, default="train_filenames_clean.txt")
    p.add_argument("--val-file", type=str, default="val_filenames_clean.txt")
    p.add_argument("--test-file", type=str, default="test_filenames_clean.txt")
    p.add_argument("--no-label-filter", action="store_true")
    p.add_argument("--no-test-eval", action="store_true",
                   help="Skip final test-set evaluation (e.g. no held-out AIRS yet)")
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Remote-sensing weight loading
# --------------------------------------------------------------------------- #

def load_remote_weights(encoder: nn.Module, init: str) -> dict:
    """Load RS-SSL weights into an SMP encoder. Returns a report dict.

    Falls back silently (random init) if weights are missing so training never
    hard-fails — but the report makes the mismatch loud.
    """
    report = {"init": init, "loaded": False, "matched": 0, "encoder_keys": 0, "missing": []}
    enc_keys = set(encoder.state_dict().keys())
    report["encoder_keys"] = len(enc_keys)

    if init in ("none",):
        report["note"] = "random init"
        return report

    if init in SECO_PATHS:
        path = SECO_PATHS[init]
        if not path.exists():
            report["note"] = (f"SeCo weights not found at {path}. "
                               f"Run: python scripts/fetch_remote_weights.py --source {init}")
            print(f"[encoder-init] WARNING: {report['note']}", file=sys.stderr)
            return report
    elif init in ("imagenet",):
        report["note"] = "imagenet handled at model build time"
        return report
    else:
        path = Path(init)
        if not path.exists():
            report["note"] = f"weights not found at {path}"
            print(f"[encoder-init] WARNING: {report['note']}", file=sys.stderr)
            return report

    ckpt = _load_lenient_ckpt(path)
    sd = ckpt.get("state_dict", ckpt.get("model", ckpt))
    remapped = _remap_to_encoder(sd, enc_keys)
    missing = enc_keys - set(remapped.keys())
    report.update({"loaded": True, "matched": len(remapped), "missing_count": len(missing),
                   "source_keys": len(sd)})
    if remapped:
        missing_shapes = [k for k in list(missing)[:5]]
        msg = (f"[encoder-init] Loaded {len(remapped)}/{len(enc_keys)} encoder params from {path.name} "
               f"(src_keys={len(sd)}, covered={100*len(remapped)/max(len(enc_keys),1):.1f}%). "
               f"Sample unmatched encoder keys: {missing_shapes}.")
        print(msg, file=sys.stderr)
        encoder.load_state_dict(remapped, strict=False)
    else:
        report["note"] = "no keys matched after remapping (check checkpoint format)"
        print(f"[encoder-init] WARNING: {report['note']}", file=sys.stderr)
    return report


# --------------------------------------------------------------------------- #
# Losses + metrics
# --------------------------------------------------------------------------- #

def dice_loss(logits: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    pred = logits.sigmoid().view(-1)
    target = target.view(-1)
    inter = (pred * target).sum()
    return 1.0 - (2.0 * inter + smooth) / (pred.sum() + target.sum() + smooth)


def binary_focal_loss(logits: torch.Tensor, target: torch.Tensor,
                      alpha: float = 0.25, gamma: float = 2.0) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    p = logits.sigmoid()
    p_t = p * target + (1 - p) * (1 - target)
    alpha_t = alpha * target + (1 - alpha) * (1 - target)
    return (alpha_t * (1 - p_t) ** gamma * bce).mean()


def seg_loss(logits: torch.Tensor, target: torch.Tensor, mode: str) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, target)
    d = dice_loss(logits, target)
    if mode == "bce_dice":
        return bce + d
    return 0.5 * bce + d + 0.5 * binary_focal_loss(logits, target)


def intersection_union(logits: torch.Tensor, target: torch.Tensor, thresh: float = 0.5):
    pred = (logits.sigmoid() > thresh).float().view(-1)
    target = target.view(-1)
    inter = (pred * target).sum()
    return inter, pred.sum() + target.sum() - inter


# --------------------------------------------------------------------------- #
# Augmentation
# --------------------------------------------------------------------------- #

def build_transforms(image_size: int):
    """Returns (train_tf, val_tf). Version-portable across albumentations 1.x/2.x
    (2.x changed size args to tuples and removed `from albumentations import A`).
    Each transform is added defensively so one API quirk can't sink the run.
    """
    try:
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
    except ImportError:
        print("[aug] albumentations not available — falling back to torch normalize.", file=sys.stderr)
        return None, None

    def _try(kwargs_variants):
        """Return the first successfully constructed transform, or None."""
        for kw in kwargs_variants:
            try:
                return A.RandomResizedCrop(**kw) if "ratio" in kw else A.Resize(**kw)
            except Exception:
                continue
        return None

    def add(tf_list, item):
        if item is not None:
            tf_list.append(item)
        return tf_list

    sz = image_size
    train = []
    add(train, _try([
        {"size": (sz, sz), "scale": (0.5, 1.0), "ratio": (0.75, 1.33), "p": 1.0},   # albumentations 2.x
        {"height": sz, "width": sz, "scale": (0.5, 1.0), "ratio": (0.75, 1.33), "p": 1.0},  # 1.x
    ]))
    add(train, A.HorizontalFlip(p=0.5))
    add(train, A.VerticalFlip(p=0.5))
    add(train, A.RandomRotate90(p=0.5))
    add(train, _try_safe_color(A, "ColorJitter", brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.5))
    add(train, _try_safe_color(A, "RandomBrightnessContrast", brightness_limit=0.2, contrast_limit=0.2, p=0.3))
    # GaussNoise signature changed across versions; try both.
    gauss = None
    for gkw in ({"std_range": (0.02, 0.08), "p": 0.2}, {"var_limit": (5.0, 20.0), "p": 0.2}):
        try:
            gauss = A.GaussNoise(**gkw)
            break
        except Exception:
            continue
    if gauss is not None:
        train.append(gauss)
    train.append(A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
    train.append(ToTensorV2())

    resize = _try([{"height": sz, "width": sz}, {"size": (sz, sz)}])  # Resize: 2.x uses h/w here
    val = [resize, A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()]
    val = [t for t in val if t is not None]
    return A.Compose(train), A.Compose(val)


def _try_safe_color(A, name, **kwargs):
    try:
        return getattr(A, name)(**kwargs)
    except Exception as e:  # pragma: no cover - defensive
        print(f"[aug] skipping {name}: {e}", file=sys.stderr)
        return None


# --------------------------------------------------------------------------- #
# Train / validate
# --------------------------------------------------------------------------- #

def train_one_epoch(model, loader, optimizer, scheduler, scaler, mode, device,
                    grad_clip, use_amp, onecycle):
    model.train()
    total_loss = 0.0
    n_batches = 0
    pbar = tqdm(loader, desc="Train", leave=False)
    for images, masks in pbar:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", enabled=use_amp):
            logits = model(images)
            loss = seg_loss(logits, masks, mode)
        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        if onecycle:
            scheduler.step()
        total_loss += loss.item()
        n_batches += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def validate(model, loader, mode, device, use_amp):
    model.eval()
    total_loss = 0.0
    total_inter, total_union = 0.0, 0.0
    n_batches = 0
    for images, masks in tqdm(loader, desc="Val", leave=False):
        images, masks = images.to(device), masks.to(device)
        with torch.autocast("cuda", enabled=use_amp):
            logits = model(images)
            loss = seg_loss(logits, masks, mode)
        inter, union = intersection_union(logits, masks)
        total_loss += loss.item()
        total_inter += inter.item()
        total_union += union.item()
        n_batches += 1
    smooth = 1e-6
    return total_loss / max(n_batches, 1), (total_inter + smooth) / (total_union + smooth)


@torch.no_grad()
def threshold_sweep(model, loader, device, use_amp, n_bins: int = 100):
    """Find the threshold maximizing global IoU on the val set via probability histograms."""
    model.eval()
    edges = np.linspace(0, 1, n_bins + 1)
    hist_pos = np.zeros(n_bins, dtype=np.float64)  # target==1 prob histogram
    hist_neg = np.zeros(n_bins, dtype=np.float64)
    total_pos = 0.0
    for images, masks in tqdm(loader, desc="Sweep", leave=False):
        images, masks = images.to(device), masks.to(device)
        with torch.autocast("cuda", enabled=use_amp):
            logits = model(images)
        prob = logits.sigmoid().float().cpu().numpy().ravel()
        tgt = masks.float().cpu().numpy().ravel()
        pos = tgt > 0.5
        total_pos += float(pos.sum())
        hist_pos += np.histogram(prob[pos], bins=edges)[0]
        hist_neg += np.histogram(prob[~pos], bins=edges)[0]
    # For a threshold at edge[i], pred-positive = bins >= i.
    cum_pos = np.cumsum(hist_pos[::-1])[::-1]  # TP at each edge index
    cum_neg = np.cumsum(hist_neg[::-1])[::-1]  # FP
    tp = cum_pos
    fp = cum_neg
    fn = total_pos - tp
    smooth = 1e-6
    iou = (tp + smooth) / (tp + fp + fn + smooth)
    best_idx = int(np.argmax(iou))
    best_thr = float(edges[best_idx])
    return best_thr, float(iou[best_idx]), {"thresholds": edges.tolist(), "iou": iou.tolist()}


@torch.no_grad()
def evaluate_split(model, loader, device, use_amp, threshold: float):
    """Full metrics at a fixed threshold. Returns dict with TP/FP/FN/TN-derived metrics."""
    model.eval()
    tp = fp = fn = tn = 0.0
    for images, masks in tqdm(loader, desc="Eval", leave=False):
        images, masks = images.to(device), masks.to(device)
        with torch.autocast("cuda", enabled=use_amp):
            logits = model(images)
        pred = (logits.sigmoid() > threshold).float().view(-1)
        tgt = masks.float().view(-1)
        tp += float((pred * tgt).sum())
        fp += float((pred * (1 - tgt)).sum())
        fn += float(((1 - pred) * tgt).sum())
        tn += float(((1 - pred) * (1 - tgt)).sum())
    smooth = 1e-6
    iou = (tp + smooth) / (tp + fp + fn + smooth)
    dice = (2 * tp + smooth) / (2 * tp + fp + fn + smooth)
    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    prec = (tp + smooth) / (tp + fp + smooth)
    rec = (tp + smooth) / (tp + fn + smooth)
    f1 = (2 * prec * rec + smooth) / (prec + rec + smooth)
    return {"threshold": threshold, "iou": iou, "dice": dice, "accuracy": acc,
            "precision": prec, "recall": rec, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


# --------------------------------------------------------------------------- #

def main():
    args = get_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    use_amp = (args.device.startswith("cuda")) and not args.no_amp
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True  # fixed input sizes -> autotune convs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    images_dir = args.images_dir or str(args.data_root / "images")
    masks_dir = args.masks_dir or str(args.data_root / "masks")
    train_pairs, val_pairs, test_pairs = get_train_val_test_pairs(
        args.data_root,
        train_file=args.train_file, val_file=args.val_file, test_file=args.test_file,
        exclude_label_in_mask_name=not args.no_label_filter,
        images_dir=images_dir, masks_dir=masks_dir,
    )
    if not train_pairs:
        print("No training pairs found. Check --data-root and the *_filenames_clean.txt manifests.",
              file=sys.stderr)
        sys.exit(1)
    print(f"Train: {len(train_pairs)} | Val: {len(val_pairs)} | Test: {len(test_pairs)}")

    train_tf, val_tf = build_transforms(args.image_size)
    train_ds = RoofSegmentationDataset(train_pairs, transform=train_tf)
    val_ds = RoofSegmentationDataset(val_pairs, transform=val_tf) if val_pairs else None
    pw = args.workers > 0
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=(device.type == "cuda"),
                              drop_last=True, persistent_workers=pw,
                              prefetch_factor=4 if pw else None)
    val_loader = None
    if val_ds:
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.workers, pin_memory=(device.type == "cuda"),
                                persistent_workers=pw, prefetch_factor=4 if pw else None)

    # Model
    weights = "imagenet" if args.encoder_init == "imagenet" else None
    model = ARCHES[args.arch](encoder_name=args.encoder, encoder_weights=weights,
                              in_channels=3, classes=1)
    if args.encoder_init not in ("imagenet", "none"):
        load_remote_weights(model.encoder, args.encoder_init)
    model = model.to(device)
    if args.compile:
        try:
            model = torch.compile(model)
            print("[compile] torch.compile enabled (first batches will compile/slow)")
        except Exception as e:  # pragma: no cover
            print(f"[compile] disabled (torch.compile unavailable: {e})", file=sys.stderr)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    steps_per_epoch = len(train_loader)
    onecycle = args.scheduler == "onecycle"
    if onecycle:
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=args.lr, total_steps=args.epochs * steps_per_epoch)
    elif args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        scheduler = None

    best_val_iou = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, scaler,
                                     args.loss, device, args.grad_clip, use_amp, onecycle)
        log = {"epoch": epoch, "train_loss": train_loss, "lr": optimizer.param_groups[0]["lr"]}
        if val_loader:
            val_loss, val_iou = validate(model, val_loader, args.loss, device, use_amp)
            log.update({"val_loss": val_loss, "val_iou": val_iou})
            print(f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_iou={val_iou:.4f}")
            if val_iou > best_val_iou:
                best_val_iou = val_iou
                _save_ckpt(args, model, optimizer, epoch, val_iou, threshold=0.5,
                           path=args.output_dir / "best.pt")
        else:
            print(f"Epoch {epoch}: train_loss={train_loss:.4f}")
        if scheduler is not None and not onecycle:
            scheduler.step()
        history.append(log)
        if epoch % args.save_every == 0:
            _save_ckpt(args, model, optimizer, epoch, best_val_iou, threshold=0.5,
                       path=args.output_dir / f"checkpoint_epoch_{epoch}.pt")
        (args.output_dir / "history.json").write_text(json.dumps(history, indent=2))

    # Threshold tuning on val
    best_thr, thr_iou, sweep = (0.5, best_val_iou, None)
    if val_loader:
        best_thr, thr_iou, sweep = threshold_sweep(model, val_loader, device, use_amp)
        (args.output_dir / "threshold_sweep.json").write_text(json.dumps({
            "best_threshold": best_thr, "best_iou_at_threshold": thr_iou, "sweep": sweep,
        }, indent=2))
        print(f"Threshold sweep: best={best_thr:.3f} (val IoU {thr_iou:.4f} @0.5 was {best_val_iou:.4f})")
        # Re-inject tuned threshold into best.pt so inference uses it.
        best_path = args.output_dir / "best.pt"
        if best_path.exists():
            ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
            ckpt.setdefault("args", {})["threshold"] = best_thr
            ckpt["threshold"] = best_thr
            torch.save(ckpt, best_path)

    # Honest test-set evaluation
    summary = {"best_val_iou": best_val_iou, "best_threshold": best_thr, "history_epochs": len(history)}
    if test_pairs and not args.no_test_eval:
        test_loader = DataLoader(RoofSegmentationDataset(test_pairs, transform=val_tf),
                                 batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.workers, pin_memory=(device.type == "cuda"))
        # also keep the current model weights eval (last-epoch) + best weights eval
        metrics_last = evaluate_split(model, test_loader, device, use_amp, best_thr)
        summary["test_last_epoch"] = metrics_last
        # evaluate best.pt too
        best_path = args.output_dir / "best.pt"
        if best_path.exists():
            ckpt = torch.load(best_path, map_location=device, weights_only=False)
            best_model = ARCHES[args.arch](encoder_name=args.encoder, encoder_weights=None,
                                           in_channels=3, classes=1).to(device)
            best_model.load_state_dict(ckpt["model_state_dict"], strict=True)
            summary["test_best"] = evaluate_split(best_model, test_loader, device, use_amp, best_thr)
        print("Test metrics (best.pt):", json.dumps(summary.get("test_best", metrics_last), indent=2))
    elif not test_pairs:
        print("[test] No test pairs (AIRS test mosaic not prepared yet). "
              "Skipping held-out test eval — run prepare_airs.py with AIRS to enable it.")
    (args.output_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(f"\nDone. Best val IoU @0.5: {best_val_iou:.4f} | tuned threshold: {best_thr:.3f} "
          f"| output: {args.output_dir}")


def _save_ckpt(args, model, optimizer, epoch, val_iou, threshold, path: Path):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_iou": val_iou,
        "threshold": threshold,
        "args": {
            "arch": args.arch, "encoder": args.encoder, "encoder_weights": None,
            "encoder_init": args.encoder_init, "image_size": args.image_size,
            "threshold": threshold,
        },
    }, path)


if __name__ == "__main__":
    main()
