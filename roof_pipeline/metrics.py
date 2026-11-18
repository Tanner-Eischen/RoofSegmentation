"""Dataset-level segmentation metrics and threshold calibration."""

from __future__ import annotations

import numpy as np
import torch

from .validity import target_validity


def metrics_from_counts(tp: float, fp: float, fn: float, tn: float) -> dict:
    def div(a, b):
        return float(a / b) if b > 0 else 0.0

    prec = div(tp, tp + fp)
    rec = div(tp, tp + fn)
    return {
        "iou": div(tp, tp + fp + fn),
        "dice": div(2 * tp, 2 * tp + fp + fn),
        "precision": prec,
        "recall": rec,
        "f1": div(2 * prec * rec, prec + rec),
        "pixel_accuracy": div(tp + tn, tp + fp + fn + tn),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


@torch.no_grad()
def evaluate_loader(model, loader, device, threshold=0.5, with_rows=False):
    model.eval()
    tp = fp = fn = tn = 0.0
    detail = []
    total_pixels = 0
    offset = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        valid = target_validity(y)
        total_pixels += y.numel()
        available = valid.flatten(1).any(1)
        probs = torch.zeros_like(y, dtype=torch.float32)
        if available.any():
            probs[available] = model(x[available]).sigmoid().float()
        pred = probs >= threshold
        truth = y > 0.5
        batch_tp = (pred & truth & valid).flatten(1).sum(1).cpu().tolist()
        batch_fp = (pred & ~truth & valid).flatten(1).sum(1).cpu().tolist()
        batch_fn = (~pred & truth & valid).flatten(1).sum(1).cpu().tolist()
        batch_tn = (~pred & ~truth & valid).flatten(1).sum(1).cpu().tolist()
        for i, (a, b, c, d) in enumerate(zip(batch_tp, batch_fp, batch_fn, batch_tn)):
            tp += a
            fp += b
            fn += c
            tn += d
            if with_rows:
                detail.append(
                    {
                        "index": offset + i,
                        **metrics_from_counts(a, b, c, d),
                        "valid_pixel_count": a + b + c + d,
                        "invalid_pixel_count": y[i].numel() - (a + b + c + d),
                        "total_pixel_count": y[i].numel(),
                        "roof_fraction": (a + c) / max(a + b + c + d, 1),
                    }
                )
        offset += y.shape[0]
    valid_count = int(tp + fp + fn + tn)
    if not valid_count:
        raise ValueError("Evaluation split has no valid pixels")
    return {
        **metrics_from_counts(tp, fp, fn, tn),
        "valid_pixel_count": valid_count,
        "invalid_pixel_count": total_pixels - valid_count,
        "total_pixel_count": total_pixels,
    }, detail


@torch.no_grad()
def choose_threshold(model, loader, device, edges=None):
    """Tune on validation ONLY, using one forward pass over all validation images."""
    model.eval()
    if edges is None:
        edges = np.linspace(0.15, 0.85, 15)
    thresholds = [float(x) for x in edges]
    tp = np.zeros(len(thresholds))
    fp = np.zeros(len(thresholds))
    fn = np.zeros(len(thresholds))
    valid_count = 0
    for x, y in loader:
        valid = target_validity(y).cpu().numpy()
        available = valid.reshape(len(valid), -1).any(1)
        valid_count += int(valid.sum())
        if not available.any():
            continue
        prob = model(x[available].to(device)).sigmoid().cpu().numpy()
        mask = y[available].cpu().numpy() > 0.5
        valid = valid[available]
        for i, t in enumerate(thresholds):
            predicted = prob >= t
            tp[i] += (predicted & mask & valid).sum()
            fp[i] += (predicted & ~mask & valid).sum()
            fn[i] += (~predicted & mask & valid).sum()
    if not valid_count:
        raise ValueError("Threshold calibration split has no valid pixels")
    ious = np.where(tp + fp + fn > 0, tp / np.maximum(1, tp + fp + fn), 0.0)
    best = int(np.argmax(ious))
    return thresholds[best], {str(t): float(v) for t, v in zip(thresholds, ious)}
