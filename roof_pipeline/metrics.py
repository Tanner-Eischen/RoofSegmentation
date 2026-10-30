"""Dataset-level segmentation metrics and threshold calibration."""

from __future__ import annotations

import numpy as np
import torch


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
    for batch_idx, (x, y) in enumerate(loader):
        x = x.to(device)
        y = y.to(device)
        probs = model(x).sigmoid().float()
        pred = probs >= threshold
        truth = y > 0.5
        batch_tp = ((pred) & truth).flatten(1).sum(1).cpu().tolist()
        batch_fp = ((pred) & (~truth)).flatten(1).sum(1).cpu().tolist()
        batch_fn = ((~pred) & truth).flatten(1).sum(1).cpu().tolist()
        batch_tn = ((~pred) & (~truth)).flatten(1).sum(1).cpu().tolist()
        for i, (a, b, c, d) in enumerate(zip(batch_tp, batch_fp, batch_fn, batch_tn)):
            tp += a
            fp += b
            fn += c
            tn += d
            if with_rows:
                detail.append(
                    {
                        "index": batch_idx * loader.batch_size + i,
                        **metrics_from_counts(a, b, c, d),
                        "roof_fraction": (a + c) / max(a + b + c + d, 1),
                    }
                )
    return metrics_from_counts(tp, fp, fn, tn), detail


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
    for x, y in loader:
        prob = model(x.to(device)).sigmoid().cpu().numpy()
        mask = y.numpy() > 0.5
        for i, t in enumerate(thresholds):
            predicted = prob >= t
            tp[i] += np.logical_and(predicted, mask).sum()
            fp[i] += np.logical_and(predicted, ~mask).sum()
            fn[i] += np.logical_and(~predicted, mask).sum()
    ious = np.where(tp + fp + fn > 0, tp / np.maximum(1, tp + fp + fn), 0.0)
    best = int(np.argmax(ious))
    return thresholds[best], {str(t): float(v) for t, v in zip(thresholds, ious)}
