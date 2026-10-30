"""Matched auxiliary-edge pilot, with validation-only assessment and selection."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .boundary import boundary_target
from .common import dataset_fingerprint
from .dataset import RoofDataset
from .metrics import metrics_from_counts
from .models import load_model
from .refine import (
    MIN_IMPROVEMENT,
    _candidate,
    _fresh_output,
    _now,
    _sha256,
    _snapshot,
    _write_json,
)


def select_candidate(candidates: list[dict]) -> dict:
    """Use only validation IoU; conservative ties favor incumbent then control."""
    priority = {"incumbent": 2, "control": 1, "boundary": 0}
    if (
        {c.get("name") for c in candidates} != set(priority)
        or len(candidates) != 3
        or any(
            isinstance(c.get("validation_iou"), bool)
            or not isinstance(c.get("validation_iou"), (int, float))
            or not math.isfinite(c["validation_iou"])
            or not 0 <= c["validation_iou"] <= 1
            for c in candidates
        )
    ):
        raise ValueError(
            "Require incumbent/control/boundary with finite validation IoU"
        )
    incumbent = next(c for c in candidates if c["name"] == "incumbent")
    eligible = [incumbent] + [
        c
        for c in candidates
        if c["validation_iou"] > incumbent["validation_iou"] + MIN_IMPROVEMENT
    ]
    return max(eligible, key=lambda c: (c["validation_iou"], priority[c["name"]]))


def mask_edges(mask: torch.Tensor) -> torch.Tensor:
    """Two-sided 3x3 morphological band; tile borders use replicated values."""
    return boundary_target(mask.float()) > 0


def boundary_counts(prediction, truth, tolerance=2) -> tuple[int, int, int, int]:
    """Matched predicted/true boundary pixels and both independent denominators."""
    if type(tolerance) is not int or tolerance < 0:
        raise ValueError("Boundary tolerance must be a nonnegative integer")
    predicted_edges, true_edges = mask_edges(prediction), mask_edges(truth)
    kernel = 2 * tolerance + 1
    near_truth = F.max_pool2d(true_edges.float(), kernel, 1, tolerance) > 0
    near_prediction = F.max_pool2d(predicted_edges.float(), kernel, 1, tolerance) > 0
    return (
        int((predicted_edges & near_truth).sum()),
        int(predicted_edges.sum()),
        int((true_edges & near_prediction).sum()),
        int(true_edges.sum()),
    )


def boundary_scores(counts) -> dict:
    matched_prediction, predicted, matched_truth, truth = counts
    precision = matched_prediction / predicted if predicted else 0.0
    recall = matched_truth / truth if truth else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0,
        "matched_predicted_pixels": matched_prediction,
        "predicted_boundary_pixels": predicted,
        "matched_truth_pixels": matched_truth,
        "truth_boundary_pixels": truth,
    }


@torch.no_grad()
def validation_diagnostics(checkpoint, data, out, *, device, batch_size, workers):
    model, config = load_model(checkpoint, device)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    dataset = RoofDataset(data, "val", size=config["image_size"])
    loader = DataLoader(
        dataset, batch_size=batch_size, num_workers=workers, shuffle=False
    )
    threshold = float(state["threshold"])
    pixel_totals = [0, 0, 0, 0]
    outline_totals = [0, 0, 0, 0]
    details = []
    for x, y in loader:
        truth = y.to(device) > 0.5
        prediction = model(x.to(device)).sigmoid() >= threshold
        for predicted, target in zip(prediction, truth):
            predicted, target = predicted[None], target[None]
            counts = (
                int((predicted & target).sum()),
                int((predicted & ~target).sum()),
                int((~predicted & target).sum()),
                int((~predicted & ~target).sum()),
            )
            outline = boundary_counts(predicted, target)
            pixel_totals = [a + b for a, b in zip(pixel_totals, counts)]
            outline_totals = [a + b for a, b in zip(outline_totals, outline)]
            details.append(
                {
                    "sample_id": dataset.rows[len(details)]["sample_id"],
                    **metrics_from_counts(*counts),
                    "boundary_f1_tolerance_2px": boundary_scores(outline)["f1"],
                }
            )
    result = {
        "split": "val",
        "samples": len(dataset),
        "checkpoint_sha256": _sha256(checkpoint),
        "threshold_frozen_from_validation": threshold,
        "aggregate": metrics_from_counts(*pixel_totals),
        "binary_mask_boundary": boundary_scores(outline_totals),
        "boundary_definition": "Two-sided 3x3 morphological band, replicated tile border; Euclidean-free Chebyshev tolerance 2 native pixels, pooled counts. Empty denominators score zero.",
    }
    out.mkdir(parents=True, exist_ok=True)
    _write_json(out / "val_metrics.json", result)
    with (out / "val_per_image.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(details[0]))
        writer.writeheader()
        writer.writerows(details)
    return result


def run_experiment(args):
    if (
        args.epochs < 1
        or args.batch_size < 1
        or args.workers < 0
        or not math.isfinite(args.lr)
        or args.lr <= 0
        or not math.isfinite(args.boundary_weight)
        or args.boundary_weight <= 0
        or args.boundary_radius < 1
    ):
        raise ValueError(
            "Invalid epochs, batch size, workers, LR, boundary weight/radius"
        )
    data, checkpoint, out = (
        Path(p).resolve() for p in (args.data, args.incumbent, args.out)
    )
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    _fresh_output(out)
    status = {"started_at": _now(), "test_evaluated": False}

    def update(phase, **values):
        status.update(phase=phase, updated_at=_now(), **values)
        _write_json(out / "workflow_status.json", status)
        print(json.dumps(status), flush=True)

    try:
        update("validating")
        dataset_sha = dataset_fingerprint(data)
        incumbent = dict(_candidate(checkpoint, dataset_sha), name="incumbent")
        if incumbent["arch"] not in {"tiny", "unet-resnet34"}:
            raise ValueError("This matched pilot supports tiny or U-Net/ResNet34")
        frozen = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if frozen["config"].get("boundary_supervision", False):
            raise ValueError("Incumbent must use binary supervision only")
        if frozen["config"].get("dataset_manifest_sha256") != _sha256(
            data / "manifest.csv"
        ):
            raise ValueError("Incumbent manifest hash differs")
        del frozen
        registration = {
            "registered_at": _now(),
            "arguments": {
                k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
            },
            "incumbent": incumbent,
            "dataset_content_sha256": dataset_sha,
            "execution_order": ["control", "boundary"],
            "shared": "Same full weights, native input size, seed, uniform sampling, standard augmentation, AdamW restart, cosine decay, BCE+Dice and epoch budget. Epoch zero is eligible.",
            "intervention": "Additional shared-decoder edge head; binary segmentation remains one channel; BCE+Dice edge loss (positive weight 1) on mask-derived morphological band after augmentation",
            "boundary_weight": args.boundary_weight,
            "boundary_radius": args.boundary_radius,
            "selection": "Validation-calibrated global binary IoU only; >0.0001 over incumbent; exact ties prefer incumbent/control/boundary",
            "diagnostic": "Binary-mask boundary F1 with Chebyshev tolerance 2 native pixels; diagnostic only, no threshold search on boundary F1",
            "test_policy": "No test inference or selection. The existing full-dataset audit checks dimensions/encoding/leakage and writes previews for every partition; the frozen-data fingerprint hashes all contents.",
            "interpretation": "Exploratory single-seed short-budget pilot. Prior test-error inspection informed research direction. Independent geography and paired repeat seeds needed for confirmation.",
            "reference": "https://openaccess.thecvf.com/content/CVPR2021/html/Girard_Polygonal_Building_Extraction_by_Frame_Field_Learning_CVPR_2021_paper.html",
            "reference_scope": "Motivates contour-aware multitask learning; this edge head does not reproduce frame-field learning or establish gains on WHU.",
        }
        _write_json(out / "registration.json", registration)
        repo = Path(__file__).resolve().parents[1]
        _snapshot(repo, out)
        environment = os.environ.copy()
        environment.update(
            OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", CUBLAS_WORKSPACE_CONFIG=":4096:8"
        )
        for arm in ("control", "boundary"):
            command = [
                sys.executable,
                "-u",
                "-m",
                "roof_pipeline.train",
                "--data",
                str(data),
                "--run",
                str(out / arm),
                "--arch",
                incumbent["arch"],
                "--size",
                str(incumbent["image_size"]),
                "--init-checkpoint",
                str(checkpoint),
                "--no-pretrained",
                "--epochs",
                str(args.epochs),
                "--patience",
                str(args.epochs),
                "--batch-size",
                str(args.batch_size),
                "--workers",
                str(args.workers),
                "--device",
                args.device,
                "--seed",
                str(args.seed),
                "--lr",
                str(args.lr),
                "--weight-decay",
                "0.0001",
                "--pos-weight",
                "1",
                "--min-delta",
                str(MIN_IMPROVEMENT),
                "--photometric",
                "standard",
                "--boundary-weight",
                str(args.boundary_weight if arm == "boundary" else 0),
                "--boundary-radius",
                str(args.boundary_radius),
            ]
            update(arm, command=command)
            with (out / f"{arm}.log").open("w", encoding="utf-8") as log:
                subprocess.run(
                    command,
                    cwd=repo,
                    env=environment,
                    check=True,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            update(f"{arm}_complete", command=None)
        if (
            dataset_fingerprint(data) != dataset_sha
            or _sha256(checkpoint) != incumbent["checkpoint_sha256"]
        ):
            raise ValueError("Frozen dataset or incumbent changed")
        candidates = [incumbent] + [
            dict(_candidate(out / arm / "best.pt", dataset_sha, incumbent), name=arm)
            for arm in ("control", "boundary")
        ]
        selected = select_candidate(candidates)
        _write_json(
            out / "selection.json",
            {
                "selected": selected,
                "candidates": candidates,
                "used_test_metrics": False,
            },
        )
        diagnostics = {}
        for candidate in candidates:
            update("validation_diagnostics", candidate=candidate["name"])
            result = validation_diagnostics(
                Path(candidate["checkpoint"]),
                data,
                out / (candidate["name"] + "_validation"),
                device=args.device,
                batch_size=args.batch_size,
                workers=args.workers,
            )
            if not math.isclose(
                result["aggregate"]["iou"], candidate["validation_iou"], abs_tol=1e-10
            ):
                raise ValueError(
                    "Recomputed validation IoU disagrees with training summary"
                )
            diagnostics[candidate["name"]] = result
        control, boundary = (diagnostics[name] for name in ("control", "boundary"))
        report = {
            "candidates": candidates,
            "selected": selected,
            "validation": diagnostics,
            "test_evaluated": False,
            "binary_iou_change_boundary_vs_control": boundary["aggregate"]["iou"]
            - control["aggregate"]["iou"],
            "boundary_f1_change_boundary_vs_control": boundary["binary_mask_boundary"][
                "f1"
            ]
            - control["binary_mask_boundary"]["f1"],
            "frozen_incumbent_unchanged": _sha256(checkpoint)
            == incumbent["checkpoint_sha256"],
            "interpretation": registration["interpretation"],
        }
        _write_json(out / "comparison.json", report)
        lines = [
            "# Auxiliary boundary supervision: matched validation pilot",
            "",
            "| Candidate | Validation IoU | Boundary F1 (2px) | Epoch |",
            "| --- | ---: | ---: | ---: |",
        ]
        for candidate in candidates:
            result = diagnostics[candidate["name"]]
            lines.append(
                f"| {candidate['name']} | {result['aggregate']['iou']:.6f} | {result['binary_mask_boundary']['f1']:.6f} | {candidate['epoch']} |"
            )
        lines += [
            "",
            f"Validation-selected candidate: **{selected['name']}**.",
            f"Boundary minus control IoU: {report['binary_iou_change_boundary_vs_control']:+.6f}; boundary F1: {report['boundary_f1_change_boundary_vs_control']:+.6f}.",
            "",
            "The incumbent is preserved. No test inference or label edits were performed.",
            registration["interpretation"],
            "",
            "See registration.json, source_snapshot.zip, selection.json, comparison.json, per-arm histories and validation per-image CSVs.",
        ]
        (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        update("complete", selected=selected["name"], command=None)
        return report
    except Exception as exc:
        update("failed", error=str(exc), traceback=traceback.format_exc(), command=None)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--incumbent", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--boundary-weight", type=float, default=0.2)
    parser.add_argument("--boundary-radius", type=int, default=1)
    run_experiment(parser.parse_args())


if __name__ == "__main__":
    main()
