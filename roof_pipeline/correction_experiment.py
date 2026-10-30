"""Matched binary-label correction test with unchanged validation references."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import traceback
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .adjudication import load_review
from .boundary_experiment import validation_diagnostics
from .common import dataset_fingerprint, load_binary_mask
from .correction_overlay import load_training_overlay
from .dataset import normalize_rgb
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


def select_candidate(candidates):
    priority = {"incumbent": 2, "control": 1, "corrected": 0}
    if (
        len(candidates) != 3
        or {c.get("name") for c in candidates} != set(priority)
        or any(
            isinstance(c.get("validation_iou"), bool)
            or not isinstance(c.get("validation_iou"), (float, int))
            or not math.isfinite(c["validation_iou"])
            or not 0 <= c["validation_iou"] <= 1
            for c in candidates
        )
    ):
        raise ValueError(
            "Require incumbent/control/corrected with finite validation IoUs"
        )
    incumbent = next(c for c in candidates if c["name"] == "incumbent")
    eligible = [incumbent] + [
        c
        for c in candidates
        if c["validation_iou"] > incumbent["validation_iou"] + MIN_IMPROVEMENT
    ]
    return max(eligible, key=lambda c: (c["validation_iou"], priority[c["name"]]))


def _counts(prediction, truth):
    return [
        int((prediction & truth).sum()),
        int((prediction & ~truth).sum()),
        int((~prediction & truth).sum()),
        int((~prediction & ~truth).sum()),
    ]


@torch.no_grad()
def correction_case_diagnostics(checkpoint, reviewroot, out, *, device):
    """Two-label sensitivity and model-training effects on edited TRAIN cases."""
    review = load_review(reviewroot, require_sealed=True)
    model, config = load_model(checkpoint, device)
    totals = {name: [0, 0, 0, 0] for name in ("original", "proposed")}
    details = []
    out.mkdir(parents=True, exist_ok=True)
    for case in review["cases"]:
        if case["split"] != "train" or case["decision"] != "correct":
            continue
        with Image.open(reviewroot / case["image"]) as image:
            rgb = np.asarray(image.convert("RGB")).copy()
        if rgb.shape[:2] != (config["image_size"], config["image_size"]):
            raise ValueError("Correction-case grid differs from checkpoint")
        prediction = (
            model(normalize_rgb(rgb).to(device)).sigmoid()[0, 0].cpu().numpy() >= 0.5
        )
        row = {"sample_id": case["sample_id"], "split": "train"}
        for name, field in (("original", "original_mask"), ("proposed", "mask")):
            truth = load_binary_mask(reviewroot / case[field]) > 0
            counts = _counts(prediction, truth)
            totals[name] = [a + b for a, b in zip(totals[name], counts)]
            row[name] = metrics_from_counts(*counts)
        row["same_model_label_rescore_iou_change"] = (
            row["proposed"]["iou"] - row["original"]["iou"]
        )
        Image.fromarray(prediction.astype(np.uint8) * 255).save(
            out / f"{case['sample_id']}_prediction.png"
        )
        details.append(row)
    result = {
        "checkpoint_sha256": _sha256(checkpoint),
        "threshold_fixed_for_diagnostics": 0.5,
        "cases": details,
        "aggregate": {
            name: metrics_from_counts(*counts) for name, counts in totals.items()
        },
        "interpretation": "Edited training-tile diagnostics only. Same model against two masks measures annotation sensitivity; different models against the SAME masks measure local training effect. These are provisional targets and not held-out generalization evidence.",
    }
    _write_json(out / "metrics.json", result)
    return result


def _verify_arm(checkpoint, *, review_sha, effective_sha):
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    cfg = state["config"]
    if (
        cfg.get("training_review_content_sha256") != review_sha
        or cfg.get("effective_training_content_sha256") != effective_sha
    ):
        raise ValueError(
            "Trained checkpoint target provenance differs from registration"
        )
    return state


def run_experiment(args):
    if (
        args.epochs < 1
        or args.batch_size < 1
        or args.workers < 0
        or not math.isfinite(args.lr)
        or args.lr <= 0
    ):
        raise ValueError("Invalid epoch, batch, worker or learning-rate settings")
    data, reviewroot, checkpoint, out = (
        Path(p).resolve() for p in (args.data, args.review, args.incumbent, args.out)
    )
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    _fresh_output(out)
    status = {
        "started_at": _now(),
        "test_inference_performed": False,
        "provisional_label_intervention": True,
    }

    def update(phase, **values):
        status.update(phase=phase, updated_at=_now(), **values)
        _write_json(out / "workflow_status.json", status)
        print(json.dumps(status), flush=True)

    try:
        update("validating")
        source_sha = dataset_fingerprint(data)
        review = load_review(reviewroot, require_sealed=True)
        overrides, target_metadata = load_training_overlay(
            reviewroot,
            data,
            source_sha,
            checkpoint,
            allow_provisional=args.allow_provisional_review,
        )
        incumbent = dict(_candidate(checkpoint, source_sha), name="incumbent")
        initial_state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if initial_state["config"].get("boundary_supervision", False) or initial_state[
            "config"
        ].get("training_review_content_sha256"):
            raise ValueError("Incumbent must use the unmodified binary source targets")
        del initial_state
        registration = {
            "registered_at": _now(),
            "arguments": {
                k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
            },
            "incumbent": incumbent,
            "source_dataset_content_sha256": source_sha,
            "review_version": review["version"],
            "review_content_sha256": review["content_sha256"],
            "corrected_training_targets": target_metadata,
            "control_effective_training_content_sha256": source_sha,
            "changed_train_tiles": len(overrides),
            "primary_training_effect_metric": "Final fixed-budget validation global binary IoU at threshold 0.50: corrected minus control",
            "selection": "Best epoch by validation IoU@0.50, retaining epoch zero, then validation threshold calibration 0.15-0.85. Promote only >0.0001 over incumbent; exact ties incumbent/control/corrected.",
            "shared": "Fresh matched arms, same full weights, seed, full parent training partition, native resolution, uniform sampling, standard augmentation, BCE+Dice positiveweight1, AdamW restart, weightdecay0.0001, cosine decay, clipping1.0, CUDA AMP when available.",
            "execution_order": ["control", "corrected"],
            "intervention": "Only declared train-mask corrections from the sealed review; no image, validation or test target overrides; no auxiliary edge loss.",
            "local_diagnostics": "Incumbent and each final model scored at fixed0.50 on the edited training tiles against BOTH source and proposed masks, separately. Not a selection criterion or generalization evidence.",
            "test_policy": "No test inference or selection; existing integrity audit/fingerprint validates source files across all partitions.",
            "interpretation": "Exploratory single-seed provisional-label experiment on unchanged validation annotations. It measures agreement with those annotations, not independently verified building accuracy. Two selected training corrections cannot establish the value of a broad correction campaign.",
        }
        _write_json(out / "registration.json", registration)
        repo = Path(__file__).resolve().parents[1]
        _snapshot(repo, out)
        environment = os.environ.copy()
        environment.update(
            OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", CUBLAS_WORKSPACE_CONFIG=":4096:8"
        )
        candidates = [incumbent]
        final_diagnostics, local = {}, {}
        final_ious = {}
        for arm in ("control", "corrected"):
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
            ]
            if arm == "corrected":
                command += ["--training-review", str(reviewroot)]
                if args.allow_provisional_review:
                    command.append("--allow-provisional-review")
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
            expected_review = review["content_sha256"] if arm == "corrected" else None
            expected_effective = (
                target_metadata["effective_training_content_sha256"]
                if arm == "corrected"
                else source_sha
            )
            _verify_arm(
                out / arm / "best.pt",
                review_sha=expected_review,
                effective_sha=expected_effective,
            )
            final_state = _verify_arm(
                out / arm / "final.pt",
                review_sha=expected_review,
                effective_sha=expected_effective,
            )
            if (
                final_state["config"]["epoch"] != args.epochs
                or final_state["threshold"] != 0.5
            ):
                raise ValueError(
                    "Final checkpoint did not retain the registered full budget/threshold"
                )
            candidates.append(
                dict(_candidate(out / arm / "best.pt", source_sha, incumbent), name=arm)
            )
            final_ious[arm] = final_state["val_iou_at_0p5"]
            update(f"{arm}_training_complete", command=None)
        # Finish both child processes before this parent allocates CUDA diagnostic
        # models. Its caching allocator must not compete with the second trainer.
        for arm in ("control", "corrected"):
            update(f"{arm}_final_validation", command=None)
            diagnostics = validation_diagnostics(
                out / arm / "final.pt",
                data,
                out / f"{arm}_final_validation",
                device=args.device,
                batch_size=args.batch_size,
                workers=args.workers,
            )
            if not math.isclose(
                diagnostics["aggregate"]["iou"],
                final_ious[arm],
                abs_tol=1e-10,
            ):
                raise ValueError(
                    "Final reloaded validation counts disagree with training"
                )
            final_diagnostics[arm] = diagnostics
            local[arm] = correction_case_diagnostics(
                out / arm / "final.pt",
                reviewroot,
                out / f"{arm}_training_cases",
                device=args.device,
            )
            update(f"{arm}_complete")
        if (
            dataset_fingerprint(data) != source_sha
            or _sha256(checkpoint) != incumbent["checkpoint_sha256"]
            or load_review(reviewroot, require_sealed=True) != review
        ):
            raise ValueError(
                "Frozen parent, incumbent or review changed during correction experiment"
            )
        local["incumbent"] = correction_case_diagnostics(
            checkpoint, reviewroot, out / "incumbent_training_cases", device=args.device
        )
        selected = select_candidate(candidates)
        _write_json(
            out / "selection.json",
            {
                "selected": selected,
                "candidates": candidates,
                "used_test_metrics": False,
            },
        )
        control, corrected = (
            final_diagnostics[name] for name in ("control", "corrected")
        )
        result = {
            "candidates": candidates,
            "selected": selected,
            "final_validation": final_diagnostics,
            "final_validation_iou_change_corrected_minus_control": corrected[
                "aggregate"
            ]["iou"]
            - control["aggregate"]["iou"],
            "final_validation_boundary_f1_change_corrected_minus_control": corrected[
                "binary_mask_boundary"
            ]["f1"]
            - control["binary_mask_boundary"]["f1"],
            "local_training_diagnostics": local,
            "local_training_effect_iou_change_on_proposed_masks": local["corrected"][
                "aggregate"
            ]["proposed"]["iou"]
            - local["control"]["aggregate"]["proposed"]["iou"],
            "frozen_incumbent_unchanged": True,
            "source_dataset_content_sha256": source_sha,
            "review_content_sha256": review["content_sha256"],
            "effective_corrected_training_content_sha256": target_metadata[
                "effective_training_content_sha256"
            ],
            "training_review_provisional": target_metadata[
                "training_review_provisional"
            ],
            "test_inference_performed": False,
            "interpretation": registration["interpretation"],
        }
        _write_json(out / "comparison.json", result)
        lines = [
            "# Matched training-mask correction test",
            "",
            registration["interpretation"],
            "",
            "## Fixed-budget training effect (final epoch, threshold 0.50)",
            "",
            "| Arm | Validation IoU | Validation boundary F1 (2px) |",
            "| --- | ---: | ---: |",
        ]
        for arm, diagnostics in final_diagnostics.items():
            lines.append(
                f"| {arm} | {diagnostics['aggregate']['iou']:.6f} | {diagnostics['binary_mask_boundary']['f1']:.6f} |"
            )
        lines += [
            "",
            f"Corrected minus control validation IoU: **{result['final_validation_iou_change_corrected_minus_control']:+.6f}**.",
            "",
            "## Validation-selected best checkpoints",
            "",
            "| Candidate | Calibrated validation IoU | Selected epoch | Threshold |",
            "| --- | ---: | ---: | ---: |",
        ]
        for candidate in candidates:
            lines.append(
                f"| {candidate['name']} | {candidate['validation_iou']:.6f} | {candidate['epoch']} | {candidate['threshold']:.2f} |"
            )
        lines += [
            "",
            f"Selected candidate: **{selected['name']}**. The original checkpoint is preserved.",
            "",
            "## Edited training tiles (diagnostic only, fixed threshold 0.50)",
            "",
            "| Model | IoU against original labels | IoU against proposed labels |",
            "| --- | ---: | ---: |",
        ]
        for name in ("incumbent", "control", "corrected"):
            lines.append(
                f"| {name} | {local[name]['aggregate']['original']['iou']:.6f} | {local[name]['aggregate']['proposed']['iou']:.6f} |"
            )
        lines += [
            "",
            "The two label columns are annotation sensitivity, while the control/corrected rows on the same labels compare local training effects. These tiles were trained on and cannot establish generalization or independently confirm the labels.",
            "",
            "All validation reference masks are unchanged. No test inference, threshold sweep on test, image changes or source-mask mutation was performed.",
            "See registration.json, source_snapshot.zip, per-arm histories, best.pt/final.pt, final validation per-image CSVs, and comparison.json.",
        ]
        (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        update("complete", selected=selected["name"], command=None)
        return result
    except Exception as exc:
        update("failed", error=str(exc), traceback=traceback.format_exc(), command=None)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "review", "incumbent", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--allow-provisional-review", action="store_true")
    run_experiment(parser.parse_args())


if __name__ == "__main__":
    main()
