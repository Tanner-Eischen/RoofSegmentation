"""Compare frozen predictions with original and separately versioned review labels.

This evaluates reviewed validation tiles only. It does not replace the locked
full-dataset evaluator, tune a threshold, or run inference on train/test tiles.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .common import load_binary_mask
from .dataset import normalize_rgb
from .metrics import metrics_from_counts
from .models import load_model


def _load_review(root):
    from .adjudication import load_review

    return load_review(root, require_sealed=True)


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _counts(prediction, truth, valid):
    return [
        int((prediction & truth & valid).sum()),
        int((prediction & ~truth & valid).sum()),
        int((~prediction & truth & valid).sum()),
        int((~prediction & ~truth & valid).sum()),
    ]


def _human_approved(manifest):
    policy = manifest.get("policy", {})
    return (
        policy.get("status") == "approved"
        and policy.get("reviewer_kind") == "human"
        and bool(str(policy.get("reviewer_id", "")).strip())
        and all(
            case.get("reviewer_kind") == "human"
            and bool(str(case.get("reviewer_id", "")).strip())
            and case.get("decision") in {"keep", "correct", "uncertain"}
            for case in manifest["cases"]
        )
    )


def _new_bucket():
    return {
        "tiles": 0,
        "pending_tiles": 0,
        "unresolved_tiles": 0,
        "fully_ignored_tiles": 0,
        "paired_tiles": 0,
        "ignored_pixels": 0,
        "evaluated_paired_pixels": 0,
        "changed_pixels": 0,
        "changed_evaluated_pixels": 0,
        "original_all_counts": [0, 0, 0, 0],
        "paired_original_counts": [0, 0, 0, 0],
        "paired_review_counts": [0, 0, 0, 0],
    }


def _add(bucket, row, all_counts, original_counts, review_counts):
    bucket["tiles"] += 1
    bucket["pending_tiles"] += row["decision"] == "pending"
    bucket["unresolved_tiles"] += row["unresolved"]
    bucket["fully_ignored_tiles"] += row["fully_ignored"]
    bucket["paired_tiles"] += row["paired_eligible"]
    for key in (
        "ignored_pixels",
        "evaluated_paired_pixels",
        "changed_pixels",
        "changed_evaluated_pixels",
    ):
        bucket[key] += row[key]
    for key, counts in (
        ("original_all_counts", all_counts),
        ("paired_original_counts", original_counts),
        ("paired_review_counts", review_counts),
    ):
        if counts is not None:
            bucket[key] = [a + b for a, b in zip(bucket[key], counts)]


def _finish(bucket):
    result = {
        key: value for key, value in bucket.items() if not key.endswith("_counts")
    }
    result["original_all_pixels"] = (
        metrics_from_counts(*bucket["original_all_counts"]) if bucket["tiles"] else None
    )
    for name in ("paired_original", "paired_review"):
        result[name] = (
            metrics_from_counts(*bucket[name + "_counts"])
            if bucket["paired_tiles"]
            else None
        )
    result["paired_iou_change"] = (
        result["paired_review"]["iou"] - result["paired_original"]["iou"]
        if bucket["paired_tiles"]
        else None
    )
    return result


@torch.no_grad()
def score_review(
    review: Path,
    checkpoint: Path,
    out: Path,
    *,
    device="cpu",
    allow_provisional=False,
    save_predictions=True,
):
    review, checkpoint, out = (
        Path(path).resolve() for path in (review, checkpoint, out)
    )
    if out == review or review in out.parents:
        raise ValueError("Scoring output must be outside the sealed review set")
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise FileExistsError("Choose a fresh scoring output directory")
    manifest = _load_review(review)
    if manifest.get("schema_version") != "roof_review_v1":
        raise ValueError("Unsupported review schema")
    if not manifest.get("sealed") or not manifest.get("content_sha256"):
        raise ValueError("Review set must be sealed before scoring")
    human_approved = _human_approved(manifest)
    if not human_approved and not allow_provisional:
        raise ValueError(
            "Confirmed scoring requires an approved human policy and human adjudication "
            "for every case. Use --allow-provisional for explicitly provisional results."
        )
    checkpoint_sha = _sha256(checkpoint)
    if checkpoint_sha != manifest.get("checkpoint_sha256"):
        raise ValueError("Checkpoint SHA differs from the frozen review checkpoint")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config = state["config"]
    if config.get("dataset_content_sha256") != manifest.get(
        "source_dataset_content_sha256"
    ):
        raise ValueError(
            "Checkpoint source dataset fingerprint differs from review parent"
        )
    if config.get("normalization") != "imagenet_rgb_v1":
        raise ValueError("Unsupported checkpoint normalization")
    size = config.get("image_size")
    if type(size) is not int or size < 32 or size % 32:
        raise ValueError("Invalid checkpoint image size")
    threshold = state.get("threshold")
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not math.isfinite(threshold)
        or not 0 < threshold < 1
    ):
        raise ValueError("Invalid frozen checkpoint threshold")
    cases = manifest["cases"]
    if any(case.get("split") not in {"train", "val"} for case in cases):
        raise ValueError("Review sets may contain train/validation cases only")
    val_cases = [case for case in cases if case["split"] == "val"]
    if not val_cases:
        raise ValueError("No validation cases in review set")
    if any(case.get("cohort") not in {"representative", "targeted"} for case in cases):
        raise ValueError("Unknown review cohort")
    model, loaded_config = load_model(checkpoint, device)
    if loaded_config != config:
        raise ValueError("Checkpoint changed during loading")
    model.eval()
    out.mkdir(parents=True, exist_ok=True)
    if save_predictions:
        (out / "predictions").mkdir()
    cohorts = {name: _new_bucket() for name in ("representative", "targeted")}
    strata = {name: {} for name in cohorts}
    details = []
    for case in val_cases:
        with Image.open(review / case["image"]) as image:
            rgb = np.asarray(image.convert("RGB")).copy()
        if rgb.shape[:2] != (size, size):
            raise ValueError("Review image size differs from native checkpoint grid")
        original = load_binary_mask(review / case["original_mask"], (size, size)) > 0
        revised = load_binary_mask(review / case["mask"], (size, size)) > 0
        ignored = load_binary_mask(review / case["ignore_mask"], (size, size)) > 0
        changed = original != revised
        if case["decision"] == "uncertain" and changed.any():
            raise ValueError(
                "Uncertain case cannot change label values; use ignore mask"
            )
        if case["decision"] not in {"pending", "keep", "correct", "uncertain"}:
            raise ValueError("Unknown adjudication decision")
        if case["decision"] == "keep" and changed.any():
            raise ValueError("Keep decision cannot change label values")
        probability = model(normalize_rgb(rgb).to(device)).sigmoid().float()
        if (
            probability.shape != (1, 1, size, size)
            or not torch.isfinite(probability).all()
        ):
            raise ValueError("Expected finite single-channel binary probabilities")
        probability = probability[0, 0].cpu().numpy()
        prediction = probability >= threshold
        valid = ~ignored
        unresolved = case["decision"] == "uncertain" and not bool(ignored.any())
        if case["decision"] == "pending":
            exclusion_reason = "pending_decision"
        elif unresolved:
            exclusion_reason = "unlocalized_uncertainty"
        elif not valid.any():
            exclusion_reason = "fully_ignored"
        else:
            exclusion_reason = ""
        eligible = not exclusion_reason
        all_counts = _counts(prediction, original, np.ones_like(valid))
        original_counts = _counts(prediction, original, valid) if eligible else None
        review_counts = _counts(prediction, revised, valid) if eligible else None
        row = {
            "sample_id": case["sample_id"],
            "split": "val",
            "group": case["group"],
            "cohort": case["cohort"],
            "stratum": case["stratum"],
            "decision": case["decision"],
            "reviewer_kind": case["reviewer_kind"],
            "reviewer_id": case["reviewer_id"],
            "inclusion_probability": case.get("inclusion_probability"),
            "paired_eligible": eligible,
            "paired_exclusion_reason": exclusion_reason,
            "unresolved": unresolved,
            "fully_ignored": not bool(valid.any()),
            "ignored_pixels": int(ignored.sum()),
            "evaluated_paired_pixels": int(valid.sum()) if eligible else 0,
            "changed_pixels": int(changed.sum()),
            "changed_evaluated_pixels": int((changed & valid).sum()) if eligible else 0,
            "added_building_pixels_evaluated": int((~original & revised & valid).sum())
            if eligible
            else 0,
            "removed_building_pixels_evaluated": int(
                (original & ~revised & valid).sum()
            )
            if eligible
            else 0,
        }
        for prefix, counts in (
            ("original_all", all_counts),
            ("paired_original", original_counts),
            ("paired_review", review_counts),
        ):
            row.update(
                {
                    f"{prefix}_{name}": value
                    for name, value in (
                        metrics_from_counts(*counts).items()
                        if counts is not None
                        else {
                            key: None for key in metrics_from_counts(0, 0, 0, 0)
                        }.items()
                    )
                }
            )
        cohort = case["cohort"]
        _add(cohorts[cohort], row, all_counts, original_counts, review_counts)
        bucket = strata[cohort].setdefault(case["stratum"], _new_bucket())
        _add(bucket, row, all_counts, original_counts, review_counts)
        if save_predictions:
            prediction_stem = f"val_{len(details):04d}"
            row["prediction_mask"] = f"predictions/{prediction_stem}_mask.png"
            row["prediction_probability"] = (
                f"predictions/{prediction_stem}_probability.png"
            )
            Image.fromarray(prediction.astype(np.uint8) * 255).save(
                out / row["prediction_mask"]
            )
            Image.fromarray(np.rint(probability * 255).astype(np.uint8)).save(
                out / row["prediction_probability"]
            )
        details.append(row)
    if _sha256(checkpoint) != checkpoint_sha or _load_review(review) != manifest:
        raise ValueError("Frozen checkpoint or review set changed during scoring")
    result = {
        "review_version": manifest["version"],
        "review_content_sha256": manifest["content_sha256"],
        "source_dataset_content_sha256": manifest["source_dataset_content_sha256"],
        "checkpoint_sha256": checkpoint_sha,
        "threshold_frozen_from_checkpoint": threshold,
        "status": "human_adjudicated" if human_approved else "provisional",
        "human_adjudication_complete": human_approved,
        "allow_provisional": bool(allow_provisional),
        "prediction_probability_encoding": "Diagnostic PNGs are uint8 round(probability*255); counts use full-precision model probabilities before quantization.",
        "validation_cases": len(val_cases),
        "training_cases_not_inferred": sum(case["split"] == "train" for case in cases),
        "validation_decision_counts": dict(
            Counter(case["decision"] for case in val_cases)
        ),
        "test_inference_performed": False,
        "cohorts": {},
        "interpretation": "Raw validation review-sample metrics, not full-WHU scores or population prevalence. Targeted and representative cases remain separate. Paired original/review scores use identical non-ignored pixels and the unchanged model/threshold. Pending cases and uncertain cases without localized ignore regions appear only in original-label baseline. Fully ignored cases have no paired score. No training, threshold tuning or test inference performed.",
    }
    for name, bucket in cohorts.items():
        result["cohorts"][name] = {
            **_finish(bucket),
            "coverage_strata": {
                key: _finish(value) for key, value in strata[name].items()
            },
        }
    (out / "comparison.json").write_text(
        json.dumps(result, indent=2, allow_nan=False), encoding="utf-8"
    )
    with (out / "val_per_image.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(details[0]))
        writer.writeheader()
        writer.writerows(details)
    lines = [
        "# Frozen U-Net review-label comparison",
        "",
        f"Review version: `{manifest['version']}`. Status: **{result['status']}**.",
        f"Frozen threshold: {threshold:.2f}. Validation cases: {len(val_cases)}.",
        "",
        "| Cohort | Tiles | Paired tiles | Original all-pixel IoU | Paired original IoU | Paired review IoU | Delta |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, values in result["cohorts"].items():
        numbers = []
        for key in ("original_all_pixels", "paired_original", "paired_review"):
            numbers.append(
                f"{values[key]['iou']:.6f}" if values[key] is not None else "—"
            )
        delta = values["paired_iou_change"]
        delta_text = f"{delta:+.6f}" if delta is not None else "—"
        lines.append(
            f"| {name} | {values['tiles']} | {values['paired_tiles']} | "
            + " | ".join(numbers)
            + f" | {delta_text} |"
        )
    lines += ["", result["interpretation"]]
    if not human_approved:
        lines += [
            "",
            "These results use provisional review decisions. They do not establish independently confirmed annotation accuracy or justify a production class change.",
        ]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--allow-provisional", action="store_true")
    parser.add_argument("--no-save-predictions", action="store_true")
    args = parser.parse_args()
    result = score_review(
        args.review,
        args.checkpoint,
        args.out,
        device=args.device,
        allow_provisional=args.allow_provisional,
        save_predictions=not args.no_save_predictions,
    )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
