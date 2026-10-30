"""Explicit, frozen training-mask interventions from a sealed review version.

The dataset fingerprint continues to identify the unchanged physical parent
dataset. Effective training targets have a second fingerprint binding the exact
applied mask hashes. No image, validation label or test label is overridden.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from PIL import Image

from .adjudication import load_review, safe_path, sha256
from .common import load_binary_mask, read_manifest


def source_training_metadata(dataset_sha: str) -> dict:
    """With no intervention, the effective training content is the parent data."""
    return {
        "effective_training_content_sha256": dataset_sha,
        "training_review_content_sha256": None,
        "training_review_version": None,
        "training_review_path": None,
        "training_review_provisional": False,
        "training_review_checkpoint_sha256": None,
        "training_review_mask_changes": [],
    }


def _human_ready(review):
    policy = review["policy"]
    return (
        policy.get("status") == "approved"
        and policy.get("reviewer_kind") == "human"
        and bool(str(policy.get("reviewer_id", "")).strip())
        and all(
            case.get("reviewer_kind") == "human"
            and bool(str(case.get("reviewer_id", "")).strip())
            and case.get("decision") in {"keep", "correct", "uncertain"}
            for case in review["cases"]
        )
    )


def load_training_overlay(
    reviewroot: Path,
    data: Path,
    dataset_sha: str,
    init_checkpoint: Path | None,
    *,
    allow_provisional=False,
) -> tuple[dict[str, Path], dict]:
    """Validate a reviewed intervention against its original data and checkpoint.

    The caller supplies the full, freshly computed parent dataset fingerprint.
    This function also verifies every reviewed image/original mask against the
    physical source and checks all review files through the sealed version.
    Loading an overlay consumes no Python, NumPy or PyTorch random draws.
    """
    reviewroot, data = Path(reviewroot).resolve(), Path(data).resolve()
    review = load_review(reviewroot, require_sealed=True)
    if review["source_dataset_content_sha256"] != dataset_sha:
        raise ValueError("Training review source dataset fingerprint differs")
    if review.get("source_manifest_sha256") != sha256(data / "manifest.csv"):
        raise ValueError("Training review source manifest differs")
    provisional = bool(review.get("provisional", True)) or not _human_ready(review)
    if provisional and not allow_provisional:
        raise ValueError(
            "Training review is provisional; --allow-provisional-review is required"
        )
    if init_checkpoint is None:
        raise ValueError(
            "Training review requires its frozen initialization checkpoint"
        )
    checkpoint_sha = sha256(Path(init_checkpoint))
    if checkpoint_sha != review["checkpoint_sha256"]:
        raise ValueError("Training review initialization checkpoint SHA differs")
    checkpoint = torch.load(init_checkpoint, map_location="cpu", weights_only=True)
    if checkpoint["config"].get("dataset_content_sha256") != dataset_sha:
        raise ValueError("Training review checkpoint parent dataset differs")
    manifest = {row["sample_id"]: row for row in read_manifest(data)}
    overrides, changes = {}, []
    for case in review["cases"]:
        ident = case["sample_id"]
        source = manifest.get(ident)
        if source is None or any(
            source.get(key) != case.get(key)
            for key in ("sample_id", "split", "group", "source")
        ):
            raise ValueError("Training review source identity/partition/group differs")
        if sha256(data / source["image"]) != case["image_sha256"]:
            raise ValueError("Training review source image SHA differs")
        if sha256(data / source["mask"]) != case["original_mask_sha256"]:
            raise ValueError("Training review source original-mask SHA differs")
        with Image.open(data / source["image"]) as image:
            size = image.size
        original = load_binary_mask(data / source["mask"], size) > 0
        proposed = load_binary_mask(safe_path(reviewroot, case["mask"]), size) > 0
        ignored = load_binary_mask(safe_path(reviewroot, case["ignore_mask"]), size) > 0
        if ignored.any():
            raise ValueError("Training review overlays do not support ignored pixels")
        changed = original != proposed
        if not changed.any():
            continue
        if source["split"] != "train" or case["decision"] != "correct":
            raise ValueError("Only declared training-mask corrections may be applied")
        overrides[ident] = safe_path(reviewroot, case["mask"])
        changes.append(
            {
                "sample_id": ident,
                "split": "train",
                "original_mask_sha256": case["original_mask_sha256"],
                "mask_sha256": case["mask_sha256"],
                "changed_pixels": int(changed.sum()),
                "added_building_pixels": int((proposed & ~original).sum()),
                "removed_building_pixels": int((original & ~proposed).sum()),
            }
        )
    if not overrides:
        raise ValueError("Training review contains no declared training corrections")
    changes.sort(key=lambda row: row["sample_id"])
    effective = hashlib.sha256(
        json.dumps(
            {
                "source_dataset_content_sha256": dataset_sha,
                "training_mask_overrides": changes,
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    metadata = {
        "effective_training_content_sha256": effective,
        "training_review_content_sha256": review["content_sha256"],
        "training_review_version": review["version"],
        "training_review_path": str(reviewroot),
        "training_review_provisional": provisional,
        "training_review_checkpoint_sha256": checkpoint_sha,
        "training_review_mask_changes": changes,
    }
    return overrides, metadata
