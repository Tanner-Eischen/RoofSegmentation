"""Explicit local model preparation; no downloads or changes to training artifacts."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
from pathlib import Path

import torch


def sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def checkpoint_config(state):
    if not isinstance(state, dict) or not isinstance(state.get("config"), dict):
        raise ValueError("Expected a roof_pipeline state-dictionary checkpoint")
    config = state["config"]
    if config.get("arch") not in {
        "tiny",
        "unet-resnet34",
        "unetplusplus-efficientnet-b0",
        "deeplabv3plus-resnet34",
    }:
        raise ValueError("Unsupported checkpoint architecture")
    size = config.get("image_size")
    if type(size) is not int or not 32 <= size <= 1024 or size % 32:
        raise ValueError("image_size must be a multiple of 32 in [32,1024]")
    if config.get("normalization") != "imagenet_rgb_v1":
        raise ValueError("Unsupported normalization")
    threshold = state.get("threshold")
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not math.isfinite(threshold)
        or not 0 < threshold < 1
    ):
        raise ValueError("Checkpoint needs a frozen numeric threshold in (0,1)")
    return config, float(threshold)


def read_checkpoint(checkpoint):
    path = Path(checkpoint)
    if not path.is_file() or path.stat().st_size > 512_000_000:
        raise ValueError("Checkpoint must be a regular file smaller than 512 MB")
    with path.open("rb") as stream:
        content = stream.read(512_000_001)
    if len(content) > 512_000_000:
        raise ValueError("Checkpoint exceeds size limit")
    return content


def prepare_release(checkpoint, output, source_manifest=None):
    checkpoint, output = Path(checkpoint), Path(output)
    if output.exists():
        raise FileExistsError("Model release exists; choose a fresh output path")
    content = read_checkpoint(checkpoint)
    config, threshold = checkpoint_config(
        torch.load(io.BytesIO(content), map_location="cpu", weights_only=True)
    )
    source = None
    if source_manifest is not None:
        source_bytes = Path(source_manifest).read_bytes()
        recorded = json.loads(source_bytes)
        source = {
            "base_revision": recorded["base_commit"],
            "worktree_dirty": recorded["worktree_dirty"],
            "manifest_sha256": sha256(source_bytes),
            "file_sha256": recorded["sha256"],
        }
    record = {
        "schema_version": 1,
        "model_id": "roof-pipeline-" + sha256(content)[:16],
        "artifact_format": "roof_pipeline_state_dict_v1",
        "weights_sha256": sha256(content),
        "architecture": config["arch"],
        "input_channels": 3,
        "image_size": config["image_size"],
        "normalization": config["normalization"],
        "threshold": threshold,
        "aspp_rates": config.get("aspp_rates"),
        "boundary_supervision": config.get("boundary_supervision", False),
        "training_dataset_content_sha256": config.get("dataset_content_sha256"),
        "training_dataset_manifest_sha256": config.get("dataset_manifest_sha256"),
        "training_source": source,
        "preprocessing_version": "imagenet-rgb-edgepad-tiled-v1",
        "overlap": min(64, config["image_size"] // 4),
        "image_policy": "No full-image resize; edge padding; weighted probability blending before thresholding",
        "alpha_policy": "Pillow RGB conversion discards alpha without compositing; grayscale repeats into RGB",
        "gsd_policy": "Native image pixels; GSD is not inferred or validated",
        "label_definition": "binary building extent",
        "limitations": [
            "Image-space outputs only; no material, damage, instance or physical-area claim.",
            "Preparation verifies metadata and artifact identity, not model accuracy or serving readiness.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def validate_release(record):
    if (
        record.get("schema_version") != 1
        or record.get("artifact_format") != "roof_pipeline_state_dict_v1"
    ):
        raise ValueError("Unsupported model release format")
    if not isinstance(record.get("model_id"), str) or not record["model_id"]:
        raise ValueError("Missing model release identity")
    weight_hash = record.get("weights_sha256", "")
    if len(weight_hash) != 64 or any(c not in "0123456789abcdef" for c in weight_hash):
        raise ValueError("Missing checkpoint SHA256")
    if (
        record.get("input_channels") != 3
        or record.get("preprocessing_version") != "imagenet-rgb-edgepad-tiled-v1"
    ):
        raise ValueError("Unsupported preprocessing")
    size = record.get("image_size")
    if type(size) is not int or not 32 <= size <= 1024 or size % 32:
        raise ValueError("Unsupported release image size")
    if record.get("overlap") != min(64, size // 4):
        raise ValueError("Unsupported tiling overlap")
    if (
        not record.get("limitations")
        or record.get("label_definition") != "binary building extent"
    ):
        raise ValueError("Missing label definition or limitations")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            prepare_release(args.checkpoint, args.out, args.source_manifest), indent=2
        )
    )


if __name__ == "__main__":
    main()
