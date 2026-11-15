#!/usr/bin/env python3
"""Render seeded validation-only examples after the complete architecture pilot.

CPU is the default. Does not tune thresholds, select examples by performance,
or run test inference. Use only after training and verified checkpoint retrieval.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from pathlib import Path

if __package__:
    from .analyze_architecture_comparison import ARCHITECTURES, LABELS, load_results
else:
    from analyze_architecture_comparison import ARCHITECTURES, LABELS, load_results


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def render(samples, predictions, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with plt.rc_context({"font.size": 9, "svg.fonttype": "none"}):
        figure, axes = plt.subplots(
            len(samples),
            5,
            squeeze=False,
            figsize=(12, 2.55 * len(samples)),
            constrained_layout=True,
        )
        for row, sample in enumerate(samples):
            images = [
                sample["rgb"],
                sample["truth"],
                *[predictions[arch][row] for arch in ARCHITECTURES],
            ]
            for column, image in enumerate(images):
                axes[row, column].imshow(
                    image,
                    cmap="gray" if column else None,
                    vmin=0 if column else None,
                    vmax=1 if column else None,
                    interpolation="nearest",
                )
                axes[row, column].set_xticks([])
                axes[row, column].set_yticks([])
                if row == 0:
                    axes[row, column].set_title(
                        ("Input RGB", "Building ground truth", *LABELS)[column]
                    )
            axes[row, 0].set_ylabel(sample["sample_id"], fontsize=8)
        figure.suptitle(
            "Seeded random validation examples | primary threshold 0.50 | n=1 pilot\n"
            "White = building; black = background. Same examples for every model.",
            fontsize=11,
        )
        for extension in ("png", "svg"):
            figure.savefig(
                out / f"qualitative_validation.{extension}",
                dpi=300,
                metadata={"Creator": "RoofSegmentation validation diagnostics"},
            )
            if extension == "svg":
                path = out / "qualitative_validation.svg"
                path.write_text(
                    "\n".join(line.rstrip() for line in path.read_text().splitlines())
                    + "\n",
                    encoding="utf-8",
                )
        plt.close(figure)


def generate(results, data, out, examples=4, seed=42, device="cpu"):
    summary, _ = load_results(results)
    results, data, out = results.resolve(), data.resolve(), out.resolve()
    if out in (results, data) or any(
        (out / name).exists() for name in ("plan.json", "manifest.csv", "history.json")
    ):
        raise ValueError("Choose a separate qualitative-output directory")
    if (
        type(examples) is not int
        or examples < 1
        or type(seed) is not int
        or not 0 <= seed < 2**32
    ):
        raise ValueError("Use a positive example count and a seed in [0, 2**32)")
    if device not in {"cpu", "cuda"}:
        raise ValueError("Choose cpu or explicitly request cuda")
    # A script launched from scripts/ needs the repository for pipeline imports.
    repository = str(Path(__file__).resolve().parents[1])
    if repository not in sys.path:
        sys.path.insert(0, repository)
    import numpy as np
    import torch
    from PIL import Image

    from roof_pipeline.common import dataset_fingerprint
    from roof_pipeline.dataset import RoofDataset
    from roof_pipeline.models import load_model

    expected_data = summary["dataset"]
    # Fingerprinting reads all partition bytes for integrity; only VAL is inferred.
    if (
        sha256(data / "manifest.csv") != expected_data["manifest_sha256"]
        or dataset_fingerprint(data) != expected_data["content_sha256"]
    ):
        raise ValueError("Local imagery/labels differ from the registered WHU dataset")
    dataset = RoofDataset(data, "val", size=512, augment=False)
    if len(dataset) != 1036 or examples > len(dataset):
        raise ValueError("Expected full WHU validation and no more examples than tiles")
    ordered = sorted(
        range(len(dataset)), key=lambda index: dataset.rows[index]["sample_id"]
    )
    indices = random.Random(seed).sample(ordered, examples)
    selection = {
        "split": "val",
        "sample_ids": [dataset.rows[index]["sample_id"] for index in indices],
        "selection_seed": seed,
        "examples": examples,
        "training_seed": 42,
        "selection_policy": "seeded random sample of sorted validation IDs, frozen before model loading or prediction",
        "plan_sha256": summary["source_json_sha256"]["plan.json"],
        "comparison_sha256": summary["source_json_sha256"]["comparison.json"],
        "dataset_content_sha256": expected_data["content_sha256"],
        "threshold": 0.5,
        "checkpoint_sha256": {
            row["architecture"]: row["checkpoint_sha256"] for row in summary["rows"]
        },
    }
    out.mkdir(parents=True, exist_ok=True)
    selection_file = out / "selection.json"
    if (
        selection_file.exists()
        and json.loads(selection_file.read_text(encoding="utf-8")) != selection
    ):
        raise ValueError("Existing frozen selection differs; use its original options")
    write_json(
        selection_file, selection
    )  # Persist selection before checkpoint loading.
    samples = []
    for index in indices:
        x, truth = dataset[index]
        source = dataset.rows[index]
        with Image.open(data / source["image"]) as image:
            rgb = np.asarray(
                image.convert("RGB").resize((512, 512), Image.Resampling.BILINEAR)
            ).copy()
        samples.append(
            {
                "sample_id": source["sample_id"],
                "x": x,
                "rgb": rgb,
                "truth": truth[0].numpy().astype(bool),
            }
        )
    predictions, measurements = {}, []
    for row in summary["rows"]:
        arch = row["architecture"]
        run = results / "runs" / f"{arch}_seed42"
        checkpoint = run / "best.pt"
        if sha256(checkpoint) != row["checkpoint_sha256"]:
            raise ValueError(
                f"{arch}: local checkpoint hash differs from verified assessment"
            )
        model, config = load_model(checkpoint, device)
        expected = {
            "arch": arch,
            "seed": 42,
            "epoch": row["best_epoch"],
            "image_size": 512,
            "dataset_content_sha256": expected_data["content_sha256"],
            "dataset_manifest_sha256": expected_data["manifest_sha256"],
            "normalization": "imagenet_rgb_v1",
            "boundary_supervision": False,
        }
        if any(config.get(key) != value for key, value in expected.items()):
            raise ValueError(f"{arch}: loaded checkpoint identity differs")
        server_file = run / "validation_primary_0p5_per_image.csv"
        server = {}
        if server_file.is_file():
            with server_file.open(newline="", encoding="utf-8") as stream:
                entries = list(csv.DictReader(stream))
            server = {entry["sample_id"]: entry for entry in entries}
            if len(server) != len(entries) or len(server) != len(dataset):
                raise ValueError(
                    f"{arch}: server validation CSV is partial or duplicated"
                )
        predictions[arch] = []
        with torch.inference_mode():
            for sample in samples:
                probability = (
                    model(sample["x"].unsqueeze(0).to(device))
                    .sigmoid()[0, 0]
                    .cpu()
                    .numpy()
                )
                if (
                    probability.shape != (512, 512)
                    or not np.isfinite(probability).all()
                ):
                    raise ValueError(f"{arch}: invalid prediction")
                prediction, truth = probability >= 0.5, sample["truth"]
                counts = {
                    "tp": int((prediction & truth).sum()),
                    "fp": int((prediction & ~truth).sum()),
                    "fn": int((~prediction & truth).sum()),
                    "tn": int((~prediction & ~truth).sum()),
                }
                reference = server.get(sample["sample_id"])
                if server and reference is None:
                    raise ValueError(
                        "Selected validation ID is missing from server CSV"
                    )
                server_counts = (
                    {key: int(float(reference[key])) for key in counts}
                    if reference
                    else None
                )
                measurements.append(
                    {
                        "architecture": arch,
                        "sample_id": sample["sample_id"],
                        "counts": counts,
                        "server_counts": server_counts,
                        "server_counts_match": counts == server_counts
                        if server_counts is not None
                        else None,
                        "server_csv_sha256": sha256(server_file) if reference else None,
                    }
                )
                predictions[arch].append(prediction)
        del model
    metadata = {
        **selection,
        "device": device,
        "torch": str(torch.__version__),
        "inference_dtype": "float32; no autocast",
        "measurements": measurements,
        "test_inference": False,
        "count_comparison_note": "CPU and server CUDA/batched kernels can differ at threshold-adjacent pixels; exact-count agreement and mismatches are retained explicitly.",
        "interpretation": "Descriptive random validation examples, not independent geographic confirmation or model selection.",
    }
    write_json(out / "qualitative_metadata.json", metadata)
    render(samples, predictions, out)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--examples", type=int, default=4)
    parser.add_argument(
        "--seed", type=int, default=42, help="Validation sample-selection seed"
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    metadata = generate(
        args.results, args.data, args.out, args.examples, args.seed, args.device
    )
    mismatches = sum(
        row["server_counts_match"] is False for row in metadata["measurements"]
    )
    print(
        f"Wrote {len(metadata['measurements'])} validation predictions, figures and metadata; {mismatches} server count mismatches."
    )


if __name__ == "__main__":
    main()
