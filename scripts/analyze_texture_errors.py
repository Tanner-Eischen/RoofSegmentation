"""Explain frozen model errors and replay post-hoc high-FP test examples.

Examples are selected after inspecting test errors. They are neither random nor
representative and must not be used for tuning or as fresh training examples.
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import math
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from roof_pipeline.metrics import metrics_from_counts  # noqa: E402
from scripts.analyze_architecture_comparison import (  # noqa: E402
    ARCHITECTURES,
    LABELS,
    load_results,
)
from scripts.evaluate_architecture_test import (  # noqa: E402
    COUNTS,
    checkpoint_contract,
    packet_checksums,
    sha256,
    verify_test_result,
    write_json,
)


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def analyze_rows(model_rows):
    """Align immutable rows by ID and reconcile empty/nonempty subgroup counts."""
    if set(model_rows) != set(ARCHITECTURES):
        raise ValueError("Expected all three registered architectures")
    aligned, identities, truth = {}, None, {}
    for arch, rows in model_rows.items():
        by_id = {}
        for row in rows:
            identity = row["sample_id"]
            if not identity or identity in by_id:
                raise ValueError("Duplicate or empty sample IDs")
            counts = {}
            for key in COUNTS:
                value = float(row[key])
                if not math.isfinite(value) or value < 0 or not value.is_integer():
                    raise ValueError("Counts must be finite nonnegative integers")
                counts[key] = int(value)
            pixels, positive = sum(counts.values()), counts["tp"] + counts["fn"]
            if not pixels:
                raise ValueError("A tile has no pixels")
            if identity in truth and truth[identity] != (pixels, positive):
                raise ValueError("Ground-truth pixel counts differ across models")
            truth[identity] = pixels, positive
            by_id[identity] = counts
        if not by_id or (identities is not None and set(by_id) != identities):
            raise ValueError("Sample IDs differ across models or rows are empty")
        identities = set(by_id)
        aligned[arch] = by_id
    models, per_tile = {}, []
    for arch in ARCHITECTURES:
        groups = {}
        for name, empty in (("empty_gt", True), ("contains_buildings", False)):
            subset = [i for i in sorted(identities) if (truth[i][1] == 0) == empty]
            totals = {key: sum(aligned[arch][i][key] for i in subset) for key in COUNTS}
            pixels = sum(totals.values())
            groups[name] = {
                "samples": len(subset),
                "pixels": pixels,
                **totals,
                "gt_positive_pixels": totals["tp"] + totals["fn"],
                "iou": None
                if empty or not subset
                else metrics_from_counts(*(totals[k] for k in COUNTS))["iou"],
                "false_positive_pixel_fraction": totals["fp"] / pixels
                if pixels
                else None,
                "tiles_with_false_positives": sum(
                    aligned[arch][i]["fp"] > 0 for i in subset
                ),
            }
        totals = {key: sum(group[key] for group in groups.values()) for key in COUNTS}
        models[arch] = {
            "samples": len(identities),
            "overall": metrics_from_counts(*(totals[k] for k in COUNTS)),
            "groups": groups,
        }
    for identity in sorted(identities):
        row = {
            "sample_id": identity,
            "pixels": truth[identity][0],
            "gt_positive_pixels": truth[identity][1],
        }
        for arch in ARCHITECTURES:
            row.update(
                {f"{arch}_{key}": aligned[arch][identity][key] for key in COUNTS}
            )
        row["unetplusplus_minus_unet_fp"] = (
            row["unetplusplus-resnet34_fp"] - row["unet-resnet34_fp"]
        )
        per_tile.append(row)
    return {"models": models, "per_tile": per_tile}


def select_examples(rows, count):
    if count < 1 or count > len(rows):
        raise ValueError("Example count must be positive and within tile coverage")
    return sorted(
        rows, key=lambda row: (-row["unetplusplus_minus_unet_fp"], row["sample_id"])
    )[:count]


def mask_coverage(data, manifest):
    """Count actual decoded empty masks instead of trusting a cached audit."""
    from concurrent.futures import ThreadPoolExecutor

    from PIL import Image

    def inspect(row):
        with Image.open(data / row["mask"]) as mask:
            return row["split"], mask.getextrema()[1] == 0

    coverage = {
        split: {"samples": 0, "empty_gt_tiles": 0} for split in ("train", "val", "test")
    }
    with ThreadPoolExecutor(max_workers=8) as pool:
        for split, empty in pool.map(inspect, manifest):
            coverage[split]["samples"] += 1
            coverage[split]["empty_gt_tiles"] += int(empty)
    for group in coverage.values():
        group["empty_gt_tile_fraction"] = (
            group["empty_gt_tiles"] / group["samples"] if group["samples"] else None
        )
    return {
        "method": "Decode every source mask with PIL; maximum pixel value == 0; eight CPU threads",
        "splits": coverage,
    }


def verify_packet(packet):
    manifest = json.loads((packet / "checksums.json").read_text(encoding="utf-8"))
    actual = {
        p.name for p in packet.iterdir() if p.is_file() and p.name != "checksums.json"
    }
    if set(manifest) != actual:
        raise ValueError("Test packet files differ from its checksum manifest")
    for name, digest in manifest.items():
        if Path(name).name != name or sha256(packet / name) != digest:
            raise ValueError("Test packet checksum differs")
    return {str(p.resolve()): sha256(p) for p in packet.iterdir() if p.is_file()}


def replay_examples(data, results, out, selected, summary, device):
    """Recreate each original batch of four, preserving prediction numerics."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from PIL import Image

    from roof_pipeline.dataset import RoofDataset
    from roof_pipeline.models import load_model

    dataset = RoofDataset(data, "test", size=512, augment=False)
    index_by_id = {row["sample_id"]: index for index, row in enumerate(dataset.rows)}
    batches, examples = {}, []
    for item in selected:
        index = index_by_id[item["sample_id"]]
        start = index // 4 * 4
        if start not in batches:
            batch = [dataset[i] for i in range(start, min(start + 4, len(dataset)))]
            batches[start] = (
                torch.stack([x for x, _ in batch]),
                torch.stack([y for _, y in batch]),
            )
        with Image.open(data / dataset.rows[index]["image"]) as image:
            rgb = np.array(image.convert("RGB"))
        examples.append(
            {
                "item": item,
                "index": index,
                "start": start,
                "rgb": rgb,
                "truth": batches[start][1][index - start, 0].numpy() > 0.5,
                "predictions": {},
            }
        )
    records = []
    for row in summary["rows"]:
        arch = row["architecture"]
        checkpoint = results / "runs" / f"{arch}_seed42" / "best.pt"
        checkpoint_contract(checkpoint, row, summary["dataset"])
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        saved_summary = json.loads(
            (checkpoint.parent / "training_summary.json").read_text()
        )
        if state["config"] != saved_summary["config"]:
            raise ValueError("Checkpoint config differs from its original summary")
        del state
        model, _ = load_model(checkpoint, device)
        model.float()
        model.eval()
        predictions = {}
        with torch.inference_mode(), torch.autocast(device_type=device, enabled=False):
            for start, (images, _) in batches.items():
                logits = model(images.to(device))
                if not torch.isfinite(logits).all():
                    raise ValueError("Nonfinite replay logits")
                predictions[start] = (logits.sigmoid() >= 0.5).cpu().numpy()[:, 0]
        for example in examples:
            prediction = predictions[example["start"]][
                example["index"] - example["start"]
            ]
            truth, item = example["truth"], example["item"]
            values = (
                np.count_nonzero(prediction & truth),
                np.count_nonzero(prediction & ~truth),
                np.count_nonzero(~prediction & truth),
                np.count_nonzero(~prediction & ~truth),
            )
            official = {k: item[f"{arch}_{k}"] for k in COUNTS}
            counts = dict(zip(COUNTS, map(int, values)))
            records.append(
                {
                    "architecture": arch,
                    "sample_id": item["sample_id"],
                    "dataset_index": example["index"],
                    "original_batch_ids": [
                        dataset.rows[i]["sample_id"]
                        for i in range(
                            example["start"], min(example["start"] + 4, len(dataset))
                        )
                    ],
                    "checkpoint_sha256": row["checkpoint_sha256"],
                    "official_counts": official,
                    "replay_counts": counts,
                    "count_deltas": {k: counts[k] - official[k] for k in COUNTS},
                    "exact_count_match": counts == official,
                }
            )
            example["predictions"][arch] = prediction
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
    figure, axes = plt.subplots(
        len(examples),
        5,
        figsize=(15, 3.6 * len(examples)),
        squeeze=False,
        constrained_layout=True,
    )
    for index, example in enumerate(examples):
        item, rgb, truth = example["item"], example["rgb"], example["truth"]
        axes[index, 0].imshow(rgb)
        axes[index, 0].set_title(
            f"{item['sample_id']}\nGT building pixels: {item['gt_positive_pixels']}",
            fontsize=9,
        )
        axes[index, 1].imshow(truth, cmap="gray", vmin=0, vmax=1)
        axes[index, 1].set_title("Ground truth", fontsize=9)
        for column, (arch, label) in enumerate(zip(ARCHITECTURES, LABELS), start=2):
            prediction = example["predictions"][arch]
            overlay = rgb.astype(np.float32).copy()
            for mask, color in (
                (prediction & truth, (40, 210, 70)),
                (prediction & ~truth, (245, 45, 45)),
                (~prediction & truth, (40, 110, 255)),
            ):
                overlay[mask] = 0.4 * overlay[mask] + 0.6 * np.asarray(color)
            axes[index, column].imshow(np.rint(overlay).astype(np.uint8))
            fp = int(np.count_nonzero(prediction & ~truth))
            axes[index, column].set_title(
                f"{label}\nReplay FP: {fp:,} ({100 * fp / truth.size:.2f}% of tile)",
                fontsize=9,
            )
        for axis in axes[index]:
            axis.axis("off")
    figure.suptitle(
        "Post-hoc test examples: largest U-Net++ minus U-Net false-positive increases\nGreen: TP; red: FP; blue: FN. Selected errors, not representative samples or training data.",
        fontsize=11,
    )
    for extension in ("png", "svg"):
        path = out / f"field_examples.{extension}"
        figure.savefig(path, dpi=220)
        if extension == "svg":
            path.write_text(
                "\n".join(
                    line.rstrip()
                    for line in path.read_text(encoding="utf-8").splitlines()
                )
                + "\n",
                encoding="utf-8",
            )
    plt.close(figure)
    return {
        "device": device,
        "precision": "float32, no autocast",
        "threshold": 0.5,
        "original_batch_size": 4,
        "all_counts_match": all(row["exact_count_match"] for row in records),
        "records": records,
    }


def generate(test_packet, results, data, out, *, device="cuda", examples=2):
    import torch

    from roof_pipeline.common import dataset_fingerprint, read_manifest

    test_packet, results, data, out = (
        Path(p).resolve() for p in (test_packet, results, data, out)
    )
    if out.exists():
        raise FileExistsError("Choose a fresh error-analysis output directory")
    if any(
        out == source or out.is_relative_to(source)
        for source in (test_packet, results, data)
    ):
        raise ValueError("Error analysis must not write into original evidence/data")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    protected = verify_packet(test_packet)
    summary, _ = load_results(results)
    comparison = json.loads((test_packet / "comparison.json").read_text())
    registration = json.loads((test_packet / "registration.json").read_text())
    if (
        comparison["training_comparison_sha256"] != sha256(results / "comparison.json")
        or any(
            comparison["dataset"].get(key) != summary["dataset"].get(key)
            for key in ("manifest_sha256", "content_sha256", "split_counts")
        )
        or registration["inference_batch_size"] != 4
        or registration["threshold"] != 0.5
    ):
        raise ValueError("Test packet and original comparison identities differ")
    dataset_sha = dataset_fingerprint(data)
    if (
        dataset_sha != summary["dataset"]["content_sha256"]
        or sha256(data / "manifest.csv") != summary["dataset"]["manifest_sha256"]
    ):
        raise ValueError("Original dataset changed")
    manifest = read_manifest(data)
    coverage = mask_coverage(data, manifest)
    protected.update(
        {
            str(results / name): digest
            for name, digest in summary["source_json_sha256"].items()
        }
    )
    analyses = {}
    for split in ("test", "val"):
        expected_ids = [row["sample_id"] for row in manifest if row["split"] == split]
        if len(expected_ids) != (2416 if split == "test" else 1036):
            raise ValueError("Official split coverage differs")
        model_rows = {}
        for row in summary["rows"]:
            arch = row["architecture"]
            run = results / "runs" / f"{arch}_seed42"
            paths = [
                run / name
                for name in ("best.pt", "training_summary.json", "validation_best.json")
            ]
            path = (
                test_packet / f"{arch}_per_image.csv"
                if split == "test"
                else run / "validation_primary_0p5_per_image.csv"
            )
            if split == "val" and not path.exists():
                break
            model_rows[arch] = read_csv(path)
            if split == "test":
                result = json.loads((test_packet / f"{arch}_metrics.json").read_text())
            else:
                aggregate = json.loads((run / "validation_best.json").read_text())[
                    "primary_0p5"
                ]["aggregate"]
                result = {
                    "split": "test",
                    "samples": len(expected_ids),
                    "threshold_frozen_from_validation": 0.5,
                    "checkpoint_sha256": row["checkpoint_sha256"],
                    "dataset_content_sha256": dataset_sha,
                    "aggregate": aggregate,
                }
            verify_test_result(
                result,
                model_rows[arch],
                expected_ids,
                row["checkpoint_sha256"],
                dataset_sha,
            )
            protected.update({str(p): sha256(p) for p in (*paths, path)})
        analyses[split] = (
            analyze_rows(model_rows) if len(model_rows) == 3 else {"available": False}
        )
    selected = select_examples(analyses["test"]["per_tile"], examples)
    out.mkdir(parents=True)
    selection = {
        "selected_utc": datetime.now(timezone.utc).isoformat(),
        "split": "test",
        "policy": "descending U-Net++ minus U-Net FP count; sample-ID ascending tie break",
        "posthoc": True,
        "representative": False,
        "training_data": False,
        "threshold": 0.5,
        "selected": selected,
    }
    write_json(out / "selection.json", selection)  # Freeze before any model replay.
    with (out / "per_tile_error_comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(analyses["test"]["per_tile"][0])
        )
        writer.writeheader()
        writer.writerows(analyses["test"]["per_tile"])
    replay = replay_examples(data, results, out, selected, summary, device)
    write_json(out / "replay.json", replay)
    if (
        any(sha256(Path(path)) != digest for path, digest in protected.items())
        or dataset_fingerprint(data) != dataset_sha
    ):
        raise ValueError("Original evidence, weights or data changed during analysis")
    for result in analyses.values():
        result.pop("per_tile", None)
    source_files = [
        Path(__file__),
        ROOT / "scripts/analyze_architecture_comparison.py",
        ROOT / "scripts/evaluate_architecture_test.py",
        ROOT / "roof_pipeline/models.py",
        ROOT / "roof_pipeline/dataset.py",
    ]
    write_json(
        out / "analysis.json",
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "splits": analyses,
            "source_mask_coverage": coverage,
            "original_artifacts_unchanged": True,
            "replay_counts_match": replay["all_counts_match"],
            "posthoc_selection": selection["policy"],
            "test_packet_checksums_sha256": sha256(test_packet / "checksums.json"),
            "protected_artifact_sha256": {
                str(Path(p).relative_to(ROOT))
                if Path(p).is_relative_to(ROOT)
                else p: digest
                for p, digest in protected.items()
            },
            "source_sha256": {
                str(p.relative_to(ROOT)): sha256(p) for p in source_files
            },
            "environment": {
                "python": platform.python_version(),
                **{
                    name: importlib.metadata.version(name)
                    for name in (
                        "torch",
                        "numpy",
                        "Pillow",
                        "matplotlib",
                        "segmentation-models-pytorch",
                    )
                },
            },
            "limitations": [
                "Descriptive post-hoc analysis of the previously inspected test split.",
                "Selected examples explain large observed errors; they are not representative prevalence estimates.",
                "Agricultural appearance is visual interpretation, not an annotated land-cover class.",
                "No training, threshold change, label correction or model selection occurred.",
            ],
        },
    )
    packet_checksums(out)
    if not replay["all_counts_match"]:
        raise ValueError(
            "Replay counts differ from official CSV; see replay.json, official counts unchanged"
        )
    return analyses


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-packet", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--examples", type=int, default=2)
    args = parser.parse_args()
    generate(
        args.test_packet,
        args.results,
        args.data,
        args.out,
        device=args.device,
        examples=args.examples,
    )


if __name__ == "__main__":
    main()
