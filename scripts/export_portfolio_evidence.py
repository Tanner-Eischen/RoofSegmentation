"""Export and verify compact original-run evidence without loading a model.

Run from the repository root. Uses only the Python standard library, never
modifies the training run or dataset, and refuses to overwrite an export.
"""

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

RELEASE = "whu-baseline-2026-10-28"
EXPECTED_CHECKPOINT = "2c5757995dafaa206abac4e22e384cdef78b4d93ab9026975050307ac2d9c7c5"
SOURCE = "https://github.com/Tanner-Eischen/RoofSegmentation"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sanitize(value):
    if isinstance(value, dict):
        return {key: sanitize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize(item) for item in value]
    if isinstance(value, str):
        if re.match(r"^[A-Za-z]:[\\/]", value) or value.startswith(
            ("/home/", "/Users/")
        ):
            return "<local path omitted>"
        return value.replace("\\", "/")
    return value


def write_json(path, value):
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )


def metric_check(path, metrics):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    require(len(rows) == metrics["samples"] == 2416, "Expected all 2,416 test rows")
    require(
        len({row["sample_id"] for row in rows}) == len(rows), "Duplicate sample IDs"
    )
    totals = {
        key: sum(int(row[key]) for row in rows) for key in ("tp", "fp", "fn", "tn")
    }
    for row in rows:
        require(all(int(row[key]) >= 0 for key in totals), "Negative confusion count")
        require(
            sum(int(row[key]) for key in totals) == 512 * 512, "Wrong tile pixel count"
        )
    tp, fp, fn, tn = (totals[key] for key in ("tp", "fp", "fn", "tn"))
    computed = {
        "iou": tp / (tp + fp + fn),
        "dice": 2 * tp / (2 * tp + fp + fn),
        "precision": tp / (tp + fp),
        "recall": tp / (tp + fn),
        "f1": 2 * tp / (2 * tp + fp + fn),
        "pixel_accuracy": (tp + tn) / (tp + fp + fn + tn),
    }
    for key, value in {**totals, **computed}.items():
        require(
            math.isclose(value, metrics["aggregate"][key], rel_tol=0, abs_tol=1e-12),
            f"Aggregate mismatch: {key}",
        )
    return {"samples": len(rows), "confusion_counts": totals, "metrics": computed}


def dataset_fingerprint(root):
    # Same byte-level contract as roof_pipeline.common.dataset_fingerprint.
    digest = hashlib.sha256()
    digest.update((root / "manifest.csv").read_bytes())
    digest.update((root / "dataset_meta.json").read_bytes())
    with (root / "manifest.csv").open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    for row in sorted(rows, key=lambda item: item["sample_id"]):
        for field in ("image", "mask"):
            relative = row[field]
            digest.update(
                json.dumps(
                    [row["sample_id"], field, relative], ensure_ascii=True
                ).encode()
            )
            digest.update(bytes.fromhex(sha256(root / relative)))
    return digest.hexdigest(), dict(Counter(row["split"] for row in rows))


def verify_packet(output):
    entries = {}
    for line in (output / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        expected, name = line.split("  ", 1)
        require(name not in entries, "Duplicate checksum entry")
        path = (output / name).resolve()
        require(path.is_relative_to(output.resolve()), "Unsafe checksum path")
        require(sha256(path) == expected, f"Checksum mismatch: {name}")
        entries[name] = expected
    actual = {
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS.txt"
    }
    require(actual == set(entries), "Checksum inventory differs from packet files")
    result = metric_check(
        output / "test_per_image.csv", read_json(output / "test_metrics.json")
    )
    for path in output.rglob("*"):
        if path.suffix in {".json", ".csv", ".md", ".txt"}:
            require(
                not re.search(
                    r"(?:^|[\s\"'])[A-Za-z]:[\\/]|file:///|/Users/|/home/",
                    path.read_text(encoding="utf-8"),
                ),
                f"Local path found: {path.name}",
            )
    return {"verified_files": len(entries), **result}


def export(args):
    run, data, output = args.run, args.data, args.out
    require(
        not output.exists(),
        "Output exists; verify it or choose a fresh output directory",
    )
    metrics = read_json(run / "test/test_metrics.json")
    summary = read_json(run / "training_summary.json")
    history = read_json(run / "history.json")
    source = read_json(run / "source_manifest.json")
    checkpoint_hash = sha256(run / "best.pt")
    require(
        checkpoint_hash == metrics["checkpoint_sha256"] == EXPECTED_CHECKPOINT,
        "Original baseline checkpoint hash mismatch",
    )
    require(
        sha256(data / "manifest.csv") == summary["config"]["dataset_manifest_sha256"],
        "Dataset manifest hash mismatch",
    )
    require(
        summary["config"]["dataset_content_sha256"]
        == metrics["dataset_content_sha256"],
        "Training/evaluation dataset identity mismatch",
    )
    require(
        metrics["threshold_frozen_from_validation"]
        == summary["val_tuned_threshold"]
        == 0.5,
        "Calibration threshold mismatch",
    )
    require(
        len(history) == 40 and [row["epoch"] for row in history] == list(range(1, 41)),
        "Incomplete epoch history",
    )
    require(summary["best_epoch"] == 32, "Unexpected baseline selected epoch")
    checked = metric_check(run / "test/test_per_image.csv", metrics)
    with zipfile.ZipFile(run / "source_snapshot.zip") as archive:
        for name, expected in source["sha256"].items():
            require(
                hashlib.sha256(archive.read(name.replace("\\", "/"))).hexdigest()
                == expected,
                f"Source snapshot mismatch: {name}",
            )
    data_status = "recorded identity; full image/mask bytes not rehashed during export"
    if args.verify_data_content:
        actual, counts = dataset_fingerprint(data)
        require(
            actual == metrics["dataset_content_sha256"], "Dataset content hash mismatch"
        )
        require(
            counts == {"train": 4736, "val": 1036, "test": 2416}, "Split counts changed"
        )
        data_status = "full original image/mask content hash recomputed and matched"
    verified_at = datetime.now(timezone.utc).isoformat()
    output.mkdir(parents=True)
    copied = {
        "test/test_metrics.json": "test_metrics.json",
        "training_summary.json": "training_summary.json",
        "history.json": "history.json",
        "experiment.json": "experiment.json",
        "source_manifest.json": "source_manifest.json",
        "environment_versions.json": "environment_versions.json",
        "figures/example_selection.json": "example_selection.json",
    }
    origins = {}
    for name, target in copied.items():
        write_json(output / target, sanitize(read_json(run / name)))
        origins[target] = {
            "original_run_relative_path": name,
            "original_sha256": sha256(run / name),
        }
    for name, target in {
        "test/test_per_image.csv": "test_per_image.csv",
        "figures/learning_curve.png": "learning_curve.png",
    }.items():
        shutil.copyfile(run / name, output / target)
        origins[target] = {
            "original_run_relative_path": name,
            "original_sha256": sha256(run / name),
        }
    write_json(
        output / "verification.json",
        {
            "verified_at_utc": verified_at,
            "checks": {
                "per_image_aggregate": checked,
                "checkpoint_sha256": checkpoint_hash,
                "dataset_manifest_sha256": sha256(data / "manifest.csv"),
                "dataset_content_status": data_status,
                "source_snapshot_members_verified": len(source["sha256"]),
                "source_snapshot_sha256": sha256(run / "source_snapshot.zip"),
                "epoch_history_count": len(history),
            },
            "original_artifacts": origins,
            "not_run": [
                "new model inference",
                "new training",
                "browser verification",
                "container verification",
                "remote publication",
            ],
        },
    )
    limits = [
        "WHU binary building extent; no roof material, damage, instance or Minnesota validation.",
        "Original test used frozen validation threshold; later test reuse makes follow-up comparisons development work.",
        "Training source was dirty on its recorded base commit; per-file hashes identify the exact snapshot.",
        "Metrics were recomputed from original rows; inference and training were not rerun for this packet.",
        "Imagery redistribution rights are unconfirmed; no imagery, masks or qualitative image panels are included.",
        "Legacy address demo is not verified against this checkpoint.",
    ]
    write_json(
        output / "portfolio-evidence.json",
        {
            "schema_version": 1,
            "project_id": "building-segmentation",
            "release_id": RELEASE,
            "source_revision": source["base_commit"],
            "source_worktree_dirty": source["worktree_dirty"],
            "tested_revision": None,
            "tested_source_identity": "source_manifest.json",
            "evidence_source_revision": "4cf6fa12c635610d1b2832e0e0b5aa96329b4561",
            "contribution": "Paired-data training, content-bound evaluation, validation calibration and WHU baseline analysis.",
            "status": "local_evidence_packet_verified_not_published",
            "source_url": SOURCE,
            "evidence_urls": {
                "report": f"{SOURCE}/blob/4cf6fa12c635610d1b2832e0e0b5aa96329b4561/docs/WHU_BASELINE_2026-10-28.md"
            },
            "packet_path": f"reports/{RELEASE}/",
            "published_packet_url": None,
            "checks": {
                "runtime": "original GPU run recorded; no new inference",
                "metrics": "independently_artifact_checked",
                "browser": "not_run",
                "model_sha256": checkpoint_hash,
                "dataset_content": data_status,
            },
            "headline_results": [
                {
                    "metric": "global_pixel_iou",
                    "value": checked["metrics"]["iou"],
                    "unit": "fraction",
                    "denominator": "TP + FP + FN = 74,230,723 pixels",
                    "samples": 2416,
                    "scope": "Official WHU test partition, 512 x 512 tiles",
                    "selection_protocol": "Validation-selected epoch 32, threshold 0.50",
                    "evidence": ["test_metrics.json", "test_per_image.csv"],
                }
            ],
            "limitations": limits,
            "assets": [
                {
                    "path": "learning_curve.png",
                    "sha256": sha256(output / "learning_curve.png"),
                    "attribution": "Tanner Eischen; original-run training/validation metric plot; no source imagery",
                }
            ],
            "access": {
                "source": "public",
                "packet": "local_pending_publication",
                "imagery": "excluded_pending_rights_review",
                "weights": "local_not_copied",
            },
            "last_verified_at": verified_at,
        },
    )
    (output / "README.md").write_text(
        "# Original WHU baseline evidence\n\n"
        "Local RS-01 packet; not yet committed, published or deployed.\n\n"
        "All 2,416 original per-image rows reproduce global pixel IoU 0.8921192509468081 "
        "and Dice 0.942984170263471. The validation-selected epoch was 32 of 40, "
        "with threshold 0.50. See `verification.json` for checks actually executed.\n\n"
        "`source_manifest.json` identifies dirty training source based on commit "
        f"`{source['base_commit']}`. This is not a claim that pristine public commit "
        "4cf6fa1 was the training source. Every recorded source member was checked "
        "against the retained original source snapshot; that archive is not redistributed here.\n\n"
        "`example_selection.json` preserves original sample IDs and selection policy. "
        "Qualitative panels, imagery and masks are omitted pending redistribution-rights review. "
        "The original sample policy did not register a separate empty-background group. "
        "Retrieve the official data and follow the baseline report to reproduce rendering; "
        "join sample IDs to `test_per_image.csv` and the prepared dataset manifest.\n\n"
        "![Original training and validation learning curve](learning_curve.png)\n\n"
        "Reproduction and original result: [baseline report](../../docs/WHU_BASELINE_2026-10-28.md). "
        "No retraining is required to inspect these metrics. Original run and data remain unchanged.\n\n"
        "Verify with `python scripts/export_portfolio_evidence.py --verify reports/"
        f"{RELEASE}` from the repository root. `SHA256SUMS.txt` covers every other packet file.\n\n"
        + "\n".join(f"- {limit}" for limit in limits)
        + "\n",
        encoding="utf-8",
    )
    files = sorted(path for path in output.rglob("*") if path.is_file())
    (output / "SHA256SUMS.txt").write_text(
        "".join(
            f"{sha256(path)}  {path.relative_to(output).as_posix()}\n" for path in files
        ),
        encoding="utf-8",
    )
    return verify_packet(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run", type=Path, default=Path("runs/whu_r34_official_2026-10-28")
    )
    parser.add_argument("--data", type=Path, default=Path("data/whu_official_prepared"))
    parser.add_argument("--out", type=Path, default=Path("reports") / RELEASE)
    parser.add_argument(
        "--verify",
        type=Path,
        help="Verify an existing packet without original artifacts",
    )
    parser.add_argument(
        "--verify-data-content",
        action="store_true",
        help="Rehash every original paired file",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            verify_packet(args.verify) if args.verify else export(args), indent=2
        )
    )


if __name__ == "__main__":
    main()
