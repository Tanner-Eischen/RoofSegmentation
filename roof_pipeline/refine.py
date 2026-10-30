"""Run one preregistered hard-example experiment and its matched control.

The original test errors informed this experiment. Its final test result is a
development benchmark; a fresh independent holdout is needed for confirmation.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import traceback
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import torch

from .common import dataset_fingerprint, read_manifest

TARGET_IOU = 0.95
MIN_IMPROVEMENT = 0.0001
SELECTION_METRIC = "validation_calibrated_global_iou"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _number(value, *, probability: bool = False) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
        and value >= 0
        and (not probability or value <= 1)
    )


def choose_candidate(candidates: list[dict]) -> dict:
    """Select using calibrated validation IoU alone, with conservative ties."""
    priorities = {"baseline": 2, "control": 1, "targeted": 0}
    names = [candidate.get("name") for candidate in candidates]
    if (
        len(set(names)) != len(names)
        or "baseline" not in names
        or any(name not in priorities for name in names)
        or any(
            not _number(c.get("validation_iou"), probability=True) for c in candidates
        )
    ):
        raise ValueError(
            "Candidates require unique known names and finite validation IoUs"
        )
    baseline = next(c for c in candidates if c["name"] == "baseline")
    eligible = [baseline] + [
        c
        for c in candidates
        if c["name"] != "baseline"
        and c["validation_iou"] > baseline["validation_iou"] + MIN_IMPROVEMENT
    ]
    return max(eligible, key=lambda c: (c["validation_iou"], priorities[c["name"]]))


def valid_cached_test(
    result: dict,
    *,
    checkpoint_sha: str,
    dataset_sha: str,
    threshold: float,
    test_samples: int,
) -> bool:
    """Accept test metrics only when frozen inputs, partitions and counts agree."""
    if not isinstance(result, dict):
        return False
    frozen_threshold = result.get("threshold_frozen_from_validation")
    if (
        result.get("split") != "test"
        or type(result.get("samples")) is not int
        or result["samples"] != test_samples
        or result.get("checkpoint_sha256") != checkpoint_sha
        or result.get("dataset_content_sha256") != dataset_sha
        or not _number(frozen_threshold, probability=True)
        or not math.isclose(frozen_threshold, threshold, rel_tol=0, abs_tol=1e-12)
    ):
        return False
    aggregate = result.get("aggregate")
    if not isinstance(aggregate, dict):
        return False
    if any(not _number(aggregate.get(k)) for k in ("tp", "fp", "fn", "tn")):
        return False
    if sum(aggregate[k] for k in ("tp", "fp", "fn", "tn")) <= 0:
        return False
    tp, fp, fn = (aggregate[k] for k in ("tp", "fp", "fn"))
    expected = {
        "iou": tp / (tp + fp + fn) if tp + fp + fn else 0.0,
        "dice": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
    }
    return all(
        _number(aggregate.get(key), probability=True)
        and math.isclose(aggregate[key], value, rel_tol=0, abs_tol=1e-10)
        for key, value in expected.items()
    )


def _fresh_output(out: Path) -> None:
    if out.exists() and not out.is_dir():
        raise FileExistsError("Experiment output must be a fresh directory")
    phase_logs = {"mining.log", "targeted.log", "control.log", "evaluation.log"}
    if out.exists():
        for path in out.iterdir():
            launcher_file = path.name in {"process.json", "verification.json"} or (
                path.suffix == ".log" and path.name not in phase_logs
            )
            if not path.is_file() or not launcher_file:
                raise FileExistsError(
                    f"Experiment output already contains artifacts: {path}"
                )
    out.mkdir(parents=True, exist_ok=True)


def _candidate(
    checkpoint: Path, dataset_sha: str, reference: dict | None = None
) -> dict:
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config = state["config"]
    summary = json.loads(
        (checkpoint.parent / "training_summary.json").read_text(encoding="utf-8")
    )
    required = ("arch", "image_size", "normalization", "dataset_content_sha256")
    if (
        config.get("dataset_content_sha256") != dataset_sha
        or any(summary["config"].get(k) != config.get(k) for k in required)
        or summary["config"].get("epoch") != config.get("epoch")
        or summary.get("best_epoch") != config.get("epoch")
        or not _number(state.get("val_iou_at_0p5"), probability=True)
        or state.get("val_iou_at_0p5") != summary.get("val_iou_at_0p5")
        or not _number(state.get("threshold"), probability=True)
        or not 0 < state["threshold"] < 1
        or state["threshold"] != summary.get("val_tuned_threshold")
        or not _number(
            summary.get("val_metrics_tuned", {}).get("iou"), probability=True
        )
        or config.get("normalization") != "imagenet_rgb_v1"
        or config.get("arch")
        not in {
            "tiny",
            "unet-resnet34",
            "unetplusplus-efficientnet-b0",
            "deeplabv3plus-resnet34",
        }
        or type(config.get("epoch")) is not int
        or config["epoch"] < 0
        or type(config.get("image_size")) is not int
        or config["image_size"] < 32
        or config["image_size"] % 32
    ):
        raise ValueError(
            f"Checkpoint and validation summary disagree or use incompatible data: {checkpoint}"
        )
    if reference is not None and (
        config["arch"] != reference["arch"]
        or config["image_size"] != reference["image_size"]
        or config.get("initial_checkpoint_sha256") != reference["checkpoint_sha256"]
    ):
        raise ValueError(
            "Fine-tuned candidate does not match the registered warm start"
        )
    return {
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": _sha256(checkpoint),
        "arch": config["arch"],
        "image_size": config["image_size"],
        "epoch": config["epoch"],
        "val_iou_at_0p5": state["val_iou_at_0p5"],
        "validation_iou": summary["val_metrics_tuned"]["iou"],
        "threshold": state["threshold"],
    }


def _read_optional_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _snapshot(repo: Path, out: Path) -> None:
    paths = set()
    for pattern in (
        "roof_pipeline/*.py",
        "scripts/*.py",
        "tests/*.py",
        "requirements*.txt",
        "integration_optional/**/*.py",
        "docs/*.md",
    ):
        paths.update(repo.glob(pattern))
    paths.update(
        repo / name for name in ("ruff.toml", "README.md", ".github/workflows/ci.yml")
    )
    paths = sorted(path for path in paths if path.is_file())
    _write_json(
        out / "source_manifest.json",
        {
            "sha256": {
                path.relative_to(repo).as_posix(): _sha256(path) for path in paths
            }
        },
    )
    with zipfile.ZipFile(
        out / "source_snapshot.zip", "w", zipfile.ZIP_DEFLATED
    ) as archive:
        for path in paths:
            archive.write(path, path.relative_to(repo).as_posix())
    versions = {"python": platform.python_version(), "torch": str(torch.__version__)}
    for package in (
        "numpy",
        "Pillow",
        "torchvision",
        "segmentation-models-pytorch",
        "rasterio",
        "shapely",
    ):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    _write_json(out / "environment_versions.json", versions)


def _verify_per_image(path: Path, result: dict) -> bool:
    if not path.is_file():
        return False
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != result["samples"] or any(
        sum(float(row[key]) for row in rows) != result["aggregate"][key]
        for key in ("tp", "fp", "fn", "tn")
    ):
        raise ValueError("Per-image test counts do not match aggregate metrics")
    return True


def run_refinement(args) -> dict:
    data, baseline_path, out = (
        Path(p).resolve() for p in (args.data, args.baseline, args.out)
    )
    _fresh_output(out)
    status = {"started_at": _now(), "target_iou": TARGET_IOU, "test_evaluated": False}

    def update(phase, **values):
        status.update(phase=phase, updated_at=_now(), **values)
        _write_json(out / "workflow_status.json", status)
        print(json.dumps(status), flush=True)

    repo = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment.update(
        OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", CUBLAS_WORKSPACE_CONFIG=":4096:8"
    )

    def execute(phase, arguments):
        command = [sys.executable, "-u", "-m", *arguments]
        update(phase, command=command)
        with (out / f"{phase}.log").open("w", encoding="utf-8") as log:
            subprocess.run(
                command,
                cwd=repo,
                env=environment,
                check=True,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        update(f"{phase}_complete", command=None)

    try:
        if os.name == "nt":
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
        update("validating")
        if (
            args.epochs < 1
            or args.batch_size < 1
            or args.workers < 0
            or not math.isfinite(args.lr)
            or args.lr <= 0
        ):
            raise ValueError(
                "Epochs/batch size/lr must be positive; workers must be nonnegative"
            )
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        dataset_sha = dataset_fingerprint(data)
        manifest = read_manifest(data)
        test_samples = sum(row["split"] == "test" for row in manifest)
        baseline = dict(_candidate(baseline_path, dataset_sha), name="baseline")
        state = torch.load(baseline_path, map_location="cpu", weights_only=True)
        if state["config"].get("dataset_manifest_sha256") != _sha256(
            data / "manifest.csv"
        ):
            raise ValueError("Baseline manifest hash differs from the frozen dataset")
        baseline_cache_path = baseline_path.parent / "test/test_metrics.json"
        baseline_test = _read_optional_json(baseline_cache_path)
        if not valid_cached_test(
            baseline_test,
            checkpoint_sha=baseline["checkpoint_sha256"],
            dataset_sha=dataset_sha,
            threshold=baseline["threshold"],
            test_samples=test_samples,
        ):
            baseline_test = None
        registration = {
            "registered_at": _now(),
            "baseline": baseline,
            "dataset_content_sha256": dataset_sha,
            "arguments": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            "arms_in_execution_order": ["targeted", "control"],
            "targeted": "Train-only bounded hard-tile sampling plus robust photometry",
            "control": "Uniform sampling plus standard photometry",
            "shared": {
                "warm_start": "full baseline checkpoint",
                "optimizer": "restarted AdamW",
                "weight_decay": 0.0001,
                "positive_weight": 1.0,
                "loss": "BCE+Dice",
                "epochs_per_arm": args.epochs,
                "image_size": baseline["image_size"],
            },
            "selection_metric": SELECTION_METRIC,
            "selection": "Highest validation-calibrated global IoU, improvement >0.0001 over baseline; exact ties favor baseline/control/targeted",
            "test_policy": "Only validation-selected candidate evaluated; no test threshold sweep or repeated search for 0.95",
            "interpretation": "DEVELOPMENT benchmark: baseline test errors informed this experiment. A fresh independent holdout is required for confirmation.",
            "target_iou": TARGET_IOU,
            "strict_target_comparison": ">",
        }
        _write_json(out / "experiment.json", registration)
        _snapshot(repo, out)
        update("validated", command=None)
        execute(
            "mining",
            [
                "roof_pipeline.hard_examples",
                "--data",
                str(data),
                "--checkpoint",
                str(baseline_path),
                "--out",
                str(out / "hard_examples.json"),
                "--device",
                args.device,
                "--batch-size",
                str(args.batch_size),
                "--workers",
                str(args.workers),
            ],
        )
        for arm, photometric in (("targeted", "robust"), ("control", "standard")):
            command = [
                "roof_pipeline.train",
                "--data",
                str(data),
                "--run",
                str(out / arm),
                "--arch",
                baseline["arch"],
                "--size",
                str(baseline["image_size"]),
                "--init-checkpoint",
                str(baseline_path),
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
                photometric,
            ]
            if arm == "targeted":
                command += ["--sampling-report", str(out / "hard_examples.json")]
            execute(arm, command)
        if (
            dataset_fingerprint(data) != dataset_sha
            or _sha256(baseline_path) != baseline["checkpoint_sha256"]
        ):
            raise ValueError("Frozen data or baseline changed during refinement")
        update("selecting")
        candidates = [baseline] + [
            dict(_candidate(out / arm / "best.pt", dataset_sha, baseline), name=arm)
            for arm in ("targeted", "control")
        ]
        selected = choose_candidate(candidates)
        _write_json(
            out / "selection.json",
            {
                "selected": selected,
                "candidates": candidates,
                "selected_at": _now(),
                "used_test_metrics": False,
                "selection_metric": SELECTION_METRIC,
            },
        )
        update("selected", selected_candidate=selected["name"])
        if selected["name"] == "baseline" and baseline_test is not None:
            update("evaluation_cached")
            (out / "test").mkdir()
            _write_json(out / "test/test_metrics.json", baseline_test)
            csv_source = baseline_cache_path.with_name("test_per_image.csv")
            if csv_source.is_file():
                shutil.copyfile(csv_source, out / "test/test_per_image.csv")
            update("evaluation_cached_complete")
        else:
            execute(
                "evaluation",
                [
                    "roof_pipeline.evaluate",
                    "--data",
                    str(data),
                    "--checkpoint",
                    selected["checkpoint"],
                    "--out",
                    str(out / "test"),
                    "--split",
                    "test",
                    "--batch-size",
                    str(args.batch_size),
                    "--device",
                    args.device,
                ],
            )
        result = json.loads(
            (out / "test/test_metrics.json").read_text(encoding="utf-8")
        )
        if not valid_cached_test(
            result,
            checkpoint_sha=selected["checkpoint_sha256"],
            dataset_sha=dataset_sha,
            threshold=selected["threshold"],
            test_samples=test_samples,
        ):
            raise ValueError(
                "Selected test result does not match the frozen checkpoint/data/threshold"
            )
        if selected["name"] == "baseline":
            baseline_test = result
        per_image_verified = _verify_per_image(out / "test/test_per_image.csv", result)
        baseline_iou = (
            baseline_test["aggregate"]["iou"] if baseline_test is not None else None
        )
        test_iou = result["aggregate"]["iou"]
        comparison = {
            "candidates": candidates,
            "selected": selected,
            "selection_used_test_metrics": False,
            "selection_metric": SELECTION_METRIC,
            "test": result,
            "baseline_test_iou": baseline_iou,
            "test_iou_change_vs_baseline": test_iou - baseline_iou
            if baseline_iou is not None
            else None,
            "target_iou": TARGET_IOU,
            "target_reached": test_iou > TARGET_IOU,
            "per_image_counts_verified": per_image_verified,
            "interpretation": registration["interpretation"],
        }
        _write_json(out / "comparison.json", comparison)
        table = "\n".join(
            f"| {c['name']} | {c['val_iou_at_0p5']:.6f} | {c['validation_iou']:.6f} | {c['threshold']:.2f} | {c['epoch']} |"
            for c in candidates
        )
        baseline_text = (
            f"{baseline_iou:.6f}"
            if baseline_iou is not None
            else "unavailable: baseline cache did not match frozen inputs"
        )
        change_text = (
            f"{comparison['test_iou_change_vs_baseline']:+.6f}"
            if baseline_iou is not None
            else "unavailable"
        )
        report = f"""# Targeted roof-segmentation refinement

Status: completed. This is a **DEVELOPMENT benchmark**, because the previous test
errors informed this experiment. A fresh independent holdout is needed for a
confirmatory generalization claim.

Both arms started from the same frozen baseline weights, used {args.epochs} epochs,
input size {baseline["image_size"]}px, batch size {args.batch_size},
seed {args.seed}, learning rate {args.lr}, and restarted AdamW with BCE+Dice.
The targeted arm used train-only hard-tile sampling and robust photometry; the
matched control used uniform sampling and standard photometry. No test examples
were added to training and no test threshold sweep was performed.

Selection metric: `{SELECTION_METRIC}`. Each arm still selects its checkpoint by
validation IoU at 0.50, then calibrates that checkpoint using validation only.

| Candidate | Validation IoU at 0.50 | Calibrated validation IoU | Threshold | Selected arm epoch |
| --- | ---: | ---: | ---: | ---: |
{table}

Validation selected **{selected["name"]}**. Its calibrated threshold was
{selected["threshold"]:.6f}. Only this selected candidate received final test
evaluation (matching cached baseline evaluation may be reused).

- Baseline test IoU: {baseline_text}
- Selected test IoU: **{test_iou:.6f}**
- Change versus baseline: {change_text}
- Target: strictly above {TARGET_IOU:.2f}
- Target reached: **{comparison["target_reached"]}**
- Test images: {test_samples}
- Per-image TP/FP/FN/TN sums verified: {per_image_verified}
- Selected checkpoint: `{selected["checkpoint"]}`

Data SHA256: `{dataset_sha}`.
Checkpoint SHA256: `{selected["checkpoint_sha256"]}`.

See `experiment.json`, `source_snapshot.zip`, `source_manifest.json`,
`environment_versions.json`, `hard_examples.json`, arm histories and summaries,
`selection.json`, `comparison.json`, and `test/` for reproducibility evidence.
This is one bounded experiment; it does not continue searching until 0.95.
"""
        (out / "report.md").write_text(report, encoding="utf-8")
        update(
            "complete",
            command=None,
            completed_at=_now(),
            test_evaluated=True,
            selected_test_iou=test_iou,
            target_reached=comparison["target_reached"],
        )
        return comparison
    except BaseException as exc:
        failed_phase = status.get("phase", "initializing")
        update(
            "failed",
            failed_phase=failed_phase,
            error=str(exc),
            traceback=traceback.format_exc(),
        )
        raise
    finally:
        if os.name == "nt":
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=0.0001)
    run_refinement(parser.parse_args())


if __name__ == "__main__":
    main()
