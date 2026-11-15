"""Prepare a controlled WHU architecture comparison; fitting is explicit opt-in.

Preparation freezes the dataset, ImageNet encoder identities and source snapshot.
Full fitting requires --execute and a runtime estimate bound to that exact plan.
Only training and validation inference are performed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from .common import dataset_fingerprint, read_manifest

ARCHITECTURES = ("unet-resnet34", "unetplusplus-resnet34", "segformer-b0")
OFFICIAL_COUNTS = {"train": 4736, "val": 1036, "test": 2416}
THREAD_ENVIRONMENT = {
    "OMP_NUM_THREADS": "4",
    "MKL_NUM_THREADS": "4",
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value):
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def dataset_identity(data: Path) -> dict:
    rows = read_manifest(data)
    counts = dict(Counter(row["split"] for row in rows))
    metadata = read_json(data / "dataset_meta.json")
    source = metadata.get("source_provenance", {})
    if (
        counts != OFFICIAL_COUNTS
        or metadata.get("limited_subset") is not False
        or source.get("source") != "official"
        or source.get("spatial_independence") != "publisher_geographic_partitions"
    ):
        raise ValueError(
            "Comparison requires complete official geographic WHU partitions"
        )
    return {
        "path": str(data.resolve()),
        "manifest_sha256": sha256(data / "manifest.csv"),
        "content_sha256": dataset_fingerprint(data),
        "split_counts": counts,
    }


def runtime_environment() -> dict:
    import torch

    return {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "numpy": importlib.metadata.version("numpy"),
        "Pillow": importlib.metadata.version("Pillow"),
        "segmentation_models_pytorch": importlib.metadata.version(
            "segmentation-models-pytorch"
        ),
        "cuda_runtime": torch.version.cuda,
        "thread_environment": dict(THREAD_ENVIRONMENT),
    }


def encoder_initializations() -> dict:
    """Inspect fresh pretrained encoders on CPU without fitting any model."""
    import torch

    from .models import create_model, encoder_fingerprint
    from .train import seed_everything

    identities = {}
    for arch in ARCHITECTURES:
        seed_everything(42)
        model = create_model(arch, pretrained=True)
        identities[arch] = {
            "encoder_name": "mit_b0" if arch == "segformer-b0" else "resnet34",
            "encoder_weights": "imagenet",
            "encoder_sha256": encoder_fingerprint(model),
            "parameter_count": sum(
                parameter.numel() for parameter in model.parameters()
            ),
        }
        del model
    if (
        identities[ARCHITECTURES[0]]["encoder_sha256"]
        != identities[ARCHITECTURES[1]]["encoder_sha256"]
    ):
        raise ValueError(
            "U-Net and U-Net++ do not share identical ResNet-34 initialization"
        )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return identities


def settings(epochs=40, batch_size=4, workers=4, device="cuda") -> dict:
    if epochs < 1 or batch_size < 1 or workers < 0 or device not in {"cpu", "cuda"}:
        raise ValueError("Invalid fixed training budget or device")
    return {
        "epochs": epochs,
        "size": 512,
        "batch_size": batch_size,
        "workers": workers,
        "device": device,
        "lr": 0.0003,
        "weight_decay": 0.0001,
        "pos_weight": 1.0,
        "loss": "BCE_with_logits_plus_soft_Dice",
        "optimizer": "AdamW",
        "scheduler": "CosineAnnealingLR_eta_min_0.02_lr",
        "photometric": "standard",
        "sampling": "uniform_shuffle_without_replacement",
        "normalization": "imagenet_rgb_v1",
        "pretrained": True,
        "full_checkpoint_warm_start": False,
        "boundary_weight": 0.0,
        "min_delta": 0.0,
        "patience": epochs + 1,
        "primary_threshold": 0.5,
        "primary_selection": "highest_validation_global_pixel_IoU_at_0.5_earliest_tie",
        "secondary_calibration": "validation_only_0.15_to_0.85_step_0.05",
    }


def source_files(repo: Path) -> list[Path]:
    return sorted((repo / "roof_pipeline").glob("*.py"))


def prepare(args) -> dict:
    out, data = args.out.resolve(), args.data.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(
            "Choose a fresh comparison directory; existing artifacts are preserved"
        )
    fixed = settings(
        40 if args.epochs is None else args.epochs,
        4 if args.batch_size is None else args.batch_size,
        4 if args.workers is None else args.workers,
        args.device or "cuda",
    )
    seeds = args.seeds or [42]
    validate_seeds(seeds)
    dataset = dataset_identity(data)
    initializations = encoder_initializations()
    repo = Path(__file__).resolve().parents[1]
    sources = source_files(repo)
    source_hashes = {
        path.relative_to(repo).as_posix(): sha256(path) for path in sources
    }
    plan = {
        "schema_version": 1,
        "study": "controlled_whu_architecture_comparison",
        "prepared_utc": datetime.now(timezone.utc).isoformat(),
        "repository_path": str(repo),
        "dataset": dataset,
        "architectures": list(ARCHITECTURES),
        "seeds": seeds,
        "settings": fixed,
        "initializations": initializations,
        "source_sha256": source_hashes,
        "environment": runtime_environment(),
        "test_inference": False,
        "interpretation": "exploratory_existing_WHU_geography; no independent geographic confirmation",
    }
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        out / "source_snapshot.zip", "w", zipfile.ZIP_DEFLATED
    ) as archive:
        for path in sources:
            archive.write(path, path.relative_to(repo).as_posix())
    plan["source_snapshot_sha256"] = sha256(out / "source_snapshot.zip")
    write_json(out / "plan.json", plan)
    write_status(out, "prepared_awaiting_runtime_estimate")
    return plan


def training_command(plan: dict, arch: str, seed: int, out: Path) -> list[str]:
    fixed = plan["settings"]
    run = out.resolve() / "runs" / f"{arch}_seed{seed}"
    return [
        sys.executable,
        "-u",
        "-m",
        "roof_pipeline.train",
        "--data",
        plan["dataset"]["path"],
        "--run",
        str(run),
        "--arch",
        arch,
        "--pretrained",
        "--epochs",
        str(fixed["epochs"]),
        "--size",
        str(fixed["size"]),
        "--batch-size",
        str(fixed["batch_size"]),
        "--workers",
        str(fixed["workers"]),
        "--device",
        fixed["device"],
        "--seed",
        str(seed),
        "--lr",
        str(fixed["lr"]),
        "--weight-decay",
        str(fixed["weight_decay"]),
        "--pos-weight",
        "1",
        "--patience",
        str(fixed["patience"]),
        "--min-delta",
        "0",
        "--photometric",
        "standard",
        "--boundary-weight",
        "0",
        "--expected-encoder-sha256",
        plan["initializations"][arch]["encoder_sha256"],
    ]


def restore_snapshot(plan: dict, out: Path) -> Path:
    snapshot = out / "source_snapshot.zip"
    if sha256(snapshot) != plan["source_snapshot_sha256"]:
        raise ValueError("Frozen source snapshot changed")
    frozen = out / "frozen_source"
    with zipfile.ZipFile(snapshot) as archive:
        if set(archive.namelist()) != set(plan["source_sha256"]):
            raise ValueError("Snapshot members differ from registered source")
        for name in archive.namelist():
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Unsafe frozen source path")
            target = frozen / relative
            content = archive.read(name)
            if hashlib.sha256(content).hexdigest() != plan["source_sha256"][name]:
                raise ValueError("Snapshot source hash differs")
            if target.exists() and target.read_bytes() != content:
                raise ValueError("Restored source changed")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
    return frozen


def validate_estimate(estimate: dict, plan: dict, plan_sha: str):
    if estimate.get("schema_version") != 1 or estimate.get("plan_sha256") != plan_sha:
        raise ValueError("Runtime estimate is missing or belongs to a different plan")
    if set(estimate.get("architectures", {})) != set(plan["architectures"]):
        raise ValueError(
            "Runtime estimate does not cover every registered architecture"
        )
    for record in [
        estimate["total_seconds"],
        *[r["run_seconds"] for r in estimate["architectures"].values()],
    ]:
        lower, upper = record.get("lower"), record.get("upper")
        if (
            isinstance(lower, bool)
            or isinstance(upper, bool)
            or not isinstance(lower, (int, float))
            or not isinstance(upper, (int, float))
            or not math.isfinite(lower)
            or not math.isfinite(upper)
            or lower <= 0
            or upper < lower
        ):
            raise ValueError(
                "Runtime estimate bounds must be finite, positive and ordered"
            )


def validate_hardware(estimate: dict, device: str):
    if device != "cuda":
        return
    import torch

    if not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable on the execution machine")
    if (
        estimate.get("device") != torch.cuda.get_device_name()
        or estimate.get("device_total_bytes")
        != torch.cuda.get_device_properties(0).total_memory
    ):
        raise ValueError("Runtime estimate belongs to different GPU hardware")


def write_status(out: Path, phase: str, **detail):
    write_json(
        out / "status.json",
        {
            "phase": phase,
            "updated_utc": datetime.now(timezone.utc).isoformat(),
            "test_inference": False,
            **detail,
        },
    )


def execute_command(command: list[str], frozen: Path, log: Path):
    environment = dict(
        os.environ,
        **THREAD_ENVIRONMENT,
        PYTHONPATH=str(frozen),
        PYTHONIOENCODING="utf-8",
    )
    with log.open("w", encoding="utf-8") as stream:
        result = subprocess.run(
            command,
            cwd=frozen,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
    if result.returncode:
        raise RuntimeError(
            f"Comparison command failed ({result.returncode}); inspect {log}"
        )


def load_checkpoint(path: Path) -> dict:
    import torch

    return torch.load(path, map_location="cpu", weights_only=True)


def finite_score(value, label):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0 <= value <= 1
    ):
        raise ValueError(f"Invalid {label}")
    return float(value)


def verify_run(plan: dict, run: Path, arch: str, seed: int) -> dict:
    fixed = plan["settings"]
    summary, experiment, history = [
        read_json(run / name)
        for name in ("training_summary.json", "experiment.json", "history.json")
    ]
    if [row.get("epoch") for row in history] != list(range(1, fixed["epochs"] + 1)):
        raise ValueError("Run did not complete the common epoch budget")
    scores = [
        finite_score(row.get("val_iou_0p5"), "history validation score")
        for row in history
    ]
    best_index = max(range(len(scores)), key=scores.__getitem__)
    state = load_checkpoint(run / "best.pt")
    expected = {
        "arch": arch,
        "seed": seed,
        "image_size": 512,
        "normalization": "imagenet_rgb_v1",
        "dataset_content_sha256": plan["dataset"]["content_sha256"],
        "dataset_manifest_sha256": plan["dataset"]["manifest_sha256"],
        "effective_training_content_sha256": plan["dataset"]["content_sha256"],
        "pretrained_encoder": True,
        "encoder_initialization_sha256": plan["initializations"][arch][
            "encoder_sha256"
        ],
        "sampling_report_sha256": None,
        "uniform_replacement": False,
        "photometric_profile": "standard",
        "boundary_weight": 0.0,
        "boundary_supervision": False,
        "training_review_content_sha256": None,
    }
    for config in (summary["config"], experiment["config"], state["config"]):
        if any(config.get(key) != value for key, value in expected.items()):
            raise ValueError("Run configuration differs from matched comparison")
        if config.get("initial_checkpoint_sha256") is not None:
            raise ValueError("Architecture comparison must not warm-start full weights")
    arguments = experiment["arguments"]
    expected_args = {
        key: fixed[key]
        for key in (
            "epochs",
            "size",
            "batch_size",
            "workers",
            "device",
            "lr",
            "weight_decay",
            "pos_weight",
            "patience",
            "min_delta",
            "photometric",
            "boundary_weight",
        )
    }
    expected_args.update(
        {
            "arch": arch,
            "seed": seed,
            "pretrained": True,
            "no_amp": False,
            "init_checkpoint": None,
            "sampling_report": None,
            "uniform_replacement": False,
            "training_review": None,
            "expected_encoder_sha256": expected["encoder_initialization_sha256"],
            "data": plan["dataset"]["path"],
            "run": str(run.resolve()),
        }
    )
    if any(arguments.get(key) != value for key, value in expected_args.items()):
        raise ValueError("Actual training arguments differ from frozen plan")
    if (
        summary.get("best_epoch") != best_index + 1
        or state["config"].get("epoch") != best_index + 1
        or summary["config"].get("epoch") != best_index + 1
        or finite_score(state["val_iou_at_0p5"], "saved checkpoint IoU")
        != scores[best_index]
        or finite_score(summary["val_iou_at_0p5"], "summary selected IoU")
        != scores[best_index]
    ):
        raise ValueError("Saved checkpoint is not the earliest validation-best epoch")
    threshold = state.get("threshold")
    if (
        not isinstance(threshold, (int, float))
        or isinstance(threshold, bool)
        or not any(
            math.isclose(threshold, 0.15 + index * 0.05, abs_tol=1e-12)
            for index in range(15)
        )
        or threshold != summary.get("val_tuned_threshold")
        or summary.get("test_evaluated") is not False
        or summary.get("final_epoch") != fixed["epochs"]
        or summary.get("train_samples") != plan["dataset"]["split_counts"]["train"]
        or summary.get("val_samples") != plan["dataset"]["split_counts"]["val"]
    ):
        raise ValueError("Calibration, sample counts or test policy differs from plan")
    if any(
        row.get("train_draws") != plan["dataset"]["split_counts"]["train"]
        or row.get("unique_train_samples") != plan["dataset"]["split_counts"]["train"]
        for row in history
    ):
        raise ValueError("Uniform no-replacement training coverage differs")
    seconds = [row.get("seconds") for row in history]
    if any(
        not isinstance(value, (float, int))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or value < 0
        for value in seconds
    ):
        raise ValueError("Invalid epoch timing")
    return {
        "architecture": arch,
        "seed": seed,
        "best_epoch": best_index + 1,
        "primary_validation_iou_0p5": scores[best_index],
        "secondary_validation_threshold": threshold,
        "secondary_calibrated_validation_iou": finite_score(
            summary["val_metrics_tuned"]["iou"], "calibrated score"
        ),
        "epoch_minutes": sum(seconds) / 60,
        "checkpoint_sha256": sha256(run / "best.pt"),
    }


def reassess_validation(plan: dict, run: Path, result: dict):
    import torch
    from torch.utils.data import DataLoader

    from .dataset import RoofDataset
    from .metrics import evaluate_loader
    from .models import load_model

    fixed = plan["settings"]
    dataset = RoofDataset(Path(plan["dataset"]["path"]), "val", size=512, augment=False)
    loader = DataLoader(
        dataset,
        batch_size=fixed["batch_size"],
        shuffle=False,
        num_workers=fixed["workers"],
    )
    model, _ = load_model(run / "best.pt", fixed["device"])
    records = {}
    primary_details = []
    for name, threshold, expected_score in (
        ("primary_0p5", 0.5, result["primary_validation_iou_0p5"]),
        (
            "secondary_calibrated",
            result["secondary_validation_threshold"],
            result["secondary_calibrated_validation_iou"],
        ),
    ):
        if name == "secondary_calibrated" and threshold == 0.5:
            aggregate, details = records["primary_0p5"]["aggregate"], primary_details
        else:
            aggregate, details = evaluate_loader(
                model,
                loader,
                torch.device(fixed["device"]),
                threshold=threshold,
                with_rows=True,
            )
        if abs(aggregate["iou"] - expected_score) > 1e-6 or len(details) != len(
            dataset
        ):
            raise ValueError(
                "Reloaded validation-best checkpoint does not reproduce its registered score"
            )
        for row in details:
            source = dataset.rows[row["index"]]
            row.update(sample_id=source["sample_id"], group=source["group"])
        if name == "primary_0p5":
            primary_details = details
        records[name] = {"threshold": threshold, "aggregate": aggregate}
        with (run / f"validation_{name}_per_image.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=list(details[0]))
            writer.writeheader()
            writer.writerows(details)
    write_json(
        run / "validation_best.json",
        {
            **result,
            "samples": len(dataset),
            "split": "val",
            "test_inference": False,
            **records,
        },
    )


def execute_plan(out: Path):
    out = out.resolve()
    plan = read_json(out / "plan.json")
    estimate = read_json(out / "runtime_estimate.json")
    validate_estimate(estimate, plan, sha256(out / "plan.json"))
    validate_hardware(estimate, plan["settings"]["device"])
    if (out / "runs").exists() and any((out / "runs").iterdir()):
        raise FileExistsError(
            "Existing or partial comparison runs cannot be overwritten or silently resumed"
        )
    if dataset_identity(Path(plan["dataset"]["path"])) != plan["dataset"]:
        raise ValueError("WHU data changed since preparation")
    if runtime_environment() != plan["environment"]:
        raise ValueError("Runtime libraries changed since preparation")
    repo = Path(plan["repository_path"])
    for relative, expected in plan["source_sha256"].items():
        if sha256(repo / relative) != expected:
            raise ValueError("Registered source changed; prepare a new comparison")
    frozen = restore_snapshot(plan, out)
    (out / "runs").mkdir(exist_ok=True)
    results = []
    try:
        for seed in plan["seeds"]:
            for arch in plan["architectures"]:
                run = out / "runs" / f"{arch}_seed{seed}"
                run.mkdir()
                write_status(
                    out,
                    "training",
                    architecture=arch,
                    seed=seed,
                    completed_runs=len(results),
                )
                execute_command(
                    training_command(plan, arch, seed, out),
                    frozen,
                    run / "training.log",
                )
                if dataset_identity(Path(plan["dataset"]["path"])) != plan["dataset"]:
                    raise ValueError("Dataset changed during fitting")
                write_status(
                    out,
                    "validating_best",
                    architecture=arch,
                    seed=seed,
                    completed_runs=len(results),
                )
                execute_command(
                    [
                        sys.executable,
                        "-u",
                        "-m",
                        "roof_pipeline.architecture_comparison",
                        "--out",
                        str(out),
                        "--assess-run",
                        "--architecture",
                        arch,
                        "--seed",
                        str(seed),
                    ],
                    frozen,
                    run / "validation.log",
                )
                result = read_json(run / "validation_best.json")
                if (
                    result.get("architecture") != arch
                    or result.get("seed") != seed
                    or result.get("checkpoint_sha256") != sha256(run / "best.pt")
                    or result.get("split") != "val"
                    or result.get("test_inference") is not False
                    or result.get("samples") != plan["dataset"]["split_counts"]["val"]
                ):
                    raise ValueError("Frozen validation assessment identity differs")
                results.append(result)
        write_json(
            out / "comparison.json",
            {
                "plan_sha256": sha256(out / "plan.json"),
                "test_inference": False,
                "runs": results,
            },
        )
        lines = [
            "# Controlled WHU architecture comparison",
            "",
            "Primary: validation-best checkpoint, global IoU at threshold 0.50.",
            "Secondary: the same checkpoint with validation-only threshold calibration. No test inference.",
            "",
            "| Architecture | Mean primary IoU | Seed SD | Mean secondary calibrated IoU |",
            "| --- | ---: | ---: | ---: |",
        ]
        for arch in plan["architectures"]:
            group = [row for row in results if row["architecture"] == arch]
            scores = [row["primary_validation_iou_0p5"] for row in group]
            sd = statistics.stdev(scores) if len(scores) > 1 else None
            lines.append(
                f"| {arch} | {statistics.mean(scores):.6f} | {sd:.6f} | {statistics.mean(row['secondary_calibrated_validation_iou'] for row in group):.6f} |"
                if sd is not None
                else f"| {arch} | {statistics.mean(scores):.6f} | not estimable (one seed) | {group[0]['secondary_calibrated_validation_iou']:.6f} |"
            )
        (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        write_status(out, "complete", completed_runs=len(results))
    except Exception as error:
        write_status(out, "failed", completed_runs=len(results), error=str(error))
        raise
    return results


def assess_run(out: Path, arch: str, seed: int):
    """Internal frozen-source subprocess: verify selection and re-evaluate validation."""
    out = out.resolve()
    plan = read_json(out / "plan.json")
    if Path(__file__).resolve().parents[1] != out / "frozen_source":
        raise ValueError(
            "Validation assessment must run from the registered source snapshot"
        )
    if arch not in plan["architectures"] or seed not in plan["seeds"]:
        raise ValueError("Assessment is not a registered run")
    for relative, expected in plan["source_sha256"].items():
        if sha256(out / "frozen_source" / relative) != expected:
            raise ValueError("Frozen assessment source changed")
    if dataset_identity(Path(plan["dataset"]["path"])) != plan["dataset"]:
        raise ValueError("Dataset changed before validation assessment")
    run = out / "runs" / f"{arch}_seed{seed}"
    result = verify_run(plan, run, arch, seed)
    reassess_validation(plan, run, result)


def validate_seeds(values):
    if (
        not values
        or len(set(values)) != len(values)
        or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in values)
    ):
        raise ValueError("Seeds must be distinct integers in [0, 2**32)")


def parse_seeds(raw):
    try:
        values = [int(value.strip()) for value in raw.split(",")]
        validate_seeds(values)
        return values
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def get_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        help="Prepared official WHU dataset; required for preparation",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", type=parse_seeds)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--architecture", choices=ARCHITECTURES, help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, help=argparse.SUPPRESS)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands only; do not load data/models or write files",
    )
    mode.add_argument("--assess-run", action="store_true", help=argparse.SUPPRESS)
    mode.add_argument(
        "--execute",
        action="store_true",
        help="Fit the prepared plan only after a matching runtime estimate exists",
    )
    return parser.parse_args()


def main():
    args = get_args()
    if args.assess_run:
        if args.architecture is None or args.seed is None:
            raise SystemExit("Internal assessment requires architecture and seed")
        assess_run(args.out, args.architecture, args.seed)
        return
    if args.execute:
        if any(
            getattr(args, key) is not None
            for key in (
                "data",
                "seeds",
                "epochs",
                "batch_size",
                "workers",
                "device",
                "architecture",
                "seed",
            )
        ):
            raise SystemExit(
                "--execute uses the saved plan; do not pass training overrides"
            )
        execute_plan(args.out)
        return
    if args.data is None:
        raise SystemExit("--data is required for preparation or dry-run")
    if args.dry_run:
        fixed = settings(
            40 if args.epochs is None else args.epochs,
            4 if args.batch_size is None else args.batch_size,
            4 if args.workers is None else args.workers,
            args.device or "cuda",
        )
        plan = {
            "dataset": {"path": str(args.data.resolve())},
            "settings": fixed,
            "initializations": {
                arch: {"encoder_sha256": "<frozen-on-prepare>"}
                for arch in ARCHITECTURES
            },
        }
        for seed in args.seeds or [42]:
            for arch in ARCHITECTURES:
                print(
                    subprocess.list2cmdline(
                        training_command(plan, arch, seed, args.out)
                    )
                )
        print("Dry-run only: no files, data access, downloads or fitting.")
        return
    plan = prepare(args)
    print(
        f"Prepared {len(plan['architectures']) * len(plan['seeds'])} matched runs: {args.out.resolve() / 'plan.json'}"
    )
    print(
        "Generate and review runtime_estimate.json before invoking --execute. No full training started."
    )


if __name__ == "__main__":
    main()
