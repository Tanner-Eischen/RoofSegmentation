#!/usr/bin/env python3
"""Run a 2x2 matched-seed factorial study against RoofSegmentation.

No real data or checkpoints are bundled. Requires the sampler-control patch.
Runs only TRAINING and VALIDATION; never invokes the test partition.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ARMS = (
    ("a0b0_uniform_standard", False, "standard"),
    ("a1b0_hard_standard", True, "standard"),
    ("a0b1_uniform_robust", False, "robust"),
    ("a1b1_hard_robust", True, "robust"),
)


def source_files(repo: Path) -> list[Path]:
    files = list((repo / "roof_pipeline").glob("*.py"))
    files += list((repo / "ablation_study" / "scripts").glob("*.py"))
    files += [repo / "ablation_study" / "experiment_plan.json"]
    return sorted(files)


def preflight(data: Path, checkpoint: Path, device: str) -> dict:
    """Reject incompatible assets before registering or spending GPU time."""
    from collections import Counter

    import torch

    from roof_pipeline.common import dataset_fingerprint, read_manifest

    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    rows = read_manifest(data)
    counts = dict(Counter(row["split"] for row in rows))
    if counts != {"train": 4736, "val": 1036, "test": 2416}:
        raise ValueError("Study requires complete official WHU partitions")
    meta = json.loads((data / "dataset_meta.json").read_text(encoding="utf-8"))
    provenance = meta.get("source_provenance", {})
    if (
        provenance.get("source") != "official"
        or provenance.get("spatial_independence") != "publisher_geographic_partitions"
        or meta.get("limited_subset") is not False
    ):
        raise ValueError("Study requires documented official geographic WHU provenance")
    dataset_sha = dataset_fingerprint(data)
    config = torch.load(checkpoint, map_location="cpu", weights_only=True)["config"]
    if (
        config.get("arch") != "unet-resnet34"
        or config.get("image_size") != 512
        or config.get("normalization") != "imagenet_rgb_v1"
        or config.get("dataset_content_sha256") != dataset_sha
        or config.get("dataset_manifest_sha256") != sha256(data / "manifest.csv")
        or config.get("boundary_supervision", False)
        or config.get("training_review_content_sha256") is not None
        or config.get("effective_training_content_sha256", dataset_sha) != dataset_sha
    ):
        raise ValueError(
            "Starting checkpoint does not match frozen source-label WHU study"
        )
    return {"dataset_content_sha256": dataset_sha, "split_counts": counts}


def freeze_mining(mine: Path, out: Path, reg: dict) -> None:
    from roof_pipeline.common import read_manifest
    from roof_pipeline.hard_examples import load_sampling_weights

    report = json.loads(mine.read_text(encoding="utf-8"))
    if (
        report.get("checkpoint_sha256") != reg["checkpoint_sha256"]
        or report.get("image_size") != reg["image_size"]
    ):
        raise ValueError("Mining checkpoint or image size differs from study")
    rows = [r for r in read_manifest(Path(reg["data"])) if r["split"] == "train"]
    load_sampling_weights(mine, rows, reg["dataset_content_sha256"])
    fingerprint = {"sha256": sha256(mine)}
    frozen = out / "mining_fingerprint.json"
    if frozen.exists():
        if json.loads(frozen.read_text(encoding="utf-8")) != fingerprint:
            raise ValueError("Mining report changed after freezing")
    else:
        frozen.write_text(json.dumps(fingerprint, indent=2) + "\n", encoding="utf-8")


def assert_frozen_inputs(reg: dict, repo: Path, data: Path, checkpoint: Path):
    if sha256(checkpoint) != reg["checkpoint_sha256"]:
        raise ValueError("Starting checkpoint changed during study")
    current_sources = {
        str(p.relative_to(repo)).replace("\\", "/"): sha256(p)
        for p in source_files(repo)
    }
    if current_sources != reg["source_sha256"]:
        raise ValueError("Study source changed; use a new output root")
    if (
        preflight(data, checkpoint, reg["device"])["dataset_content_sha256"]
        != reg["dataset_content_sha256"]
    ):
        raise ValueError("Dataset changed during study")


def status(out: Path, phase: str, **detail):
    record = {
        "phase": phase,
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "test_evaluated": False,
        **detail,
    }
    (out / "workflow_status.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8"
    )


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def command_text(command: list[str]) -> str:
    return subprocess.list2cmdline(command)


def execute(command: list[str], cwd: Path, logfile: Path) -> None:
    logfile.parent.mkdir(parents=True, exist_ok=True)
    print("RUN:", command_text(command), flush=True)
    with logfile.open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=dict(os.environ, PYTHONIOENCODING="utf-8"),
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            stream.write(line)
            stream.flush()
        result = process.wait()
    if result:
        raise RuntimeError(f"Command exited {result}; inspect {logfile}")


def registration(args: argparse.Namespace, data: Path, checkpoint: Path, repo: Path):
    try:
        git = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            capture_output=True,
            text=True,
            check=False,
        )
        git_sha = git.stdout.strip() if git.returncode == 0 else "unknown"
    except OSError:
        git_sha = "unknown"
    return {
        "study": "whu_2x2_hard_sampling_by_photometric_augmentation_v1",
        "phase": "exploratory_validation_only",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "hypotheses": {
            "H_A": "Hard-example weighting may improve global spatial-val IoU over uniform replacement.",
            "H_B": "Robust photometric augmentation may improve spatial-val IoU.",
            "H_AB": "Sampling and augmentation may interact (positive or negative).",
        },
        "repository": str(repo),
        "git_head": git_sha,
        "data": str(data),
        "data_manifest_sha256": sha256(data / "manifest.csv"),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        **preflight(data, checkpoint, args.device),
        "source_sha256": {
            str(p.relative_to(repo)).replace("\\", "/"): sha256(p)
            for p in source_files(repo)
        },
        "environment": {
            key: os.environ.get(key)
            for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "CUBLAS_WORKSPACE_CONFIG")
        },
        "library_versions": {
            name: importlib.metadata.version(name)
            for name in ("torch", "numpy", "Pillow")
        },
        "factors": {
            "hard_example_weighting": [False, True],
            "photometric": ["standard", "robust"],
            "sampler": "weighted-with-replacement for ALL arms (control weights uniform)",
        },
        "seeds": args.seeds,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "workers": args.workers,
        "device": args.device,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "image_size": 512,
        "arch": "unet-resnet34",
        "positive_weight": 1.0,
        "boundary_weight": 0.0,
        "min_delta": 0.0001,
        "endpoint": "final.pt at fixed epoch count / threshold=0.5",
        "primary_metric": "global_validation_pixel_iou_at_0.5",
        "design_note": "WHU val has one coarse partition group; no tile-independence confidence intervals.",
        "test_use": "NONE: prior WHU test already inspected; no fresh confirmatory claim",
    }


def comparable_registration(a: dict, b: dict) -> bool:
    ignore = {"created_utc"}
    return {k: v for k, v in a.items() if k not in ignore} == {
        k: v for k, v in b.items() if k not in ignore
    }


def parse_seeds(raw: str) -> list[int]:
    try:
        values = [int(s.strip()) for s in raw.split(",")]
    except ValueError as ex:
        raise argparse.ArgumentTypeError(
            "seeds must be comma-separated integers"
        ) from ex
    if (
        not values
        or len(set(values)) != len(values)
        or any(v < 0 or v >= 2**32 for v in values)
    ):
        raise argparse.ArgumentTypeError("seeds must be distinct nonnegative integers")
    return values


def get_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--repo", type=Path, default=Path.cwd(), help="RoofSegmentation repository root"
    )
    p.add_argument(
        "--data", type=Path, required=True, help="Prepared official WHU data"
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Frozen existing 512px U-Net/ResNet34 checkpoint",
    )
    p.add_argument(
        "--out", type=Path, required=True, help="Fresh experiment output root"
    )
    p.add_argument("--seeds", type=parse_seeds, default=[42, 1337, 2026])
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument(
        "--workers", type=int, default=0, help="0 is safest with Windows PowerShell"
    )
    p.add_argument("--lr", type=float, default=0.0001)
    p.add_argument("--weight-decay", type=float, default=0.0001)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p.add_argument(
        "--resume",
        action="store_true",
        help="Skip *completed* runs; reject partial results",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Show commands without touching data or training",
    )
    return p.parse_args()


def main():
    args = get_args()
    if (
        args.epochs < 1
        or args.batch_size < 1
        or args.workers < 0
        or not math.isfinite(args.lr)
        or args.lr <= 0
        or not math.isfinite(args.weight_decay)
        or args.weight_decay < 0
    ):
        raise SystemExit("Invalid epochs, batch size, workers, or learning rate")
    repo, data, checkpoint, out = (
        args.repo.resolve(),
        args.data.resolve(),
        args.checkpoint.resolve(),
        args.out.resolve(),
    )
    mine = out / "hard_examples.json"
    mine_cmd = [
        sys.executable,
        "-u",
        "-m",
        "roof_pipeline.hard_examples",
        "--data",
        str(data),
        "--checkpoint",
        str(checkpoint),
        "--out",
        str(mine),
        "--device",
        args.device,
        "--batch-size",
        str(args.batch_size),
        "--workers",
        str(args.workers),
    ]
    specs = []
    for seed in args.seeds:
        for arm, hard, photometric in ARMS:
            run = out / "runs" / f"{arm}_seed{seed}"
            cmd = [
                sys.executable,
                "-u",
                "-m",
                "roof_pipeline.train",
                "--data",
                str(data),
                "--run",
                str(run),
                "--arch",
                "unet-resnet34",
                "--size",
                "512",
                "--init-checkpoint",
                str(checkpoint),
                "--no-pretrained",
                "--epochs",
                str(args.epochs),
                "--patience",
                str(args.epochs + 1),
                "--batch-size",
                str(args.batch_size),
                "--workers",
                str(args.workers),
                "--device",
                args.device,
                "--seed",
                str(seed),
                "--lr",
                str(args.lr),
                "--weight-decay",
                str(args.weight_decay),
                "--pos-weight",
                "1",
                "--min-delta",
                "0.0001",
                "--boundary-weight",
                "0",
                "--photometric",
                photometric,
            ]
            if hard:
                cmd += ["--sampling-report", str(mine)]
            else:
                cmd += ["--uniform-replacement"]
            valout = run / "eval_final_val"
            valcmd = [
                sys.executable,
                "-u",
                "-m",
                "roof_pipeline.evaluate",
                "--data",
                str(data),
                "--checkpoint",
                str(run / "final.pt"),
                "--out",
                str(valout),
                "--split",
                "val",
                "--batch-size",
                str(args.batch_size),
                "--device",
                args.device,
            ]
            specs.append((seed, arm, hard, photometric, run, cmd, valcmd, valout))
    if args.dry_run:
        print("MINING (TRAIN ONLY):", command_text(mine_cmd), "\n")
        for seed, arm, hard, photo, run, traincmd, valcmd, _ in specs:
            print(f"[{arm} / seed={seed}] TRAIN:\n{command_text(traincmd)}")
            print("VALIDATE:\n" + command_text(valcmd) + "\n")
        print("No files created, no model trained, and no test evaluated.")
        return 0
    if not (repo / "roof_pipeline" / "train.py").is_file():
        raise SystemExit(f"Not a RoofSegmentation repo root: {repo}")
    text = (repo / "roof_pipeline" / "train.py").read_text(encoding="utf-8")
    if '"--uniform-replacement"' not in text:
        raise SystemExit(
            "Missing uniform-control patch. From repo root: git apply --check <patch>; git apply <patch>"
        )
    if not (data / "manifest.csv").is_file() or not checkpoint.is_file():
        raise SystemExit("Real WHU prepared manifest or frozen checkpoint is missing")
    sys.path.insert(0, str(repo))
    reg = registration(args, data, checkpoint, repo)
    registration_file = out / "registration.json"
    if registration_file.exists():
        recorded = json.loads(registration_file.read_text(encoding="utf-8"))
        if not args.resume or not comparable_registration(recorded, reg):
            raise SystemExit(
                "Study already exists or preregistration changed; use --resume with exactly the same design"
            )
    else:
        if out.exists() and any(out.iterdir()):
            raise SystemExit(
                "Output directory is nonempty without registration. Choose a fresh --out"
            )
        out.mkdir(parents=True, exist_ok=True)
        registration_file.write_text(json.dumps(reg, indent=2) + "\n", encoding="utf-8")
        with zipfile.ZipFile(
            out / "source_snapshot.zip", "w", zipfile.ZIP_DEFLATED
        ) as snapshot:
            for path in source_files(repo):
                snapshot.write(path, path.relative_to(repo))
    if not mine.is_file() and (out / "mining_fingerprint.json").exists():
        raise ValueError("Frozen mining report is missing; do not regenerate it")
    if not mine.is_file():
        status(out, "mining")
        execute(mine_cmd, repo, out / "mining.log")
    freeze_mining(mine, out, reg)
    print(f"EXPLORATORY VALIDATION: {len(specs)} training runs; NO test evaluation")
    for seed, arm, hard, photo, run, traincmd, valcmd, valout in specs:
        assert_frozen_inputs(reg, repo, data, checkpoint)
        freeze_mining(mine, out, reg)
        armreg = {
            "arm": arm,
            "seed": seed,
            "hard": hard,
            "photometric": photo,
            "study_registration_sha256": sha256(registration_file),
            "epochs": args.epochs,
        }
        armreg_file = run / "registration.json"
        if run.exists():
            valid = (
                armreg_file.is_file()
                and json.loads(armreg_file.read_text(encoding="utf-8")) == armreg
            )
            done = all(
                p.is_file()
                for p in (
                    run / "training_summary.json",
                    run / "final.pt",
                    valout / "val_metrics.json",
                    valout / "val_per_image.csv",
                )
            )
            if not valid or not done or not args.resume:
                raise SystemExit(
                    f"Partial, mismatched, or existing run: {run}. Do not overwrite it; use a new --out"
                )
            from analyze_factorial import read_run

            read_run(out, reg, seed, arm, sha256(registration_file))
            print(f"SKIP verified complete: {run}")
            continue
        run.mkdir(parents=True, exist_ok=False)
        armreg_file.write_text(json.dumps(armreg, indent=2) + "\n", encoding="utf-8")
        status(out, "training", arm=arm, seed=seed)
        execute(traincmd, repo, run / "training.log")
        assert_frozen_inputs(reg, repo, data, checkpoint)
        freeze_mining(mine, out, reg)
        status(out, "validating", arm=arm, seed=seed)
        execute(valcmd, repo, run / "validation.log")
        from analyze_factorial import read_run

        read_run(out, reg, seed, arm, sha256(registration_file))
    status(out, "analysis")
    execute(
        [
            sys.executable,
            str(repo / "ablation_study/scripts/analyze_factorial.py"),
            "--out",
            str(out),
        ],
        repo,
        out / "analysis.log",
    )
    execute(
        [
            sys.executable,
            str(repo / "ablation_study/scripts/plot_factorial.py"),
            "--out",
            str(out),
            "--device",
            args.device,
        ],
        repo,
        out / "figures.log",
    )
    status(out, "complete", completed_runs=len(specs))
    print("Finished. Report:", out / "analysis/report.md")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (Exception, KeyboardInterrupt) as exc:
        # Keep the exact failing command and traceback in the launcher log.
        print(f"STUDY FAILED: {exc}", file=sys.stderr, flush=True)
        arguments = sys.argv[1:]
        if "--out" in arguments and "--dry-run" not in arguments:
            output = Path(arguments[arguments.index("--out") + 1]).resolve()
            if (output / "workflow_status.json").is_file():
                previous = json.loads(
                    (output / "workflow_status.json").read_text(encoding="utf-8")
                )
                if previous.get("phase") != "complete":
                    status(
                        output,
                        "failed",
                        failed_phase=previous.get("phase"),
                        error=str(exc),
                    )
        raise
