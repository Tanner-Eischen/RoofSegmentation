"""Bounded, disposable GPU timing for a prepared architecture comparison.

No checkpoint or accuracy result is produced. Temporary optimizer steps measure
training cost; their weights are discarded before the next architecture.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import random
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from .common import dataset_fingerprint
from .dataset import RoofDataset
from .models import create_model, encoder_fingerprint
from .train import bce_dice_loss, seed_everything, seed_worker


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def timing_summary(seconds):
    if not seconds or any(not math.isfinite(x) or x <= 0 for x in seconds):
        raise ValueError("Timing observations must be finite and positive")
    ordered = sorted(seconds)
    return {
        "median": statistics.median(ordered),
        "p90": ordered[math.ceil(0.9 * len(ordered)) - 1],
        "min": min(ordered),
        "max": max(ordered),
    }


def project_runtime(train_seconds, validation_seconds, settings, counts, seeds):
    """Planning allowances, not statistical confidence bounds or GPU billing."""
    train_steps = math.ceil(counts["train"] / settings["batch_size"])
    val_steps = math.ceil(counts["val"] / settings["batch_size"])
    base = (
        train_seconds["median"] * train_steps + validation_seconds["median"] * val_steps
    )
    slower = train_seconds["p90"] * train_steps + validation_seconds["p90"] * val_steps
    epoch = {"lower": base * 1.15, "upper": max(base, slower) * 1.75}
    # Audits/hashes, process/loader startup, checkpoint writes, calibration and
    # explicit selected-checkpoint validation are outside per-epoch timings.
    run = {
        "lower": epoch["lower"] * settings["epochs"] + 300,
        "upper": epoch["upper"] * settings["epochs"] + 900,
    }
    return {
        "train_batches_per_epoch": train_steps,
        "validation_batches_per_epoch": val_steps,
        "epoch_seconds": epoch,
        "run_seconds": run,
        "all_seeds_seconds": {key: value * len(seeds) for key, value in run.items()},
    }


def measure_phase(model, loader, *, training, optimizer, scaler, warmup, measured):
    iterator_start = time.perf_counter()
    iterator = iter(loader)
    model.train(training)
    timings = []
    for step in range(warmup + measured):
        torch.cuda.synchronize()
        start = time.perf_counter()
        x, y = next(iterator)
        x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=True):
                logits = model(x)
                if logits.shape != y.shape:
                    raise ValueError(
                        "Model logits do not match full-resolution binary target"
                    )
                loss = bce_dice_loss(logits, y, 1.0)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite profiling loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            loss.item()
        else:
            with torch.no_grad():
                prediction = model(x).sigmoid().float() >= 0.5
                truth = y > 0.5
                # Match evaluate_loader's full-precision inference and counts.
                for values in (
                    prediction & truth,
                    prediction & ~truth,
                    ~prediction & truth,
                    ~prediction & ~truth,
                ):
                    values.flatten(1).sum(1).cpu().tolist()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        if step >= warmup:
            timings.append(elapsed)
    del iterator
    return {
        **timing_summary(timings),
        "observations": timings,
        "phase_wall_seconds": time.perf_counter() - iterator_start,
    }


def profile(plan_path, warmup=4, train_batches=32, validation_batches=12):
    if warmup < 1 or train_batches < 10 or validation_batches < 5:
        raise ValueError(
            "Use at least one warmup, ten training and five validation batches"
        )
    plan_path = Path(plan_path).resolve()
    plan_sha = sha256(plan_path)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if sha256(plan_path) != plan_sha:
        raise ValueError("Plan changed while reading")
    from .architecture_comparison import runtime_environment, write_status

    for key, value in plan["environment"]["thread_environment"].items():
        os.environ[key] = value
    torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
    if runtime_environment() != plan["environment"]:
        raise ValueError("Profiling runtime differs from prepared plan")
    if not torch.cuda.is_available():
        raise RuntimeError("GPU profiling requires CUDA")
    repo = Path(__file__).resolve().parents[1]
    for relative, expected in plan["source_sha256"].items():
        if sha256(repo / relative) != expected:
            raise ValueError(f"Profiling source changed after preparation: {relative}")
    settings = plan["settings"]
    if settings["device"] != "cuda":
        raise ValueError("This profiler requires a CUDA comparison plan")
    out = plan_path.parent / "runtime_estimate.json"
    if out.exists():
        raise FileExistsError(
            "Runtime estimate already exists; preserve the original probe"
        )
    data = Path(plan["dataset"]["path"])
    if dataset_fingerprint(data) != plan["dataset"]["content_sha256"]:
        raise ValueError("Profiling dataset differs from prepared plan")
    train = RoofDataset(
        data, "train", settings["size"], augment=True, photometric="standard"
    )
    val = RoofDataset(data, "val", settings["size"], augment=False)
    selected = {}
    for split, dataset, measured in (
        ("train", train, train_batches),
        ("val", val, validation_batches),
    ):
        count = (warmup + measured) * settings["batch_size"]
        if count > len(dataset):
            raise ValueError("Probe exceeds the registered partition")
        selected[split] = random.Random(42).sample(range(len(dataset)), count)
    record = {
        "schema_version": 1,
        "plan_sha256": plan_sha,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "device": torch.cuda.get_device_name(0),
        "device_total_bytes": torch.cuda.get_device_properties(0).total_memory,
        "library_versions": {
            name: importlib.metadata.version(name)
            for name in ("torch", "numpy", "Pillow", "segmentation-models-pytorch")
        },
        "probe": {
            "warmup_batches_per_phase": warmup,
            "measured_train_batches": train_batches,
            "measured_validation_batches": validation_batches,
            "batch_size": settings["batch_size"],
            "size": settings["size"],
            "workers": settings["workers"],
            "amp_training": True,
            "validation_precision": "float32",
            "weights_saved": False,
            "sample_ids": {
                split: [dataset.rows[i]["sample_id"] for i in selected[split]]
                for split, dataset in (("train", train), ("val", val))
            },
        },
        "method": {
            "epoch_allowance_lower": 1.15,
            "epoch_allowance_upper": 1.75,
            "extra_run_seconds_lower": 300,
            "extra_run_seconds_upper": 900,
            "note": "Planning range from median/p90 batch times, not a confidence interval. Short probes cannot guarantee sustained laptop throughput.",
        },
        "architectures": {},
    }
    for arch in plan["architectures"]:
        print(
            f"Profiling {arch}: disposable steps only; no checkpoint retained",
            flush=True,
        )
        seed_everything(42)
        model = create_model(arch, pretrained=True)
        actual_sha = encoder_fingerprint(model)
        if actual_sha != plan["initializations"][arch]["encoder_sha256"]:
            raise ValueError(f"Pretrained encoder identity differs from plan: {arch}")
        parameters = sum(p.numel() for p in model.parameters())
        if parameters != plan["initializations"][arch]["parameter_count"]:
            raise ValueError("Model parameter count differs from plan")
        model = model.cuda()
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=settings["lr"], weight_decay=settings["weight_decay"]
        )
        scaler = torch.amp.GradScaler("cuda", enabled=True)
        torch.cuda.reset_peak_memory_stats()
        phases = {}
        for split, dataset, measured in (
            ("train", train, train_batches),
            ("val", val, validation_batches),
        ):
            loader = DataLoader(
                Subset(dataset, selected[split]),
                batch_size=settings["batch_size"],
                shuffle=False,
                num_workers=settings["workers"],
                pin_memory=True,
                worker_init_fn=seed_worker,
                generator=torch.Generator().manual_seed(42),
            )
            phases[split] = measure_phase(
                model,
                loader,
                training=split == "train",
                optimizer=optimizer,
                scaler=scaler,
                warmup=warmup,
                measured=measured,
            )
            del loader
        training_validation_peak = torch.cuda.max_memory_allocated()
        # The trainer retains the fitted model and Adam states when reloading
        # its selected checkpoint. Exercise that additional model residency.
        selected_copy = create_model(arch, pretrained=False).cuda().eval()
        images = torch.stack(
            [val[i][0] for i in selected["val"][: settings["batch_size"]]]
        )
        with torch.inference_mode():
            selected_copy(images.cuda()).sigmoid().sum().item()
        torch.cuda.synchronize()
        result = {
            "encoder_sha256": actual_sha,
            "parameter_count": parameters,
            "train_step_seconds": phases["train"],
            "validation_step_seconds": phases["val"],
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "training_validation_peak_allocated_bytes": training_validation_peak,
            "selection_model_residency_exercised": True,
            **project_runtime(
                phases["train"],
                phases["val"],
                settings,
                plan["dataset"]["split_counts"],
                plan["seeds"],
            ),
        }
        record["architectures"][arch] = result
        print(
            f"{arch}: train {phases['train']['median']:.3f}s/batch; val {phases['val']['median']:.3f}s/batch; peak {result['peak_allocated_bytes'] / 2**30:.2f} GiB",
            flush=True,
        )
        del model, selected_copy, images, optimizer, scaler
        torch.cuda.empty_cache()
    record["total_seconds"] = {
        key: sum(
            item["all_seeds_seconds"][key] for item in record["architectures"].values()
        )
        for key in ("lower", "upper")
    }
    record["three_seed_total_seconds"] = {
        key: value * 3 / len(plan["seeds"])
        for key, value in record["total_seconds"].items()
    }
    if dataset_fingerprint(data) != plan["dataset"]["content_sha256"]:
        raise ValueError("Dataset changed during timing probe")
    for relative, expected in plan["source_sha256"].items():
        if sha256(repo / relative) != expected:
            raise ValueError("Source changed during timing probe")
    if sha256(plan_path) != plan_sha:
        raise ValueError("Plan changed during timing probe")
    out.write_text(
        json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    write_status(
        plan_path.parent,
        "ready_not_started",
        full_training_started=False,
        planned_runs=len(plan["architectures"]) * len(plan["seeds"]),
        runtime_estimate_sha256=sha256(out),
    )
    print(
        json.dumps(
            {
                "estimate": str(out),
                "total_hours": {
                    key: value / 3600 for key, value in record["total_seconds"].items()
                },
            },
            indent=2,
        )
    )
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--warmup-batches", type=int, default=4)
    parser.add_argument("--train-batches", type=int, default=32)
    parser.add_argument("--validation-batches", type=int, default=12)
    args = parser.parse_args()
    try:
        profile(
            args.plan, args.warmup_batches, args.train_batches, args.validation_batches
        )
    except torch.cuda.OutOfMemoryError as error:
        failure = {
            "full_training_started": False,
            "error": str(error),
            "decision": "Re-prepare a new plan with a smaller common batch size for ALL models, then re-profile.",
        }
        (args.plan.parent / "runtime_probe_failure.json").write_text(
            json.dumps(failure, indent=2), encoding="utf-8"
        )
        raise


if __name__ == "__main__":
    main()
