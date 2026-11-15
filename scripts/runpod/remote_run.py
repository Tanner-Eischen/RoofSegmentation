"""Derive, profile and optionally execute the frozen comparison on a Linux Pod.

This helper does not contain Runpod credentials and cannot stop Pod billing.
The owner must configure an external stop/termination control for the Pod.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def same_controls(parent, derived):
    """Only platform/path/registration identity may differ from the local plan."""
    for name in (
        "architectures",
        "seeds",
        "settings",
        "initializations",
        "source_sha256",
    ):
        if parent[name] != derived[name]:
            raise ValueError(f"Derived plan changed scientific control: {name}")
    for name in ("content_sha256", "manifest_sha256", "split_counts"):
        if parent["dataset"][name] != derived["dataset"][name]:
            raise ValueError(f"Derived plan changed dataset identity: {name}")
    if derived.get("test_inference") is not False:
        raise ValueError("Derived plan must remain validation-only")


def write_status(out, phase, **detail):
    (out / "remote_status.json").write_text(
        json.dumps(
            {
                "phase": phase,
                "updated_utc": datetime.now(timezone.utc).isoformat(),
                **detail,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def execution_budget(estimate, rate, ceiling, setup_allowance, timeout=None):
    upper = estimate["total_seconds"]["upper"]
    if not math.isfinite(upper) or upper <= 0:
        raise ValueError("Remote estimate must have a positive finite upper duration")
    if timeout is not None and upper > timeout:
        raise ValueError(
            "Remote upper estimate exceeds the remaining job time budget; no full training started"
        )
    projected = upper / 3600 * rate + setup_allowance
    if projected > ceiling:
        raise ValueError(
            "Remote estimate exceeds the approved cost ceiling; no full training started"
        )
    return projected


def run(command, log, *, timeout=None):
    with log.open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise RuntimeError(
                "Job process time limit reached; Pod billing must still be stopped externally"
            ) from None
    if code:
        raise RuntimeError(f"Command failed ({code}); inspect {log}")


def export_results(out):
    """Keep checkpoints, logs, plan, timing and source together; no dataset copy."""
    destination = out.parent / f"{out.name}_results.tar"
    if destination.exists():
        raise FileExistsError(f"Result archive already exists: {destination}")
    files = sorted(p for p in out.rglob("*") if p.is_file())
    manifest = {
        p.relative_to(out).as_posix(): {"sha256": sha256(p), "bytes": p.stat().st_size}
        for p in files
    }
    (out / "remote_result_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    with tarfile.open(destination.with_suffix(".part"), "w") as archive:
        for path in [*files, out / "remote_result_manifest.json"]:
            if path.is_symlink():
                raise ValueError("Refusing a linked result artifact")
            archive.add(
                path,
                arcname=f"{out.name}/{path.relative_to(out).as_posix()}",
                recursive=False,
            )
    destination.with_suffix(".part").replace(destination)
    destination.with_suffix(".tar.sha256").write_text(
        f"{sha256(destination)}  {destination.name}\n", encoding="utf-8"
    )
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, default=ROOT / "runs/whu_architecture_runpod_2026-11-14"
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--hourly-rate", type=float)
    parser.add_argument("--max-estimated-cost", type=float)
    parser.add_argument("--setup-cost-allowance", type=float, default=0.0)
    parser.add_argument("--wall-timeout-seconds", type=float)
    args = parser.parse_args()
    for key in ("hourly_rate", "max_estimated_cost", "wall_timeout_seconds"):
        value = getattr(args, key)
        if value is not None and (not math.isfinite(value) or value <= 0):
            raise ValueError(f"{key} must be positive and finite")
    if not math.isfinite(args.setup_cost_allowance) or args.setup_cost_allowance < 0:
        raise ValueError("Setup-cost allowance must be nonnegative and finite")
    if args.execute and (args.hourly_rate is None or args.max_estimated_cost is None):
        raise ValueError(
            "Execution requires the quoted hourly rate and an approved estimated-cost ceiling"
        )
    out = args.out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Choose a fresh remote output directory")
    parent_path = ROOT / "runpod_parent_plan.json"
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    environment = parent["environment"]["thread_environment"]
    os.environ.update(environment)
    os.environ["PYTHONIOENCODING"] = "utf-8"
    from roof_pipeline import architecture_comparison as comparison

    namespace = argparse.Namespace(
        out=out,
        data=ROOT / "data/whu_official_prepared",
        epochs=parent["settings"]["epochs"],
        batch_size=parent["settings"]["batch_size"],
        workers=parent["settings"]["workers"],
        device=parent["settings"]["device"],
        seeds=parent["seeds"],
    )
    derived = comparison.prepare(namespace)
    try:
        same_controls(parent, derived)
        comparison.write_json(
            out / "remote_derivation.json",
            {
                "parent_plan_sha256": sha256(parent_path),
                "derived_plan_sha256": sha256(out / "plan.json"),
                "scientific_controls_identical": True,
                "differences": {
                    "parent_environment": parent["environment"],
                    "remote_environment": derived["environment"],
                    "parent_repository_path": parent["repository_path"],
                    "remote_repository_path": str(ROOT),
                },
                "note": "Linux-specific path/runtime registration; original Windows plan is not changed.",
            },
        )
        write_status(out, "profiling", full_training_started=False)
        run(
            [
                sys.executable,
                "-u",
                "-m",
                "roof_pipeline.profile_architectures",
                "--plan",
                str(out / "plan.json"),
            ],
            out / "remote_profile.log",
        )
        estimate = comparison.read_json(out / "runtime_estimate.json")
        projected = (
            None
            if args.hourly_rate is None
            else estimate["total_seconds"]["upper"] / 3600 * args.hourly_rate
            + args.setup_cost_allowance
        )
        comparison.write_json(
            out / "remote_budget.json",
            {
                "hourly_rate_usd": args.hourly_rate,
                "estimated_upper_total_usd": projected,
                "setup_cost_allowance_usd": args.setup_cost_allowance,
                "approved_estimated_ceiling_usd": args.max_estimated_cost,
                "pod_billing_stop": "External Runpod stop/termination control required; process timeout does not stop billing.",
            },
        )
        if not args.execute:
            write_status(
                out,
                "profiled_not_started",
                full_training_started=False,
                estimated_upper_cost_usd=projected,
            )
            return
        execution_budget(
            estimate,
            args.hourly_rate,
            args.max_estimated_cost,
            args.setup_cost_allowance,
            args.wall_timeout_seconds,
        )
        write_status(
            out,
            "training",
            full_training_started=True,
            estimated_upper_cost_usd=projected,
        )
        start = time.monotonic()
        run(
            [
                sys.executable,
                "-u",
                "-m",
                "roof_pipeline.architecture_comparison",
                "--out",
                str(out),
                "--execute",
            ],
            out / "remote_training.log",
            timeout=args.wall_timeout_seconds,
        )
        write_status(
            out,
            "complete",
            full_training_started=True,
            training_wall_seconds=time.monotonic() - start,
        )
        archive = export_results(out)
        print(
            json.dumps(
                {
                    "complete": True,
                    "results_archive": str(archive),
                    "sha256": sha256(archive),
                }
            ),
            flush=True,
        )
    except Exception as error:
        write_status(
            out,
            "failed",
            error=str(error),
            full_training_started=(out / "runs").exists(),
        )
        if not (out.parent / f"{out.name}_results.tar").exists():
            export_results(out)
        raise


if __name__ == "__main__":
    main()
