#!/usr/bin/env python3
"""Reviewed opt-in single-file change for RoofSegmentation's train.py.

--check prints exactly what would change. --apply changes only train.py and
writes one backup. Fails closed if the expected original text has changed.
"""

from __future__ import annotations

import argparse
import difflib
from pathlib import Path

SWAPS = [
    (
        """    sampling_report = getattr(args, "sampling_report", None)
    sampler = None
    if sampling_report is not None:
""",
        """    sampling_report = getattr(args, "sampling_report", None)
    uniform_replacement = bool(getattr(args, "uniform_replacement", False))
    if uniform_replacement and sampling_report is not None:
        raise ValueError("--uniform-replacement cannot be combined with --sampling-report")
    sampler = None
    if uniform_replacement:
        sampler = WeightedRandomSampler(
            torch.ones(len(train_ds), dtype=torch.double),
            len(train_ds),
            replacement=True,
            generator=generator,
        )
    elif sampling_report is not None:
""",
    ),
    (
        """        "sampling_report_sha256": hashlib.sha256(
""",
        """        "uniform_replacement": uniform_replacement,
        "sampling_report_sha256": hashlib.sha256(
""",
    ),
    (
        """    p.add_argument(
        "--sampling-report",
        type=Path,
        help="Train-only hard-example probabilities from roof_pipeline.hard_examples",
    )
""",
        """    p.add_argument(
        "--sampling-report",
        type=Path,
        help="Train-only hard-example probabilities from roof_pipeline.hard_examples",
    )
    p.add_argument(
        "--uniform-replacement",
        action="store_true",
        help="Uniform with-replacement control to match hard-example sampler's replacement policy",
    )
""",
    ),
]


def make_change(source: str) -> str:
    out = source
    if '"--uniform-replacement"' in source:
        raise ValueError("Sampling control is already present. No files changed.")
    for old, new in SWAPS:
        matches = out.count(old)
        if matches != 1:
            raise ValueError(
                f"Expected one exact anchor, found {matches}. No files changed. Anchor: {old[:65]!r}"
            )
        out = out.replace(old, new, 1)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, default=Path.cwd())
    action = p.add_mutually_exclusive_group(required=True)
    action.add_argument("--check", action="store_true")
    action.add_argument("--apply", action="store_true")
    args = p.parse_args()
    dest = args.repo.resolve() / "roof_pipeline" / "train.py"
    if not dest.is_file():
        raise SystemExit(f"Missing {dest}")
    old = dest.read_text(encoding="utf-8")
    try:
        new = make_change(old)
    except ValueError as ex:
        raise SystemExit(str(ex)) from ex
    diff = "".join(
        difflib.unified_diff(
            old.splitlines(keepends=True),
            new.splitlines(keepends=True),
            fromfile="a/roof_pipeline/train.py",
            tofile="b/roof_pipeline/train.py",
        )
    )
    print(diff)
    if args.check:
        print("DRY CHECK ONLY: no files changed")
        return
    backup = dest.with_suffix(".py.pre_ablation.bak")
    if backup.exists():
        raise SystemExit(f"Refusing to overwrite existing backup {backup}")
    backup.write_text(old, encoding="utf-8")
    dest.write_text(new, encoding="utf-8")
    print(
        f"Updated: {dest}\nBackup: {backup}\nReview: git diff -- roof_pipeline/train.py"
    )


if __name__ == "__main__":
    main()
