"""No torch or real data required. These test package logic, not segmentation accuracy."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


analysis = module("ablation_analysis", ROOT / "scripts" / "analyze_factorial.py")
patch = module("ablation_patch", ROOT / "scripts" / "install_sampler_control.py")
runner = module("ablation_runner", ROOT / "scripts" / "run_factorial.py")


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def create_numeric_fixture(out):
    """Fake files and counts exercise verification; never segmentation measurements."""
    reg = {
        "phase": "exploratory_validation_only",
        "seeds": [42, 1337, 2026],
        "epochs": 2,
        "arch": "unet-resnet34",
        "data": "fake",
        "checkpoint": "fake/base.pt",
        "checkpoint_sha256": "base",
        "dataset_content_sha256": "sha",
        "data_manifest_sha256": "manifest",
        "image_size": 512,
        "batch_size": 4,
        "workers": 0,
        "device": "cpu",
        "learning_rate": 0.0001,
        "weight_decay": 0.0001,
        "positive_weight": 1.0,
        "min_delta": 0.0001,
        "split_counts": {"val": 1},
    }
    write_json(out / "registration.json", reg)
    write_json(
        out / "hard_examples.json",
        {
            "split": "train",
            "checkpoint_sha256": "base",
            "dataset_content_sha256": "sha",
            "image_size": 512,
        },
    )
    mining_sha = analysis.sha256(out / "hard_examples.json")
    write_json(out / "mining_fingerprint.json", {"sha256": mining_sha})
    for seed in reg["seeds"]:
        for arm, (hard, robust) in analysis.ARMS.items():
            run = out / "runs" / f"{arm}_seed{seed}"
            ev = run / "eval_final_val"
            ev.mkdir(parents=True)
            tp = 80 + 2 * hard + robust + hard * robust
            iou = tp / 100
            photo = "robust" if robust else "standard"
            config = {
                "arch": reg["arch"],
                "image_size": 512,
                "seed": seed,
                "normalization": "imagenet_rgb_v1",
                "dataset_manifest_sha256": "manifest",
                "dataset_content_sha256": "sha",
                "effective_training_content_sha256": "sha",
                "initial_checkpoint_sha256": "base",
                "photometric_profile": photo,
                "uniform_replacement": not hard,
                "sampling_report_sha256": mining_sha if hard else None,
                "boundary_supervision": False,
                "boundary_weight": 0.0,
                "training_review_content_sha256": None,
                "optimizer_restarted": True,
            }
            arguments = {
                "arch": reg["arch"],
                "size": 512,
                "seed": seed,
                "epochs": 2,
                "patience": 3,
                "batch_size": 4,
                "workers": 0,
                "device": "cpu",
                "lr": 0.0001,
                "weight_decay": 0.0001,
                "pos_weight": 1.0,
                "min_delta": 0.0001,
                "boundary_weight": 0.0,
                "photometric": photo,
                "uniform_replacement": not hard,
                "training_review": None,
                "no_amp": False,
                "pretrained": False,
                "init_checkpoint": "fake/base.pt",
                "data": "fake",
                "sampling_report": str((out / "hard_examples.json").resolve())
                if hard
                else None,
            }
            write_json(
                run / "registration.json",
                {
                    "arm": arm,
                    "seed": seed,
                    "hard": bool(hard),
                    "photometric": photo,
                    "study_registration_sha256": analysis.sha256(
                        out / "registration.json"
                    ),
                    "epochs": 2,
                },
            )
            write_json(
                run / "history.json",
                [{"epoch": i, "seconds": 1.5, "val_iou_0p5": iou} for i in (1, 2)],
            )
            write_json(
                run / "experiment.json", {"arguments": arguments, "config": config}
            )
            write_json(
                run / "training_summary.json",
                {
                    "final_val_iou_at_0p5": iou,
                    "final_epoch": 2,
                    "config": config,
                    "final_threshold": 0.5,
                    "test_evaluated": False,
                    "val_samples": 1,
                },
            )
            (run / "final.pt").write_bytes(b"numeric-fixture-not-a-model")
            (run / "best.pt").write_bytes(b"numeric-fixture-not-a-model")
            write_json(
                ev / "val_metrics.json",
                {
                    "split": "val",
                    "threshold_frozen_from_validation": 0.5,
                    "dataset_content_sha256": "sha",
                    "checkpoint_sha256": analysis.sha256(run / "final.pt"),
                    "samples": 1,
                    "aggregate": {
                        "iou": iou,
                        "dice": 0.9,
                        "precision": 0.92,
                        "recall": 0.88,
                        "tp": tp,
                        "fp": 100 - tp,
                        "fn": 0,
                        "tn": 0,
                    },
                },
            )
            (ev / "val_per_image.csv").write_text(
                f"sample_id,tp,fp,fn,tn\ntile,{tp},{100 - tp},0,0\n", encoding="utf-8"
            )


class StudyTests(unittest.TestCase):
    def test_factorial_contrasts(self):
        values = {(0, 0): 0.8, (1, 0): 0.82, (0, 1): 0.81, (1, 1): 0.84}
        effect = analysis.effects(values)
        self.assertAlmostEqual(effect["sampling"], 0.025)
        self.assertAlmostEqual(effect["augmentation"], 0.015)
        self.assertAlmostEqual(effect["interaction"], 0.01)
        self.assertAlmostEqual(effect["combined_vs_control"], 0.04)

    def test_patch_only_exact_anchors(self):
        source = "\n".join(old for old, _ in patch.SWAPS) + "\n"
        new = patch.make_change(source)
        self.assertIn('"--uniform-replacement"', new)
        self.assertIn("replacement=True", new)
        self.assertIn('"uniform_replacement": uniform_replacement', new)
        with self.assertRaises(ValueError):
            patch.make_change(new)
        with self.assertRaises(ValueError):
            patch.make_change("changed code")

    def test_seed_validation(self):
        self.assertEqual(runner.parse_seeds("42,1337"), [42, 1337])
        from argparse import ArgumentTypeError

        with self.assertRaises(ArgumentTypeError):
            runner.parse_seeds("3,3")
        with self.assertRaises(ArgumentTypeError):
            runner.parse_seeds("x")

    def test_synthetic_results_aggregation(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            create_numeric_fixture(out)
            _, rows = analysis.read_results(out)
            arms, contrasts, effects = analysis.summaries(rows)
            self.assertEqual(len(rows), 12)
            self.assertEqual(len(arms), 4)
            self.assertEqual(len(contrasts), 3)
            self.assertAlmostEqual(
                next(
                    x["mean_effect_iou"] for x in effects if x["contrast"] == "sampling"
                ),
                0.025,
            )
            self.assertAlmostEqual(
                next(
                    x["mean_effect_iou"]
                    for x in effects
                    if x["contrast"] == "interaction"
                ),
                0.01,
            )
            self.assertAlmostEqual(rows[0]["train_minutes"], 0.05)
            (
                out
                / "runs"
                / "a0b0_uniform_standard_seed42"
                / "eval_final_val"
                / "val_metrics.json"
            ).unlink()
            with self.assertRaises(ValueError):
                analysis.read_results(out)

    def test_rejects_modified_or_incomplete_run_artifacts(self):
        mutations = {
            "stale final": ("final.pt", None, b"changed weights"),
            "changed batch": ("experiment.json", ("arguments", "batch_size"), 8),
            "changed seed": ("training_summary.json", ("config", "seed"), 99),
            "changed augmentation": (
                "training_summary.json",
                ("config", "photometric_profile"),
                "robust",
            ),
            "changed weights": (
                "training_summary.json",
                ("config", "initial_checkpoint_sha256"),
                "other",
            ),
            "changed sampling": (
                "training_summary.json",
                ("config", "uniform_replacement"),
                False,
            ),
            "label overlay": (
                "training_summary.json",
                ("config", "effective_training_content_sha256"),
                "overlay",
            ),
            "nonfinite IoU": (
                "training_summary.json",
                ("final_val_iou_at_0p5",),
                float("nan"),
            ),
            "epoch order": ("history.json", (1, "epoch"), 1),
            "negative timing": ("history.json", (0, "seconds"), -1),
            "incomplete tiles": ("eval_final_val/val_metrics.json", ("samples",), 2),
            "nonfrozen threshold": (
                "eval_final_val/val_metrics.json",
                ("threshold_frozen_from_validation",),
                0.6,
            ),
            "registration link": (
                "registration.json",
                ("study_registration_sha256",),
                "different",
            ),
        }
        for label, (filename, keys, value) in mutations.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp)
                create_numeric_fixture(out)
                path = out / "runs/a0b0_uniform_standard_seed42" / filename
                if keys is None:
                    path.write_bytes(value)
                else:
                    obj = json.loads(path.read_text())
                    item = obj
                    for key in keys[:-1]:
                        item = item[key]
                    item[keys[-1]] = value
                    write_json(path, obj)
                with self.assertRaises(ValueError):
                    analysis.read_results(out)

    def test_rejects_changed_mining_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            create_numeric_fixture(out)
            with (out / "hard_examples.json").open("a") as stream:
                stream.write(" ")
            with self.assertRaisesRegex(ValueError, "changed after freezing"):
                analysis.read_results(out)

    def test_dry_run_no_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "out"
            cp = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "run_factorial.py"),
                    "--data",
                    str(Path(tmp) / "data"),
                    "--checkpoint",
                    str(Path(tmp) / "baseline.pt"),
                    "--out",
                    str(dest),
                    "--seeds",
                    "42",
                    "--dry-run",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            self.assertIn("--uniform-replacement", cp.stdout)
            self.assertIn("--sampling-report", cp.stdout)
            self.assertIn("No files created", cp.stdout)
            self.assertFalse(dest.exists())


if __name__ == "__main__":
    unittest.main()
