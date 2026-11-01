"""Scientific figure guards and CPU inference with tiny fixture checkpoints."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from ablation_study.scripts import plot_factorial as plots
from roof_pipeline.common import dataset_fingerprint
from roof_pipeline.models import create_model
from scripts.make_smoke_dataset import make_fixture


def test_example_selection_is_order_independent_and_validation_only():
    rows = [{"sample_id": str(i), "split": "val"} for i in range(20)]
    selected = [
        rows[index]["sample_id"] for index in plots.selected_indices(rows, 4, 42)
    ]
    reordered = list(reversed(rows))
    assert selected == [
        reordered[index]["sample_id"]
        for index in plots.selected_indices(reordered, 4, 42)
    ]
    assert len(set(selected)) == 4
    with pytest.raises(ValueError, match="validation-only"):
        plots.selected_indices([{"sample_id": "test", "split": "test"}], 1, 42)
    with pytest.raises(ValueError, match="examples"):
        plots.selected_indices(rows, 21, 42)


def test_error_overlay_distinguishes_false_positives_and_negatives():
    rgb = Image.new("RGB", (2, 2), (100, 100, 100))
    result = np.asarray(
        plots.error_overlay(rgb, np.array([[1, 1], [0, 0]]), np.array([[1, 0], [1, 0]]))
    )
    assert result[0, 0, 1] > result[0, 0, 0]  # True positive is green.
    assert result[0, 1, 0] > result[0, 1, 1]  # False positive is red.
    assert result[1, 0, 2] > result[1, 0, 1]  # False negative is blue.
    assert result[1, 1].tolist() == [100, 100, 100]


def fixture_study(tmp_path):
    data = make_fixture(tmp_path / "data", 2, 4, 1, 64)
    out = tmp_path / "study"
    out.mkdir()
    reg = {
        "repository": str(Path(__file__).resolve().parents[2]),
        "data": str(data),
        "dataset_content_sha256": dataset_fingerprint(data),
        "seeds": [42],
        "epochs": 2,
        "image_size": 64,
        "arch": "tiny",
    }
    (out / "registration.json").write_text(json.dumps(reg), encoding="utf-8")
    torch.set_num_threads(2)
    model = create_model("tiny")
    for arm in plots.ARMS:
        run = out / "runs" / f"{arm}_seed42"
        run.mkdir(parents=True)
        (run / "history.json").write_text(
            json.dumps(
                [
                    {"epoch": i, "val_iou_0p5": 0.7 + i / 100, "train_loss": 1 / i}
                    for i in range(1, 3)
                ]
            ),
            encoding="utf-8",
        )
        torch.save(
            {
                "state_dict": model.state_dict(),
                "threshold": 0.5,
                "config": {
                    "arch": "tiny",
                    "image_size": 64,
                    "dataset_content_sha256": reg["dataset_content_sha256"],
                    "seed": 42,
                    "epoch": 2,
                },
            },
            run / "final.pt",
        )
    return out, reg


def test_plots_render_and_frozen_selection_precedes_inference(tmp_path, monkeypatch):
    out, reg = fixture_study(tmp_path)
    # Analyzer integrity is covered separately; this isolates real figure inference.
    monkeypatch.setattr(plots, "read_results", lambda root: (reg, []))
    from roof_pipeline import models

    original_loader = models.load_model
    observed = []

    def checked_loader(checkpoint, device):
        selection = out / "analysis/figures/qualitative_seed42_selection.json"
        assert selection.is_file(), "Sample IDs must be frozen before model inference"
        observed.append(checkpoint)
        return original_loader(checkpoint, device)

    monkeypatch.setattr(models, "load_model", checked_loader)
    paths = plots.plot_study(out, examples=2)
    assert len(observed) == 4
    for path in paths:
        with Image.open(path) as image:
            image.verify()
    selection = json.loads(
        (paths[0].parent / "qualitative_seed42_selection.json").read_text()
    )
    assert selection["split"] == "val"
    assert len(selection["sample_ids"]) == 2
    manifest = json.loads((paths[0].parent / "figure_manifest.json").read_text())
    assert manifest["artifacts"] == {path.name: plots.sha256(path) for path in paths}
    with pytest.raises(ValueError, match="Frozen qualitative"):
        plots.qualitative_examples(out, reg, paths[0].parent, 42, 3, "cpu")


def test_figures_reject_changed_data_and_nonfinite_history(tmp_path):
    out, reg = fixture_study(tmp_path)
    with pytest.raises(ValueError, match="Dataset content"):
        plots.qualitative_examples(
            out,
            {**reg, "dataset_content_sha256": "changed"},
            tmp_path / "figures",
            42,
            2,
            "cpu",
        )
    history = out / "runs/a0b0_uniform_standard_seed42/history.json"
    history.write_text(
        json.dumps(
            [
                {"epoch": 1, "val_iou_0p5": float("nan"), "train_loss": 1},
                {"epoch": 2, "val_iou_0p5": 0.8, "train_loss": 0.9},
            ]
        )
    )
    with pytest.raises(ValueError, match="Invalid val_iou"):
        plots.learning_curves(out, reg, tmp_path / "figures")
