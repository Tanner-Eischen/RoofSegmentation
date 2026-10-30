"""Correction comparisons retain finals, original references and conservative selection."""

import argparse
import json

import pytest
import torch

from roof_pipeline.adjudication import apply_proposals, build_review, seal_review
from roof_pipeline.annotation_audit import make_packet
from roof_pipeline.common import dataset_fingerprint
from roof_pipeline.correction_experiment import run_experiment, select_candidate
from roof_pipeline.train import train
from scripts.make_smoke_dataset import make_fixture


def candidates(incumbent=0.9, control=0.9, corrected=0.9):
    return [
        {"name": name, "validation_iou": score}
        for name, score in zip(
            ("incumbent", "control", "corrected"), (incumbent, control, corrected)
        )
    ]


def test_selection_is_conservative_and_ignores_training_or_test_scores():
    assert select_candidate(candidates())["name"] == "incumbent"
    assert select_candidate(candidates(corrected=0.90005))["name"] == "incumbent"
    assert (
        select_candidate(candidates(control=0.92, corrected=0.92))["name"] == "control"
    )
    values = candidates(control=0.91, corrected=0.92)
    values[0]["test_iou"] = 1
    values[0]["training_iou"] = 1
    assert select_candidate(values)["name"] == "corrected"


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, -0.1, 1.1])
def test_invalid_selection_rejected(value):
    with pytest.raises(ValueError):
        select_candidate(candidates(corrected=value))


def test_matched_workflow_runs_real_overlay_and_keeps_both_finals(
    tmp_path, monkeypatch
):
    torch.set_num_threads(2)
    data = make_fixture(tmp_path / "data", 4, 2, 1, 64)
    source_sha = dataset_fingerprint(data)
    initial = tmp_path / "initial"
    train(
        argparse.Namespace(
            data=data,
            run=initial,
            arch="tiny",
            pretrained=False,
            epochs=1,
            batch_size=2,
            size=64,
            workers=0,
            lr=0.001,
            weight_decay=0.0001,
            pos_weight=1.0,
            device="cpu",
            seed=42,
            patience=1,
            min_delta=0.0001,
            no_amp=True,
        )
    )
    checkpoint = initial / "best.pt"
    frozen = checkpoint.read_bytes()
    audit = tmp_path / "audit"
    make_packet(data, audit)
    draft = tmp_path / "draft"
    review = build_review(data, audit, checkpoint, draft, "fixture-draft")
    edit_id = next(c["sample_id"] for c in review["cases"] if c["split"] == "train")
    proposals = tmp_path / "proposal.json"
    rows = []
    for case in review["cases"]:
        rows.append(
            {
                "sample_id": case["sample_id"],
                "decision": "correct" if case["sample_id"] == edit_id else "keep",
                "rationale": "Synthetic AI review fixture",
                "edits": [
                    {
                        "action": "add",
                        "points": [[0, 0], [4, 0], [4, 4], [0, 4]],
                        "reason": "Synthetic missing foreground",
                    }
                ]
                if case["sample_id"] == edit_id
                else [],
            }
        )
    proposals.write_text(
        json.dumps({"reviewer_kind": "ai", "reviewer_id": "Fixture AI", "cases": rows})
    )
    apply_proposals(draft, proposals)
    sealed = tmp_path / "sealed"
    sealed_value = seal_review(
        draft, sealed, "fixture-provisional", allow_provisional=True
    )
    out = tmp_path / "comparison"
    from roof_pipeline import correction_experiment as module

    actual_diagnostics = module.validation_diagnostics

    def diagnostics_after_both_trainers(*args, **kwargs):
        assert (out / "control/final.pt").is_file()
        assert (out / "corrected/final.pt").is_file()
        return actual_diagnostics(*args, **kwargs)

    monkeypatch.setattr(
        module, "validation_diagnostics", diagnostics_after_both_trainers
    )
    args = argparse.Namespace(
        data=data,
        review=sealed,
        incumbent=checkpoint,
        out=out,
        epochs=1,
        batch_size=2,
        workers=0,
        device="cpu",
        seed=42,
        lr=0.0001,
        allow_provisional_review=True,
    )
    report = run_experiment(args)
    assert report["test_inference_performed"] is False
    assert report["training_review_provisional"] is True
    assert report["review_content_sha256"] == sealed_value["content_sha256"]
    assert report["effective_corrected_training_content_sha256"] != source_sha
    assert checkpoint.read_bytes() == frozen
    assert dataset_fingerprint(data) == source_sha
    for arm in ("control", "corrected"):
        assert (out / arm / "best.pt").is_file()
        state = torch.load(
            out / arm / "final.pt", map_location="cpu", weights_only=True
        )
        assert state["threshold"] == 0.5 and state["config"]["epoch"] == 1
        assert bool(state["config"]["training_review_content_sha256"]) == (
            arm == "corrected"
        )
        assert report["final_validation"][arm]["samples"] == 2
        assert len(report["local_training_diagnostics"][arm]["cases"]) == 1
    assert not (out / "test").exists()
    assert json.loads((out / "workflow_status.json").read_text())["phase"] == "complete"
    with pytest.raises(FileExistsError):
        run_experiment(args)
