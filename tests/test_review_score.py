"""Frozen review comparisons must separate label changes, ignore support and cohorts."""

import copy
import hashlib
import json

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from roof_pipeline import review_score


class HalfTileModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, image):
        self.calls += 1
        logits = image.new_full((1, 1, 32, 32), -10)
        logits[..., :16] = 10
        return logits


def setup_review(tmp_path, monkeypatch, cases, *, human=True, threshold=0.5):
    root = tmp_path / "review"
    root.mkdir()
    config = {
        "arch": "tiny",
        "image_size": 32,
        "normalization": "imagenet_rgb_v1",
        "dataset_content_sha256": "parent-data-fingerprint",
    }
    checkpoint = tmp_path / "frozen.pt"
    torch.save({"config": config, "threshold": threshold}, checkpoint)
    manifest = {
        "schema_version": "roof_review_v1",
        "version": "review-test-v1",
        "sealed": True,
        "content_sha256": "sealed-review-fingerprint",
        "source_dataset_content_sha256": "parent-data-fingerprint",
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "policy": {
            "status": "approved" if human else "draft",
            "reviewer_kind": "human" if human else "ai",
            "reviewer_id": "fixture-reviewer",
            "rules": {"building": "fixed roofed structure"},
        },
        "cases": [],
    }
    for index, spec in enumerate(cases):
        original = np.zeros((32, 32), dtype=np.uint8)
        original[:, : spec.get("original_width", 8)] = 255
        revised = np.zeros((32, 32), dtype=np.uint8)
        revised[:, : spec.get("review_width", 16)] = 255
        ignored = np.zeros((32, 32), dtype=np.uint8)
        ignored[:, : spec.get("ignored_width", 0)] = 255
        fields = {
            "image": np.zeros((32, 32, 3), dtype=np.uint8),
            "original_mask": original,
            "mask": revised,
            "ignore_mask": ignored,
        }
        case = {
            "sample_id": f"case_{index}",
            "split": spec.get("split", "val"),
            "group": f"site-{index}",
            "cohort": spec.get("cohort", "representative"),
            "stratum": "at_least_10pct",
            "inclusion_probability": 0.25,
            "decision": spec.get("decision", "correct"),
            "reviewer_kind": "human" if human else "ai",
            "reviewer_id": "fixture-reviewer",
            "rationale": "Synthetic unit-test decision",
        }
        for field, array in fields.items():
            path = root / f"{index}_{field}.png"
            Image.fromarray(array).save(path)
            case[field] = path.name
            case[field + "_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest["cases"].append(case)
    model = HalfTileModel()
    monkeypatch.setattr(
        review_score, "_load_review", lambda path: copy.deepcopy(manifest)
    )
    monkeypatch.setattr(
        review_score, "load_model", lambda path, device: (model, config)
    )
    return root, checkpoint, manifest, model


def test_known_predictions_separate_annotation_change_from_model_change(
    tmp_path, monkeypatch
):
    root, checkpoint, _, model = setup_review(tmp_path, monkeypatch, [{}])
    result = review_score.score_review(root, checkpoint, tmp_path / "score")
    values = result["cohorts"]["representative"]
    assert result["status"] == "human_adjudicated"
    assert model.calls == 1
    assert values["paired_original"]["iou"] == 0.5
    assert values["paired_review"]["iou"] == 1
    assert values["paired_iou_change"] == 0.5
    assert values["changed_evaluated_pixels"] == 256
    assert values["paired_original"]["fp"] == 256
    assert values["paired_review"]["fp"] == 0
    assert result["test_inference_performed"] is False
    assert len(list((tmp_path / "score/predictions").glob("*_mask.png"))) == 1


def test_ignore_uses_identical_support_for_original_and_review(tmp_path, monkeypatch):
    root, checkpoint, _, _ = setup_review(tmp_path, monkeypatch, [{"ignored_width": 4}])
    result = review_score.score_review(root, checkpoint, tmp_path / "score")
    values = result["cohorts"]["representative"]
    assert values["original_all_pixels"]["iou"] == 0.5
    assert values["paired_original"]["iou"] == pytest.approx(1 / 3)
    assert values["paired_review"]["iou"] == 1
    assert values["ignored_pixels"] == 128
    assert values["evaluated_paired_pixels"] == 896
    for key in ("paired_original", "paired_review"):
        assert sum(values[key][count] for count in ("tp", "fp", "fn", "tn")) == 896


def test_targeted_cohort_and_training_are_not_mixed_or_inferred(tmp_path, monkeypatch):
    root, checkpoint, manifest, model = setup_review(
        tmp_path,
        monkeypatch,
        [
            {},
            {"cohort": "targeted", "original_width": 16, "decision": "keep"},
            {"split": "train"},
        ],
    )
    for field in ("image", "mask", "original_mask", "ignore_mask"):
        (root / manifest["cases"][2][field]).write_bytes(b"not decoded")
    result = review_score.score_review(root, checkpoint, tmp_path / "score")
    assert model.calls == 2
    assert result["training_cases_not_inferred"] == 1
    assert result["cohorts"]["targeted"]["paired_original"]["iou"] == 1
    assert result["cohorts"]["representative"]["paired_original"]["iou"] == 0.5
    assert "aggregate" not in result


def test_pending_cases_never_become_corrected_evidence(tmp_path, monkeypatch):
    root, checkpoint, _, model = setup_review(
        tmp_path, monkeypatch, [{"decision": "pending"}], human=False
    )
    with pytest.raises(ValueError, match="human"):
        review_score.score_review(root, checkpoint, tmp_path / "rejected")
    result = review_score.score_review(
        root, checkpoint, tmp_path / "score", allow_provisional=True
    )
    assert model.calls == 1
    values = result["cohorts"]["representative"]
    assert result["status"] == "provisional"
    assert values["original_all_pixels"]["iou"] == 0.5
    assert values["pending_tiles"] == 1
    assert values["paired_tiles"] == 0
    assert values["paired_review"] is None
    assert values["paired_iou_change"] is None


def test_fully_ignored_case_has_no_fabricated_perfect_score(tmp_path, monkeypatch):
    root, checkpoint, _, _ = setup_review(
        tmp_path,
        monkeypatch,
        [{"ignored_width": 32, "review_width": 8, "decision": "uncertain"}],
    )
    result = review_score.score_review(root, checkpoint, tmp_path / "score")
    values = result["cohorts"]["representative"]
    assert values["fully_ignored_tiles"] == 1
    assert values["paired_tiles"] == 0
    assert values["paired_original"] is None
    assert values["paired_review"] is None


@pytest.mark.parametrize("decision", ["keep", "uncertain"])
def test_keep_or_uncertain_case_cannot_silently_change_masks(
    tmp_path, monkeypatch, decision
):
    root, checkpoint, _, _ = setup_review(
        tmp_path, monkeypatch, [{"decision": decision}]
    )
    with pytest.raises(ValueError, match="cannot change"):
        review_score.score_review(root, checkpoint, tmp_path / "score")


def test_checkpoint_hash_mismatch_rejected_before_inference(tmp_path, monkeypatch):
    root, checkpoint, manifest, model = setup_review(tmp_path, monkeypatch, [{}])
    manifest["checkpoint_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="Checkpoint SHA"):
        review_score.score_review(root, checkpoint, tmp_path / "score")
    assert model.calls == 0


def test_parent_fingerprint_mismatch_rejected_before_inference(tmp_path, monkeypatch):
    root, checkpoint, manifest, model = setup_review(tmp_path, monkeypatch, [{}])
    manifest["source_dataset_content_sha256"] = "wrong parent"
    with pytest.raises(ValueError, match="fingerprint"):
        review_score.score_review(root, checkpoint, tmp_path / "score")
    assert model.calls == 0


def test_test_split_is_rejected_before_any_inference(tmp_path, monkeypatch):
    root, checkpoint, _, model = setup_review(
        tmp_path, monkeypatch, [{}, {"split": "test"}]
    )
    with pytest.raises(ValueError, match="train/validation"):
        review_score.score_review(root, checkpoint, tmp_path / "score")
    assert model.calls == 0


def test_frozen_threshold_is_used_without_tuning(tmp_path, monkeypatch):
    root, checkpoint, _, model = setup_review(
        tmp_path, monkeypatch, [{}], threshold=0.8
    )

    def low_confidence(image):
        model.calls += 1
        logits = image.new_full((1, 1, 32, 32), -10)
        logits[..., :16] = 1  # sigmoid(1)=0.731; below frozen threshold 0.8.
        return logits

    monkeypatch.setattr(model, "forward", low_confidence)
    result = review_score.score_review(root, checkpoint, tmp_path / "score")
    assert result["threshold_frozen_from_checkpoint"] == 0.8
    assert result["cohorts"]["representative"]["paired_review"]["tp"] == 0
    assert result["cohorts"]["representative"]["paired_review"]["iou"] == 0
    assert model.calls == 1


def test_existing_output_and_output_inside_review_are_rejected(tmp_path, monkeypatch):
    root, checkpoint, _, _ = setup_review(tmp_path, monkeypatch, [{}])
    with pytest.raises(ValueError, match="outside"):
        review_score.score_review(root, checkpoint, root / "score")
    out = tmp_path / "existing"
    out.mkdir()
    (out / "report.md").write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        review_score.score_review(root, checkpoint, out)
    assert (out / "report.md").read_text() == "keep"


def test_native_grid_mismatch_is_rejected(tmp_path, monkeypatch):
    root, checkpoint, manifest, model = setup_review(tmp_path, monkeypatch, [{}])
    path = root / manifest["cases"][0]["image"]
    Image.new("RGB", (64, 64)).save(path)
    with pytest.raises(ValueError, match="native"):
        review_score.score_review(root, checkpoint, tmp_path / "score")
    assert model.calls == 0


def test_output_json_is_machine_readable_and_retains_empty_cohort(
    tmp_path, monkeypatch
):
    root, checkpoint, _, _ = setup_review(tmp_path, monkeypatch, [{}])
    out = tmp_path / "score"
    result = review_score.score_review(root, checkpoint, out, save_predictions=False)
    assert json.loads((out / "comparison.json").read_text()) == result
    assert result["cohorts"]["targeted"]["original_all_pixels"] is None
    assert not (out / "predictions").exists()


def test_unlocalized_uncertainty_is_original_only(tmp_path, monkeypatch):
    root, checkpoint, _, _ = setup_review(
        tmp_path,
        monkeypatch,
        [{"review_width": 8, "decision": "uncertain"}],
    )
    result = review_score.score_review(root, checkpoint, tmp_path / "score")
    values = result["cohorts"]["representative"]
    assert values["unresolved_tiles"] == 1
    assert values["original_all_pixels"]["iou"] == 0.5
    assert values["paired_tiles"] == 0
    assert values["paired_review"] is None
    import csv

    with (tmp_path / "score/val_per_image.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        row = next(csv.DictReader(stream))
    assert row["paired_exclusion_reason"] == "unlocalized_uncertainty"


def test_localized_uncertainty_pairs_only_known_support(tmp_path, monkeypatch):
    root, checkpoint, _, _ = setup_review(
        tmp_path,
        monkeypatch,
        [{"review_width": 8, "decision": "uncertain", "ignored_width": 4}],
    )
    result = review_score.score_review(root, checkpoint, tmp_path / "score")
    values = result["cohorts"]["representative"]
    assert values["unresolved_tiles"] == 0
    assert values["paired_tiles"] == 1
    assert values["evaluated_paired_pixels"] == 896
    assert values["paired_original"] == values["paired_review"]


def test_real_sealed_review_and_checkpoint_integrate_without_fingerprint_bypass(
    tmp_path,
):
    from roof_pipeline.adjudication import apply_proposals, build_review, seal_review
    from roof_pipeline.annotation_audit import make_packet
    from roof_pipeline.common import dataset_fingerprint
    from roof_pipeline.models import create_model
    from scripts.make_smoke_dataset import make_fixture

    torch.set_num_threads(2)
    data = make_fixture(tmp_path / "data", 2, 2, 1, 64)
    audit = tmp_path / "audit"
    make_packet(data, audit, per_stratum=1)
    checkpoint = tmp_path / "frozen.pt"
    torch.save(
        {
            "config": {
                "arch": "tiny",
                "image_size": 64,
                "normalization": "imagenet_rgb_v1",
                "dataset_content_sha256": dataset_fingerprint(data),
            },
            "state_dict": create_model("tiny").state_dict(),
            "threshold": 0.5,
        },
        checkpoint,
    )
    checkpoint_before = checkpoint.read_bytes()
    draft = tmp_path / "draft"
    review = build_review(data, audit, checkpoint, draft, "synthetic-draft")
    proposals = {"reviewer_kind": "ai", "reviewer_id": "test-proposer", "cases": []}
    changed_val = False
    for case in review["cases"]:
        decision, edits = "keep", []
        if case["split"] == "val" and not changed_val:
            with Image.open(draft / case["mask"]) as image:
                positions = np.argwhere(np.asarray(image) > 0)
            y, x = map(int, positions[0])
            edits = [
                {
                    "action": "remove",
                    "points": [[x, y], [x + 1, y], [x + 1, y + 1], [x, y + 1]],
                    "reason": "Synthetic correction for integration verification",
                }
            ]
            decision, changed_val = "correct", True
        proposals["cases"].append(
            {
                "sample_id": case["sample_id"],
                "decision": decision,
                "rationale": "Synthetic AI proposal; never independent expert truth",
                "edits": edits,
            }
        )
    proposal_path = tmp_path / "proposals.json"
    proposal_path.write_text(json.dumps(proposals), encoding="utf-8")
    apply_proposals(draft, proposal_path)
    sealed = tmp_path / "sealed"
    seal_review(draft, sealed, "synthetic-sealed", allow_provisional=True)
    result = review_score.score_review(
        sealed, checkpoint, tmp_path / "score", allow_provisional=True
    )
    assert changed_val
    assert result["status"] == "provisional"
    assert result["cohorts"]["representative"]["changed_evaluated_pixels"] > 0
    assert result["cohorts"]["representative"]["paired_tiles"] > 0
    assert checkpoint.read_bytes() == checkpoint_before
