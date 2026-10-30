"""Correction versions retain frozen identities and never imply expert review."""

import csv
import json
import os

import numpy as np
import pytest
import torch
from PIL import Image

from roof_pipeline.adjudication import (
    apply_proposals,
    build_review,
    export_decisions,
    import_decisions,
    load_review,
    seal_review,
    sha256,
)
from roof_pipeline.annotation_audit import make_packet
from roof_pipeline.common import dataset_fingerprint
from roof_pipeline.models import create_model
from scripts.make_smoke_dataset import make_fixture


@pytest.fixture
def review_fixture(tmp_path):
    data = make_fixture(tmp_path / "source", 3, 2, 1, 64)
    # Source fingerprint may hash test bytes; the review must never decode them.
    (data / "images/synthetic_test_000.png").write_bytes(b"no test image decoding")
    audit = tmp_path / "audit"
    make_packet(data, audit, per_stratum=4)
    checkpoint = tmp_path / "frozen.pt"
    torch.save(
        {
            "state_dict": create_model("tiny").state_dict(),
            "config": {
                "arch": "tiny",
                "image_size": 64,
                "normalization": "imagenet_rgb_v1",
                "dataset_content_sha256": dataset_fingerprint(data),
            },
            "threshold": 0.5,
        },
        checkpoint,
    )
    root = tmp_path / "draft"
    value = build_review(data, audit, checkpoint, root, "test-draft")
    return data, audit, checkpoint, root, value


def save_review(root, value):
    (root / "review.json").write_text(json.dumps(value), encoding="utf-8")


def proposal_file(root, cases):
    path = root.parent / "proposals.json"
    path.write_text(
        json.dumps(
            {
                "reviewer_kind": "ai",
                "reviewer_id": "AI provisional reviewer",
                "cases": cases,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_build_preserves_originals_and_partition_without_test_decode(review_fixture):
    data, audit, checkpoint, root, value = review_fixture
    assert len(value["cases"]) == 5
    assert {c["split"] for c in value["cases"]} == {"train", "val"}
    assert all(c["decision"] == "pending" for c in value["cases"])
    assert value["policy"]["status"] == "draft"
    assert all(
        (root / "panels" / (c["sample_id"] + ".png")).is_file() for c in value["cases"]
    )
    for case in value["cases"]:
        assert sha256(root / case["original_mask"]) == sha256(
            data / "masks" / (case["sample_id"] + ".png")
        )
    with pytest.raises(FileExistsError):
        build_review(data, audit, checkpoint, root, "test-draft")


@pytest.mark.parametrize(
    "key,value",
    [("split", "val"), ("cohort", "targeted"), ("inclusion_probability", 0.1)],
)
def test_draft_cannot_reassign_locked_selection(review_fixture, key, value):
    *_, root, review = review_fixture
    review["cases"][0][key] = value
    save_review(root, review)
    with pytest.raises(ValueError, match="identity changed"):
        load_review(root)


def test_proposal_changes_only_versioned_mask_and_requires_provisional_seal(
    review_fixture,
):
    data, _, _, root, review = review_fixture
    before = dataset_fingerprint(data)
    case = review["cases"][0]
    original_hash = sha256(root / case["original_mask"])
    proposals = proposal_file(
        root,
        [
            {
                "sample_id": case["sample_id"],
                "decision": "correct",
                "rationale": "Proposed omitted foreground, expert confirmation pending",
                "edits": [
                    {
                        "action": "add",
                        "points": [[0, 0], [4, 0], [4, 4], [0, 4]],
                        "reason": "Synthetic roof region",
                    }
                ],
            }
        ],
    )
    updated = apply_proposals(root, proposals)
    assert updated["cases"][0]["reviewer_kind"] == "ai"
    assert sha256(root / case["original_mask"]) == original_hash
    assert sha256(root / case["mask"]) != original_hash
    assert dataset_fingerprint(data) == before
    with pytest.raises(ValueError, match="Human adjudication"):
        seal_review(root, root.parent / "confirmed", "test-confirmed")
    sealed = root.parent / "provisional"
    result = seal_review(root, sealed, "test-provisional", allow_provisional=True)
    assert result["sealed"] and result["provisional"]
    assert (
        load_review(sealed, require_sealed=True)["content_sha256"]
        == result["content_sha256"]
    )
    with pytest.raises(ValueError, match="cannot be edited"):
        apply_proposals(sealed, proposals)
    Image.new("L", (64, 64), 255).save(sealed / case["mask"])
    with pytest.raises(ValueError, match="hash mismatch"):
        load_review(sealed, require_sealed=True)


def test_human_approval_can_seal_but_cannot_be_inferred_from_ai(review_fixture):
    *_, root, value = review_fixture
    value["policy"].update(status="approved", reviewer_kind="ai", reviewer_id="AI")
    save_review(root, value)
    with pytest.raises(ValueError, match="identified human"):
        load_review(root)
    value["policy"].update(reviewer_kind="human", reviewer_id="Fixture human")
    for case in value["cases"]:
        case.update(
            decision="keep",
            reviewer_kind="human",
            reviewer_id="Fixture human",
            rationale="Synthetic verified fixture",
        )
    save_review(root, value)
    sealed = seal_review(root, root.parent / "human", "human-v1")
    assert sealed["provisional"] is False


@pytest.mark.parametrize(
    "points",
    [
        [[0, 0], [80, 0], [0, 4]],
        [[0, 0], [4, 4], [0, 4], [4, 0]],
        [[True, 0], [4, 0], [0, 4]],
    ],
)
def test_invalid_polygon_is_rejected_without_partial_changes(review_fixture, points):
    *_, root, review = review_fixture
    before = sha256(root / "review.json")
    first, second = review["cases"][:2]
    path = proposal_file(
        root,
        [
            {
                "sample_id": first["sample_id"],
                "decision": "correct",
                "rationale": "Valid proposal",
                "edits": [
                    {
                        "action": "add",
                        "points": [[0, 0], [4, 0], [0, 4]],
                        "reason": "fixture",
                    }
                ],
            },
            {
                "sample_id": second["sample_id"],
                "decision": "correct",
                "rationale": "Invalid proposal",
                "edits": [{"action": "add", "points": points, "reason": "fixture"}],
            },
        ],
    )
    with pytest.raises(ValueError, match="polygon|Polygon"):
        apply_proposals(root, path)
    assert sha256(root / "review.json") == before
    assert sha256(root / first["mask"]) == first["mask_sha256"]


def test_uncertain_regions_use_separate_ignore_mask(review_fixture):
    *_, root, review = review_fixture
    case = review["cases"][0]
    path = proposal_file(
        root,
        [
            {
                "sample_id": case["sample_id"],
                "decision": "uncertain",
                "rationale": "Cannot determine identity",
                "edits": [
                    {
                        "action": "ignore",
                        "points": [[0, 0], [4, 0], [4, 4], [0, 4]],
                        "reason": "Unresolved object",
                    }
                ],
            }
        ],
    )
    apply_proposals(root, path)
    assert sha256(root / case["mask"]) == case["mask_sha256"]
    with Image.open(root / case["ignore_mask"]) as mask:
        assert np.count_nonzero(np.asarray(mask)) == 25


def test_seal_refuses_nested_output(review_fixture):
    *_, root, _ = review_fixture
    with pytest.raises(FileExistsError, match="independent"):
        seal_review(root, root / "nested", "nested", allow_provisional=True)


def test_original_hash_and_sealed_metadata_are_checked(review_fixture):
    *_, root, review = review_fixture
    sealed = root.parent / "sealed"
    seal_review(root, sealed, "sealed-v1", allow_provisional=True)
    value = json.loads((sealed / "review.json").read_text())
    value["version"] = "tampered-version"
    save_review(sealed, value)
    with pytest.raises(ValueError, match="metadata/content"):
        load_review(sealed)
    case = review["cases"][0]
    Image.new("L", (64, 64), 255).save(root / case["original_mask"])
    with pytest.raises(ValueError, match="changes disagree|hash mismatch"):
        load_review(root)


def human_worksheet(root, *, correct_id=None):
    path = root.parent / "human.csv"
    export_decisions(root, path)
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fields, rows = reader.fieldnames, list(reader)
    for row in rows:
        row.update(
            decision="correct" if row["sample_id"] == correct_id else "keep",
            reviewer_id="Synthetic human reviewer",
            rationale="Fixture adjudication",
        )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_import_human_decisions_handles_manually_corrected_draft(review_fixture):
    *_, root, review = review_fixture
    case = review["cases"][0]
    mask = np.asarray(Image.open(root / case["mask"])).copy()
    mask[:5, :5] = 255
    Image.fromarray(mask).save(root / case["mask"])
    # Manual pixel edits can precede importing their corresponding decisions.
    with pytest.raises(ValueError, match="changes disagree"):
        load_review(root)
    # Export before edits is the usual flow; allow an in-memory fixture worksheet.
    original = load_review(root, allow_draft_edits=True)
    save_review(root, original)
    path = root.parent / "human.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "sample_id",
                "split",
                "cohort",
                "decision",
                "reviewer_id",
                "rationale",
            ),
        )
        writer.writeheader()
        for item in review["cases"]:
            writer.writerow(
                {
                    "sample_id": item["sample_id"],
                    "split": item["split"],
                    "cohort": item["cohort"],
                    "decision": "correct" if item is case else "keep",
                    "reviewer_id": "Synthetic human reviewer",
                    "rationale": "Fixture adjudication",
                }
            )
    value = import_decisions(root, path, policy_approved_by="Synthetic policy approver")
    assert value["policy"]["status"] == "approved"
    assert all(c["reviewer_kind"] == "human" for c in value["cases"])
    assert (
        seal_review(root, root.parent / "confirmed", "confirmed-v1")["provisional"]
        is False
    )
    with pytest.raises(ValueError, match="cannot replace"):
        apply_proposals(root, proposal_file(root, []))


def test_human_keep_rejects_proposal_by_restoring_original(review_fixture):
    *_, root, review = review_fixture
    case = review["cases"][0]
    path = proposal_file(
        root,
        [
            {
                "sample_id": case["sample_id"],
                "decision": "correct",
                "rationale": "AI proposal",
                "edits": [
                    {
                        "action": "add",
                        "points": [[0, 0], [4, 0], [0, 4]],
                        "reason": "fixture",
                    }
                ],
            }
        ],
    )
    apply_proposals(root, path)
    decisions = human_worksheet(root)
    result = import_decisions(root, decisions, policy_approved_by="Synthetic approver")
    assert result["cases"][0]["decision"] == "keep"
    assert sha256(root / case["mask"]) == sha256(root / case["original_mask"])
    assert result["cases"][0]["previous_proposal"]["decision"] == "correct"


def test_empty_worksheet_is_not_approval_and_is_not_overwritten(review_fixture):
    *_, root, _ = review_fixture
    path = root.parent / "worksheet.csv"
    export_decisions(root, path)
    with pytest.raises(FileExistsError, match="already exists"):
        export_decisions(root, path)
    before = sha256(root / "review.json")
    with pytest.raises(ValueError, match="Every human decision"):
        import_decisions(root, path, policy_approved_by="Fixture approver")
    assert sha256(root / "review.json") == before


def test_editable_mask_cannot_alias_original_file(review_fixture):
    *_, root, review = review_fixture
    case = review["cases"][0]
    target = root / case["mask"]
    target.unlink()
    os.link(root / case["original_mask"], target)
    with pytest.raises(ValueError, match="distinct files"):
        load_review(root)


def test_editable_mask_cannot_alias_external_source_dataset(review_fixture):
    data, _, _, root, review = review_fixture
    case = review["cases"][0]
    source_mask = data / "masks" / (case["sample_id"] + ".png")
    before = sha256(source_mask)
    target = root / case["mask"]
    target.unlink()
    os.link(source_mask, target)
    path = proposal_file(
        root,
        [
            {
                "sample_id": case["sample_id"],
                "decision": "correct",
                "rationale": "Candidate",
                "edits": [
                    {
                        "action": "add",
                        "points": [[0, 0], [4, 0], [0, 4]],
                        "reason": "fixture",
                    }
                ],
            }
        ],
    )
    with pytest.raises(ValueError, match="hard-linked"):
        apply_proposals(root, path)
    assert sha256(source_mask) == before
