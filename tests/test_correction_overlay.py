"""Reviewed mask interventions preserve data/selection isolation and provenance."""

import argparse
import json
import random

import numpy as np
import pytest
import torch
from PIL import Image

from roof_pipeline.adjudication import (
    apply_proposals,
    build_review,
    seal_review,
    sha256,
)
from roof_pipeline.annotation_audit import make_packet
from roof_pipeline.common import dataset_fingerprint
from roof_pipeline.correction_overlay import (
    load_training_overlay,
    source_training_metadata,
)
from roof_pipeline.dataset import MEAN, STD, RoofDataset
from roof_pipeline.models import create_model, load_model
from roof_pipeline.train import train
from scripts.make_smoke_dataset import make_fixture


def make_review(tmp_path, *, corrected_split="train", ignored=False):
    torch.set_num_threads(2)
    data = make_fixture(tmp_path / "data", 4, 2, 2, 64)
    dataset_sha = dataset_fingerprint(data)
    checkpoint = tmp_path / "initial.pt"
    torch.save(
        {
            "state_dict": create_model("tiny").state_dict(),
            "config": {
                "arch": "tiny",
                "image_size": 64,
                "normalization": "imagenet_rgb_v1",
                "dataset_content_sha256": dataset_sha,
                "epoch": 3,
            },
            "threshold": 0.5,
        },
        checkpoint,
    )
    audit, draft = tmp_path / "audit", tmp_path / "draft"
    make_packet(data, audit)
    review = build_review(data, audit, checkpoint, draft, "fixture-draft")
    selected = next((c for c in review["cases"] if c["split"] == corrected_split), None)
    ignored_case = next(c for c in review["cases"] if c["split"] == "val")
    proposals = []
    for case in review["cases"]:
        edits, decision = [], "keep"
        if selected is not None and case["sample_id"] == selected["sample_id"]:
            decision = "correct"
            edits.append(
                {
                    "action": "add",
                    "points": [[0, 0], [3, 0], [3, 3], [0, 3]],
                    "reason": "Synthetic omitted foreground",
                }
            )
        if ignored and case["sample_id"] == ignored_case["sample_id"]:
            decision = "uncertain"
            edits = [
                {
                    "action": "ignore",
                    "points": [[0, 0], [3, 0], [3, 3], [0, 3]],
                    "reason": "Synthetic uncertainty",
                }
            ]
        proposals.append(
            {
                "sample_id": case["sample_id"],
                "decision": decision,
                "rationale": "AI-provisional synthetic fixture decision",
                "edits": edits,
            }
        )
    path = tmp_path / "proposals.json"
    path.write_text(
        json.dumps(
            {"reviewer_kind": "ai", "reviewer_id": "Fixture AI", "cases": proposals}
        ),
        encoding="utf-8",
    )
    apply_proposals(draft, path)
    sealed = tmp_path / "sealed"
    review = seal_review(draft, sealed, "fixture-provisional", allow_provisional=True)
    return data, checkpoint, dataset_sha, sealed, review


@pytest.fixture
def overlay_fixture(tmp_path):
    return make_review(tmp_path)


def load_fixture(fixture, *, allow=True):
    data, checkpoint, dataset_sha, sealed, _ = fixture
    return load_training_overlay(
        sealed, data, dataset_sha, checkpoint, allow_provisional=allow
    )


def test_only_declared_training_mask_changes_and_source_is_unchanged(overlay_fixture):
    data, checkpoint, dataset_sha, sealed, review = overlay_fixture
    original_sha = sha256(checkpoint)
    overrides, metadata = load_fixture(overlay_fixture)
    assert len(overrides) == 1
    ident = next(iter(overrides))
    assert "train" in ident
    assert overrides[ident] == (sealed / "masks" / (ident + ".png")).resolve()
    assert metadata["training_review_mask_changes"] == [
        {
            "sample_id": ident,
            "split": "train",
            "original_mask_sha256": sha256(data / "masks" / (ident + ".png")),
            "mask_sha256": sha256(overrides[ident]),
            "changed_pixels": 16,
            "added_building_pixels": 16,
            "removed_building_pixels": 0,
        }
    ]
    assert metadata["training_review_provisional"] is True
    assert metadata["training_review_content_sha256"] == review["content_sha256"]
    assert metadata["effective_training_content_sha256"] != dataset_sha
    assert (
        source_training_metadata(dataset_sha)["effective_training_content_sha256"]
        == dataset_sha
    )
    assert dataset_fingerprint(data) == dataset_sha
    assert sha256(checkpoint) == original_sha
    # Reading with/without an intervention leaves RGB and all untouched targets identical.
    default = RoofDataset(data, "train", 64)
    intervention = RoofDataset(data, "train", 64, mask_overrides=overrides)
    for index, case in enumerate(default.rows):
        x, original = default[index]
        x2, proposed = intervention[index]
        torch.testing.assert_close(x, x2, rtol=0, atol=0)
        if case["sample_id"] == ident:
            assert int((original != proposed).sum()) == 16
        else:
            torch.testing.assert_close(original, proposed, rtol=0, atol=0)


def test_overlay_loading_preserves_all_rng_streams(overlay_fixture):
    random.seed(123)
    np.random.seed(456)
    torch.manual_seed(789)
    before_python, before_numpy = random.getstate(), np.random.get_state()
    before_torch = torch.get_rng_state().clone()
    load_fixture(overlay_fixture)
    assert random.getstate() == before_python
    after_numpy = np.random.get_state()
    assert before_numpy[0] == after_numpy[0]
    np.testing.assert_array_equal(before_numpy[1], after_numpy[1])
    assert before_numpy[2:] == after_numpy[2:]
    torch.testing.assert_close(torch.get_rng_state(), before_torch, rtol=0, atol=0)


def test_ai_overlay_needs_explicit_flag_and_frozen_checkpoint(overlay_fixture):
    with pytest.raises(ValueError, match="allow-provisional-review"):
        load_fixture(overlay_fixture, allow=False)
    data, _, dataset_sha, sealed, _ = overlay_fixture
    with pytest.raises(ValueError, match="requires its frozen"):
        load_training_overlay(sealed, data, dataset_sha, None, allow_provisional=True)


@pytest.mark.parametrize(
    "change", ["parent", "manifest", "image", "original_mask", "checkpoint"]
)
def test_overlay_rejects_changed_source_or_checkpoint(overlay_fixture, change):
    data, checkpoint, dataset_sha, sealed, review = overlay_fixture
    if change == "parent":
        dataset_sha = "0" * 64
    elif change == "manifest":
        with (data / "manifest.csv").open("a", encoding="utf-8") as stream:
            stream.write("\n")
    elif change == "image":
        Image.new("RGB", (64, 64), "red").save(data / review["cases"][0]["image"])
    elif change == "original_mask":
        Image.new("L", (64, 64), 0).save(
            data / "masks" / (review["cases"][0]["sample_id"] + ".png")
        )
    else:
        with checkpoint.open("ab") as stream:
            stream.write(b"changed checkpoint bytes")
    with pytest.raises(ValueError, match="differs"):
        load_training_overlay(
            sealed, data, dataset_sha, checkpoint, allow_provisional=True
        )


@pytest.mark.parametrize("split", ["val", None])
def test_nontraining_or_empty_corrections_rejected(tmp_path, split):
    fixture = make_review(tmp_path, corrected_split=split)
    with pytest.raises(ValueError, match="Only declared training|no declared training"):
        load_fixture(fixture)


def test_ignored_pixels_are_not_silently_used_for_training(tmp_path):
    fixture = make_review(tmp_path, ignored=True)
    with pytest.raises(ValueError, match="do not support ignored pixels"):
        load_fixture(fixture)


@pytest.mark.parametrize("split", ["val", "test"])
def test_dataset_rejects_nontraining_overrides(overlay_fixture, split):
    data, *_ = overlay_fixture
    overrides, _ = load_fixture(overlay_fixture)
    with pytest.raises(ValueError, match="restricted to the training"):
        RoofDataset(data, split, 64, mask_overrides=overrides)
    with pytest.raises(ValueError, match="unknown training"):
        RoofDataset(
            data,
            "train",
            64,
            mask_overrides={"unknown": next(iter(overrides.values()))},
        )


def test_override_geometry_tracks_augmentation_and_native_dimensions(tmp_path):
    data = make_fixture(tmp_path / "data", 1, 1, 1, 64)
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[7:26, 13:44] = 255
    Image.fromarray(np.repeat(mask[..., None], 3, axis=2)).save(
        data / "images/synthetic_train_000.png"
    )
    override = tmp_path / "override.png"
    Image.fromarray(mask).save(override)
    dataset = RoofDataset(
        data,
        "train",
        64,
        augment=True,
        mask_overrides={"synthetic_train_000": override},
    )
    for seed in range(12):
        random.seed(seed)
        image, target = dataset[0]
        red = image[0].numpy() * STD[0, 0, 0] + MEAN[0, 0, 0]
        np.testing.assert_array_equal(
            red > (red.max() + red.min()) / 2, target[0].numpy() > 0
        )
    Image.new("L", (32, 32), 0).save(override)
    with pytest.raises(ValueError, match="Mask/image size mismatch"):
        dataset[0]


def training_args(fixture, run):
    data, checkpoint, _, sealed, _ = fixture
    return argparse.Namespace(
        data=data,
        run=run,
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
        patience=2,
        min_delta=1.0,
        no_amp=True,
        init_checkpoint=checkpoint,
        sampling_report=None,
        photometric="standard",
        training_review=sealed,
        allow_provisional_review=True,
    )


def test_training_saves_final_endpoint_even_when_initial_candidate_retained(
    overlay_fixture, tmp_path
):
    data, checkpoint, dataset_sha, _, _ = overlay_fixture
    original_sha = sha256(checkpoint)
    args = training_args(overlay_fixture, tmp_path / "run")
    initial = torch.load(checkpoint, map_location="cpu", weights_only=True)
    summary = train(args)
    final = torch.load(args.run / "final.pt", map_location="cpu", weights_only=True)
    best = torch.load(args.run / "best.pt", map_location="cpu", weights_only=True)
    assert summary["best_epoch"] == best["config"]["epoch"] == 0
    assert summary["final_epoch"] == final["config"]["epoch"] == 1
    assert final["threshold"] == summary["final_threshold"] == 0.5
    assert final["val_iou_at_0p5"] == summary["final_val_iou_at_0p5"]
    assert not torch.equal(
        final["state_dict"]["head.weight"], initial["state_dict"]["head.weight"]
    )
    assert final["config"]["dataset_content_sha256"] == dataset_sha
    assert final["config"]["effective_training_content_sha256"] != dataset_sha
    assert final["config"]["training_review_provisional"] is True
    model, _ = load_model(args.run / "final.pt")
    assert model(torch.zeros(1, 3, 64, 64)).shape == (1, 1, 64, 64)
    assert dataset_fingerprint(data) == dataset_sha
    assert sha256(checkpoint) == original_sha


def test_training_rechecks_overlay_before_saving_endpoint(
    overlay_fixture, tmp_path, monkeypatch
):
    from roof_pipeline import train as module

    args = training_args(overlay_fixture, tmp_path / "run")
    _, _, _, sealed, review = overlay_fixture
    changed = next(c for c in review["cases"] if c["decision"] == "correct")
    evaluate = module.evaluate_loader
    calls = 0

    def mutate_after_last_validation(*args, **kwargs):
        nonlocal calls
        result = evaluate(*args, **kwargs)
        calls += 1
        if calls == 2:
            Image.new("L", (64, 64), 0).save(sealed / changed["mask"])
        return result

    monkeypatch.setattr(module, "evaluate_loader", mutate_after_last_validation)
    with pytest.raises(ValueError, match="changes disagree|hash mismatch"):
        train(args)
    assert not (args.run / "final.pt").exists()
