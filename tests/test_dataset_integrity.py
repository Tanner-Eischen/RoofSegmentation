"""Regression checks for freezing paired data and detecting cross-split leakage."""

import hashlib
import shutil

import numpy as np
import pytest
import torch
from PIL import Image

from roof_pipeline.common import dataset_audit, dataset_fingerprint, load_binary_mask
from roof_pipeline.dataset import RoofDataset
from roof_pipeline.evaluate import evaluate
from roof_pipeline.models import create_model
from scripts.make_smoke_dataset import make_fixture


@pytest.mark.parametrize("field", ["images", "masks", "metadata"])
def test_evaluation_rejects_changed_paired_content(tmp_path, field):
    data = make_fixture(tmp_path / "data", 2, 1, 1, 64)
    checkpoint = tmp_path / "frozen.pt"
    original_manifest = (data / "manifest.csv").read_bytes()
    fingerprint = dataset_fingerprint(data)
    torch.save(
        {
            "state_dict": create_model("tiny").state_dict(),
            "config": {
                "arch": "tiny",
                "image_size": 64,
                "dataset_manifest_sha256": hashlib.sha256(
                    original_manifest
                ).hexdigest(),
                "dataset_content_sha256": fingerprint,
            },
            "threshold": 0.5,
        },
        checkpoint,
    )
    if field == "metadata":
        (data / "dataset_meta.json").write_text('{"dataset": "different provenance"}')
    else:
        path = data / field / "synthetic_test_000.png"
        with Image.open(path) as source:
            changed = np.array(source)
        changed[0, 0] = 255 - changed[0, 0]
        Image.fromarray(changed).save(path)
    assert (data / "manifest.csv").read_bytes() == original_manifest
    assert dataset_fingerprint(data) != fingerprint
    with pytest.raises(ValueError, match="Dataset content SHA differs"):
        evaluate(checkpoint, data, tmp_path / "evaluation")
    assert not (tmp_path / "evaluation").exists()


def test_audit_rejects_duplicate_pixels_across_splits(tmp_path):
    data = make_fixture(tmp_path / "data", 2, 1, 1, 64)
    shutil.copyfile(
        data / "images/synthetic_train_000.png", data / "images/synthetic_test_000.png"
    )
    with pytest.raises(ValueError, match="LEAKAGE: identical image across splits"):
        dataset_audit(data, preview_count=0)


def test_palette_mask_uses_class_indices(tmp_path):
    mask = Image.fromarray(np.array([[0, 1], [1, 0]], dtype=np.uint8)).convert("P")
    mask.putpalette([0, 0, 0, 255, 0, 0] + [0] * 762)
    path = tmp_path / "indexed_mask.png"
    mask.save(path)
    np.testing.assert_array_equal(load_binary_mask(path), [[0, 255], [255, 0]])


def test_dataset_rejects_size_incompatible_with_smp_encoders(tmp_path):
    data = make_fixture(tmp_path / "data", 1, 1, 1, 64)
    with pytest.raises(ValueError, match="divisible by 32"):
        RoofDataset(data, "train", size=48)
