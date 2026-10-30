"""Annotation review sampling must be reproducible and never inspect test pixels."""

import numpy as np
import pytest

from roof_pipeline.annotation_audit import (
    coverage_stratum,
    make_packet,
    mask_outline,
    select_samples,
)
from scripts.make_smoke_dataset import make_fixture


def test_audit_excludes_test_pixels_and_is_reproducible(tmp_path):
    root = make_fixture(tmp_path / "data", 8, 5, 1, 64)
    (root / "masks/synthetic_test_000.png").write_bytes(b"do not decode test pixels")
    first, counts = select_samples(root, per_stratum=2, seed=71)
    assert first == select_samples(root, per_stratum=2, seed=71)[0]
    assert all(row["split"] in {"train", "val"} for row in first)
    assert sum(row["population"] for row in counts) == 13
    assert all(0 < row["inclusion_probability"] <= 1 for row in first)
    packet = make_packet(root, tmp_path / "packet", per_stratum=2, seed=71)
    assert len(list((tmp_path / "packet/panels").glob("*.png"))) == len(first)
    assert all(len(row["mask_sha256"]) == 64 for row in packet["samples"])
    with pytest.raises(FileExistsError, match="Refusing to replace"):
        make_packet(root, tmp_path / "packet")


def test_strata_and_native_contour():
    assert [coverage_stratum(value) for value in (0, 0.001, 0.01, 0.1)] == [
        "empty",
        "positive_lt_1pct",
        "1_to_10pct",
        "at_least_10pct",
    ]
    mask = np.zeros((5, 5), dtype=np.uint8)
    mask[0:3, 0:3] = 255
    contour = mask_outline(mask)
    assert contour.sum() == 8
    assert not contour[1, 1]
    assert contour[0, 0]
    assert not mask_outline(np.zeros((5, 5))).any()


def test_invalid_sample_size(tmp_path):
    with pytest.raises(ValueError, match="positive"):
        select_samples(tmp_path, 0)
