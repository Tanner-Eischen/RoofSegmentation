"""Validity is metadata, independent of numeric black/background/roof values."""

import numpy as np
import pytest
import rasterio
from PIL import Image
from rasterio.transform import from_origin

from roof_pipeline.validity import read_label, read_rgb, resize_rgb_valid


def write_tiff(path, bands, **kwargs):
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=bands.shape[1],
        width=bands.shape[2],
        count=len(bands),
        dtype=bands.dtype,
        transform=from_origin(0, 10, 1, 1),
        **kwargs,
    ) as dst:
        dst.write(bands)


def test_plain_zero_background_and_255_roof_are_valid(tmp_path):
    path = tmp_path / "mask.png"
    Image.fromarray(np.array([[0, 1, 255]], np.uint8)).save(path)
    mask, valid = read_label(path)
    assert valid.all()
    assert mask.tolist() == [[0, 255, 255]]


def test_declared_label_nodata_255_is_ignored(tmp_path):
    path = tmp_path / "mask.tif"
    write_tiff(path, np.array([[[0, 1, 255]]], np.uint8), nodata=255)
    mask, valid = read_label(path)
    assert valid.tolist() == [[True, True, False]]
    assert mask.tolist() == [[0, 255, 0]]


def test_float_stretch_excludes_nodata_and_nonfinite(tmp_path):
    path = tmp_path / "image.tif"
    values = np.array([[[10, 20, 30, -9999, np.nan]]] * 3, np.float32)
    write_tiff(path, values, nodata=-9999)
    rgb, valid, _ = read_rgb(path)
    assert valid.tolist() == [[True, True, True, False, False]]
    assert rgb[0, :, 0].tolist() == [0, 127, 255, 0, 0]


def test_valid_black_and_transparent_pixels_are_distinct(tmp_path):
    path = tmp_path / "alpha.png"
    Image.fromarray(np.array([[[0, 0, 0, 255], [255, 255, 255, 0]]], np.uint8)).save(
        path
    )
    _, valid, _ = read_rgb(path)
    assert valid.tolist() == [[True, False]]


def test_weighted_resize_cannot_bleed_invalid_colors():
    valid = np.array([[True, False], [True, False]])
    a = np.full((2, 2, 3), 100, np.uint8)
    b = a.copy()
    a[~valid] = 0
    b[~valid] = 255
    x, xv = resize_rgb_valid(a, valid, (8, 8))
    y, yv = resize_rgb_valid(b, valid, (8, 8))
    np.testing.assert_array_equal(xv, yv)
    np.testing.assert_array_equal(x, y)
    assert (x[xv] == 100).all()


def test_all_valid_resize_exact_legacy_path():
    rgb = np.random.default_rng(4).integers(0, 256, (5, 7, 3), dtype=np.uint8)
    out, valid = resize_rgb_valid(rgb, np.ones((5, 7), bool), (9, 11))
    np.testing.assert_array_equal(
        out, np.asarray(Image.fromarray(rgb).resize((9, 11), Image.Resampling.BILINEAR))
    )
    assert valid.all()


def test_invalid_nan_does_not_poison_weighted_resize():
    valid = np.array([[True, False], [True, False]])
    rgb = np.full((2, 2, 3), 100, np.float32)
    rgb[~valid] = np.nan
    resized, available = resize_rgb_valid(rgb, valid, (8, 8))
    assert (resized[available] == 100).all()
    assert not resized[~available].any()


def test_label_unknown_class_requires_explicit_nodata(tmp_path):
    path = tmp_path / "mask.png"
    Image.fromarray(np.array([[0, 2]], np.uint8)).save(path)
    with pytest.raises(ValueError, match="non-binary"):
        read_label(path)
