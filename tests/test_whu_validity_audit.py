"""Independent original-raster audit guards: declared validity and PNG parity."""

import numpy as np
import pytest
import rasterio
from PIL import Image
from rasterio.enums import ColorInterp
from rasterio.transform import from_origin

from scripts import audit_whu_validity as audit


def raster(path, values, *, nodata=None, alpha=False):
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=values.shape[2],
        height=values.shape[1],
        count=len(values),
        dtype=values.dtype,
        nodata=nodata,
        transform=from_origin(0, 4, 1, 1),
    ) as target:
        target.write(values)
        if alpha:
            target.colorinterp = (
                ColorInterp.red,
                ColorInterp.green,
                ColorInterp.blue,
                ColorInterp.alpha,
            )
    return path


def paired(tmp_path, *, rgb=None, image_nodata=None, label_nodata=None):
    rgb = np.zeros((3, 4, 4), dtype=np.uint8) if rgb is None else rgb
    labels = np.zeros((1, 4, 4), dtype=np.uint8)
    labels[0, 1, 1] = 1
    image = raster(tmp_path / "rgb.tif", rgb, nodata=image_nodata)
    label = raster(tmp_path / "label.tif", labels, nodata=label_nodata)
    prepared_image, prepared_label = tmp_path / "image.png", tmp_path / "label.png"
    Image.fromarray(np.moveaxis(rgb[:3], 0, -1)).save(prepared_image)
    Image.fromarray(labels[0] * 255).save(prepared_label)
    return image, label, prepared_image, prepared_label


def test_black_rgb_is_valid_and_label_zero_is_background(tmp_path):
    evidence = audit.audit_pair(*paired(tmp_path))
    assert evidence["source_invalid_pixels"] == 0
    assert evidence["image_rgb_pixel_parity"]
    assert evidence["normalized_label_pixel_parity"]
    assert not evidence["label_nodata_class_collision"]


def test_declared_rgb_nodata_is_detected_even_with_pixel_parity(tmp_path):
    rgb = np.full((3, 4, 4), 20, dtype=np.uint8)
    rgb[:, 0, 0] = 0
    evidence = audit.audit_pair(*paired(tmp_path, rgb=rgb, image_nodata=0))
    assert evidence["image_rgb_pixel_parity"]
    assert evidence["image_declared_invalid_pixels"] == 1
    assert evidence["prepared_lost_declared_or_nonfinite_invalid_pixels"] == 1


def test_label_class_nodata_collision_is_flagged_without_relabeling(tmp_path):
    evidence = audit.audit_pair(*paired(tmp_path, label_nodata=0))
    assert evidence["label_nodata_class_collision"]
    assert evidence["label_declared_invalid_pixels"] == 15
    assert evidence["normalized_label_pixel_parity"]  # Class-zero remains background.


def test_alpha_validity_survives_gdal_read_but_is_lost_in_rgb_png(tmp_path):
    rgba = np.full((4, 4, 4), 255, dtype=np.uint8)
    rgba[:3] = 42
    rgba[3, 0, 0] = 0
    image, label, prepared_image, prepared_label = paired(tmp_path, rgb=rgba)
    raster(image, rgba, alpha=True)
    evidence = audit.audit_pair(image, label, prepared_image, prepared_label)
    assert evidence["image_alpha_bands"] == 1
    assert evidence["image_alpha_invalid_pixels"] == 1
    assert evidence["source_invalid_pixels"] == 1
    assert evidence["prepared_lost_declared_or_nonfinite_invalid_pixels"] == 1


def test_external_mask_is_honored_from_extracted_path(tmp_path):
    paths = paired(tmp_path)
    mask = np.full((4, 4), 255, dtype=np.uint8)
    mask[2, 3] = 0
    with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=False):
        with rasterio.open(paths[0], "r+") as image:
            image.write_mask(mask)
    assert paths[0].with_suffix(".tif.msk").is_file()
    evidence = audit.audit_pair(*paths)
    assert evidence["image_declared_invalid_pixels"] == 1
    assert evidence["image_sidecars"][0]["name"] == "rgb.tif.msk"


@pytest.mark.parametrize("kind", ["image", "label"])
def test_prepared_pixel_changes_are_not_accepted_as_parity(tmp_path, kind):
    paths = paired(tmp_path)
    path = paths[2 if kind == "image" else 3]
    with Image.open(path) as original:
        changed = np.asarray(original).copy()
    changed[0, 0] = 255
    Image.fromarray(changed).save(path)
    evidence = audit.audit_pair(*paths)
    key = (
        "image_rgb_pixel_parity" if kind == "image" else "normalized_label_pixel_parity"
    )
    assert not evidence[key]


def test_nonfinite_rgb_is_invalid_without_nodata_metadata(tmp_path):
    values = np.ones((3, 4, 4), dtype=np.float32)
    values[1, 3, 2] = np.nan
    path = raster(tmp_path / "float.tif", values)
    _, valid, metadata = audit.inspect_raster(path)
    assert metadata["declared_invalid_pixels"] == 0
    assert metadata["nonfinite_pixels"] == 1
    assert not valid[3, 2]


def test_external_aux_nodata_metadata_is_honored(tmp_path):
    paths = paired(tmp_path)
    auxiliary = paths[0].with_suffix(".tif.aux.xml")
    auxiliary.write_text(
        "<PAMDataset>"
        + "".join(
            f'<PAMRasterBand band="{band}"><NoDataValue>0</NoDataValue></PAMRasterBand>'
            for band in (1, 2, 3)
        )
        + "</PAMDataset>"
    )
    evidence = audit.audit_pair(*paths)
    assert evidence["image_nodata"] == [0.0, 0.0, 0.0]
    assert evidence["image_declared_invalid_pixels"] == 16
    assert any(
        sidecar["name"] == auxiliary.name for sidecar in evidence["image_sidecars"]
    )


def test_one_bit_white_is_zero_uses_original_importer_semantics(tmp_path):
    paths = paired(tmp_path)
    raw_codes = np.ones((1, 4, 4), dtype=np.uint8)
    raw_codes[0, 1, 1] = 0  # White/building, despite a raw code of zero.
    with rasterio.open(
        paths[1],
        "w",
        driver="GTiff",
        width=4,
        height=4,
        count=1,
        dtype="uint8",
        nbits=1,
        photometric="MINISWHITE",
        transform=from_origin(0, 4, 1, 1),
    ) as target:
        target.write(raw_codes)
    with Image.open(paths[1]) as image:
        assert image.mode == "1" and image.tag_v2[262] == 0
        assert np.asarray(image.convert("L"))[1, 1] == 255
    evidence = audit.audit_pair(*paths)
    assert evidence["normalized_label_pixel_parity"]
    assert evidence["source_label_pil_mode"] == "1"
    assert evidence["source_label_photometric"] == 0
    assert evidence["label_palette"] == {"0": [255, 255, 255, 255], "1": [0, 0, 0, 255]}


def test_eight_bit_palette_preserves_class_indices_despite_colors(tmp_path):
    paths = paired(tmp_path)
    codes = np.zeros((4, 4), dtype=np.uint8)
    codes[1, 1] = 1
    image = Image.fromarray(codes).convert("P")
    image.putpalette([255, 255, 255, 0, 0, 0] + [0] * (256 * 3 - 6))
    image.save(paths[1])
    with Image.open(paths[1]) as source:
        assert source.mode == "P"
    evidence = audit.audit_pair(*paths)
    assert evidence["source_label_pil_mode"] == "P"
    assert evidence["normalized_label_pixel_parity"]
