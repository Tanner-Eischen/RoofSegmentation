"""Source validity stays separate from background and edge padding in inference."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from roof_pipeline import predict as prediction
from roof_pipeline.validity import read_rgb


class ConstantModel(torch.nn.Module):
    def __init__(self, logit=2.0):
        super().__init__()
        self.logit = logit
        self.inputs = []

    def forward(self, value):
        self.inputs.append(value.detach().cpu().clone())
        return value.new_full(
            (value.shape[0], 1, value.shape[2], value.shape[3]), self.logit
        )


def fake_checkpoint(tmp_path, monkeypatch, model, size=32):
    checkpoint = tmp_path / "constant.pt"
    torch.save({"threshold": 0.5}, checkpoint)
    monkeypatch.setattr(
        prediction, "load_model", lambda *args: (model, {"image_size": size})
    )
    return checkpoint


def test_invalid_pixels_are_neutral_before_forward_and_empty_chips_are_skipped():
    rgb = np.zeros((32, 96, 3), np.float32)
    rgb[:, 32:] = np.nan
    valid = np.zeros((32, 96), bool)
    valid[:, :32] = True
    valid[4:8, 4:8] = False
    rgb[4:8, 4:8] = 1e30
    model = ConstantModel()
    probability = prediction.predict_array(model, rgb, "cpu", 32, 0, valid_mask=valid)
    assert len(model.inputs) == 1
    assert torch.equal(model.inputs[0][0, :, 4:8, 4:8], torch.zeros((3, 4, 4)))
    assert torch.all(model.inputs[0][0, :, 0, 0] < 0), (
        "Valid black pixels must retain their RGB normalization"
    )
    assert np.all(probability[~valid] == 0)
    assert np.all(probability[valid] > 0.5)
    assert probability.shape == valid.shape


def test_validity_argument_preserves_all_valid_rgb_predictions():
    rgb = np.random.default_rng(34).integers(0, 256, (29, 43, 3), np.uint8)
    model = torch.nn.Conv2d(3, 1, 1).eval()
    ordinary = prediction.predict_array(model, rgb, "cpu", 32, 8)
    explicit = prediction.predict_array(
        model, rgb, "cpu", 32, 8, valid_mask=np.ones(rgb.shape[:2], bool)
    )
    np.testing.assert_array_equal(ordinary, explicit)


def test_all_invalid_image_and_wrong_mask_shape_fail_before_forward():
    model = ConstantModel()
    rgb = np.zeros((32, 32, 3), np.uint8)
    with pytest.raises(ValueError, match="no valid pixels"):
        prediction.predict_array(
            model, rgb, "cpu", 32, 0, valid_mask=np.zeros((32, 32), bool)
        )
    with pytest.raises(ValueError, match="dimensions"):
        prediction.predict_array(
            model, rgb, "cpu", 32, 0, valid_mask=np.ones((16, 16), bool)
        )
    assert not model.inputs


def test_cli_alpha_and_cropped_padding_do_not_add_foreground_area(
    tmp_path, monkeypatch
):
    rgba = np.zeros((29, 43, 4), np.uint8)
    rgba[3:26, 5:38, 3] = 255
    rgba[:3, :, :3] = 255
    valid = rgba[..., 3] > 0
    image = tmp_path / "alpha.png"
    Image.fromarray(rgba).save(image)
    checkpoint = fake_checkpoint(tmp_path, monkeypatch, ConstantModel())
    result = prediction.predict(checkpoint, image, tmp_path / "out", overlap=8)
    mask = np.asarray(Image.open(result["mask_png"]))
    validity = np.asarray(Image.open(result["validity_png"]))
    assert mask.shape == (29, 43) and np.array_equal(mask > 0, valid)
    assert np.array_equal(validity > 0, valid)
    assert result["roof_area_pixels"] == result["valid_pixel_count"] == int(valid.sum())
    assert result["invalid_pixel_count"] + result["valid_pixel_count"] == 29 * 43
    assert result["total_pixel_count"] == 29 * 43 and result["pixel_coverage"] == 1
    assert "placeholder" in result["validity_definition"]


def make_tiff(path, rgb, *, nodata=None):
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        count=3,
        width=rgb.shape[1],
        height=rgb.shape[0],
        dtype=rgb.dtype,
        nodata=nodata,
        crs="EPSG:32615",
        transform=from_origin(500000, 5000000, 0.3, 0.3),
    ) as dst:
        dst.write(np.moveaxis(rgb, -1, 0))


def test_geotiff_mask_and_polygons_exclude_nodata_and_preserve_holes(
    tmp_path, monkeypatch
):
    rasterio = pytest.importorskip("rasterio")
    rgb = np.full((48, 64, 3), 65535, np.uint16)
    valid = np.zeros((48, 64), bool)
    valid[4:44, 6:58] = True
    valid[18:24, 20:27] = False
    rgb[valid] = 800
    image = tmp_path / "nodata.tif"
    make_tiff(image, rgb, nodata=65535)
    checkpoint = fake_checkpoint(tmp_path, monkeypatch, ConstantModel())
    result = prediction.predict(checkpoint, image, tmp_path / "out", overlap=8)
    with rasterio.open(result["mask_geotiff"]) as mask:
        np.testing.assert_array_equal(mask.read_masks(1) > 0, valid)
        np.testing.assert_array_equal(mask.read(1) > 0, valid)
        assert mask.nodata is None, "Class zero must not be tagged as NoData"
    geojson = json.loads(Path(result["geojson"]).read_text())
    assert len(geojson["features"]) == 1
    assert len(geojson["features"][0]["geometry"]["coordinates"]) == 2
    assert result["area_m2"] == pytest.approx(valid.sum() * 0.3**2, abs=1e-6)
    assert result["roof_area_pixels"] == int(valid.sum())


@pytest.mark.parametrize("value", [0, 255])
def test_unflagged_black_or_white_geotiff_pixels_remain_valid(
    tmp_path, monkeypatch, value
):
    rasterio = pytest.importorskip("rasterio")
    image = tmp_path / "valid.tif"
    make_tiff(image, np.full((17, 23, 3), value, np.uint8))
    checkpoint = fake_checkpoint(tmp_path, monkeypatch, ConstantModel(-2.0))
    result = prediction.predict(checkpoint, image, tmp_path / "out", overlap=8)
    assert result["valid_pixel_count"] == 17 * 23 and result["invalid_pixel_count"] == 0
    assert result["roof_area_pixels"] == 0 and result["pixel_coverage"] == 0
    with rasterio.open(result["mask_geotiff"]) as output:
        assert output.nodata is None and np.all(output.read_masks(1) == 255)
        assert not output.read(1).any()


def test_invalid_float_extremes_and_nan_cannot_change_percentile_stretch(tmp_path):
    arrays = []
    for index, sentinel in enumerate((-9999.0, -1e30)):
        rgb = np.full((20, 20, 3), sentinel, np.float32)
        valid = np.zeros((20, 20), bool)
        valid[3:17, 3:17] = True
        values = np.arange(valid.sum() * 3, dtype=np.float32).reshape(-1, 3) + 50
        rgb[valid] = values
        rgb[0, 0] = np.nan
        image = tmp_path / f"float{index}.tif"
        make_tiff(image, rgb, nodata=sentinel)
        converted, actual_valid, _ = read_rgb(image)
        np.testing.assert_array_equal(actual_valid, valid)
        arrays.append(converted)
    np.testing.assert_array_equal(arrays[0][valid], arrays[1][valid])
