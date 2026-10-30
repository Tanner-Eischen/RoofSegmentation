"""Regression coverage for real-pixel geospatial labels and importer arguments."""

import csv
import json

import numpy as np
import pytest
from PIL import Image

from roof_pipeline.common import read_manifest
from roof_pipeline.prepare import import_geopairs, import_whu


@pytest.mark.parametrize("width,height", [(96, 64), (70, 65), (16, 20)])
def test_geopair_border_chips_keep_only_real_pixels(tmp_path, width, height):
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    transform = from_origin(-93.3, 45.0, 0.00001, 0.00001)
    left, top = transform * (-1, -1)
    right, bottom = transform * (width + 1, height + 1)
    polygon = {
        "type": "Polygon",
        "coordinates": [
            [[left, top], [right, top], [right, bottom], [left, bottom], [left, top]]
        ],
    }
    (tmp_path / "labels.geojson").write_text(json.dumps(polygon), encoding="utf-8")
    sources = {}
    pairs = []
    for index, split in enumerate(("train", "val", "test")):
        rgb = (np.arange(height * width * 3).reshape(height, width, 3) + index).astype(
            np.uint8
        )
        sources[split] = rgb
        with rasterio.open(
            tmp_path / f"{split}.tif",
            "w",
            driver="GTiff",
            count=3,
            width=width,
            height=height,
            dtype="uint8",
            crs="EPSG:4326",
            transform=transform,
        ) as dst:
            dst.write(np.moveaxis(rgb, -1, 0))
        pairs.append(
            {
                "image": f"{split}.tif",
                "vector": "labels.geojson",
                "split": split,
                "group": f"region_{split}",
            }
        )
    with (tmp_path / "pairs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("image", "vector", "split", "group")
        )
        writer.writeheader()
        writer.writerows(pairs)

    out = tmp_path / "prepared"
    import_geopairs(tmp_path / "pairs.csv", out, tile_size=64)
    rows = read_manifest(out)
    for split, source in sources.items():
        covered = np.zeros((height, width), dtype=np.uint8)
        for row in (item for item in rows if item["split"] == split):
            row0, col0 = map(int, row["sample_id"].rsplit("_", 2)[1:])
            with Image.open(out / row["image"]) as image:
                chip = np.array(image)
            with Image.open(out / row["mask"]) as mask:
                labels = np.array(mask)
            chip_height, chip_width = labels.shape
            assert chip_height == min(64, height - row0)
            assert chip_width == min(64, width - col0)
            assert np.all(labels == 255), (
                "Padding must not create false background labels"
            )
            np.testing.assert_array_equal(
                chip, source[row0 : row0 + chip_height, col0 : col0 + chip_width]
            )
            covered[row0 : row0 + chip_height, col0 : col0 + chip_width] += 1
        assert np.all(covered == 1), "Every source pixel must be included exactly once"
    metadata = json.loads((out / "dataset_meta.json").read_text(encoding="utf-8"))
    assert metadata["border_policy"] == "retain_unpadded_partial_chips"
    assert metadata["tile_size"] == 64


@pytest.mark.parametrize("tile_size", [0, -1, -64])
def test_geopair_rejects_invalid_tile_size_before_io(tmp_path, tile_size):
    with pytest.raises(ValueError, match="tile_size must be positive"):
        import_geopairs(tmp_path / "missing.csv", tmp_path / "out", tile_size=tile_size)
    assert not (tmp_path / "out").exists()


def test_whu_rejects_negative_limit_before_io(tmp_path):
    with pytest.raises(ValueError, match="limit_per_split must be nonnegative"):
        import_whu(tmp_path / "missing", tmp_path / "out", limit_per_split=-1)
    assert not (tmp_path / "out").exists()
