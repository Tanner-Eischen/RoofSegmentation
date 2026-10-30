"""Tests for masks, split leakage, geo rasterization and actual tiny model train/eval."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from roof_pipeline.common import dataset_audit, load_binary_mask, read_manifest
from roof_pipeline.evaluate import evaluate
from roof_pipeline.predict import predict
from roof_pipeline.prepare import import_geopairs, import_whu, labelme_mask
from roof_pipeline.train import train
from scripts.make_smoke_dataset import make_fixture


def test_binary_mask_and_labelme():
    anno = {
        "imageWidth": 32,
        "imageHeight": 32,
        "shapes": [
            {
                "label": "roof",
                "shape_type": "polygon",
                "points": [[4, 4], [20, 4], [20, 20], [4, 20]],
            },
            {
                "label": "background",
                "shape_type": "rectangle",
                "points": [[8, 8], [12, 12]],
            },
        ],
    }
    mask = labelme_mask(anno)
    assert mask[6, 6] == 255 and mask[10, 10] == 0 and mask[0, 0] == 0
    with pytest.raises(ValueError):
        labelme_mask(
            dict(
                anno,
                shapes=[
                    {
                        "label": "tree",
                        "shape_type": "polygon",
                        "points": [[1, 1], [5, 1], [5, 5]],
                    }
                ],
            )
        )


def test_smoke_manifest_and_visual_audit(tmp_path):
    root = make_fixture(tmp_path / "data", 4, 2, 2, 64)
    rows = read_manifest(root)
    assert len(rows) == 8
    result = dataset_audit(root, preview_count=1)
    assert result["counts"] == {"train": 4, "val": 2, "test": 2}
    assert (root / "preview_test.png").exists()
    original = (root / "manifest.csv").read_text(encoding="utf-8")
    # Malicious leakage: train and test sharing a source group must fail.
    changed = original.replace("synthetic_site_test_0", "synthetic_site_train_0")
    (root / "manifest.csv").write_text(changed, encoding="utf-8")
    with pytest.raises(ValueError, match="LEAKAGE"):
        read_manifest(root)


def test_whu_import_preserves_splits(tmp_path):
    src = tmp_path / "source"
    out = tmp_path / "processed"
    for split in ("train", "val", "test"):
        (src / split / "Image").mkdir(parents=True)
        (src / split / "Mask").mkdir()
        Image.new("RGB", (32, 32), "green").save(src / split / "Image" / "001.png")
        m = Image.new("L", (32, 32), 0)
        p = np.array(m)
        p[3:12, 3:12] = 255
        Image.fromarray(p).save(src / split / "Mask" / "001.png")
    import_whu(src, out)
    assert len(read_manifest(out)) == 3
    with Image.open(out / "masks/whu_test_000000.png") as m:
        assert set(np.unique(m)).issubset({0, 255})


def test_geopair_geojson_rasterization(tmp_path):
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    root = tmp_path / "geo"
    root.mkdir()
    transform = from_origin(-93.3, 45.0, 0.00001, 0.00001)
    for split in ("train", "val", "test"):
        tif = root / f"{split}.tif"
        with rasterio.open(
            tif,
            "w",
            driver="GTiff",
            count=3,
            width=64,
            height=64,
            dtype="uint8",
            crs="EPSG:4326",
            transform=transform,
        ) as d:
            d.write(np.full((3, 64, 64), 120, dtype=np.uint8))
        geom = {
            "type": "Polygon",
            "coordinates": [
                [
                    [-93.29985, 44.99985],
                    [-93.29965, 44.99985],
                    [-93.29965, 44.99965],
                    [-93.29985, 44.99965],
                    [-93.29985, 44.99985],
                ]
            ],
        }
        (root / f"{split}.geojson").write_text(
            json.dumps(
                {
                    "type": "FeatureCollection",
                    "features": [
                        {"type": "Feature", "geometry": geom, "properties": {}}
                    ],
                }
            )
        )
    with (root / "pairs.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "vector", "split", "group"])
        writer.writeheader()
        for split in ("train", "val", "test"):
            writer.writerow(
                {
                    "image": f"{split}.tif",
                    "vector": f"{split}.geojson",
                    "split": split,
                    "group": f"{split}_region",
                }
            )
    out = root / "dataset"
    import_geopairs(root / "pairs.csv", out, tile_size=64)
    assert len(read_manifest(out)) == 3
    assert (load_binary_mask(out / "masks/geo_0000_00000_00000.png") > 0).sum() > 0


def test_training_calibration_evaluation_prediction(tmp_path):
    import torch

    torch.set_num_threads(2)
    data = make_fixture(tmp_path / "data", 6, 3, 3, 64)
    run = tmp_path / "run"
    args = argparse.Namespace(
        data=data,
        run=run,
        arch="tiny",
        pretrained=False,
        epochs=1,
        batch_size=3,
        size=64,
        workers=0,
        lr=0.001,
        weight_decay=0.0001,
        pos_weight=1.0,
        device="cpu",
        seed=42,
        patience=8,
        min_delta=0.0001,
        no_amp=True,
    )
    summary = train(args)
    assert summary["test_evaluated"] is False
    checkpoint = run / "best.pt"
    assert checkpoint.is_file()
    result = evaluate(checkpoint, data, tmp_path / "eval", batch_size=2, device="cpu")
    assert 0 <= result["aggregate"]["iou"] <= 1
    assert (
        len(list(csv.DictReader((tmp_path / "eval" / "test_per_image.csv").open())))
        == 3
    )
    pred = predict(
        checkpoint,
        data / "images" / "synthetic_test_000.png",
        tmp_path / "prediction",
        overlap=16,
    )
    assert Path(pred["mask_png"]).is_file()


def test_backend_adapter_accepts_new_checkpoint(tmp_path):
    import io
    import sys

    import torch
    from PIL import Image

    from roof_pipeline.models import create_model

    path = tmp_path / "tiny.pt"
    model = create_model("tiny")
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": {"arch": "tiny", "image_size": 64},
            "threshold": 0.5,
            "val_iou_at_0p5": 0.0,
        },
        path,
    )
    sys.path.insert(
        0, str(Path(__file__).resolve().parents[1] / "integration_optional" / "api")
    )
    from roof_inference_compat import image_bytes_to_mask_and_polygons, load_model

    assert load_model(path)[2] == 64
    img = Image.new("RGB", (64, 64), "gray")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    result = image_bytes_to_mask_and_polygons(buf.getvalue(), path)
    assert len(result) == 4
    decoded = np.array(Image.open(io.BytesIO(result[0])))
    assert decoded.shape == (64, 64)
    assert set(np.unique(decoded)).issubset({0, 255})


def test_predict_geotiff_output_with_metric_crs(tmp_path):
    import torch

    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    from roof_pipeline.models import create_model

    src = tmp_path / "tile.tif"
    with rasterio.open(
        src,
        "w",
        driver="GTiff",
        count=3,
        width=64,
        height=64,
        dtype="uint8",
        crs="EPSG:32615",
        transform=from_origin(500000, 5000000, 0.3, 0.3),
    ) as dst:
        dst.write(np.full((3, 64, 64), 130, dtype=np.uint8))
    model = create_model("tiny")
    ckpt = tmp_path / "best.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": {"arch": "tiny", "image_size": 64},
            "threshold": 0.4,
            "val_iou_at_0p5": 0.0,
        },
        ckpt,
    )
    result = predict(ckpt, src, tmp_path / "geo_out", overlap=16)
    assert result["geo_products"] is True
    assert (tmp_path / "geo_out" / "tile_mask.tif").exists()
    assert (tmp_path / "geo_out" / "tile_roofs.geojson").exists()
