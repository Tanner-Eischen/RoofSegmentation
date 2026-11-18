"""NoData metadata is availability, never an invented building/background class."""

import base64
import csv
import hashlib
import json
import random

import numpy as np
import pytest
import torch
from PIL import Image, ImageEnhance

from roof_pipeline.common import (
    FIELDS,
    dataset_audit,
    dataset_fingerprint,
    read_manifest,
    read_paired_sample,
    save_pair,
    write_manifest,
)
from roof_pipeline.dataset import MEAN, STD, RoofDataset
from roof_pipeline.prepare import import_geopairs, import_labelme, import_whu
from roof_pipeline.validity import IGNORE_LABEL, _uint8_rgb, read_label, read_valid_mask


def prepared(root, image=None, valid=None):
    rows = []
    for index, split in enumerate(("train", "val", "test")):
        source = (
            image
            if image is not None
            else Image.new("RGB", (32, 32), (20 + index, 30, 40))
        )
        labels = np.full((source.height, source.width), 255, dtype=np.uint8)
        rows.append(
            save_pair(
                root, source, labels, f"case_{split}", split, split, "fixture", valid
            )
        )
    write_manifest(root, rows, {"fixture": "validity"})
    return root, rows


def test_alpha_zero_is_ignored_but_opaque_black_is_real_data(tmp_path):
    rgba = np.zeros((32, 32, 4), dtype=np.uint8)
    rgba[..., 3] = 255
    rgba[:, :16, 3] = 0
    data, rows = prepared(tmp_path / "data", Image.fromarray(rgba))
    assert all(row.get("valid_mask") for row in rows)
    x, y = RoofDataset(data, "train", size=32)[0]
    assert torch.all(y[:, :, :16] == IGNORE_LABEL)
    assert torch.all(x[:, :, :16] == 0)
    assert torch.all(y[:, :, 16:] == 1)
    assert torch.all(x[:, :, 16:] < 0), (
        "Valid black RGB must not be guessed to be NoData"
    )
    with Image.open(data / rows[0]["mask"]) as labels:
        assert np.all(np.asarray(labels)[:, :16] == 0)


def test_legacy_all_valid_fingerprint_is_byte_for_byte_unchanged(tmp_path):
    # Frozen bytes and SHA were calculated with the pre-validity fingerprint
    # implementation, rather than regenerating PNGs with a newer image codec.
    image = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGPgEpEDAABoAD1UCKP3AAAAAElFTkSuQmCC"
    )
    mask = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAAAAAA6fptVAAAACklEQVR4nGP4DwABAQEAsTj2FAAAAABJRU5ErkJggg=="
    )
    (tmp_path / "images").mkdir()
    (tmp_path / "masks").mkdir()
    lines = ["sample_id,split,image,mask,group,source"]
    for split in ("train", "val", "test"):
        (tmp_path / f"images/{split}.png").write_bytes(image)
        (tmp_path / f"masks/{split}.png").write_bytes(mask)
        lines.append(
            f"{split}_0,{split},images/{split}.png,masks/{split}.png,{split},golden"
        )
    (tmp_path / "manifest.csv").write_bytes(("\r\n".join(lines) + "\r\n").encode())
    (tmp_path / "dataset_meta.json").write_bytes(b'{"fixture":"legacy"}\n')
    assert (
        dataset_fingerprint(tmp_path)
        == "15ad2db157519e95a1285a815523c5e938b7d3eba9c72774eba8392ba442a40b"
    )


def test_all_valid_standard_resize_and_photometry_stay_numerically_legacy(
    tmp_path, monkeypatch
):
    rgb = np.arange(64 * 64 * 3, dtype=np.uint8).reshape(64, 64, 3)
    image = Image.fromarray(rgb)
    data, rows = prepared(tmp_path / "data", image)
    assert all(set(row) == set(FIELDS) for row in rows)
    assert not (data / "valid_masks").exists()
    assert next(csv.reader((data / "manifest.csv").open())) == list(FIELDS)
    samples = iter((1.0, 1.0, 0.0, 0.0))
    factors = iter((1.1, 1.15))
    monkeypatch.setattr(random, "random", lambda: next(samples))
    monkeypatch.setattr(random, "randrange", lambda _: 0)
    monkeypatch.setattr(random, "uniform", lambda *_: next(factors))
    x, y = RoofDataset(data, "train", size=96, augment=True)[0]
    expected = image.resize((96, 96), Image.Resampling.BILINEAR)
    expected = ImageEnhance.Brightness(expected).enhance(1.1)
    expected = ImageEnhance.Contrast(expected).enhance(1.15)
    normalized = (np.asarray(expected, dtype=np.float32) / 255 - MEAN) / STD
    np.testing.assert_array_equal(x.numpy(), np.moveaxis(normalized, -1, 0))
    assert torch.all(y == 1)


def test_partial_contrast_ignores_invisible_rgb_and_validity_follows_geometry(
    tmp_path, monkeypatch
):
    valid = np.zeros((32, 32), dtype=bool)
    valid[:16] = True
    outputs = []
    for index, invisible in enumerate((0, 255)):
        rgb = np.full((32, 32, 3), invisible, dtype=np.uint8)
        rgb[:16] = (60, 100, 140)
        data, _ = prepared(tmp_path / str(index), Image.fromarray(rgb), valid)
        # Flip vertically, rotate once, then contrast only. Those same geometric
        # operations must apply to the image, labels, and ignored region.
        draws = iter((1.0, 0.0, 1.0, 0.0))
        monkeypatch.setattr(random, "random", lambda: next(draws))
        monkeypatch.setattr(random, "randrange", lambda _: 1)
        monkeypatch.setattr(random, "uniform", lambda *_: 1.15)
        outputs.append(RoofDataset(data, "train", size=32, augment=True)[0])
    torch.testing.assert_close(outputs[0][0], outputs[1][0], rtol=0, atol=0)
    torch.testing.assert_close(outputs[0][1], outputs[1][1], rtol=0, atol=0)
    expected = (
        Image.fromarray(valid).transpose(Image.Transpose.FLIP_TOP_BOTTOM).rotate(90)
    )
    np.testing.assert_array_equal(outputs[0][1][0].numpy() >= 0, np.asarray(expected))


@pytest.mark.parametrize("nodata,dtype", [(255, "uint8"), (float("nan"), "float32")])
def test_declared_tiff_label_nodata_is_not_roof_and_overlay_cannot_revalidate_it(
    tmp_path, nodata, dtype
):
    rasterio = pytest.importorskip("rasterio")
    data, rows = prepared(tmp_path / "data")
    path = data / "label.tif"
    values = np.zeros((32, 32), dtype=dtype)
    values[:, :8] = nodata
    values[:, 16:] = 1
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=32,
        height=32,
        count=1,
        dtype=dtype,
        nodata=nodata,
        transform=rasterio.transform.from_origin(10, 20, 1, 1),
    ) as target:
        target.write(values, 1)
    rows[0]["mask"] = "label.tif"
    write_manifest(data, rows)
    overlay = data / "overlay.png"
    Image.new("L", (32, 32), 255).save(overlay)
    x, y = RoofDataset(data, "train", size=32, mask_overrides={"case_train": overlay})[
        0
    ]
    assert torch.all(y[:, :, :8] == IGNORE_LABEL)
    assert torch.all(y[:, :, 8:] == 1)
    assert torch.all(x[:, :, :8] == 0)


def test_palette_indices_and_undeclared_255_keep_building_semantics(tmp_path):
    indices = np.array([[0, 1, 255]], dtype=np.uint8)
    image = Image.fromarray(indices).convert("P")
    image.putpalette([255, 0, 0] * 256)
    path = tmp_path / "indexed.png"
    image.save(path)
    labels, valid = read_label(path)
    np.testing.assert_array_equal(labels, [[0, 255, 255]])
    assert valid.all()


def test_bilevel_white_is_zero_tiff_preserves_visual_whu_mask_semantics(tmp_path):
    pytest.importorskip("rasterio")
    shape = np.zeros((32, 32), dtype=bool)
    shape[4:12, 5:17] = True
    path = tmp_path / "whu_bilevel.tif"
    Image.fromarray(shape).save(path, tiffinfo={262: 0})
    with Image.open(path) as image:
        assert image.mode == "1" and image.tag_v2.get(262) == 0
        expected = np.asarray(image.convert("L"))
    labels, valid = read_label(path)
    np.testing.assert_array_equal(labels, expected)
    assert valid.all()
    assert np.count_nonzero(labels) == 8 * 12


def test_bilevel_png_with_external_metadata_has_no_tiff_photometric_tag(tmp_path):
    pytest.importorskip("rasterio")
    shape = np.zeros((32, 32), dtype=bool)
    shape[2:9, 4:12] = True
    path = tmp_path / "mask.png"
    Image.fromarray(shape).save(path)
    (tmp_path / "mask.png.aux.xml").write_text(
        '<PAMDataset><Metadata><MDI key="fixture">one</MDI></Metadata></PAMDataset>'
    )
    labels, valid = read_label(path)
    np.testing.assert_array_equal(labels > 0, shape)
    assert valid.all()


def test_sidecars_are_required_sized_and_content_bound(tmp_path):
    valid = np.ones((32, 32), dtype=bool)
    valid[0] = False
    data, rows = prepared(tmp_path / "data", valid=valid)
    before = dataset_fingerprint(data)
    sidecar = data / rows[0]["valid_mask"]
    changed = valid.copy()
    changed[1] = False
    Image.fromarray(changed).save(sidecar)
    assert dataset_fingerprint(data) != before
    Image.new("L", (16, 16), 255).save(sidecar)
    with pytest.raises(ValueError, match="size mismatch"):
        read_manifest(data)
    sidecar.unlink()
    with pytest.raises(FileNotFoundError, match="valid_mask"):
        read_manifest(data)


@pytest.mark.parametrize("field", ["image", "mask"])
@pytest.mark.parametrize("suffix", [".msk", ".aux.xml"])
def test_external_validity_companion_bytes_are_content_bound(tmp_path, field, suffix):
    data, rows = prepared(tmp_path / "data")
    before = dataset_fingerprint(data)
    companion = data / (rows[0][field] + suffix)
    companion.write_bytes(b"first metadata fixture")
    with_companion = dataset_fingerprint(data)
    assert before != with_companion
    companion.write_bytes(b"changed metadata fixture")
    assert dataset_fingerprint(data) != with_companion


def test_all_valid_nonuint8_scene_stretch_keeps_legacy_float32_arithmetic():
    rgb = np.random.default_rng(6).uniform(-500, 10000, (33, 35, 3)).astype(np.float64)
    legacy = np.zeros(rgb.shape, dtype=np.uint8)
    for band in range(3):
        channel = rgb[..., band].astype(np.float32)
        low, high = np.percentile(channel, (2, 98))
        legacy[..., band] = np.clip(
            (channel - low) * 255 / max(high - low, 1e-8), 0, 255
        ).astype(np.uint8)
    np.testing.assert_array_equal(_uint8_rgb(rgb, np.ones(rgb.shape[:2], bool)), legacy)


def test_audit_excludes_invalid_targets_and_does_not_call_all_invalid_empty(tmp_path):
    valid = np.ones((32, 32), dtype=bool)
    valid[:, :16] = False
    data, rows = prepared(tmp_path / "data", valid=valid)
    Image.new("L", (32, 32), 0).save(data / rows[1]["valid_mask"])
    report = dataset_audit(data, preview_count=1, verify_hashes=False)
    assert report["roof_fraction_by_split"]["train"]["mean"] == 1
    assert report["valid_pixels_by_split"]["train"] == 512
    assert report["invalid_pixels_by_split"]["train"] == 512
    assert report["all_invalid_tiles_by_split"]["val"] == 1
    assert report["empty_masks_by_split"]["val"] == 0
    assert report["roof_fraction_by_split"]["val"]["mean"] is None
    assert (data / "preview_val.png").is_file()


def test_annotation_sampling_uses_valid_coverage_and_omits_unavailable_cases(tmp_path):
    from roof_pipeline.annotation_audit import make_packet, select_samples

    valid = np.ones((32, 32), dtype=bool)
    valid[:, :16] = False
    data, rows = prepared(tmp_path / "data", valid=valid)
    Image.new("L", (32, 32), 0).save(data / rows[1]["valid_mask"])
    selected, population = select_samples(data, per_stratum=1)
    assert len(selected) == 1 and selected[0]["sample_id"] == "case_train"
    assert selected[0]["coverage"] == 1
    assert sum(row["population"] for row in population) == 1
    packet = make_packet(data, tmp_path / "packet", per_stratum=1)
    assert (
        packet["samples"][0]["valid_mask_sha256"]
        == hashlib.sha256((data / rows[0]["valid_mask"]).read_bytes()).hexdigest()
    )


def test_review_export_fails_before_copying_partial_validity_source(tmp_path):
    from roof_pipeline.adjudication import build_review
    from roof_pipeline.annotation_audit import make_packet

    valid = np.ones((32, 32), dtype=bool)
    valid[0] = False
    data, _ = prepared(tmp_path / "data", valid=valid)
    audit = tmp_path / "audit"
    make_packet(data, audit, per_stratum=1)
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(
        {"config": {"dataset_content_sha256": dataset_fingerprint(data)}}, checkpoint
    )
    out = tmp_path / "review"
    with pytest.raises(ValueError, match="cannot carry partial pixel validity"):
        build_review(data, audit, checkpoint, out, "validity-draft")
    assert not out.exists()


@pytest.mark.parametrize("mode", ["review", "correction"])
def test_manual_diagnostics_rescore_on_joint_valid_support_and_neutral_inputs(
    tmp_path, monkeypatch, mode
):
    rasterio = pytest.importorskip("rasterio")
    from roof_pipeline import correction_experiment, review_score

    root = tmp_path / "review"
    root.mkdir()
    rgb = np.full((32, 32, 4), 255, dtype=np.uint8)
    rgb[24:, :, 3] = 0
    Image.fromarray(rgb).save(root / "image.png")
    for name, columns in (("original", slice(0, 8)), ("proposed", slice(8, 16))):
        label = np.ones((32, 32), dtype=np.uint8)
        label[:, columns] = 255
        with rasterio.open(
            root / f"{name}.tif",
            "w",
            driver="GTiff",
            width=32,
            height=32,
            count=1,
            dtype="uint8",
            nodata=255,
            transform=rasterio.transform.from_origin(10, 20, 1, 1),
        ) as target:
            target.write(label, 1)
    Image.new("L", (32, 32), 0).save(root / "ignore.png")
    keep = np.full((32, 32), 255, dtype=np.uint8)
    keep[:8] = 0
    Image.fromarray(keep).save(root / "valid.png")
    config = {
        "arch": "tiny",
        "image_size": 32,
        "normalization": "imagenet_rgb_v1",
        "dataset_content_sha256": "parent",
    }
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({"config": config, "threshold": 0.5}, checkpoint)
    case = {
        "sample_id": "one",
        "split": "val" if mode == "review" else "train",
        "group": "site",
        "cohort": "representative",
        "stratum": "at_least_10pct",
        "decision": "correct",
        "reviewer_kind": "human",
        "reviewer_id": "fixture",
        "image": "image.png",
        "original_mask": "original.tif",
        "mask": "proposed.tif",
        "ignore_mask": "ignore.png",
        "valid_mask": "valid.png",
    }
    manifest = {
        "schema_version": "roof_review_v1",
        "version": "validity-regression",
        "sealed": True,
        "content_sha256": "review",
        "source_dataset_content_sha256": "parent",
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "cases": [case],
        "policy": {
            "status": "approved",
            "reviewer_kind": "human",
            "reviewer_id": "fixture",
        },
    }

    class PositiveModel(torch.nn.Module):
        def forward(self, tensor):
            self.input = tensor.detach().clone()
            return tensor.new_full((1, 1, 32, 32), 10)

    model = PositiveModel()
    if mode == "review":
        monkeypatch.setattr(review_score, "_load_review", lambda *args: manifest)
        monkeypatch.setattr(review_score, "load_model", lambda *args: (model, config))
        result = review_score.score_review(root, checkpoint, tmp_path / "score")
        counts = result["cohorts"]["representative"]
        assert counts["evaluated_paired_pixels"] == 256
        assert counts["original_all_pixels"]["tp"] == 384
        assert counts["paired_original"]["tp"] == counts["paired_review"]["tp"] == 256
    else:
        monkeypatch.setattr(
            correction_experiment, "load_review", lambda *args, **kwargs: manifest
        )
        monkeypatch.setattr(
            correction_experiment, "load_model", lambda *args: (model, config)
        )
        result = correction_experiment.correction_case_diagnostics(
            checkpoint, root, tmp_path / "score", device="cpu"
        )
        assert (
            result["aggregate"]["original"]["tp"]
            == result["aggregate"]["proposed"]["tp"]
            == 256
        )
    expected_valid = np.zeros((32, 32), dtype=bool)
    expected_valid[8:24, 8:] = True
    assert torch.all(model.input[0, :, ~torch.from_numpy(expected_valid)] == 0)
    assert torch.all(model.input[0, :, torch.from_numpy(expected_valid)] > 0)


def test_labelme_alpha_is_exported_as_validity_instead_of_background(tmp_path):
    source = tmp_path / "manual"
    source.mkdir()
    rows = []
    for split in ("train", "val", "test"):
        image = Image.new("RGBA", (32, 32), (255, 255, 255, 255))
        image.putpixel((0, 0), (255, 255, 255, 0))
        image.save(source / f"{split}.png")
        (source / f"{split}.json").write_text(
            json.dumps(
                {
                    "imagePath": f"{split}.png",
                    "imageWidth": 32,
                    "imageHeight": 32,
                    "shapes": [],
                }
            )
        )
        rows.append({"stem": split, "split": split, "group": split})
    mapping = tmp_path / "splits.csv"
    with mapping.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("stem", "split", "group"))
        writer.writeheader()
        writer.writerows(rows)
    out = tmp_path / "prepared"
    import_labelme(source, out, mapping)
    item = read_manifest(out)[0]
    assert item.get("valid_mask")
    _, labels, valid = read_paired_sample(out, item)
    assert not valid[0, 0] and valid[0, 1]
    assert not labels.any()


def test_whu_import_preserves_declared_rgb_and_label_validity(tmp_path):
    rasterio = pytest.importorskip("rasterio")
    source, out = tmp_path / "source", tmp_path / "prepared"
    for split in ("train", "val", "test"):
        (source / split / "Image").mkdir(parents=True)
        (source / split / "Mask").mkdir()
        rgb = np.full((3, 32, 32), 100, dtype=np.uint8)
        rgb[:, :4] = 255
        label = np.ones((32, 32), dtype=np.uint8)
        label[:, :4] = 255
        for folder, array, bands in (("Image", rgb, 3), ("Mask", label, 1)):
            with rasterio.open(
                source / split / folder / "one.tif",
                "w",
                driver="GTiff",
                width=32,
                height=32,
                count=bands,
                dtype="uint8",
                nodata=255,
                transform=rasterio.transform.from_origin(10, 20, 1, 1),
            ) as target:
                target.write(array if bands == 3 else array[None])
    import_whu(source, out)
    item = read_manifest(out)[0]
    valid = read_valid_mask(out / item["valid_mask"], (32, 32))
    assert not valid[:4].any() and not valid[:, :4].any()
    assert valid[4:, 4:].all()


def test_geopairs_excludes_alpha_nodata_from_polygons_and_percentile_stretch(tmp_path):
    rasterio = pytest.importorskip("rasterio")
    from rasterio.enums import ColorInterp
    from rasterio.transform import from_origin

    transform = from_origin(0, 32, 1, 1)
    polygon = {
        "type": "Polygon",
        "coordinates": [[[0, 32], [32, 32], [32, 0], [0, 0], [0, 32]]],
    }
    (tmp_path / "labels.json").write_text(json.dumps(polygon))
    pairs = []
    for index, split in enumerate(("train", "val", "test")):
        bands = np.zeros((4, 32, 32), dtype=np.float32)
        bands[:3] = np.arange(1024).reshape(32, 32) + 100 + index
        bands[3] = 255
        bands[:, :, :8] = -9999
        bands[3, :, :8] = 0
        bands[3, :4, 16:24] = 0  # Alpha alone invalidates otherwise finite RGB.
        with rasterio.open(
            tmp_path / f"{split}.tif",
            "w",
            driver="GTiff",
            width=32,
            height=32,
            count=4,
            dtype="float32",
            crs="EPSG:4326",
            transform=transform,
            nodata=-9999,
        ) as target:
            target.write(bands)
            target.colorinterp = (
                ColorInterp.red,
                ColorInterp.green,
                ColorInterp.blue,
                ColorInterp.alpha,
            )
        pairs.append(
            {
                "image": f"{split}.tif",
                "vector": "labels.json",
                "split": split,
                "group": split,
            }
        )
    with (tmp_path / "pairs.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=("image", "vector", "split", "group")
        )
        writer.writeheader()
        writer.writerows(pairs)
    out = tmp_path / "prepared"
    import_geopairs(tmp_path / "pairs.csv", out, tile_size=16)
    for item in read_manifest(out):
        rgb, labels, valid = read_paired_sample(out, item)
        assert np.all(labels[~valid] == 0)
        assert np.all(labels[valid] == 255)
        assert np.isfinite(rgb).all()
    metadata = json.loads((out / "scene_metadata.json").read_text())
    assert all(scene["invalid_pixels"] == 32 * 8 + 4 * 8 for scene in metadata)
    # Rasterized polygons lying only over declared NoData are not positive
    # label overlap and must not turn invalid scenery into supervised targets.
    polygon["coordinates"] = [[[0, 32], [8, 32], [8, 0], [0, 0], [0, 32]]]
    (tmp_path / "labels.json").write_text(json.dumps(polygon))
    with pytest.raises(ValueError, match="valid raster pixels"):
        import_geopairs(tmp_path / "pairs.csv", tmp_path / "invalid_only", tile_size=16)
