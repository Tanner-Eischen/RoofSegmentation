"""Real CPU serving excludes alpha-invalid pixels from masks and measurements."""

import base64
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from api.main import Settings, create_app
from roof_pipeline.inference_service import (
    InvalidImage,
    ModelService,
    ModelUnavailable,
    decode_image_with_validity,
    predict_file,
)
from roof_pipeline.models import create_model
from roof_pipeline.release import (
    ALPHA_POLICY,
    PREPROCESSING_VERSION,
    prepare_release,
    read_checkpoint,
    validate_release,
)


@pytest.fixture
def constant_release(tmp_path):
    torch.set_num_threads(2)
    model = create_model("tiny")
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.head.bias.fill_(2)
    checkpoint = tmp_path / "constant.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": {
                "arch": "tiny",
                "image_size": 32,
                "normalization": "imagenet_rgb_v1",
            },
            "threshold": 0.5,
        },
        checkpoint,
    )
    manifest = checkpoint.with_suffix(".release.json")
    prepare_release(checkpoint, manifest)
    return checkpoint, manifest


def png_bytes(array):
    content = io.BytesIO()
    Image.fromarray(array).save(content, format="PNG")
    return content.getvalue()


def test_alpha_validity_counts_and_cli_api_artifact_parity(constant_release, tmp_path):
    checkpoint, manifest = constant_release
    rgba = np.zeros((29, 43, 4), np.uint8)
    rgba[3:26, 5:38, 3] = 255
    rgba[4, 6, 3] = 128
    rgba[0, 0, :3] = 255
    valid = rgba[..., 3] > 0
    content = png_bytes(rgba)
    image = tmp_path / "alpha.png"
    image.write_bytes(content)
    cli = predict_file(checkpoint, manifest, image, tmp_path / "cli")
    with TestClient(
        create_app(Settings(str(checkpoint), model_release=str(manifest)))
    ) as client:
        response = client.post(
            "/api/inference", files={"file": ("alpha.png", content, "image/png")}
        )
    assert response.status_code == 200
    api = response.json()
    mask = np.asarray(Image.open(io.BytesIO(base64.b64decode(api["mask_base64"]))))
    validity_bytes = base64.b64decode(api["validity_base64"])
    validity = np.asarray(Image.open(io.BytesIO(validity_bytes)))
    probability = np.load(
        io.BytesIO(base64.b64decode(api["probability_base64"])), allow_pickle=False
    )
    assert mask.shape == valid.shape and np.array_equal(mask > 0, valid)
    np.testing.assert_array_equal(validity > 0, valid)
    assert not probability[~valid].any()
    assert api["roof_area_px"] == api["valid_pixel_count"] == int(valid.sum())
    assert api["pixel_coverage"] == 1.0 and api[
        "mean_foreground_probability"
    ] == pytest.approx(float(probability[valid].mean()))
    assert api["total_pixel_count"] == 29 * 43
    assert api["invalid_pixel_count"] == 29 * 43 - int(valid.sum())
    assert validity_bytes == (tmp_path / "cli/validity.png").read_bytes()
    assert (
        base64.b64decode(api["mask_base64"]) == (tmp_path / "cli/mask.png").read_bytes()
    )
    assert (
        base64.b64decode(api["probability_base64"])
        == (tmp_path / "cli/probability.npy").read_bytes()
    )
    assert (
        hashlib.sha256(validity_bytes).hexdigest()
        == api["artifacts"]["validity"]["sha256"]
    )
    assert cli["valid_pixel_count"] == api["valid_pixel_count"]


def test_all_transparent_input_is_rejected_before_model_loading(
    constant_release, monkeypatch
):
    checkpoint, manifest = constant_release
    service = ModelService(checkpoint, release_manifest=manifest)
    monkeypatch.setattr(
        service, "load", lambda: pytest.fail("invalid image must be rejected first")
    )
    content = png_bytes(np.zeros((17, 23, 4), np.uint8))
    with pytest.raises(InvalidImage, match="no valid pixels"):
        service.predict(content)
    with TestClient(
        create_app(Settings(str(checkpoint), model_release=str(manifest)))
    ) as client:
        response = client.post(
            "/api/inference", files={"file": ("empty.png", content, "image/png")}
        )
    assert (
        response.status_code == 400 and "no valid pixels" in response.json()["detail"]
    )


@pytest.mark.parametrize("value", [0, 255])
def test_unflagged_black_and_white_uploads_are_valid(constant_release, value):
    checkpoint, manifest = constant_release
    content = png_bytes(np.full((17, 23, 3), value, np.uint8))
    rgb, valid = decode_image_with_validity(content, max_bytes=10000, max_pixels=10000)
    assert valid.all() and np.all(rgb == value)
    result = ModelService(checkpoint, release_manifest=manifest).predict(content)
    assert result["valid_pixel_count"] == result["roof_area_px"] == 17 * 23
    assert result["invalid_pixel_count"] == 0 and result["pixel_coverage"] == 1


def test_flat_polygons_do_not_claim_invalid_holes(constant_release):
    checkpoint, manifest = constant_release
    rgba = np.zeros((32, 32, 4), np.uint8)
    rgba[..., 3] = 255
    rgba[12:20, 12:20, 3] = 0
    result = ModelService(checkpoint, release_manifest=manifest).predict(
        png_bytes(rgba)
    )
    assert result["roof_area_px"] == 32 * 32 - 8 * 8
    assert (
        result["polygons"] == [] and result["polygon_components_omitted_invalid"] == 1
    )
    assert "authoritative" in result["polygon_limitation"]


def test_old_alpha_release_is_rejected_and_regeneration_keeps_weights(
    constant_release, tmp_path
):
    checkpoint, manifest = constant_release
    original_weight_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    record = json.loads(manifest.read_text())
    assert record["preprocessing_version"] == PREPROCESSING_VERSION
    assert record["alpha_policy"] == ALPHA_POLICY
    record["preprocessing_version"] = "imagenet-rgb-edgepad-tiled-v1"
    record["alpha_policy"] = (
        "Pillow RGB conversion discards alpha without compositing; grayscale repeats into RGB"
    )
    record.pop("validity_policy")
    manifest.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="fresh v2 release"):
        validate_release(record)
    with pytest.raises(ModelUnavailable):
        ModelService(checkpoint, release_manifest=manifest).load()
    fresh = tmp_path / "fresh.release.json"
    prepared = prepare_release(checkpoint, fresh)
    assert prepared["weights_sha256"] == original_weight_hash
    assert (
        ModelService(checkpoint, release_manifest=fresh).load()[1][
            "preprocessing_version"
        ]
        == PREPROCESSING_VERSION
    )
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == original_weight_hash


@pytest.mark.parametrize("grow", [False, True])
def test_checkpoint_reader_bounds_allocation_and_rejects_growth(
    constant_release, monkeypatch, grow
):
    checkpoint, _ = constant_release
    original = checkpoint.read_bytes()
    original_open = Path.open
    reads = []

    class RecordedFile:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            self.stream.__enter__()
            return self

        def __exit__(self, *args):
            return self.stream.__exit__(*args)

        def fileno(self):
            return self.stream.fileno()

        def read(self, size):
            reads.append(size)
            if grow:
                with original_open(checkpoint, "ab") as append:
                    append.write(b"changed")
            return self.stream.read(size)

    def recording_open(path, *args, **kwargs):
        stream = original_open(path, *args, **kwargs)
        return (
            RecordedFile(stream) if path == checkpoint and args == ("rb",) else stream
        )

    monkeypatch.setattr(Path, "open", recording_open)
    if grow:
        with pytest.raises(ValueError, match="changed while reading"):
            read_checkpoint(checkpoint)
    else:
        assert read_checkpoint(checkpoint) == original
    assert reads == [len(original) + 1]
