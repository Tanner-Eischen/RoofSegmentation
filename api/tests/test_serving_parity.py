"""Real CPU computations verify the release boundary and shared serving behavior."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from api.main import Settings, create_app
from roof_pipeline import inference_service
from roof_pipeline.inference_service import ModelService, ModelUnavailable, predict_file
from roof_pipeline.models import create_model
from roof_pipeline.release import prepare_release


@pytest.fixture
def release(tmp_path):
    torch.set_num_threads(2)
    torch.manual_seed(207)
    checkpoint = tmp_path / "weights.pt"
    torch.save(
        {
            "state_dict": create_model("tiny").state_dict(),
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


@pytest.mark.parametrize("mode", ["RGB", "L", "RGBA"])
def test_non_square_cli_api_probability_and_artifact_parity(release, tmp_path, mode):
    checkpoint, manifest = release
    rgb = np.random.default_rng(84).integers(0, 256, (29, 43, 3), dtype=np.uint8)
    image = tmp_path / "input.png"
    Image.fromarray(rgb).convert(mode).save(image)
    cli = predict_file(checkpoint, manifest, image, tmp_path / "cli")
    with TestClient(
        create_app(Settings(str(checkpoint), model_release=str(manifest)))
    ) as client:
        response = client.post(
            "/api/inference",
            files={"file": ("input.png", image.read_bytes(), "image/png")},
        )
    assert response.status_code == 200
    api = response.json()
    api_probability_bytes = base64.b64decode(api["probability_base64"])
    api_probability = np.load(io.BytesIO(api_probability_bytes), allow_pickle=False)
    cli_probability = np.load(tmp_path / "cli/probability.npy", allow_pickle=False)
    assert api_probability.dtype == np.float32 and api_probability.shape == (29, 43)
    np.testing.assert_allclose(api_probability, cli_probability, rtol=0, atol=1e-6)
    assert (
        base64.b64decode(api["mask_base64"]) == (tmp_path / "cli/mask.png").read_bytes()
    )
    assert api["model"] == cli["model"]
    assert api["input_sha256"] == hashlib.sha256(image.read_bytes()).hexdigest()
    assert api["valid_pixel_count"] == 29 * 43
    assert api["area"] is None
    assert "confidence" not in api
    assert (
        hashlib.sha256(api_probability_bytes).hexdigest()
        == api["artifacts"]["probability"]["sha256"]
    )
    preview_bytes = base64.b64decode(api["probability_preview_base64"])
    preview = np.array(Image.open(io.BytesIO(preview_bytes)))
    np.testing.assert_array_equal(
        preview, np.rint(api_probability * 255).astype(np.uint8)
    )
    assert preview_bytes == (tmp_path / "cli/probability.png").read_bytes()
    assert (
        hashlib.sha256(preview_bytes).hexdigest()
        == api["artifacts"]["probability_preview"]["sha256"]
    )


@pytest.mark.parametrize(
    "key,value",
    [
        ("threshold", 0.7),
        ("normalization", "unscaled_rgb"),
        ("preprocessing_version", "different-channel-order"),
        ("overlap", 1),
        ("weights_sha256", "0" * 64),
    ],
)
def test_release_metadata_drift_is_not_ready(release, key, value):
    checkpoint, manifest = release
    record = json.loads(manifest.read_text())
    record[key] = value
    manifest.write_text(json.dumps(record))
    with TestClient(create_app(Settings(str(checkpoint)))) as client:
        assert client.get("/api/health/ready").status_code == 503


def test_missing_release_is_unavailable_and_never_downloaded(release):
    checkpoint, manifest = release
    manifest.unlink()
    with pytest.raises(ModelUnavailable):
        ModelService(checkpoint).load()


def test_cached_checkpoint_hash_is_rechecked_even_with_unchanged_stat(release):
    checkpoint, _ = release
    service = ModelService(checkpoint)
    service.load()
    before = checkpoint.stat()
    content = bytearray(checkpoint.read_bytes())
    content[-1] ^= 1
    checkpoint.write_bytes(content)
    os.utime(checkpoint, ns=(before.st_atime_ns, before.st_mtime_ns))
    with pytest.raises(ModelUnavailable):
        service.load()


def test_zero_foreground_is_valid_not_an_outage(release, monkeypatch):
    checkpoint, _ = release
    monkeypatch.setattr(
        inference_service,
        "predict_array",
        lambda *a, **kw: np.zeros((29, 43), np.float32),
    )
    image = io.BytesIO()
    Image.new("RGB", (43, 29)).save(image, format="PNG")
    result = ModelService(checkpoint).predict(image.getvalue())
    assert result["roof_area_px"] == 0 and result["pixel_coverage"] == 0
    assert result["mean_foreground_probability"] is None
    assert result["mode"] == "real"


def test_readiness_requires_usable_state_dict_not_only_manifest(release):
    checkpoint, manifest = release
    state = torch.load(checkpoint, weights_only=True)
    state["state_dict"] = {}
    torch.save(state, checkpoint)
    manifest.unlink()
    prepare_release(checkpoint, manifest)
    with pytest.raises(ModelUnavailable):
        ModelService(checkpoint).load()


def test_probability_parity_detects_injected_channel_mismatch(
    release, tmp_path, monkeypatch
):
    checkpoint, manifest = release
    rgb = np.random.default_rng(5).integers(0, 256, (29, 43, 3), dtype=np.uint8)
    image = tmp_path / "rgb.png"
    Image.fromarray(rgb).save(image)
    predict_file(checkpoint, manifest, image, tmp_path / "reference")
    expected = np.load(tmp_path / "reference/probability.npy", allow_pickle=False)
    original = inference_service.predict_array

    def swapped(model, actual_rgb, *args, **kwargs):
        return original(model, actual_rgb[..., ::-1].copy(), *args, **kwargs)

    monkeypatch.setattr(inference_service, "predict_array", swapped)
    changed = ModelService(checkpoint).predict(image.read_bytes())
    actual = np.load(io.BytesIO(changed["probability_npy"]), allow_pickle=False)
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-6)


def test_unsupported_image_plugin_cannot_parse_upload_headers(monkeypatch):
    """Reject before plugin dispatch, rather than after parsing a foreign header."""
    Image.init()
    parsed = []

    def foreign_plugin(*args, **kwargs):
        parsed.append(True)
        raise RuntimeError("Unsupported image header was parsed")

    monkeypatch.setitem(
        Image.OPEN,
        "XBM",
        (foreign_plugin, lambda prefix: prefix.startswith(b"#define")),
    )
    with pytest.raises(inference_service.InvalidImage):
        inference_service.decode_image(
            b"#define hostile_header", max_bytes=1024, max_pixels=1024
        )
    assert parsed == []
