"""Behavior tests of the real ASGI app with real tiny checkpoint computations."""

from __future__ import annotations

import base64
import io
import threading
import time

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from api.main import Settings, create_app
from roof_pipeline.inference_service import ModelService, ModelUnavailable
from roof_pipeline.models import create_model
from roof_pipeline.release import prepare_release


def png(size=(32, 32)):
    out = io.BytesIO()
    Image.new("RGB", size, (127, 80, 20)).save(out, format="PNG")
    return out.getvalue()


@pytest.fixture
def checkpoint(tmp_path):
    torch.set_num_threads(2)
    torch.manual_seed(123)
    model = create_model("tiny")
    path = tmp_path / "model.pt"
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
        path,
    )
    prepare_release(path, path.with_suffix(".release.json"))
    return path


def client(checkpoint, **kwargs):
    return TestClient(create_app(Settings(str(checkpoint), **kwargs)))


def post(c, content=None):
    return c.post(
        "/api/inference",
        files={
            "file": ("input.png", png() if content is None else content, "image/png")
        },
    )


def test_actual_prediction_mask_area_polygons_and_readiness(checkpoint):
    with client(checkpoint) as c:
        assert c.get("/api/ready").status_code == 200
        r = post(c)
        assert r.status_code == 200
        d = r.json()
        mask = np.array(Image.open(io.BytesIO(base64.b64decode(d["mask_base64"]))))
        assert set(np.unique(mask)) <= {0, 255}
        assert d["roof_area_px"] == int((mask > 0).sum())
        assert d["mode"] == "real" and d["model"]["device"] == "cpu"
        assert (
            d["area_unit"] == "pixels" and "uncalibrated" in d["probability_definition"]
        )
        assert (
            d["mean_foreground_probability"] is None
            or 0 <= d["mean_foreground_probability"] <= 1
        )
        assert all(
            0 <= x < 32 and 0 <= y < 32 for poly in d["polygons"] for x, y in poly
        )


@pytest.mark.parametrize(
    "kind",
    ["missing", "corrupt", "legacy", "bad_config", "bad_threshold", "bad_weights"],
)
def test_bad_checkpoint_never_becomes_ready_or_success(tmp_path, checkpoint, kind):
    if kind == "missing":
        checkpoint = tmp_path / "absent.pt"
    elif kind == "corrupt":
        checkpoint.write_bytes(b"not a torch checkpoint")
    else:
        state = torch.load(checkpoint, weights_only=True)
        if kind == "legacy":
            state = {"model_state_dict": state["state_dict"], "args": {}}
        if kind == "bad_config":
            state["config"]["normalization"] = "unknown"
        if kind == "bad_threshold":
            state["threshold"] = float("nan")
        if kind == "bad_weights":
            state["state_dict"] = {}
        torch.save(state, checkpoint)
    with client(checkpoint) as c:
        assert c.get("/api/health").status_code == 200
        assert c.get("/api/ready").status_code == 503
        r = post(c)
        assert r.status_code == 503 and "mask_base64" not in r.json()


@pytest.mark.parametrize(
    "content", [b"", b"corrupt", b'<svg xmlns="http://www.w3.org/2000/svg"/>']
)
def test_invalid_images_are_failures(checkpoint, content):
    with client(checkpoint) as c:
        assert post(c, content).status_code == 400


def test_bytes_pixels_batch_and_validation_limits(checkpoint):
    with client(checkpoint, max_upload_bytes=1024, max_pixels=1024) as c:
        assert post(c, b"x" * 1025).status_code == 413
        assert post(c, png((33, 33))).status_code == 413
        assert c.post("/api/inference").status_code == 422
        files = [("files", ("x.png", png(), "image/png"))] * 5
        assert c.post("/api/inference/batch", files=files).status_code == 413
        assert c.post("/api/batch", json={"addresses": ["x"] * 9}).status_code == 422
        assert c.post("/api/inference", content=b"x" * 70000).status_code == 413

        def chunks():
            yield b"x" * 40000
            yield b"x" * 40000

        assert c.post("/api/inference", content=chunks()).status_code == 413


def test_real_local_batch_and_unavailable_remote_are_honest(checkpoint):
    with client(checkpoint) as c:
        files = [("files", ("x.png", png(), "image/png"))] * 2
        r = c.post("/api/inference/batch", files=files)
        assert r.status_code == 200 and r.json()["count"] == 2
        assert all(x["mode"] == "real" for x in r.json()["results"])
        assert c.get("/api/geocode?address=test").status_code == 503
        assert c.get("/api/satellite/image?lat=1&lng=2").status_code == 503
        assert c.post("/api/batch", json={"addresses": ["x"]}).status_code == 503
        assert c.get("/api/unknown").status_code == 404


def test_explicit_demo_never_runs_real_or_fake_inference(checkpoint):
    with client(checkpoint, demo_mode=True) as c:
        assert c.get("/api/health").json()["mode"] == "demo"
        assert c.get("/api/ready").status_code == 503
        assert post(c).status_code == 503


def test_cache_rechecks_replaced_or_missing_checkpoint(checkpoint):
    service = ModelService(checkpoint)
    service.load()
    checkpoint.write_bytes(b"corrupt replacement")
    with pytest.raises(ModelUnavailable):
        service.load()
    assert not service.loaded


def test_nonfinite_output_cannot_be_a_success(checkpoint, monkeypatch):
    from roof_pipeline import inference_service as inference

    monkeypatch.setattr(
        inference, "predict_array", lambda *args, **kwargs: np.full((32, 32), np.nan)
    )
    with client(checkpoint) as c:
        response = post(c)
        assert response.status_code == 503
        assert "mask_base64" not in response.json()


def test_timeout_retains_native_worker_slot(checkpoint, monkeypatch):
    from roof_pipeline import inference_service as inference

    entered, release = threading.Event(), threading.Event()
    original = inference.predict_array

    def delayed(*args, **kwargs):
        entered.set()
        release.wait(5)
        return original(*args, **kwargs)

    monkeypatch.setattr(inference, "predict_array", delayed)
    with client(checkpoint, inference_timeout=0.2) as c:
        assert c.get("/api/ready").status_code == 200
        try:
            assert post(c).status_code == 504
            assert entered.is_set()
            assert post(c).status_code == 503
        finally:
            release.set()
        for _ in range(100):
            if c.app.state.model_service._slot.acquire(blocking=False):
                c.app.state.model_service._slot.release()
                break
            time.sleep(0.01)
        else:
            pytest.fail("Worker did not release slot")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_pixels": 0},
        {"cpu_threads": 5},
        {"max_batch_images": 9},
        {"inference_timeout": float("nan")},
    ],
)
def test_bad_configuration_rejected(checkpoint, kwargs):
    with pytest.raises(ValueError):
        Settings(str(checkpoint), **kwargs)


def test_environment_limits_are_applied(monkeypatch):
    monkeypatch.setenv("ROOF_CPU_THREADS", "1")
    monkeypatch.setenv("ROOF_MAX_BATCH_IMAGES", "2")
    settings = Settings.from_env()
    assert settings.cpu_threads == 1 and settings.max_batch_images == 2


@pytest.mark.parametrize(
    "name,value",
    [
        ("ROOF_CPU_THREADS", "5"),
        ("ROOF_MAX_BATCH_IMAGES", "9"),
        ("ROOF_DEMO_MODE", "yes"),
    ],
)
def test_invalid_environment_rejected(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError):
        Settings.from_env()
