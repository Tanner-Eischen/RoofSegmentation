"""Real-checkpoint CPU CLI/HTTP parity on a generated image; no quality claim.

The one-window independent forward pass detects normalization/channel drift.
Uses the configured local checkpoint only, never dataset imagery or a browser.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.metadata
import io
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import numpy as np
import torch
from PIL import Image

from roof_pipeline.models import load_model


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(checkpoint, release, out):
    checkpoint, release, out = checkpoint.resolve(), release.resolve(), out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    # Deliberately non-square, with distinct RGB channels and reproducible texture.
    rows, cols = np.indices((97, 131))
    rgb = np.stack(
        (
            (3 * rows + cols) % 256,
            (rows + 5 * cols) % 256,
            (7 * rows + 11 * cols) % 256,
        ),
        axis=-1,
    ).astype(np.uint8)
    image = out / "generated-fixture.png"
    Image.fromarray(rgb).save(image)
    record = json.loads(release.read_text(encoding="utf-8"))
    chip = record["image_size"]
    if chip < max(rgb.shape[:2]):
        raise ValueError(
            "Integration proof requires the registered tile to fit this fixture"
        )
    torch.set_num_threads(2)
    model, _ = load_model(checkpoint, "cpu")
    padded = np.pad(rgb, ((0, chip - 97), (0, chip - 131), (0, 0)), mode="edge")
    normalized = padded.astype(np.float32) / 255
    normalized = (normalized - np.array([0.485, 0.456, 0.406], np.float32)) / np.array(
        [0.229, 0.224, 0.225], np.float32
    )
    tensor = torch.from_numpy(normalized).permute(2, 0, 1)[None]
    with torch.inference_mode():
        direct = model(tensor).sigmoid()[0, 0].numpy()[:97, :131]
    np.save(out / "independent-probability.npy", direct, allow_pickle=False)
    del model, tensor
    environment = dict(
        os.environ,
        MODEL_PATH=str(checkpoint),
        ROOF_MODEL_RELEASE=str(release),
        ROOF_DEMO_MODE="false",
        ROOF_CPU_THREADS="2",
        CUDA_VISIBLE_DEVICES="",
    )
    creation = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    cli = subprocess.run(
        [
            sys.executable,
            "-m",
            "roof_pipeline.inference_service",
            "--checkpoint",
            str(checkpoint),
            "--model-release",
            str(release),
            "--image",
            str(image),
            "--out",
            str(out / "cli"),
        ],
        cwd=root,
        env=environment,
        text=True,
        capture_output=True,
        creationflags=creation,
        check=True,
    )
    (out / "cli.log").write_text(cli.stdout + cli.stderr, encoding="utf-8")
    cli_probability = np.load(out / "cli/probability.npy", allow_pickle=False)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    with (out / "server.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "api.main:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
            ],
            cwd=root,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            creationflags=creation,
        )
        try:
            with httpx.Client(
                base_url=f"http://127.0.0.1:{port}", timeout=60, trust_env=False
            ) as client:
                for _ in range(160):
                    if process.poll() is not None:
                        raise RuntimeError("HTTP server exited; inspect server.log")
                    try:
                        ready = client.get("/api/health/ready")
                        if ready.status_code == 200:
                            break
                    except httpx.TransportError:
                        pass
                    time.sleep(0.25)
                else:
                    raise RuntimeError("Readiness deadline exceeded")
                response = client.post(
                    "/api/inference",
                    files={"file": (image.name, image.read_bytes(), "image/png")},
                )
                response.raise_for_status()
                payload = response.json()
                mask_bytes = base64.b64decode(payload.pop("mask_base64"), validate=True)
                probability_bytes = base64.b64decode(
                    payload.pop("probability_base64"), validate=True
                )
                (out / "http-mask.png").write_bytes(mask_bytes)
                (out / "http-probability.npy").write_bytes(probability_bytes)
                (out / "response.json").write_text(
                    json.dumps(payload, indent=2) + "\n", encoding="utf-8"
                )
                probability = np.load(io.BytesIO(probability_bytes), allow_pickle=False)
                np.testing.assert_allclose(
                    probability, cli_probability, rtol=0, atol=1e-6
                )
                np.testing.assert_allclose(probability, direct, rtol=0, atol=1e-6)
                actual_mask = np.array(Image.open(io.BytesIO(mask_bytes)))
                expected_mask = (cli_probability >= record["threshold"]).astype(
                    np.uint8
                ) * 255
                assert np.array_equal(actual_mask, expected_mask)
                assert mask_bytes == (out / "cli/mask.png").read_bytes()
                assert payload["model"]["checkpoint_sha256"] == digest(checkpoint)
                assert payload["model"]["model_release_sha256"] == digest(release)
                assert payload["input_sha256"] == digest(image)
                assert payload["valid_pixel_count"] == 97 * 131
                assert payload["roof_area_px"] == int((actual_mask > 0).sum())
                assert (
                    hashlib.sha256(mask_bytes).hexdigest()
                    == payload["artifacts"]["mask"]["sha256"]
                )
                assert (
                    hashlib.sha256(probability_bytes).hexdigest()
                    == payload["artifacts"]["probability"]["sha256"]
                )
                assert (
                    client.post(
                        "/api/inference",
                        files={"file": ("bad.png", b"invalid", "image/png")},
                    ).status_code
                    == 400
                )
                assert client.get("/api/satellite/image").status_code == 503
                proof = {
                    "schema_version": 1,
                    "status": "passed",
                    "checkpoint_sha256": digest(checkpoint),
                    "model_release_sha256": digest(release),
                    "input_sha256": digest(image),
                    "input_shape": list(rgb.shape),
                    "input_kind": "generated numerical fixture; no source imagery",
                    "scope": "Single-input float32 CPU integration; no model accuracy or geographic performance measurement",
                    "cli_api_probability_max_abs_diff": float(
                        np.abs(probability - cli_probability).max()
                    ),
                    "independent_forward_probability_max_abs_diff": float(
                        np.abs(probability - direct).max()
                    ),
                    "tolerance_abs": 1e-6,
                    "thresholded_masks_identical": True,
                    "probability_artifact_sha256": digest(out / "http-probability.npy"),
                    "mask_artifact_sha256": digest(out / "http-mask.png"),
                    "pixel_coverage": payload["pixel_coverage"],
                    "valid_pixel_count": payload["valid_pixel_count"],
                    "checks": {
                        "hash_bound_model_release": True,
                        "corrupt_upload_rejected": True,
                        "unavailable_imagery_is_failure": True,
                        "artifact_hashes_match": True,
                    },
                    "cpu_threads": 2,
                    "runtime_versions": {
                        name: importlib.metadata.version(name)
                        for name in (
                            "torch",
                            "torchvision",
                            "numpy",
                            "Pillow",
                            "segmentation-models-pytorch",
                            "fastapi",
                            "starlette",
                            "python-multipart",
                            "httpx",
                        )
                    },
                    "source_sha256": {
                        name: digest(root / name)
                        for name in (
                            "api/main.py",
                            "roof_pipeline/inference_service.py",
                            "roof_pipeline/release.py",
                            "roof_pipeline/runtime.py",
                            "roof_pipeline/models.py",
                            "roof_pipeline/dataset.py",
                            "roof_pipeline/predict.py",
                            "scripts/verify_serving_parity.py",
                        )
                    },
                    "not_run": [
                        "browser",
                        "container",
                        "deployment",
                        "training",
                        "dataset-wide evaluation",
                    ],
                }
                (out / "proof.json").write_text(
                    json.dumps(proof, indent=2) + "\n", encoding="utf-8"
                )
                return proof
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-release", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.checkpoint, args.model_release, args.out), indent=2))


if __name__ == "__main__":
    main()
