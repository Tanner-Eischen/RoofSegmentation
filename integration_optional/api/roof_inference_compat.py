"""Optional compatibility layer for the existing RoofSegmentation FastAPI API.

Replace ONLY the import in api/main.py:
    from inference import image_bytes_to_mask_and_polygons, load_model
with:
    from roof_inference_compat import image_bytes_to_mask_and_polygons, load_model

New roof_pipeline checkpoint is preferred. Existing old-format checkpoint is
handled by the unchanged legacy inference module (only for trusted files).
"""

from __future__ import annotations

import io
import sys
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if not (ROOT / "roof_pipeline").exists():
    # When copied to repo/api, parents[1] will point to repository root.
    ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _new_checkpoint(path: Path) -> bool:
    if not path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {path}")
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        # Old checkpoints can contain unsupported pickles; treat only as legacy.
        return False
    return isinstance(state, dict) and "state_dict" in state and "config" in state


@lru_cache(maxsize=2)
def _load_new(path_string: str):
    from roof_pipeline.models import load_model as load_roof_model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg = load_roof_model(path_string, device)
    state = torch.load(path_string, map_location="cpu", weights_only=True)
    args = {
        "threshold": float(state["threshold"]),
        "arch": cfg["arch"],
        "image_size": int(cfg["image_size"]),
        "confidence_definition": "mean model probability over predicted positive pixels; not calibrated",
    }
    return model, torch.device(device), int(cfg["image_size"]), args


def load_model(model_path, auto_download=False):
    path = Path(model_path).resolve()
    if _new_checkpoint(path):
        return _load_new(str(path))
    # Legacy checkpoints remain the responsibility of the legacy loader.
    # It uses weights_only=False, so it must only be run on your own trusted files.
    from inference import load_model as legacy_load

    return legacy_load(path, auto_download=auto_download)


def image_bytes_to_mask_and_polygons(image_bytes, model_path):
    path = Path(model_path).resolve()
    if not _new_checkpoint(path):
        from inference import image_bytes_to_mask_and_polygons as legacy_infer

        return legacy_infer(image_bytes, path)
    from roof_pipeline.predict import predict_array

    model, device, size, args = load_model(path)
    try:
        rgb = np.asarray(Image.open(io.BytesIO(image_bytes)).convert("RGB"))
    except Exception as e:
        raise ValueError("File is not a supported RGB image") from e
    if rgb.shape[0] * rgb.shape[1] > 24_000_000:
        raise ValueError("Image is too large for this in-memory inference endpoint")
    prob = predict_array(model, rgb, device, chip=size, overlap=min(64, size // 4))
    binary = (prob >= args["threshold"]).astype(np.uint8)
    cnt, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = [c.reshape(-1, 2).astype(float).tolist() for c in cnt if c.shape[0] >= 3]
    mask = (binary * 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(mask).save(buf, format="PNG")
    positive = prob[binary > 0]
    mean_positive_prob = float(positive.mean() * 100) if positive.size else 0.0
    return buf.getvalue(), polys, float(binary.sum()), mean_positive_prob
