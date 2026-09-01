"""
Load trained U-Net and run inference: image -> binary mask + optional polygons.
Matches train.py preprocessing (resize, ImageNet normalize).
"""

import io
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
import segmentation_models_pytorch as smp

# Defaults from train.py
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
DEFAULT_IMAGE_SIZE = 256
THRESHOLD = 0.5

# Precomputed arrays for efficient preprocessing
_IMAGENET_MEAN_ARR = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(1, 1, 3)
_IMAGENET_STD_ARR = np.array(IMAGENET_STD, dtype=np.float32).reshape(1, 1, 3)

_model_cache: dict[str, Any] = {}  # path -> (model, device, image_size, args)


def load_model(model_path: str | Path, auto_download: bool = True) -> tuple[Any, torch.device, int, dict]:
    """Load best.pt and return (model, device, image_size, args). Cached by path.

    If model not found locally and auto_download=True, downloads from Hugging Face.
    """
    path = Path(model_path).resolve()
    path_str = str(path)
    if path_str in _model_cache:
        return _model_cache[path_str]

    if not path.exists():
        if auto_download:
            print(f"Model not found at {path}. Downloading from Hugging Face...")
            try:
                from huggingface_hub import hf_hub_download
                hf_hub_download(
                    repo_id="TannerEischen/roof-segmentation",
                    filename="best.pt",
                    local_dir=str(path.parent),
                )
                print(f"Model downloaded to {path}")
            except Exception as e:
                raise FileNotFoundError(f"Model not found and download failed: {e}")
        else:
            raise FileNotFoundError(f"Model not found: {path}")

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    args = ckpt.get("args") or {}
    encoder = args.get("encoder", "resnet34")
    encoder_weights = args.get("encoder_weights") or None
    image_size = int(args.get("image_size", DEFAULT_IMAGE_SIZE))

    # Build the architecture the checkpoint was trained with (train_v2.py saves
    # args.arch). Legacy train.py checkpoints omit it -> default to U-Net.
    arch = args.get("arch", "unet")
    arches = {
        "unet": smp.Unet,
        "unetplusplus": smp.UnetPlusPlus,
        "deeplabv3plus": smp.DeepLabV3Plus,
    }
    model_cls = arches.get(arch, smp.Unet)
    model = model_cls(
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=1,
    )
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # Surface the tuned threshold (default 0.5 for legacy checkpoints) so
    # run_inference uses it automatically.
    args = dict(args)
    args.setdefault("threshold", float(ckpt.get("threshold", THRESHOLD)))

    _model_cache[path_str] = (model, device, image_size, args)
    return model, device, image_size, args


def preprocess(image: np.ndarray, image_size: int) -> torch.Tensor:
    """RGB numpy HWC [0,255] -> tensor 1xCxHxW normalized."""
    img = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) / 255.0
    img = (img - _IMAGENET_MEAN_ARR) / _IMAGENET_STD_ARR
    x = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
    return x


def run_inference(
    image: np.ndarray,
    model_path: str | Path,
    threshold: float | None = None,
) -> tuple[np.ndarray, list[list[list[float]]], float, float]:
    """
    Run segmentation on RGB image (HWC, 0-255).
    Returns (mask_uint8, polygons, roof_area_px, confidence).
    polygons: list of contours, each contour is list of [x,y] in original image coords.
    confidence: mean probability of predicted roof pixels (0-100).

    If threshold is None, uses the tuned threshold stored in the checkpoint
    (default 0.5 for legacy checkpoints).
    """
    model, device, image_size, args = load_model(model_path)
    if threshold is None:
        threshold = float(args.get("threshold", THRESHOLD))
    orig_h, orig_w = image.shape[:2]

    x = preprocess(image, image_size).to(device)
    with torch.no_grad():
        logits = model(x)
    pred = torch.sigmoid(logits).squeeze(0).squeeze(0).cpu().numpy()
    mask_small = (pred >= threshold).astype(np.uint8) * 255

    # Calculate confidence from mean probability of roof pixels
    roof_probs = pred[pred >= threshold]
    confidence = float(roof_probs.mean() * 100) if len(roof_probs) > 0 else 0.0

    # Resize mask back to original size for polygon extraction
    mask_full = cv2.resize(
        mask_small,
        (orig_w, orig_h),
        interpolation=cv2.INTER_NEAREST,
    )

    # Contours in (x, y) image coordinates
    contours, _ = cv2.findContours(
        mask_full,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    polygons = []
    for c in contours:
        if c.size >= 6:  # at least 3 points
            poly = c.reshape(-1, 2).tolist()
            polygons.append(poly)

    roof_area_px = float(np.sum(mask_full > 0))

    return mask_full, polygons, roof_area_px, confidence


def image_bytes_to_mask_and_polygons(
    image_bytes: bytes,
    model_path: str | Path,
) -> tuple[bytes, list[list[list[float]]], float, float]:
    """
    Read image from bytes, run inference, return (mask_png_bytes, polygons, roof_area_px, confidence).
    """
    img = np.array(Image.open(io.BytesIO(image_bytes)).convert("RGB"))
    mask, polygons, area, confidence = run_inference(img, model_path)

    buf = io.BytesIO()
    Image.fromarray(mask).save(buf, format="PNG")
    return buf.getvalue(), polygons, area, confidence
