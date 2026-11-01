"""CPU-only serving for roof_pipeline checkpoints. No download/pickle/demo fallback."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import threading
import uuid
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, UnidentifiedImageError

from roof_pipeline.models import DEFAULT_ASPP_RATES, create_model
from roof_pipeline.predict import predict_array
from roof_pipeline.release import checkpoint_config, read_checkpoint, validate_release


class ModelUnavailable(RuntimeError):
    pass


class InvalidImage(ValueError):
    pass


class ImageTooLarge(InvalidImage):
    pass


class InferenceBusy(RuntimeError):
    pass


def decode_image(content: bytes, *, max_bytes: int, max_pixels: int) -> np.ndarray:
    if not content:
        raise InvalidImage("Empty image")
    if len(content) > max_bytes:
        raise ImageTooLarge("Image exceeds upload byte limit")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(
                io.BytesIO(content), formats=("PNG", "JPEG", "WEBP")
            ) as image:
                if image.format not in {"PNG", "JPEG", "WEBP"}:
                    raise InvalidImage("Supported formats are PNG, JPEG and WebP")
                if getattr(image, "n_frames", 1) != 1:
                    raise InvalidImage("Animated images are unsupported")
                if image.width * image.height > max_pixels:
                    raise ImageTooLarge("Image exceeds decoded pixel limit")
                image.load()
                return np.array(image.convert("RGB"), dtype=np.uint8)
    except ImageTooLarge:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ImageTooLarge("Image exceeds decoded pixel limit") from exc
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise InvalidImage("Invalid or unsupported image") from exc


class ModelService:
    """One bounded CPU implementation used by HTTP and the serving CLI.

    A separate, explicitly prepared release manifest pins the checkpoint before
    safe state-dictionary loading. Neither health checks nor inference download.
    """

    def __init__(
        self,
        checkpoint,
        *,
        release_manifest=None,
        max_bytes=8_000_000,
        max_pixels=4_194_304,
    ):
        self.path = Path(checkpoint).resolve()
        self.release_path = (
            Path(release_manifest).resolve()
            if release_manifest
            else self.path.with_suffix(".release.json")
        )
        self.max_bytes, self.max_pixels = max_bytes, max_pixels
        self._fingerprint = None
        self._loaded = None
        self._load_lock = threading.Lock()
        self._slot = threading.BoundedSemaphore(1)

    @property
    def loaded(self):
        return self._loaded is not None

    def load(self):
        with self._load_lock:
            try:
                if self.release_path.stat().st_size > 1_000_000:
                    raise ValueError("Release manifest exceeds size limit")
                release_bytes = self.release_path.read_bytes()
                release = json.loads(release_bytes)
                validate_release(release)
                content = read_checkpoint(self.path)
                checkpoint_hash = hashlib.sha256(content).hexdigest()
                release_hash = hashlib.sha256(release_bytes).hexdigest()
                if checkpoint_hash != release["weights_sha256"]:
                    raise ValueError("Checkpoint does not match prepared release")
                fingerprint = (checkpoint_hash, release_hash)
                if self._loaded is not None and fingerprint == self._fingerprint:
                    return self._loaded
                self._loaded = None
                state = torch.load(
                    io.BytesIO(content), map_location="cpu", weights_only=True
                )
                cfg, threshold = checkpoint_config(state)
                expected = {
                    "architecture": cfg["arch"],
                    "image_size": cfg["image_size"],
                    "normalization": cfg["normalization"],
                    "threshold": threshold,
                    "aspp_rates": cfg.get("aspp_rates"),
                    "boundary_supervision": cfg.get("boundary_supervision", False),
                    "training_dataset_content_sha256": cfg.get(
                        "dataset_content_sha256"
                    ),
                    "training_dataset_manifest_sha256": cfg.get(
                        "dataset_manifest_sha256"
                    ),
                }
                if any(release.get(key) != value for key, value in expected.items()):
                    raise ValueError("Release and checkpoint configuration disagree")
                rates = cfg.get("aspp_rates")
                if cfg["arch"] == "deeplabv3plus-resnet34" and rates is None:
                    rates = DEFAULT_ASPP_RATES
                model = (
                    create_model(
                        cfg["arch"],
                        pretrained=False,
                        aspp_rates=rates,
                        boundary_supervision=cfg.get("boundary_supervision", False),
                    )
                    .cpu()
                    .eval()
                )
                model.load_state_dict(state["state_dict"], strict=True)
                with torch.inference_mode():
                    probe = model(torch.zeros((1, 3, 32, 32)))
                if probe.shape != (1, 1, 32, 32) or not torch.isfinite(probe).all():
                    raise ValueError("Invalid model output")
                metadata = {
                    "model_id": release["model_id"],
                    "model_release_sha256": release_hash,
                    "checkpoint_sha256": checkpoint_hash,
                    "architecture": cfg["arch"],
                    "image_size": cfg["image_size"],
                    "normalization": cfg["normalization"],
                    "threshold": threshold,
                    "device": "cpu",
                    "precision": "float32",
                    "preprocessing_version": release["preprocessing_version"],
                    "overlap": release["overlap"],
                    "label_definition": release["label_definition"],
                }
                self._loaded = model, metadata
                self._fingerprint = fingerprint
                return self._loaded
            except Exception as exc:
                self._loaded = None
                self._fingerprint = None
                raise ModelUnavailable(
                    "Configured checkpoint or model release is missing or invalid"
                ) from exc

    def predict(self, content):
        if not self._slot.acquire(blocking=False):
            raise InferenceBusy("Inference busy; retry after active request finishes")
        try:
            rgb = decode_image(
                content, max_bytes=self.max_bytes, max_pixels=self.max_pixels
            )
            model, metadata = self.load()
            probability = predict_array(
                model,
                rgb,
                torch.device("cpu"),
                chip=metadata["image_size"],
                overlap=metadata["overlap"],
            )
            if (
                probability.shape != rgb.shape[:2]
                or not np.isfinite(probability).all()
                or np.any(probability < 0)
                or np.any(probability > 1)
            ):
                raise ModelUnavailable("Model produced invalid probabilities")
            binary = (probability >= metadata["threshold"]).astype(np.uint8)
            contours, _ = cv2.findContours(
                binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            polygons = [c.reshape(-1, 2).tolist() for c in contours if len(c) >= 3]
            mask_stream, probability_stream = io.BytesIO(), io.BytesIO()
            Image.fromarray(binary * 255).save(mask_stream, format="PNG")
            np.save(
                probability_stream, probability.astype(np.float32), allow_pickle=False
            )
            mask_bytes, probability_bytes = (
                mask_stream.getvalue(),
                probability_stream.getvalue(),
            )
            positive = probability[binary > 0]
            run_id = str(uuid.uuid4())
            artifacts = {
                "mask": {
                    "filename": run_id + "-mask.png",
                    "sha256": hashlib.sha256(mask_bytes).hexdigest(),
                    "media_type": "image/png",
                    "bytes": len(mask_bytes),
                },
                "probability": {
                    "filename": run_id + "-probability.npy",
                    "sha256": hashlib.sha256(probability_bytes).hexdigest(),
                    "media_type": "application/octet-stream",
                    "format": "numpy_float32_no_pickle",
                    "bytes": len(probability_bytes),
                },
            }
            return {
                "run_id": run_id,
                "mask_png": mask_bytes,
                "probability_npy": probability_bytes,
                "input_sha256": hashlib.sha256(content).hexdigest(),
                "polygons": polygons,
                "roof_area_px": int(binary.sum()),
                "pixel_coverage": float(binary.mean()),
                "valid_pixel_count": int(binary.size),
                "mean_foreground_probability": float(positive.mean())
                if positive.size
                else None,
                "probability_definition": "Mean model probability over predicted foreground pixels; uncalibrated",
                "width": rgb.shape[1],
                "height": rgb.shape[0],
                "area": None,
                "area_unit": "pixels",
                "georeferencing_status": "unavailable_image_space_only",
                "polygon_coordinates": "image_pixels_xy",
                "polygon_limitation": "External simplified contours; holes and sub-three-point components omitted",
                "mode": "real",
                "model": metadata,
                "artifacts": artifacts,
            }
        finally:
            self._slot.release()


def predict_file(checkpoint, release_manifest, image, out, *, cpu_threads=2):
    """The serving CLI calls the same service and writes its exact artifact bytes."""
    if type(cpu_threads) is not int or not 1 <= cpu_threads <= 4:
        raise ValueError("cpu_threads must be 1..4")
    torch.set_num_threads(cpu_threads)
    out = Path(out)
    if out.exists():
        raise FileExistsError("Output exists; choose a fresh run directory")
    service = ModelService(checkpoint, release_manifest=release_manifest)
    with Path(image).open("rb") as stream:
        content = stream.read(service.max_bytes + 1)
    result = service.predict(content)
    out.mkdir(parents=True)
    (out / "mask.png").write_bytes(result.pop("mask_png"))
    (out / "probability.npy").write_bytes(result.pop("probability_npy"))
    result["artifacts"]["mask"]["path"] = "mask.png"
    result["artifacts"]["probability"]["path"] = "probability.npy"
    (out / "prediction_manifest.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-release", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cpu-threads", type=int, default=2)
    args = parser.parse_args()
    print(
        json.dumps(
            predict_file(
                args.checkpoint,
                args.model_release,
                args.image,
                args.out,
                cpu_threads=args.cpu_threads,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
