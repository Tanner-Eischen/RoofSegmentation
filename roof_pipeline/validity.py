"""Explicit raster validity, kept separate from binary building class labels.

Black/white RGB values are data unless declared invalid by metadata. A valid
pixel requires all RGB bands, alpha (when present) and finite numeric values.
Prepared validity sidecars use 0=invalid and 1/255=valid; training targets use -1
for ignored pixels, preserving the existing (image, target) dataset API.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

IGNORE_LABEL = -1
VALIDITY_POLICY = "explicit_rgb_label_sidecar_validity_v1"


def target_validity(labels):
    """Validate model targets: binary classes or the explicit ignore sentinel."""
    import torch

    if (
        not torch.isfinite(labels).all()
        or not ((labels == IGNORE_LABEL) | (labels == 0) | (labels == 1)).all()
    ):
        raise ValueError("Targets must contain only 0, 1 or -1 (invalid)")
    return labels != IGNORE_LABEL


def pil_rgb(image: Image.Image) -> tuple[np.ndarray, np.ndarray]:
    """Decode PNG transparency/alpha without guessing validity from RGB colors."""
    if set(image.getbands()) & {"A", "a"} or "transparency" in image.info:
        rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8)
        return rgba[..., :3].copy(), rgba[..., 3] > 0
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    return rgb, np.ones(rgb.shape[:2], dtype=bool)


def _raster_path(path: Path) -> bool:
    return path.suffix.lower() in {".tif", ".tiff"} or any(
        Path(str(path) + suffix).is_file() for suffix in (".msk", ".aux.xml")
    )


def _alpha_valid(src) -> np.ndarray:
    from rasterio.enums import ColorInterp

    valid = np.ones((src.height, src.width), dtype=bool)
    for band, interpretation in enumerate(src.colorinterp, 1):
        if interpretation == ColorInterp.alpha:
            alpha = src.read(band)
            valid &= np.isfinite(alpha) & (alpha > 0)
    return valid


def _uint8_rgb(rgb, valid):
    """Preserve historical all-valid float32 stretching; exclude NoData jointly."""
    if rgb.dtype == np.uint8:
        result = rgb.copy()
        result[~valid] = 0
        return result
    result = np.zeros(rgb.shape, dtype=np.uint8)
    for band in range(3):
        if valid.all():
            with np.errstate(over="ignore", invalid="ignore"):
                channel = rgb[..., band].astype(np.float32)
            if np.isfinite(channel).all():
                lower, upper = np.percentile(channel, (2, 98))
                result[..., band] = np.clip(
                    (channel - lower) * 255 / max(upper - lower, 1e-8), 0, 255
                ).astype(np.uint8)
                continue
        values = rgb[..., band][valid].astype(np.float64)
        if not values.size:
            continue
        lower, upper = np.percentile(values, (2, 98))
        scaled = (values - lower) * 255 / max(upper - lower, 1e-8)
        result[..., band][valid] = np.clip(scaled, 0, 255).astype(np.uint8)
    return result


def read_rgb(path: Path) -> tuple[np.ndarray, np.ndarray, dict | None]:
    path = Path(path)
    if _raster_path(path):
        import rasterio

        with rasterio.open(path) as src:
            if src.count < 3:
                raise ValueError("Raster imagery needs at least three RGB bands")
            bands = src.read((1, 2, 3))
            valid = (src.read_masks((1, 2, 3)) > 0).all(axis=0)
            valid &= np.isfinite(bands).all(axis=0) & _alpha_valid(src)
            rgb = _uint8_rgb(np.moveaxis(bands, 0, -1), valid)
            meta = {"crs": src.crs, "transform": src.transform, "profile": src.profile}
        return rgb, valid, meta
    with Image.open(path) as image:
        rgb, valid = pil_rgb(image)
    return rgb, valid, None


def read_label(path: Path, expected_size=None) -> tuple[np.ndarray, np.ndarray]:
    """Decode class indices, respecting declared NoData instead of labeling it roof.

    NoData metadata takes precedence over label values; without that metadata,
    class 0 is valid background and classes 1/255 are valid building. One-bit
    WhiteIsZero TIFFs encode image luminance, so they are decoded through PIL
    before binarization; multibit palette indices remain class IDs. Label alpha
    is a validity declaration, not a display-only overlay convention.
    """
    path = Path(path)
    if _raster_path(path):
        import rasterio
        from rasterio.enums import ColorInterp

        with rasterio.open(path) as src:
            if src.count != 1 and not (
                src.count == 2 and src.colorinterp[1] == ColorInterp.alpha
            ):
                raise ValueError(f"Mask must have one class band: {path}")
            values = src.read(1)  # Palette indices are classes, not display colors.
            valid = (src.read_masks(1) > 0) & np.isfinite(values) & _alpha_valid(src)
            size = (src.width, src.height)
        # WHU's original masks are bilevel WhiteIsZero TIFFs. Their stored zero
        # bit displays white (building), whereas GDAL returns the raw bit value.
        # Keep raw values for validity/NoData checks above, then retain historical
        # PIL luminance semantics only for this explicit photometric encoding.
        with Image.open(path) as image:
            if image.mode == "1" and getattr(image, "tag_v2", {}).get(262) == 0:
                values = np.asarray(image.convert("L"))
    else:
        with Image.open(path) as image:
            if image.mode not in {"L", "1", "P", "LA"}:
                raise ValueError(
                    f"Mask must be single-channel: {path} (found {image.mode})"
                )
            values = np.asarray(image if image.mode == "P" else image.convert("L"))
            valid = np.ones(values.shape, dtype=bool)
            if "A" in image.getbands() or "transparency" in image.info:
                valid &= np.asarray(image.convert("RGBA"))[..., 3] > 0
            size = image.size
    if expected_size is not None and size != expected_size:
        raise ValueError(f"Mask/image size mismatch: {path}, {size} vs {expected_size}")
    if not np.isin(values[valid], (0, 1, 255)).all():
        raise ValueError(f"Mask {path} contains non-binary values on valid pixels")
    return np.where(valid & (values > 0), 255, 0).astype(np.uint8), valid


def read_valid_mask(path: Path, expected_size=None) -> np.ndarray:
    """Prepared sidecar values are boolean availability, never building classes."""
    with Image.open(path) as image:
        if image.mode not in {"L", "1", "P"}:
            raise ValueError("Validity sidecars must be single-channel")
        values = np.asarray(image if image.mode == "P" else image.convert("L"))
        if expected_size is not None and image.size != expected_size:
            raise ValueError("Validity/image size mismatch")
        if not np.isin(values, (0, 1, 255)).all():
            raise ValueError("Validity sidecars must contain only 0, 1 or 255")
    return values > 0


def resize_rgb_valid(rgb, valid, size):
    """Resize RGB without interpolating NoData values into valid border pixels."""
    rgb, valid = np.asarray(rgb), np.asarray(valid, dtype=bool)
    if rgb.shape[:2] != valid.shape or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("RGB and validity dimensions differ")
    if (rgb.shape[1], rgb.shape[0]) == tuple(size):
        return rgb.copy(), valid.copy()
    if valid.all():
        resized = np.asarray(
            Image.fromarray(rgb).resize(size, Image.Resampling.BILINEAR)
        )
        return resized.copy(), np.ones(resized.shape[:2], dtype=bool)
    weights = np.asarray(
        Image.fromarray(valid.astype(np.float32)).resize(
            size, Image.Resampling.BILINEAR
        )
    )
    available = np.asarray(
        Image.fromarray(valid.astype(np.uint8)).resize(size, Image.Resampling.NEAREST)
    ).astype(bool) & (weights > 0)
    resized = np.zeros((*weights.shape, 3), dtype=np.uint8)
    for band in range(3):
        numerator = np.asarray(
            Image.fromarray(
                np.where(valid, rgb[..., band], 0).astype(np.float32)
            ).resize(size, Image.Resampling.BILINEAR)
        )
        values = np.divide(
            numerator, weights, out=np.zeros_like(numerator), where=weights > 0
        )
        resized[..., band] = np.clip(np.rint(values), 0, 255).astype(np.uint8)
    resized[~available] = 0
    return resized, available
