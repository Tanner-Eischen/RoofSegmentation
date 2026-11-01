"""Compatibility import for the shared CPU serving implementation.

New code uses roof_pipeline.inference_service; model downloads are never implicit.
"""

from roof_pipeline.inference_service import (
    ImageTooLarge,
    InferenceBusy,
    InvalidImage,
    ModelService,
    ModelUnavailable,
)

__all__ = [
    "ImageTooLarge",
    "InferenceBusy",
    "InvalidImage",
    "ModelService",
    "ModelUnavailable",
]
