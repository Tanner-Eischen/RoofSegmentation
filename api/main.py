"""Bounded local-upload FastAPI application. Run: uvicorn api.main:app."""

from __future__ import annotations

import asyncio
import base64
import math
import os
from dataclasses import dataclass
from pathlib import Path

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .inference import (
    ImageTooLarge,
    InferenceBusy,
    InvalidImage,
    ModelService,
    ModelUnavailable,
)


def _flag(name, default=False):
    value = os.environ.get(name, str(default)).lower()
    if value not in {"true", "false"}:
        raise ValueError(f"{name} must be true or false")
    return value == "true"


@dataclass(frozen=True)
class Settings:
    model_path: str
    model_release: str | None = None
    demo_mode: bool = False
    max_upload_bytes: int = 8_000_000
    max_pixels: int = 4_194_304
    max_batch_images: int = 4
    inference_timeout: float = 60.0
    cpu_threads: int = 2

    def __post_init__(self):
        for name in (
            "max_upload_bytes",
            "max_pixels",
            "max_batch_images",
            "cpu_threads",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if (
            not 1 <= self.cpu_threads <= 4
            or not math.isfinite(self.inference_timeout)
            or not 0 < self.inference_timeout <= 300
        ):
            raise ValueError("cpu_threads must be 1..4 and timeout must be (0,300]")
        if self.max_batch_images > 8:
            raise ValueError("At most eight images per batch")

    @classmethod
    def from_env(cls):
        return cls(
            model_path=os.environ.get("MODEL_PATH", "models/roof_pipeline/best.pt"),
            model_release=os.environ.get("ROOF_MODEL_RELEASE"),
            demo_mode=_flag("ROOF_DEMO_MODE"),
            max_upload_bytes=int(os.environ.get("ROOF_MAX_UPLOAD_BYTES", "8000000")),
            max_pixels=int(os.environ.get("ROOF_MAX_PIXELS", "4194304")),
            max_batch_images=int(os.environ.get("ROOF_MAX_BATCH_IMAGES", "4")),
            cpu_threads=int(os.environ.get("ROOF_CPU_THREADS", "2")),
            inference_timeout=float(os.environ.get("ROOF_INFERENCE_TIMEOUT", "60")),
        )


class RequestLimit:
    """Bound body bytes before multipart parsing, including chunked uploads."""

    def __init__(self, app, max_bytes):
        self.app, self.max_bytes = app, max_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] not in {"POST", "PUT", "PATCH"}:
            return await self.app(scope, receive, send)
        body = bytearray()
        while True:
            event = await receive()
            if event["type"] == "http.disconnect":
                return
            part = event.get("body", b"")
            if len(body) + len(part) > self.max_bytes:
                return await JSONResponse(
                    {"detail": "Request exceeds byte limit"}, status_code=413
                )(scope, receive, send)
            body.extend(part)
            if not event.get("more_body", False):
                break
        delivered = False

        async def replay():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": bytes(body), "more_body": False}
            return await receive()

        return await self.app(scope, replay, send)


class BatchRequest(BaseModel):
    addresses: list[str] = Field(min_length=1, max_length=8)


def create_app(settings: Settings | None = None):
    settings = settings or Settings.from_env()
    torch.set_num_threads(settings.cpu_threads)
    application = FastAPI(title="Roof Segmentation Local Upload API", version="0.3.1")
    application.add_middleware(
        RequestLimit,
        max_bytes=settings.max_upload_bytes * settings.max_batch_images + 65_536,
    )
    service = ModelService(
        settings.model_path,
        release_manifest=settings.model_release,
        max_bytes=settings.max_upload_bytes,
        max_pixels=settings.max_pixels,
    )
    application.state.model_service = service
    application.state.settings = settings

    @application.get("/api/health")
    @application.get("/api/health/live")
    def health():
        return {
            "status": "ok",
            "mode": "demo" if settings.demo_mode else "real",
            "model_loaded": service.loaded,
        }

    @application.get("/api/ready")
    @application.get("/api/health/ready")
    def ready():
        if settings.demo_mode:
            return JSONResponse(
                {
                    "status": "not_ready",
                    "mode": "demo",
                    "detail": "Demo mode does not run inference",
                },
                status_code=503,
            )
        try:
            _, metadata = service.load()
            return {"status": "ready", "mode": "real", "model": metadata}
        except ModelUnavailable:
            return JSONResponse(
                {
                    "status": "not_ready",
                    "mode": "real",
                    "detail": "Configured checkpoint or model release is missing or invalid",
                },
                status_code=503,
            )

    async def infer(file):
        try:
            if settings.demo_mode:
                raise HTTPException(
                    503, "Demo mode does not run inference; configure a real checkpoint"
                )
            content = await file.read(settings.max_upload_bytes + 1)
        finally:
            await file.close()
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(service.predict, content),
                timeout=settings.inference_timeout,
            )
            result["mask_base64"] = base64.b64encode(result.pop("mask_png")).decode(
                "ascii"
            )
            result["probability_base64"] = base64.b64encode(
                result.pop("probability_npy")
            ).decode("ascii")
            return result
        except ImageTooLarge as exc:
            raise HTTPException(413, str(exc)) from exc
        except InvalidImage as exc:
            raise HTTPException(400, str(exc)) from exc
        except (ModelUnavailable, InferenceBusy) as exc:
            raise HTTPException(503, str(exc)) from exc
        except TimeoutError as exc:
            raise HTTPException(
                504, "Inference timed out; active work may still be finishing"
            ) from exc
        except Exception as exc:
            raise HTTPException(500, "Inference failed") from exc

    @application.post("/api/inference")
    async def inference(file: UploadFile = File(...)):
        return await infer(file)

    @application.post("/api/inference/batch")
    async def image_batch(files: list[UploadFile] = File(...)):
        try:
            if not 1 <= len(files) <= settings.max_batch_images:
                raise HTTPException(
                    413, f"Batch must contain 1..{settings.max_batch_images} images"
                )
            results = [await infer(file) for file in files]
        finally:
            for file in files:
                await file.close()
        return {
            "status": "complete",
            "mode": "real",
            "count": len(results),
            "results": results,
        }

    def remote_unavailable():
        raise HTTPException(
            503,
            "Remote imagery/address batches are unavailable in this local-upload release",
        )

    for route in (
        "/api/geocode",
        "/api/satellite",
        "/api/satellite/image",
        "/api/batch/{job_id}",
    ):
        application.get(route)(remote_unavailable)

    @application.post("/api/batch")
    def address_batch(body: BatchRequest):
        remote_unavailable()

    @application.post("/api/batch/upload")
    def address_csv():
        remote_unavailable()

    frontend = Path(__file__).resolve().parents[1] / "app" / "dist"
    if (frontend / "assets").is_dir():
        application.mount(
            "/assets", StaticFiles(directory=frontend / "assets"), name="assets"
        )

    @application.get("/{path:path}")
    def static_page(path: str):
        if path == "api" or path.startswith("api/"):
            raise HTTPException(404, "Unknown API route")
        if (frontend / "index.html").is_file():
            return FileResponse(frontend / "index.html")
        return JSONResponse({"detail": "Frontend build unavailable"}, status_code=404)

    return application


app = create_app()
