"""
FastAPI backend: geocode, satellite, inference, batch.
Set MODEL_PATH and GOOGLE_MAPS_API_KEY in env.
"""

# Load .env file if it exists
from dotenv import load_dotenv
load_dotenv()

import base64
import csv
import io
import os
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

from geocode_satellite import geocode, satellite_tile_url, fetch_satellite_image_bytes
from inference import image_bytes_to_mask_and_polygons, load_model

MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    str(Path(__file__).resolve().parent.parent / "models" / "cleaned-dataset" / "best.pt"),
)
# Allow directory so we can point to model dir and auto-pick best.pt
if os.path.isdir(MODEL_PATH):
    MODEL_PATH = str(Path(MODEL_PATH) / "best.pt")

# Check for built frontend (Docker deployment)
FRONTEND_DIST = Path(__file__).resolve().parent.parent / "app" / "dist"
API_ROUTE_PREFIX = "api/"

app = FastAPI(title="Roof Detection API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- In-memory batch jobs ----------
BatchJob = dict[str, Any]
batch_jobs: dict[str, BatchJob] = {}


def ensure_model_loaded():
    try:
        load_model(MODEL_PATH)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=f"Model not found: {e}")


# ---------- Routes ----------


@app.get("/api/geocode")
def api_geocode(address: str):
    """Return { lat, lng, formatted_address } for address."""
    result = geocode(address)
    if result is None:
        raise HTTPException(status_code=404, detail="Geocode failed or no API key")
    return result


@app.get("/api/satellite")
def api_satellite(lat: float, lng: float, zoom: int = 20, width: int = 640, height: int = 640):
    """Return { url } for satellite tile image (img src)."""
    url = satellite_tile_url(lat, lng, zoom=zoom, width=width, height=height)
    if url is None:
        raise HTTPException(status_code=503, detail="Satellite URL not available (check API key)")
    return {"url": url}


@app.get("/api/satellite/image")
def api_satellite_image(lat: float, lng: float, zoom: int = 20, width: int = 640, height: int = 640):
    """Proxy satellite image bytes (e.g. for inference). Returns PNG bytes."""
    data = fetch_satellite_image_bytes(lat, lng, zoom=zoom, width=width, height=height)
    if data is None:
        raise HTTPException(status_code=503, detail="Could not fetch satellite image")
    return Response(content=data, media_type="image/png")


@app.post("/api/inference")
async def api_inference(file: UploadFile = File(...)):
    """
    Upload an image file; returns mask (base64 PNG), polygons, roof_area_px, confidence.
    """
    ensure_model_loaded()
    content = await file.read()
    try:
        mask_bytes, polygons, roof_area_px, confidence = image_bytes_to_mask_and_polygons(content, MODEL_PATH)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    mask_b64 = base64.b64encode(mask_bytes).decode("utf-8")
    return {
        "mask_base64": mask_b64,
        "polygons": polygons,
        "roof_area_px": roof_area_px,
        "confidence": round(confidence, 1),
    }


class BatchRequest(BaseModel):
    addresses: list[str]


@app.post("/api/batch")
async def api_batch_create(body: BatchRequest):
    """Queue a batch job. Returns job_id. Processing runs in background."""
    job_id = str(uuid.uuid4())
    batch_jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "addresses": body.addresses,
        "results": [],
        "created_at": time.time(),
        "total": len(body.addresses),
        "done": 0,
    }
    # Start background processing (simple sequential for MVP)
    import asyncio
    asyncio.create_task(process_batch_job(job_id))
    return {"job_id": job_id, "status": "queued"}


async def process_batch_job(job_id: str):
    job = batch_jobs.get(job_id)
    if not job or job["status"] != "queued":
        return
    job["status"] = "processing"
    try:
        ensure_model_loaded()  # Uses cached model
    except HTTPException:
        job["status"] = "failed"
        job["results"] = [{"error": "Model not found", "status": "failed"}]
        return
    results = []
    for addr in job["addresses"]:
        try:
            geo = geocode(addr)
            if not geo:
                results.append({"address": addr, "status": "failed", "error": "Geocode failed"})
                continue
            img_bytes = fetch_satellite_image_bytes(geo["lat"], geo["lng"], zoom=20, width=640, height=640)
            if not img_bytes:
                results.append({"address": addr, "status": "failed", "error": "No satellite image"})
                continue
            mask_bytes, polygons, area, confidence = image_bytes_to_mask_and_polygons(img_bytes, MODEL_PATH)
            mask_b64 = base64.b64encode(mask_bytes).decode("utf-8")
            results.append({
                "address": addr,
                "formatted_address": geo.get("formatted_address", addr),
                "status": "complete",
                "mask_base64": mask_b64,
                "polygons": polygons,
                "roof_area_px": area,
                "confidence": round(confidence, 1),
            })
        except Exception as e:
            results.append({"address": addr, "status": "failed", "error": str(e)})
        job["done"] = len(results)
        job["results"] = results
    job["status"] = "complete"


@app.get("/api/batch/{job_id}")
def api_batch_status(job_id: str):
    """Return job status and results when complete."""
    job = batch_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


# ---------- CSV batch: accept file upload ----------


@app.post("/api/batch/upload")
async def api_batch_upload_csv(file: UploadFile = File(...)):
    """
    Upload CSV with address column (header: address or Address).
    Returns job_id.
    """
    content = await file.read()
    try:
        text = content.decode("utf-8")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid UTF-8")
    reader = csv.DictReader(io.StringIO(text))
    addresses = []
    for row in reader:
        addr = row.get("address") or row.get("Address") or row.get("address_1") or ""
        addr = addr.strip()
        if addr:
            addresses.append(addr)
    if not addresses:
        raise HTTPException(status_code=400, detail="No addresses found in CSV")
    return await api_batch_create(BatchRequest(addresses=addresses))


@app.get("/api/health")
def health():
    try:
        load_model(MODEL_PATH)
        return {"status": "ok", "model_loaded": True}
    except FileNotFoundError:
        return {"status": "ok", "model_loaded": False}


# ---------- Static file serving for frontend (Docker deployment) ----------

# Mount assets directory if built frontend exists
ASSETS_DIR = FRONTEND_DIST / "assets"
if ASSETS_DIR.exists():
    app.mount("/assets", StaticFiles(directory=str(ASSETS_DIR)), name="assets")


@app.get("/")
async def serve_frontend():
    """Serve the frontend index.html for root path."""
    if FRONTEND_DIST.exists():
        return FileResponse(str(FRONTEND_DIST / "index.html"))
    return {"message": "Roof Detection API. Frontend not built - run from app/ directory in dev mode."}


@app.get("/{path:path}")
async def serve_frontend_routes(path: str):
    """Serve frontend for client-side routing (SPA)."""
    # Don't intercept API routes
    if path.startswith(API_ROUTE_PREFIX):
        raise HTTPException(status_code=404, detail="Not found")

    # Serve index.html for all other routes (SPA client-side routing)
    if FRONTEND_DIST.exists():
        return FileResponse(str(FRONTEND_DIST / "index.html"))
    raise HTTPException(status_code=404, detail="Frontend not built")
