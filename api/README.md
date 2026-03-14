# Roof Detection API

FastAPI backend: geocode, satellite tile, inference (U-Net), batch jobs.

## Env vars

- `MODEL_PATH` – Path to `best.pt` (default: `../models/pytorch-training-2026-03-10-14-21-42-952/best.pt`)
- `GOOGLE_MAPS_API_KEY` – For Geocoding and Maps Static (satellite) APIs

## Run

```bash
cd api
pip install -r requirements.txt
set GOOGLE_MAPS_API_KEY=your_key
set MODEL_PATH=..\models\pytorch-training-2026-03-10-14-21-42-952\best.pt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## Endpoints

- `GET /api/geocode?address=...` – lat, lng, formatted_address
- `GET /api/satellite?lat=&lng=&zoom=20` – satellite image URL
- `POST /api/inference` – multipart image file → mask base64, polygons, roof_area_px
- `POST /api/batch` – JSON `{ "addresses": ["..."] }` → job_id
- `POST /api/batch/upload` – CSV file (column `address` or `Address`) → job_id
- `GET /api/batch/{job_id}` – job status and results
- `GET /api/health` – health check
