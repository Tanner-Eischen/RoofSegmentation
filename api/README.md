# Local image inference API

Start from the repository root, using the supported isolated Python 3.12 CPU
profile. On Windows install `requirements-serving.txt`; Linux/CI/container use
`requirements-release-linux.lock`. Both are fully pinned with required hashes.
Historical direct pins are under `requirements/history/` for reproduction only.

Prepare a model release explicitly (`python -m roof_pipeline.release`), export
`MODEL_PATH` and `ROOF_MODEL_RELEASE`, then run:

```powershell
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

No `.env` file is read implicitly. Export the variables before startup.

- `GET /api/health/live`: process liveness; no model download.
- `GET /api/health/ready` (alias `/api/ready`): actual checkpoint/release loading;
  HTTP 503 if unavailable. `supported_profile_ready` separately reports whether
  installed package versions match the declared platform lock and Python 3.12.
- `POST /api/inference`: PNG/JPEG/WebP multipart file. Returns real mask and
  float32 probability artifacts, hashes, run/input/model identities and dimensions.
- `POST /api/inference/batch`: bounded experimental image batch, not an address job.
- Address/geocode/satellite/address-batch routes return unavailable, never fake data.

`model.runtime` identifies the startup source-build digest, dependency profile,
lock digest and any version mismatches. A match identifies configuration, not a
security certification or accuracy result. Model readiness alone does not assert
supported-profile matching, browser behavior or production readiness.

Defaults: 8 MB/file, 4,194,304 pixels, at most four images/batch, one active model
computation, two CPU threads and 60-second response timeout. Timed-out native
work retains its slot until it finishes. Image-space area is null; mean foreground
probability is uncalibrated. Do not interpret simplified contours as instances.

See the current [Wave 2 evidence](../reports/wave2-serving-profile-2026-11-01/README.md)
for actual container checks and their limits. Legacy setup is retained in Git
history; arbitrary-address acquisition is not part of the supported application.
