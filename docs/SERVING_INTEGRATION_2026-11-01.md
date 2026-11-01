# RS-02 / RS-03 local serving integration — November 1, 2026

Status: shared CPU CLI/HTTP inference verified on one generated input using the
original baseline checkpoint. No new scientific accuracy evaluation, publication, browser
verification or container run is claimed.

## Existing implementation adopted

The October 31 local owner release already implemented safe CPU ModelService,
bounded upload decoding, async request offloading, explicit unavailable responses,
and a local-upload frontend. This integration adopts those parts rather than
adding a competing model engine. It preserves the current repository's newer
training, model-configuration, dataset, factorial-study and evaluation code.

The adopted release is version 0.3.0, release ID `completion-2026-10-31`, based on
`ff53742b99a537e5461e588b457f071b2062bfb4` plus its recorded dirty changes. The
new integration is uncommitted work on repository baseline
`4cf6fa12c635610d1b2832e0e0b5aa96329b4561`. See the new evidence directory's
`source-adoption.json` for precise original and resulting file hashes.

Additional delta:

- `roof_pipeline/release.py` explicitly prepares a hash-bound release manifest.
  No checkpoint download is implicit. The manifest binds architecture, input
  channels/normalization, native-pixel tile policy, overlap, frozen threshold,
  training identities and known source fingerprints.
- `roof_pipeline/inference_service.py` is the shared CPU service and serving CLI.
  It rechecks weight bytes and release metadata before cached-model reuse, loads
  only safe state dictionaries, and uses the existing `predict_array` engine.
- `api/main.py` uses package-relative imports, runs CPU work off the event loop,
  limits body bytes, decoded pixels, batch count and active model computation,
  closes uploads, and exposes separate liveness/readiness endpoints. Missing
  weights or invalid release metadata yield 503 readiness, never fake inference.
- The API returns run/input/model-release identities, original dimensions,
  threshold, pixel coverage/valid count, mask and exact float32 probability
  artifacts with hashes. The mean foreground probability is explicitly
  uncalibrated; a generic confidence percentage is not returned. Physical area
  is null for image-space uploads.
- The frontend uploads local imagery, surfaces errors, and downloads artifacts
  belonging to the displayed run. Outages do not enable sample inference.
  Legacy address/comparison/batch pages remain in source but are not routed.
- The Dockerfile uses the retained Linux lock, valid package copy paths, an
  unprivileged user and a Python readiness probe. Compose mounts model artifacts
  read only and binds its host port to loopback. The Vite dev server proxies API
  requests to the local backend.
- Required E2E source contracts now exercise actual local upload, readiness,
  artifact downloads and honest failures. A temporary real Tiny-checkpoint API
  replaces successful address mocks. These browser tests are authored but unrun;
  their supporting backend command was checked through actual HTTP independently.

## Prepare and run

Run from the repository root in a Python 3.12 environment. The local proof used
the existing CPU environment; `requirements-serving.txt` records its direct
package versions. The Linux hash lock is a separate adopted release profile,
not the profile of this new Windows CPU proof.

```powershell
python -m pip install -r requirements-serving.txt
python -m roof_pipeline.release --checkpoint runs/whu_r34_official_2026-10-28/best.pt --source-manifest runs/whu_r34_official_2026-10-28/source_manifest.json --out models/best.release.json
$env:MODEL_PATH = (Resolve-Path runs/whu_r34_official_2026-10-28/best.pt).Path
$env:ROOF_MODEL_RELEASE = (Resolve-Path models/best.release.json).Path
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Preparation refuses to overwrite an existing manifest. The API defaults to the
checkpoint's same-stem `.release.json` unless `ROOF_MODEL_RELEASE` is provided.
Neither weights nor restricted dataset imagery are bundled with this integration.

```powershell
python -m roof_pipeline.inference_service --checkpoint runs/whu_r34_official_2026-10-28/best.pt --model-release models/best.release.json --image permitted-image.png --out runs/new-serving-prediction
```

The CLI writes the same mask/probability bytes as HTTP, plus a manifest. CPU
float32 is the supported serving profile. Grayscale is expanded to RGB; alpha is
discarded by Pillow's RGB conversion, without compositing. Images are not resized
to squares: smaller edges are padded and cropped, with probabilities blended
before thresholding. PNG/JPEG/WebP only; GeoTIFF serving is not claimed. The older
`roof_pipeline.predict` CLI preserves the experimental geospatial/override path;
use the shared serving CLI for registered release parity.

## Executed checks and actual outcome

`reports/serving-integration-2026-11-01/final-real-cpu-proof/proof.json` records the
genuine model, fixture, runtime, source and output hashes. The model is the
original checkpoint SHA-256
`2c5757995dafaa206abac4e22e384cdef78b4d93ab9026975050307ac2d9c7c5`,
with frozen threshold 0.50 and the original training data/source identities.

The fixture is generated RGB arithmetic texture, 97 rows by 131 columns, and
contains no source dataset imagery. It produced zero foreground pixels, a valid
empty prediction. This measures integration, not segmentation skill.

| Check | Observed result |
| --- | --- |
| CLI versus loopback HTTP float32 probabilities | Maximum absolute difference 0 |
| Independent normalization/forward versus HTTP | Maximum absolute difference 9.313225746154785e-10 |
| Required absolute tolerance | 1e-6 |
| Thresholded masks | Identical |
| Input, model-release, weights and output hashes | Match |
| Corrupt upload | HTTP 400 |
| Unavailable satellite route | HTTP 503 |
| Targeted API, parity and original pipeline regression tests | 45 passed |
| Frontend unit tests | 8 passed |
| Frontend ESLint and production build | Passed |
| Ruff lint/format for changed Python | Passed |
| Existing CPU environment dependency consistency | `pip check` passed |
| Compose configuration parsing | Passed |
| Temporary E2E fixture backend readiness/upload/failures via HTTP | Passed |
| Authored E2E/config TypeScript syntax via esbuild | Passed; not browser/type verification |

Tests exercise real tiny-checkpoint computations for missing/corrupt/replaced
weights, manifest drift, unsupported normalization/threshold policy, invalid files,
upload/pixel/batch limits, worker-slot retention after timeouts, grayscale/alpha,
non-square outputs, valid zero foreground and artifact identities. Injected RGB
channel swapping is detected by the probability comparison. Small fixture tests
do not contribute to the scientific accuracy denominator.

Reproduce the genuine CPU proof into a fresh output directory:

```powershell
python -m scripts.verify_serving_parity --checkpoint runs/whu_r34_official_2026-10-28/best.pt --model-release reports/serving-integration-2026-11-01/model-release.json --out runs/new-serving-parity-proof
```

The driver starts and terminates its own loopback HTTP server. It does not open a
browser, inspect a UI, access a paid provider, or evaluate dataset labels.

## Outstanding gates and rollback

RS-02 is incomplete until a fresh image builds/starts with these changed sources
and actual model artifacts. Docker CLI exists but its Linux engine is unavailable
in this session. The older owner's container proof belongs to older source and
must not be relabeled as a new container result. No browser installation/E2E,
UI interactions, dependency-advisory triage, deployment, full-dataset rerun or
new baseline/research training occurred. The existing pipeline regression test
does execute a tiny synthetic one-epoch train/calibrate/evaluate fixture to check
code operation; it is not a new research experiment. Required browser E2E files
were migrated in source to the upload workflow and retain actual artifact/failure
assertions without successful inference mocks. They were not executed. Their
temporary generated Tiny checkpoint establishes integration only, not baseline
model performance. Browser work remains prohibited by the user's instructions.

RS-03 has evidence for the bounded PNG/JPEG/WebP CPU serving path. This does not
verify geographic products, large-scene streaming, calibrated confidence, polygon
holes/instance separation, restartable batch processing or GPU serving parity.
Simplified external contours explicitly omit holes and tiny components.

No dataset, label, checkpoint, training or original-run evidence was changed.
The RS-01 packet stays intact. Rollback consists of reverting this serving/app
delta and removing only its newly added files after reviewing the diff; retain
the original checkpoint, RS-01 evidence and separate historical releases.
