# RS-02 / RS-03 local implementation handoff

Baseline: `4cf6fa12c635610d1b2832e0e0b5aa96329b4561`. This is uncommitted,
unpublished work. The earlier RS-01 packet and all original scientific artifacts
remain unchanged. No force reset, source-label edit, model retraining, paid call,
browser operation, commit, push or deployment was performed.

## Outcome and exact implementation scope

Adopted the existing October 31 owner release's bounded CPU ModelService,
package-relative API and honest local-upload frontend. Extended them with a
separately prepared hash-bound model release, exact float32 probability artifacts,
registered preprocessing/threshold metadata, input/run/output identities and a
shared serving CLI. The authoritative tiled prediction engine remains the
existing `roof_pipeline.predict.predict_array`; newer model/training/evaluation
modules were preserved. The experimental GeoTIFF CLI remains separate.

Changed or added paths:

- Shared service/model contract: `roof_pipeline/inference_service.py`,
  `roof_pipeline/release.py`, `api/__init__.py`, `api/inference.py`, `api/main.py`.
- Integration tests/driver: `api/tests/conftest.py`, `api/tests/test_api.py`,
  `api/tests/test_inference.py`, `api/tests/test_serving_parity.py`,
  `scripts/verify_serving_parity.py`, `scripts/serve_test_fixture.py`.
- Frontend: `app/src/api.js`, `app/src/App.jsx`,
  `app/src/pages/Dashboard.jsx`, `app/src/components/Layout.jsx`,
  `app/src/__tests__/api.test.js`, `app/src/__tests__/App.test.jsx`,
  `app/src/__tests__/Dashboard.test.jsx`, `app/vite.config.js`,
  `app/playwright.config.ts`, the three `app/e2e/*.spec.ts` contracts and
  `app/e2e/fixtures/generated-rgb.png` (a separate numerical test fixture).
- Environment/configuration: `Dockerfile`, `docker-compose.yml`,
  `requirements-serving.txt`, `requirements-release-linux.lock`,
  `.github/workflows/ci.yml` backend setup/lint/test steps.
- Documentation/evidence: `README.md`,
  `docs/SERVING_INTEGRATION_2026-11-01.md`, this versioned report directory.

The old mocked API tests asserted successful legacy address workflows; they were
replaced with real small-checkpoint computations and truthful unavailable states.
No browser assertions were run. Old address pages remain in source but are not
routed by the adopted upload app. `source-adoption.json` preserves exact original
owner-release file hashes and resulting source hashes. A focused independent
source review of the model-release/service/API boundary reported no actionable
findings; it did not rerun tests or inspect a UI.

The required browser CI tests previously targeted removed address workflows.
Their source contracts now exercise genuine upload/readiness/artifact downloads
and explicit failures, backed by a temporary real Tiny checkpoint service. The
E2E CI job installs Python/serving dependencies and starts that API alongside
Vite. No tests were skipped or disabled. These browser contracts are authored
but unrun; TypeScript transpilation syntax checks passed using esbuild without
invoking Playwright. The actual fixture-server command was separately exercised
through HTTP and passed readiness, image upload, invalid-file and unavailable
remote-route controls; see `e2e-fixture-server-check.json` and its server log.

## Commands actually executed

From the repository root using its existing Python 3.12 CPU environment:

```powershell
.\.venv\Scripts\python.exe -m roof_pipeline.release --checkpoint runs/whu_r34_official_2026-10-28/best.pt --source-manifest runs/whu_r34_official_2026-10-28/source_manifest.json --out reports/serving-integration-2026-11-01/model-release.json
.\.venv\Scripts\python.exe -m scripts.verify_serving_parity --checkpoint runs/whu_r34_official_2026-10-28/best.pt --model-release reports/serving-integration-2026-11-01/model-release.json --out reports/serving-integration-2026-11-01/final-real-cpu-proof
.\.venv\Scripts\python.exe -m pytest api/tests tests/test_roof_pipeline.py -q
.\.venv\Scripts\ruff.exe check api/main.py api/inference.py api/tests roof_pipeline/inference_service.py roof_pipeline/release.py scripts/verify_serving_parity.py
.\.venv\Scripts\ruff.exe format --check api/main.py api/inference.py api/tests roof_pipeline/inference_service.py roof_pipeline/release.py scripts/verify_serving_parity.py
.\.venv\Scripts\python.exe -m pip check
npm ci --prefix app --no-audit --no-fund
npm --prefix app test
npm --prefix app run lint
npm --prefix app run build
docker compose config --quiet
git diff --check
```

The initial frontend test command could not find Vitest because dependencies were
absent; `npm ci` installed the lockfile, after which tests passed. Initial Ruff
import-order findings were fixed before final lint/format passes. The Docker
engine check failed because its Linux named pipe was unavailable. No image build
or startup was attempted after establishing that limitation.

## Observed evidence

- Final targeted Python checks: **45 passed**, no skips. This includes the
  pre-existing tiny synthetic one-epoch train/calibrate/evaluate regression;
  no new baseline or research training was performed.
- Frontend unit tests: **8 passed**. ESLint and production build passed.
- Ruff lint/format, dependency consistency, Compose parsing and diff whitespace
  checks passed. New frontend build: 44 transformed modules; JS 235.04 kB,
  CSS 16.94 kB, index HTML 0.74 kB as emitted by Vite.
- Actual CPU checkpoint:
  `2c5757995dafaa206abac4e22e384cdef78b4d93ab9026975050307ac2d9c7c5`.
- Actual prepared model release:
  `54832764edf43ccdfba1243bbfc3f8ad73ac97f58dd6665b9bb071b77248b9b0`.
- Fixture: generated 97-by-131 RGB texture, 12,707 valid pixels, zero predicted
  foreground. No source imagery, reference mask or dataset label was used.
- CLI versus actual loopback HTTP float32 probability difference: **0**.
- Independent explicit normalization/forward difference:
  **9.313225746154785e-10**, under absolute tolerance **1e-6**.
- Thresholded masks identical; input, model, release and artifact hashes match.
  Corrupt upload returned HTTP 400; unavailable satellite route returned 503.
- The negative suite rejects drifted release normalization/threshold/overlap,
  missing metadata, modified same-stat cached weights, invalid images, over-limit
  uploads/batches and injected RGB-channel mismatch. Empty predictions remain valid.

`final-real-cpu-proof/proof.json` is the current proof and contains runtime/source
hashes. Its CLI/server logs, response manifest and exact arrays/masks are retained.
The earlier `real-cpu-proof/` iteration remains preserved; it predates only the
final bounded-read adjustment and is not relabeled as the final source proof.
`portfolio-evidence.json` is the compact public-safe numerical handoff; its
headline accuracy list is deliberately empty. Original baseline accuracy remains
in the separate RS-01 packet.

## Gates, limitations and next dependencies

RS-03 is verified for this bounded CPU PNG/JPEG/WebP serving path. It is not a
new segmentation-quality measurement. Mean foreground probability is uncalibrated;
there is no generic confidence percentage. Area is null without georeferencing.
External contours omit holes/tiny components and are not instance guarantees.
The Docker lock is an adopted Linux release profile; this CPU proof uses the
separately recorded Windows Python 3.12 environment. A clean installation and
new image execution were not performed here.

RS-02 remains incomplete: the changed container needs a clean build/start with
real artifacts; the migrated browser E2E environment/assertions and dependency-advisory triage
remain outstanding. Existing earlier container evidence cannot satisfy this new
source gate. UI operations remain prohibited, and no exception was requested.
RS-04's actual browser journey is not complete. Restartable batch processing,
streamed GeoTIFF/NoData/geographic products and GPU parity remain out of scope.

PF may use this numerical integration supplement while retaining all these limits.
It must not merge the generated-fixture denominator with WHU accuracy, replace old
packet hashes, claim this code was published, or advertise verified browser or
container behavior from this work.

No migration changed original training inputs/labels/checkpoints. Rollback removes
only this new integration and reverts its affected files after reviewing the diff;
retain RS-01, the original runs, and the previous owner release. The existing
uncommitted RS-01 work must not be reverted as part of a serving-only rollback.
