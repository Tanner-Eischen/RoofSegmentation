# Building Segmentation Research

[![CI](https://github.com/Tanner-Eischen/RoofSegmentation/actions/workflows/ci.yml/badge.svg)](https://github.com/Tanner-Eischen/RoofSegmentation/actions/workflows/ci.yml)

Paired-image building-segmentation training and evaluation, with preserved official
WHU partitions, content fingerprints, validation-only threshold calibration, and
traceable per-image results. The target is binary building extent, not roof
material, damage, or building instances.

## Verified original-run evidence

The completed 40-epoch U-Net/ResNet-34 baseline selected epoch 32 on validation
and froze threshold 0.50. Its 2,416 official WHU test tiles yield global pixel
IoU **0.892119** and Dice **0.942984**. The compact
[baseline evidence packet](reports/whu-baseline-2026-10-28/README.md) contains all
per-image confusion counts, aggregate metrics, history, calibration, sanitized
provenance, environment versions, a learning curve, and checksums. The exporter
recomputes the aggregates and checks the retained checkpoint/source identities;
this does not rerun training or inference.

- [Original protocol and reproduction](docs/WHU_BASELINE_2026-10-28.md)
- [Training and evaluation guide](README_UPGRADE.md)
- [Machine-readable portfolio handoff](reports/whu-baseline-2026-10-28/portfolio-evidence.json)

This result measures the WHU geographic domain. It does not establish Minnesota
or arbitrary-sensor performance. Follow-up test reuse makes later comparisons
development results. Training used a recorded dirty source snapshot; the packet
retains its base commit and exact per-file fingerprints. Imagery and weights are
not redistributed in the compact packet. Public packet access remains pending
commit and publication.

```bash
# Verify the compact packet; no GPU, model loading, or original dataset required
python scripts/export_portfolio_evidence.py --verify reports/whu-baseline-2026-10-28

# Export to a fresh directory using the retained original artifacts
python scripts/export_portfolio_evidence.py --out reports/whu-baseline-copy --verify-data-content
```

## Local image inference

The application now supports local PNG/JPEG/WebP upload through a shared CPU
CLI/HTTP service. A separately prepared model-release manifest pins the exact
weights, preprocessing and threshold. Missing or invalid model artifacts return
not-ready; outages never generate substitute masks. Address and remote-imagery
routes are unavailable. Old comparison/address-batch pages remain in source but
are not routed in this release.

See [setup, reproduction and verification](docs/SERVING_INTEGRATION_2026-11-01.md).
From the repository root, prepare a fresh release manifest and configure both
`MODEL_PATH` and `ROOF_MODEL_RELEASE`; then run `python -m uvicorn api.main:app`.
Use `requirements-serving.txt` for the tested Python 3.12 CPU direct-dependency
profile. No model download is implicit, and large weights are not bundled here.

For frontend development, run `npm ci`, `npm test`, `npm run lint`, and
`npm run dev` in `app/`. Vite proxies `/api` to the loopback backend on port 8000.
The supported serving CLI is `python -m roof_pipeline.inference_service`; it
returns the same exact mask and float32 probability artifact bytes as the API.
Outputs include run/input/model-release identities, artifact hashes, original
image dimensions, threshold, pixel coverage and valid-pixel count. Image-space
uploads have no physical area. Mean foreground probability is uncalibrated.

A new genuine baseline-checkpoint integration on a generated non-square fixture
found zero difference between CLI and HTTP probabilities, an independent forward
pass difference below 1e-6, and identical masks. This is an integration check,
not a new accuracy result. API/serving/pipeline regression tests, frontend unit
checks, lint and build passed. Browser interaction and a new container run remain
unverified; the Docker engine was unavailable. Nothing was published or deployed.

The [dependency advisory follow-up](reports/dependency-audit-2026-11-01/README.md)
records narrow Router/Vite/parser fixes, early image-format rejection, fresh
serving parity and remaining runtime/tooling migration work. These checks do not
replace the original accuracy evidence or establish a vulnerability-free release.

## Container and legacy reproduction

The Dockerfile uses package-relative copy paths, the retained Linux CPU hash lock,
an unprivileged user and a Python readiness probe. Compose mounts `models/` read
only; supply `best.pt` plus its `best.release.json`. The adopted Linux lock is a
separate profile from the Windows CPU proof. `docker compose config --quiet`
passes, but does not establish a working image or model-serving container.

`python -m roof_pipeline.predict` preserves the experimental GeoTIFF and explicit
tile/threshold-override research path. Use the shared serving CLI for registered
API parity. Training, labels and historical run evidence remain unchanged.
Historical address-app code and setup are recoverable from repository baseline
`4cf6fa12c635610d1b2832e0e0b5aa96329b4561`; synthetic demo output is not model evidence.
