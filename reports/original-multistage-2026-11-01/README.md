# Original multistage Dockerfile gate — November 1, 2026

The original Dockerfile from product revision
`6c16a49683938d1a5d6c0fb5cbc56f9b9524274f` now builds and serves the real model
through the native Ubuntu WSL Docker engine. The Dockerfile, scientific source,
model release and dependency locks were not changed. Narrow context exclusions
were added for all models, reports, scratch and weight formats. The prior Wave 2
packet remains immutable; this supplement closes its original-build limitation.

## Build identity and scope

- Original multistage build completed in **181.52 seconds** using ordinary
  BuildKit caching and `--pull`. This is not the earlier source-overlay image,
  nor a claim that every base/apt layer was rebuilt without cache.
- Result image:
  `sha256:386d132222c6efd3254eb080629a709b228794cea3b20eeedf22931d6c656498`.
- Resolved frontend base, Node 22.20.0 Alpine:
  `sha256:dbcedd8aeab47fbc0f4dd4bffa55b7c3c729a707875968d467aaaea42d6225af`.
- Resolved runtime base, Python 3.12 slim Bookworm:
  `sha256:e697c81897302a4c4333bade000d98038fbb21eb064f8ee1fd42d02ebcd9d337`.
- The backend installed all 64 hash-locked CPU dependencies. Frontend `npm ci`
  and Vite build executed in the Node stage; host-built frontend output was
  excluded from the context. The reported context transfer was 780.31 kB.
- Audited candidate COPY inputs were 65 source files / 798,462 bytes. Data,
  runs, models, reports, environment files, Python environments and node_modules
  were excluded. No model weights or dataset images were downloaded.
- The log reports approximately **371,827,200 bytes** of Python wheel payloads.
  Base/workdir/apt cache reuse and the small frontend dependency step stayed
  within the approximately 1 GB download budget; network overhead was not
  separately metered. The build timeout was 900 seconds. Free host disk before
  the build was approximately 36.9 GB.

`build-attestation.json` records the resolved base/image identifiers, native
engine, working/committed Dockerfile digests, unchanged normalized Dockerfile
bytes, lock hash and gate-script hash. `build-inputs.json`, `Dockerfile.original`,
`dockerignore.snapshot` and `original-build.log` retain the actual inputs/results.
Source-code line-ending normalization is explicit; no future report commit is
substituted for the recorded tested product source.

## Actual runtime gate

The completed proof is `container-proof-v2/proof.json`. Its three own containers
used a **1 GiB memory ceiling, two CPUs, read-only root filesystem and read-only
checkpoint/release/fixture binds**, with dropped capabilities and no Docker
socket. HTTP was published on loopback only; CLI inference had no network.
Every own container stopped with exit code 0 and no OOM. Unrelated containers
were not touched, and no image/volume pruning or GUI action occurred.

| Check | Observed result |
| --- | --- |
| Process liveness | HTTP 200 |
| Actual model/release readiness | HTTP 200; supported Linux profile matches all 64 pins |
| Model, release, fixture and output hashes | Match |
| Real CLI versus HTTP probability/mask artifacts | Byte-identical |
| Homepage, built JavaScript and CSS | HTTP 200; hashes match files in the exact image |
| Invalid image | HTTP 400 |
| Over-byte-limit upload | HTTP 413 |
| Two concurrent requests | HTTP 200 and busy 503 |
| Unavailable weights/release | Readiness and inference HTTP 503 |
| Own containers after checks | All stopped, no OOM |

The homepage and its JS/CSS checks are byte-serving checks, not browser, layout
or accessibility claims. `static-frontend.json` records the exact routes/hashes.
The generated arithmetic RGB fixture is 97 by 131 pixels and produces zero
foreground. No labels or original imagery were used. This is single-fixture
integration evidence, not a new WHU accuracy or geographic measurement.

`parity-comparison.json` independently compares the completed original image with
the previous cached-derivative Linux result: probability difference 0 and
identical masks on the same generated fixture. Historical accuracy/runtime
denominators remain unchanged.

## Preserved failed attempt and harness correction

The first attempt's CLI succeeded, but its HTTP readiness failed because the
gate mounted files at `/run/model.pt` while the original image defaults to
`/app/models/best.pt`. The earlier derivative had matching `/run` defaults.
The generic image gate now explicitly sets the supplied mount paths in the
available-model environment. No production or Dockerfile change was needed.
The failed attempt remains under `container-proof/`; it is not the selected
completed proof. The unchanged original image then passed the corrected gate
under `container-proof-v2/`.

The gate gained optional `--check-static` byte/hash verification. Its direct
original-image execution verifies those additions and the explicit environment
correction; Ruff checks passed before the local commit.

## Reproduce and remaining gates

Build from the repository root with the original Dockerfile:

```bash
docker build --pull --progress=plain -t roof-original-wave2:20261101 -f Dockerfile .
python3 -m scripts.verify_cached_container_serving --image roof-original-wave2:20261101 --checkpoint runs/whu_r34_official_2026-10-28/best.pt --model-release reports/serving-integration-2026-11-01/model-release.json --fixture reports/dependency-audit-2026-11-01/real-cpu-parity-after/generated-fixture.png --out reports/new-original-image-proof --check-static
```

Use the native Linux engine and a fresh output path. The gate creates/stops only
its own UUID-named containers. It never starts Docker or opens a browser.

Browser E2E remains authored but unrun; public deployment, full-dataset scoring
and research retraining were not performed. Existing CPU-local advisory lookup
skips and remaining npm development-tool findings remain qualified in the
earlier reports; this build is not a security certification. No source/index
failure or actual Dockerfile defect was observed. No push or cloud change occurred.
