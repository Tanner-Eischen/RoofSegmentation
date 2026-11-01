# Wave 2: supported CPU profiles and actual container gate

Prepared November 1, 2026 from dirty work based on
`0c239f6a325ecdfd279e4474d34eac0f4732050a`. Exact copied source/frontend,
lock and image hashes are recorded in `build-manifest.json`; no future result
commit has been relabeled as tested. Historical packets remain unchanged.

## Installation and runtime identity

The recommended Windows Python 3.12 installation is now `requirements-serving.txt`,
which selects `requirements-serving-windows.lock` and required hashes. Linux,
backend/E2E CI and Docker select the existing `requirements-release-linux.lock`.
These are the already validated owner profiles: 65 Windows packages / 64 Linux
packages, differing only by Windows colorama. The old direct inputs remain under
`requirements/history/` for historical reproduction, with an explicit warning
that they are not the recommended serving profile. The API README and example
environment now describe package-root startup and actual image-upload endpoints.
Narrow Git attributes preserve both platform lock files byte-for-byte so their
exposed runtime digests survive checkout; package pins/hashes were not rewritten.

```powershell
# Recommended new Windows environment; do not mutate a historical research env.
python -m venv .venv-serving
.\.venv-serving\Scripts\python.exe -m pip install --require-hashes -r requirements-serving.txt
```

The clean-install gate subsequently passed in a new isolated environment. Initial
strict-offline attempts failed on cache/index lookup and missing fonttools. Before
network installation, compatible wheels for all 65 packages were matched to the
retained lock hashes and sized conservatively at **264,991,967 bytes**; disk had
38.3 GB free and the authorized additional-transfer cap was 1 GB. Binary-only,
hash-required installation prepared 20 uncached packages and installed all 65.
No version drift, hash bypass or retained-environment mutation occurred. Network
overhead was not separately measured; the full artifact estimate bounds the
uncached subset. The exact command used UV's pip-compatible best-match strategy
across the two public indexes with mandatory hashes. See `download-estimate.json`,
`clean-install-attempt.json` and `clean-install-receipt.json`.

Both the retained isolated profile and the fresh installation passed dependency
consistency and **48** targeted current-source tests, including two new identity
controls. `clean-windows-proof/` records genuine checkpoint CLI/HTTP parity: maximum
probability difference 0, independent forward difference 9.313225746154785e-10,
and identical thresholded masks. `windows-proof/` retains the earlier existing-env
check. No frontend code changed in this work. Windows versus native Linux float32
probabilities differ by at most **3.8743019104003906e-7**, below 1e-6, with identical
thresholded masks; cross-platform arrays are not claimed byte-identical.

Model metadata now includes startup `runtime` identity: profile ID, exact lock
digest, installed-version mismatches, Python/platform, and a source-build digest.
Readiness exposes `supported_profile_ready` separately. HTTP model readiness
means verified model artifacts can run; the explicit profile flag means versions
match the declared lock. Neither is an accuracy, security or browser certificate.

## Native container evidence

Docker Desktop's engine remained unavailable, but the native Ubuntu WSL engine
was discovered through CLI and was already running. No GUI or daemon startup was
performed by this work. The recorded engine is Docker 29.1.3 on Ubuntu 24.04.3.
Unrelated live containers were not touched.

The new image is a **cached-runtime source overlay**, not a fresh build of the
original multistage Dockerfile:

- Cached dependency base:
  `sha256:157e09a1d21c0967a4ffdc88083f399d22e0889e19291cee7eac6461f1b79175`.
- Result image:
  `sha256:85619cb83bb18c8061cbd3a817b7db64edd44ff92d24ce11e28d0a2ac4d55592`.
- `Dockerfile.cached-runtime` overlays current API/pipeline source, the Linux
  lock, and previously rebuilt frontend artifacts. The build uses `--pull=false`
  and `--network=none`; no dependency installation occurs. A specific ignore file
  permits the separately built frontend, normally excluded by the root ignore.
- Initial build attempts exposed command interpolation and the excluded frontend;
  the final explicit base tag/ID check and specific ignore file resolved them.
  The retained build log identifies the actual successful image construction.

Every own container used **1 GiB / two CPUs**, a read-only root filesystem,
read-only model/release/fixture binds, dropped capabilities and no Docker socket.
The CLI container had no network. HTTP ports were bound only to loopback. All
three own containers stopped with exit code 0 and no OOM; no unrelated container
was stopped, removed or pruned. Limits are configuration plus successful-run
evidence, not measured peak-memory claims.

| Actual check | Result |
| --- | --- |
| Linux supported-profile readiness | HTTP 200, profile match true, 64 locked versions |
| CLI versus real HTTP mask/probability artifacts | Byte-identical |
| Model, release, input and output hashes | Match |
| Corrupt upload | HTTP 400 |
| Over-byte-limit upload | HTTP 413 |
| Two concurrent requests | HTTP 200 and busy 503 |
| Missing weights/release | Readiness 503 and inference 503; liveness remains available |
| Container termination | All own containers stopped; no OOM |

`container-proof/proof.json`, readiness/response manifests and exact generated
output arrays/masks are retained. The original model checkpoint and release
identities are unchanged. Inputs are generated 97-by-131 RGB arithmetic texture,
not source imagery or labels; zero foreground is a valid output. This gate is
serving integration, not a new WHU/geographic accuracy measurement.

The reusable `scripts/verify_cached_container_serving.py` takes an already built
image and explicit checkpoint/release/fixture paths. It only creates UUID-named
own containers, verifies bounds/mounts, runs CLI/HTTP/negative controls, then stops
those containers. It does not start Docker, open a browser or remove shared data.

## Remaining gates and rollback

- A fresh original multistage Dockerfile build remains unrun. The cached
  derivative is not substituted for that claim. The separate fresh Windows
  hash-locked installation did complete and has its own receipts.
- Browser interaction/E2E remains authored but unrun; no UI exception was asked.
- The earlier same-day advisory audit remains qualified: CPU-local wheel names
  and the local product are skipped by PyPI lookup, with public-version queries
  kept separate. Version matching and installer hashes are not exact-wheel
  vulnerability certification. The remaining 25 npm development/tooling records
  require their planned supported-toolchain migration.
- No baseline/research training, full-dataset scoring, public deployment, push,
  cloud-plan change or restricted-data publication occurred. The old environments
  and scientific evidence were preserved.

Rollback only this source/profile/documentation delta after reviewing its diff.
Retain historical pins and evidence. The cached base and stopped own containers
are retained for inspection; no image pruning or volume deletion was performed.

Packaging correction: the existing Linux lock's cached Git index retained LF
bytes when byte-preserving attributes were added, while the tested working file
used CRLF. The staged-lock gate caught the mismatch, but a PowerShell sequence
continued into an unpublished commit. The index was explicitly re-normalized to
the tested raw bytes, both locks were checked against raw/staged/receipt hashes,
and source normalization equivalence was checked before amending that own commit.
The amended-commit gates use fail-fast subprocess checks. No older evidence,
scientific result, model artifact or source-build digest was rehashed to hide drift.
