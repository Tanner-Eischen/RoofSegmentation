# RS-02 advisory triage — November 1, 2026

Baseline: local commit `e0de06c67173d6c767b63f658da6259a722b61bf`.
These checks describe subsequent then-uncommitted work. Original scientific and
serving packets remain historical and unchanged; they were not relabeled as
tests of this new source or of the next commit.

## Bounded fixes and verification

- Four lockfile entries changed within existing major/range constraints:
  React Router DOM/Router 7.13.1 → 7.18.5, Vite 7.3.1 → 7.3.7, and
  set-cookie-parser 2.7.2 → 2.7.3. No forced audit fix was used.
- The serving parser pin/local overlay changed from python-multipart 0.0.20
  to 0.0.32, without changing Torch, NumPy, Pillow, model weights, labels,
  normalization or threshold.
- Pillow format dispatch is restricted to PNG/JPEG/WebP at `Image.open`.
  Previously, a foreign plugin could parse a header before the format check.
  A negative test proves a hostile XBM plugin is never invoked.

Afterward, **46 targeted Python tests**, all **8 frontend tests**, Ruff, ESLint,
production build and `pip check` passed. `real-cpu-parity-after/proof.json`
records new genuine frozen-checkpoint CLI/HTTP probability difference **0**,
independent normalized forward difference **9.313225746154785e-10**, identical
masks and matching artifact identities. The generated 97-by-131 fixture has zero
foreground; this proves integration only. `environment-after.json` separately
records the parser and runtime versions. No new quality evaluation, research
training, browser execution, container run or deployment occurred.

## Audit scopes and denominators

Npm 11.13.0 queried npm advisories. pip-audit 2.10.1 queried PyPI using exact
installed versions after collecting the serving dependency closure; no resolver
or automatic Python upgrade was used for the audits. Version matches do not
demonstrate exploitability, and zero matches do not establish safety.

| Scope | Before | After / qualification |
| --- | --- | --- |
| Npm runtime, `--omit=dev` | 2 affected Router package records | **0** affected records |
| Npm full tree | 27 affected records | **25** tooling records: 1 low, 10 moderate, 12 high, 2 critical |
| Windows serving closure | 42 packages; 12 affected; 128 raw entries / 90 alias groups | 42 packages; **11 affected**, **116 raw entries / 84 alias groups** |
| Python path-only audit | Only 7 local distributions inspected | Incomplete: the environment inherits system packages; not the serving denominator |
| Retained Linux lock | 64 records; 0 matches among 62 recognized versions | Torch/torchvision `+cpu` versions were **skipped**, not passed |
| Linux CPU upstream supplement | Torch 2.14.1 / torchvision 0.29.2, without local suffixes | 2 recognized upstream versions; 0 matches; not an exact-wheel/container audit |

Raw before/after JSON and closure files are retained. The PyPI feed repeats some
IDs/aliases; merged alias groups prevent treating duplicates as distinct flaws.
The parser patch removed 12 raw entries / six alias groups. The rest remain open.

## Exposure assessment

| Component | Current scope and remaining action |
| --- | --- |
| React Router | Patched. This app uses client `BrowserRouter`, fixed internal links and no Router SSR/RSC server. Server-only advisory prerequisites are absent from this path; that is not a package-wide exemption. |
| Vite/Vitest | Root Vite patched. Vitest 1.6.1 retains older nested Vite/vite-node/tinypool. Executed tests use `vitest run`, not a listening Vitest UI. Worker/tooling advisories remain and require a supported migration. |
| Other npm tooling | ESLint, glob/source-map/build and jsdom-related dependencies retain matches. They process repository/test inputs and are not part of the static production runtime. Tooling risk remains unresolved. |
| python-multipart | Patched to 0.0.32. Header/preamble CPU issues were relevant before route validation. Body/upload limits remain; no custom keep-filename/upload-directory configuration is used. |
| Pillow 11.3.0 | No major upgrade in this pass. Unsupported plugin headers are now blocked before parsing. Many findings involve unused PDF/PSD/font/TIFF/GUI APIs; allowlisted decoding still needs updated-runtime validation. |
| Starlette 0.49.3 | Request/form and Windows static-file advisories remain. Built assets are mounted, so a compatible FastAPI/Starlette upgrade is still required; byte limits do not replace it. |
| Torch 2.9.0 | Hash-bound local checkpoints are operator-prepared; HTTP users cannot upload weights. This narrows loader exposure but does not make a vulnerable `weights_only` loader safe for untrusted weights. JIT/LSTM/unpack-sequence paths are not used by the U-Net service. |
| AnyIO | Reported IDN/TLS/process-pool paths are not used by this local inference path, which offloads with `asyncio.to_thread`; maintenance remains required. |
| certifi/idna/requests/urllib3 | Present through model/client dependencies. This API disables remote acquisition and automatic weight downloads; the new runtime profile should still update them. |
| filelock/fsspec | No HTTP-supplied lock path, reference/kerchunk JSON or remote model input. Local/build uses need consideration during migration. |
| setuptools/pip | Build/install surfaces, not request actions. The partial path-only pip audit is not presented as a full inference audit. |

Primary records include [Router](https://github.com/advisories/GHSA-49rj-9fvp-4h2h),
[Vitest UI](https://github.com/advisories/GHSA-5xrq-8626-4rwp),
[tinypool](https://github.com/advisories/GHSA-85c8-ppgw-ccpr),
[multipart preamble](https://github.com/advisories/GHSA-mj87-hwqh-73pj),
[multipart headers](https://github.com/advisories/GHSA-pp6c-gr5w-3c5g),
[Pillow XBM](https://github.com/advisories/GHSA-xgh8-jp6q-p5x8),
[Starlette static paths](https://github.com/advisories/GHSA-wqp7-x3pw-xc5r),
and [Torch loader](https://github.com/advisories/GHSA-63cw-57p8-fm3p).
Their feed descriptions and affected/fixed versions remain in the JSON snapshots.

## Next migration plan

1. Prepare a separate isolated CPU environment from the retained hash-locked
   profile rather than mutate historical research/GPU environments. Its candidate
   versions include Pillow 13, FastAPI 0.162, Starlette 1.10 and Torch 2.14.
   Verify exact CPU wheel hashes and the audit skip explicitly.
2. Run pipeline/API tests and genuine frozen-checkpoint parity there, recording
   near-threshold differences. Do not attach historical accuracy to the new
   runtime without its own measurement/provenance.
3. Migrate Vitest and its worker/Vite dependencies together. Npm's proposed
   Vitest 5.0.5 requires Node `^22.12.0 || ^24.0.0 || >=26.0.0`; local Node
   22.20.0 qualifies, while CI's Node 20 needs a tested update. Do not force a
   worker major override into Vitest 1.
4. Build/start the changed container with actual pinned artifacts once Docker's
   Linux engine is available. Its configured named pipe is currently unavailable.
   No UI fallback or cloud-plan change was attempted.
5. Browser contracts remain authored but unrun and require authorized execution.
   Audit/version checks do not substitute for browser or deployment evidence.

## Reproduction and rollback

Executed `npm audit --json`, `npm audit --omit=dev --json`, and
`uvx pip-audit --disable-pip --no-deps -r <pinned-closure> --format json`.
The Linux lock and normalized upstream supplement were queried separately.
`python -m scripts.verify_serving_parity` used the original local checkpoint and
existing release manifest, writing only into this new report directory.

Rollback only the four lock entries, parser pin/local overlay and decoder guard
after reviewing the diff. Preserve both historical and new evidence. No source
imagery, labels, weights, secrets or historical experiment files were altered.
