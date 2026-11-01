# Existing isolated CPU profile validation — November 1, 2026

Current product source at commit `f9861a3d92c60b5a6ada81f10bfe05a9baed110b`
was exercised using the existing owner-release `roof-completion-patched` isolated
environment. No packages were installed, upgraded or downloaded. The older
`.venv`, GPU environments, weights, data, labels and historical packets remained
unchanged. This report was created afterward; it is not a test of its own future
report commit.

The interpreter is Python 3.12.14 with system-site inheritance disabled. All
**65** dependency versions match the retained Windows CPU hash lock exactly;
there are 66 installed distributions including the old local product package.
Actual `api.main`, shared serving and prediction imports were checked to resolve
to the current repository, not that older installed product. Module hashes and
installed distribution metadata fingerprints are in `profile-verification.json`.

## Executed results

| Check | Observed result |
| --- | --- |
| `uv pip check` against existing isolated interpreter | All 66 packages compatible |
| Current `api/tests` plus `tests/test_roof_pipeline.py` | **46 passed**, no skips |
| Actual frozen-checkpoint CPU CLI versus loopback HTTP probabilities | Maximum difference **0** |
| Independent normalized forward versus HTTP | **9.313225746154785e-10**, under absolute 1e-6 |
| Thresholded masks and artifact hashes | Match |
| Prior profile versus this profile, decoded fixture RGB | Identical pixels |
| Prior profile versus this profile, probabilities/masks | Maximum difference **0**, identical masks |

The original model checkpoint hash remains
`2c5757995dafaa206abac4e22e384cdef78b4d93ab9026975050307ac2d9c7c5`.
It uses the same registered release/threshold. The fixture is generated arithmetic
RGB texture, 97 rows by 131 columns, with zero foreground. Pillow 13 encodes its
PNG differently from Pillow 11.3; both byte hashes and the equal decoded RGB hash
are recorded. No source imagery or labels were used in this integration proof.

This confirms current-source integration in the newer profile, not WHU accuracy,
geographic transfer, broad numerical equivalence or operational readiness.
The test suite includes its pre-existing tiny synthetic one-epoch regression;
no baseline/research training or full-dataset scoring was performed.

The profile includes Torch **2.14.1+cpu**, torchvision **0.29.2+cpu**, NumPy 2.2.6,
Pillow **13.0.0**, SMP 0.6.0, FastAPI **0.162.0**, Starlette **1.10.0** and
python-multipart **0.0.32**. It does not replace the default direct dependency
file or silently transfer the historical quality result to these versions.

## Advisory coverage and qualifications

pip-audit 2.10.1 inspected all **66** distributions. It reported zero known
advisory matches among **63 recognized** records and explicitly skipped:

- The locally authored `roof-segmentation-workbench` 0.3.0 distribution.
- Torch 2.14.1+cpu and torchvision 0.29.2+cpu, whose local-version labels are
  absent from PyPI's advisory lookup.

The same-day retained supplement queries public upstream versions Torch 2.14.1
and torchvision 0.29.2 without local suffixes: two recognized records, zero known
matches. It is kept separate from the exact CPU-build audit and does not erase
the skips. No advisory IDs were ignored. Installed version/metadata comparison
does not independently verify original CPU wheel archive bytes. Interpreter,
operating-system and bundled native-code coverage are outside this package audit;
zero matches are not a security certification.

The first audit invocation finished its queries but could not write its result
before this report directory existed. It was repeated after creating the
directory; the successful retained JSON is the audit evidence used here.

## Reproduce without mutating the retained environments

The existing isolated interpreter executed these commands from the current
repository root:

```powershell
# $patchedPython identifies the existing isolated CPU interpreter.
uv pip check --python $patchedPython
& $patchedPython -m pytest api/tests tests/test_roof_pipeline.py -q
& $patchedPython -m scripts.verify_serving_parity --checkpoint runs/whu_r34_official_2026-10-28/best.pt --model-release reports/serving-integration-2026-11-01/model-release.json --out reports/isolated-cpu-profile-2026-11-01/real-cpu-proof
```

`requirements-release-windows.lock` is copied from the retained owner release.
Only its generated-command comment was sanitized; package requirements and
hashes are unchanged. The original lock hash, version comparison and CPU metadata
fingerprints are recorded separately. A future clean reproduction can create a
new Python 3.12 environment and install this file with `--require-hashes`; that
fresh installation was not executed here.

No product code migration was needed. No browser, Docker UI, container, cloud or
publication action was performed. Docker's Linux engine remains unavailable.
Browser E2E contracts remain authored but unrun. This evidence supports a bounded
CPU runtime candidate while those release gates remain explicit.
