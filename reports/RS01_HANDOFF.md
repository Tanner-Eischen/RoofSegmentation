# RS-01 local evidence handoff — November 1, 2026

Requirement: RS-01 / RS-T01. Status: compact evidence packet prepared and checked
locally; public publication and imagery-rights review remain outstanding.

## Baseline and changed files

- Repository baseline: `4cf6fa12c635610d1b2832e0e0b5aa96329b4561`, clean before this work.
- No result commit exists yet; nothing was committed, pushed or deployed.
- `scripts/export_portfolio_evidence.py`: standard-library exporter and standalone
  packet verifier. Fresh output required; original inputs are read only.
- `reports/whu-baseline-2026-10-28/`: sanitized metrics, all per-image rows, training
  history, calibration/configuration, source/environment fingerprints, original
  learning curve, portfolio handoff, verification record and checksum manifest.
- `README.md`: research-first introduction and explicit legacy-app limitation.

## Checks actually executed

From the repository root:

```powershell
.\.venv\Scripts\ruff.exe check --fix scripts/export_portfolio_evidence.py
.\.venv\Scripts\ruff.exe format scripts/export_portfolio_evidence.py
python scripts/export_portfolio_evidence.py --verify-data-content
python scripts/export_portfolio_evidence.py --verify reports/whu-baseline-2026-10-28
.\.venv\Scripts\ruff.exe format --check scripts/export_portfolio_evidence.py
.\.venv\Scripts\ruff.exe check scripts/export_portfolio_evidence.py
git diff --check
```

Ruff initially reported import ordering; it was fixed before final successful
lint and format checks. The exporter and standalone verification passed. A
separate temporary-copy Python check also verified:

- A changed CSV is rejected by packet checksums.
- An unregistered additional packet file is rejected.
- An invented aggregate IoU of 0.99 is rejected against the original CSV rows.
- Relative Markdown links in both README files resolve locally.

All 12 packet payload files match `SHA256SUMS.txt`; the manifest excludes itself.
The full content hash of all 8,188 image/mask pairs was recomputed and matched;
split counts were 4,736 train / 1,036 validation / 2,416 test. All 16 recorded
training source member hashes matched the retained source snapshot. The original
checkpoint and dataset-manifest hashes match the recorded values. History has
all 40 sequential epochs; selected epoch 32 and validation threshold 0.50 agree.

| Recomputed quantity | Value |
| --- | ---: |
| Test rows | 2416 |
| TP | 66222657 |
| FP | 3752381 |
| FN | 4255685 |
| TN | 559109181 |
| Global pixel IoU | 0.8921192509468081 |
| Dice | 0.942984170263471 |
| Precision | 0.9463754346228437 |
| Recall | 0.9396171237966977 |

These exactly reproduce the original aggregate JSON, with a numerical metric
tolerance of absolute `1e-12`. Original-to-export file hashes and executed checks
are retained in the packet's `verification.json`.

## Scientific assumptions, rights and gaps

No scientific assumptions, labels, split definitions, training inputs, thresholds,
weights or original run artifacts changed. Training used dirty source based on
`68f906e39d5bb259cb57532bdbaaa9769ec6c0b4`; the precise recorded per-file hashes
identify that source. This work does not assert that the later pristine repository
baseline was the exact training source.

WHU measures binary building extent in its domain. It does not measure roof
material/damage, building instances, arbitrary sensors or Minnesota performance.
Later WHU test reuse is development work. No new training, inference, full test
suite, container, UI/browser, remote CI or signed-out public-access check ran.
The exporter has no inference/training dependency; verification exercises actual
retained evidence rather than substituting synthetic model performance.

No imagery, source masks, qualitative imagery panels, weights or source snapshot
archive were copied. The learning curve contains only original-run metrics.
`example_selection.json` preserves the original high/low/random sample policy and
sample IDs; qualitative panels remain local pending redistribution-rights review.
The original policy did not register a separate empty-background example group.
Metrics and reproduction instructions are available without redistributing images.

## Next dependency and rollback

PF-02 may use the locally checked metric packet, explicitly marking public packet
access pending. RS-02/03 may use the retained checkpoint identity but cannot claim
serving parity until actual integration checks run. Public packet publication and
any release of additional assets remain separate work.

Rollback is limited to reverting the README introduction and removing these new
exporter/report files after reviewing the diff. There is no migration, and original
data, labels, run artifacts and existing release evidence remain unchanged.
