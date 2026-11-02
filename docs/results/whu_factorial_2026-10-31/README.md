# Factorial study evidence packet

This compact packet accompanies the [completed study report](../../WHU_FACTORIAL_2026-10-31.md).
The original primary endpoint is final-epoch global validation IoU at threshold
0.50. All 12 runs and 240 epochs are included. Numerical fixtures and smoke tests
are not mixed into these measurements.

| File | Contents |
| --- | --- |
| `validation_runs.csv` | Each seed/treatment's fixed-endpoint metrics and epoch elapsed time |
| `validation_arms_summary.csv` | Means, ranges, sample SDs, and mean elapsed minutes |
| `validation_contrasts_by_seed.csv` | Paired main effects, interaction, combined-minus-control |
| `validation_contrasts_summary.csv` | Means, ranges, and sample SDs of the paired effects |
| `validation_metrics.json` | Global confusion counts and checkpoint/data hashes for every run |
| `epoch_history.csv` | All 240 losses, validation scores, draw counts, unique-image counts, and timings |
| `provenance.json` | Original registration/source/weight/data identities and frozen-source continuation |
| `qualitative_selection.json` | Fixed random validation IDs chosen before qualitative inference |
| `qualitative_checkpoints.json` | Four seed-42 final checkpoint hashes used for overlays |
| `figure_manifest.json` | Original figure hashes and validation-only diagnostic scope |
| `checksums.json` | SHA256 hashes of the published packet and two PNG figures |
| `test_registration.json` | Separate November 2 pre-inference freeze of all 12 final checkpoints |
| `test_registered_checkpoints.csv` | Ordered checkpoint paths, hashes, epochs, and thresholds |
| `test_runs.csv` | Every supplemental test score, confusion count, and evaluation duration |
| `test_arms_summary.csv` | Mean, range, and sample SD across all three seeds per arm |
| `test_contrasts_by_seed.csv` | Paired descriptive test main effects and interaction |
| `test_contrasts_summary.csv` | Across-seed test effects; fractions and an explicitly named percentage-point column |
| `test_verification.json` | Completed checks, unchanged original artifacts, timing, and no-selection status |
| `test_baseline_cached_metrics.json` | Independently verified historical baseline cache; no new baseline inference |
| `test_checkpoint_config_verification.json` | Fixed final checkpoint/factor/label/config checks |
| `test_original_artifact_hashes.json` | Original primary artifacts checked unchanged after the follow-up |
| `test_artifact_manifest.json` | Hashes of the original supplemental artifacts, including local files not bundled here |

CSV IoUs, effect sizes, and standard deviations are **fractions**, except the
explicit `mean_effect_test_iou_pp` column in the test effect summary. Multiply
fraction columns by 100 for the percentage/percentage-point tables in the report. `seed_sd` is the
sample SD across three seeds, not a confidence interval. `train_minutes` includes
epoch fitting and per-epoch validation, excluding preparation, mining, calibration,
final evaluation, and figure generation. It is elapsed time, not dedicated GPU cost.

Checkpoint paths are repository-relative references to Git-ignored artifacts;
weights and original per-image evidence are not bundled here. Local absolute
paths were removed from published JSON or made relative in CSVs. Original source
artifact checksums refer to the unmodified local originals, not their sanitized
representations. The published files have separate checksums.
Git newline conversion is disabled for the packet so its recorded byte hashes
remain stable across Windows and Linux checkouts.

The full local validator checked the source/checkpoint/data identities,
all registered factors and fixed training budgets, and each final evaluation
against its per-image confusion counts and official validation IDs. The compact
packet retains aggregate counts and hashes; it cannot reconstruct predictions
without the dataset and saved checkpoint files.

The original study's `test_evaluated: false` record remains unchanged. Any
`test_*` files are from the explicitly separate November 2 descriptive follow-up
on the previously inspected WHU test. They are not fresh confirmatory evidence.
All 12 fixed-endpoint test evaluations are included, with 2,416 tiles each; their
exact official IDs, counts and metrics were verified against local per-image
evidence. Test scores did not change the arm, seed, checkpoint, threshold, or
deployment choice. All endpoints scored below the contextual original baseline.

Inference uses the existing evaluator, not a new training step. With the original
dataset and weights available, run from the verified source snapshot using the
registered checkpoint list. Each invocation has this form, with a fresh output
directory and the recorded four-thread/CUDA environment:

```powershell
python -m roof_pipeline.evaluate --data data/whu_official_prepared --checkpoint runs/whu_factorial_2026-10-31/runs/a1b0_hard_standard_seed42/final.pt --out runs/whu_factorial_test_replay/a1b0_hard_standard_seed42 --split test --batch-size 4 --device cuda
```

Resolve the data/checkpoint/output paths against the repository root if executing
from the restored source directory. The registration's source/data/checkpoint
hashes, rather than an unrelated current checkout, identify the evaluated code
and assets. Do not overwrite the original completed outputs when reproducing.

From repository root, verify all packet and figure bytes without a GPU or model:

```python
import hashlib
import json
from pathlib import Path

root = Path.cwd()
manifest = root / "docs/results/whu_factorial_2026-10-31/checksums.json"
for relative, expected in json.loads(manifest.read_text()).items():
    assert hashlib.sha256((root / relative).read_bytes()).hexdigest() == expected, relative
print("All packet and figure hashes match")
```
