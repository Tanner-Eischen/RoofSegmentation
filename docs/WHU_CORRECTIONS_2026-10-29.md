# WHU provisional training-mask correction test — October 29, 2026

This tests the two corrections from the user-selected **AI-only provisional**
[review](WHU_REVIEW_WORKFLOW_2026-10-29.md). It is a matched binary U-Net training
experiment. The class vocabulary, source images and validation/test masks remain
unchanged. The current U-Net checkpoints are preserved.

## Intervention and data provenance

All 4,736 original training tiles are used at native 512 × 512 resolution.
Only two training targets change in the corrected arm:

| Tile | Proposed edit | Changed pixels |
| --- | --- | ---: |
| `whu_train_000787` | Remove label over bare ground | 2,219 removed |
| `whu_train_002862` | Add eight visible shed roofs | 26,844 added |

The edits affect 2/4,736 tiles and about 0.00234% of all training pixels. They are
candidate annotations with AI provenance, not independently verified ground truth.

A sealed review overlay supplies exactly those two train masks. It validates
every review image and original mask against the unchanged source dataset,
preserves split/group identity, rejects changed validation labels and ignored
pixels, and binds the overlay to the exact initialization checkpoint. Provisional
training is explicitly enabled with `--allow-provisional-review`.

The checkpoint's existing `dataset_content_sha256` continues to identify the
physical source dataset. Additional `effective_training_content_sha256` and
`training_review_content_sha256` fields identify the actual training targets and
sealed review. The original warm-start fingerprint guard remains enforced.
Review masks are verified again after fitting. Original masks are never replaced.

## Registered comparison

Both arms start fresh from the same full validation-winning U-Net/ResNet34
checkpoint, `runs/whu_refinement_2026-10-28/control/best.pt`, validation IoU
0.917083 at 0.50. Its SHA256 begins `41b79df0`. The earlier original baseline is
also preserved. The existing five-epoch control is not reused because it lacks
the final trained weights needed for the fixed-budget comparison.

## Completed result

Both fresh arms completed five epochs. The corrections were learned on the
edited training tiles, but the full-validation gain over matched extra training
was small and did not surpass the incumbent. **Keep the incumbent U-Net.**

| Model | Unchanged validation IoU at 0.50 | Boundary F1 (2px) |
| --- | ---: | ---: |
| Frozen incumbent | 0.917083 | 0.940653 |
| Control, final epoch 5 | 0.916180 | 0.940425 |
| Corrected labels, final epoch 5 | 0.916372 | 0.940465 |

The fixed-budget correction effect is **+0.000191812 IoU**, or about **+0.0192
percentage points**. The corrected model recovered 22,043 additional true-positive
pixels while adding 17,559 false-positive pixels against unchanged validation
labels. It reduced FP+FN by 4,484 pixels overall. Tile-level changes were mixed:
362 tiles had fewer error pixels, 367 had more and 307 were unchanged. One paired
seed cannot establish repeatability or independent annotation accuracy.

Both arms retained epoch zero for validation-selected promotion; calibration kept
threshold 0.50. Their selected binary weights therefore remain the incumbent's.
The trained `final.pt` files are retained separately for the registered comparison.
The fresh control exactly reproduced all five epoch losses, validation metrics
and learning rates from the earlier control, providing a default-path check.

| Model on the two edited TRAIN tiles | IoU against original masks | IoU against AI-proposed masks |
| --- | ---: | ---: |
| Incumbent | 0.713015 | 0.834647 |
| Control final | 0.788888 | 0.753798 |
| Corrected final | 0.664457 | 0.904214 |

Against the same proposed masks, corrected training increased local IoU by
0.150416 over the control. This supports that the intervention changed the model
in the intended direction on those training examples. It does not independently
validate the proposed annotations or demonstrate generalization. Comparing a
single model across the two mask columns is label sensitivity, not training gain.

See the [generated report](../runs/whu_corrections_2026-10-29/report.md),
[learning curves and local effects](../runs/whu_corrections_2026-10-29/learning_and_correction_effects.png),
[shed-roof comparison](../runs/whu_corrections_2026-10-29/whu_train_002862_comparison.png),
[bare-ground comparison](../runs/whu_corrections_2026-10-29/whu_train_000787_comparison.png),
and [completion verification](../runs/whu_corrections_2026-10-29/completion_verification.json).

Verification: 218 tests passed in each Python environment; the seven workflow
tests passed again after the GPU-ordering refinement. Ruff lint/format passed,
as did a native 512px/batch-4 CUDA AMP optimization step using the real correction
targets. Reloaded final metrics match recorded training values on all 1,036
validation tiles, and per-image TP/FP/FN/TN sums match global counts. Source data,
the sealed review, both original U-Net checkpoints and registered training source
hashes were verified unchanged. No test inference was performed.

- **Control:** original source training masks.
- **Corrected:** identical input data and recipe, with only the two sealed
  training-mask overrides.
- **Shared recipe:** five epochs, batch size 4, seed 42, uniform shuffled
  sampling, standard synchronized geometric and photometric augmentation,
  binary BCE + soft Dice with positive weight 1, restarted AdamW at 0.0001,
  weight decay 0.0001, five-epoch cosine decay, clipping 1.0, CUDA AMP.
  Auxiliary boundary supervision is disabled. Overlay loading consumes no
  Python, NumPy or Torch random draws.

The primary intervention comparison is **epoch-5 global validation binary IoU
at fixed threshold 0.50**, corrected minus control, on all 1,036 unchanged
validation tiles. Reloaded final checkpoints must reproduce the recorded metric.
Final binary-mask boundary F1 uses the existing two-pixel Chebyshev tolerance as
a diagnostic.

Promotion is a separate decision: select each arm's epoch by validation IoU at
0.50 with minimum improvement 0.0001, retaining epoch zero. Reload its best
checkpoint and calibrate on validation using the existing 0.15–0.85 grid. A new
candidate must improve on the incumbent by more than 0.0001; exact ties favor
incumbent, then control. Both `best.pt` and fixed-threshold `final.pt` are saved.

## Local correction diagnostics

The incumbent and each final model are additionally scored at fixed 0.50 on the
two edited **training** tiles against both original and proposed masks.

- Same model, different labels: annotation sensitivity of the metric.
- Different models, same labels: local effect of training on the corrections.

Keep those effects separate. These selected tiles were trained on; their scores
do not establish generalization, validate the proposed labels independently, or
choose checkpoints. The unchanged full validation partition provides the primary
held-out comparison. It measures agreement with source annotations, whose
accuracy remains imperfectly established.

No new test inference or test-driven selection is performed. Existing dataset
integrity auditing and hashing still read source contents across all partitions.
The experiment is exploratory, with one paired seed and a small intervention.
A small gain or null result needs replication before broader conclusions.

## Reproduction and outputs

```powershell
$env:OMP_NUM_THREADS = '4'
$env:MKL_NUM_THREADS = '4'
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
.venv-gpu/Scripts/python.exe -u -m roof_pipeline.correction_experiment --data data/whu_official_prepared --review data/whu_review_v1_provisional --incumbent runs/whu_refinement_2026-10-28/control/best.pt --out runs/whu_corrections_2026-10-29 --epochs 5 --batch-size 4 --workers 4 --device cuda --seed 42 --lr 0.0001 --allow-provisional-review
```

Use a fresh run directory. The recipe is saved in `registration.json` before
fitting, with source, review and effective-target hashes; code and environment
are captured in `source_snapshot.zip` and associated manifests. Live progress is
in `workflow_status.json` and each arm's `progress.json`. Completed results are
in `report.md`, `comparison.json`, `selection.json`, per-arm histories and final
validation per-image CSVs. Local training diagnostics and prediction PNGs are
stored separately in `*_training_cases/`.
