# WHU auxiliary-boundary pilot — October 29, 2026

All earlier U-Net checkpoints are preserved. Both new arms warm-start the
**validation winner**, `runs/whu_refinement_2026-10-28/control/best.pt`
(validation IoU 0.917083, threshold 0.50). The original U-Net benchmark checkpoint
remains unchanged. Earlier test scores do not choose this initialization.

## Completed result

Both arms completed all five epochs on the full training partition. Neither
improved on the incumbent; both retained epoch zero. The incumbent remains the
selected model at threshold 0.50.

| Candidate | Best trained IoU at 0.50 | Retained validation IoU | Retained boundary F1 (2px) |
| --- | ---: | ---: | ---: |
| Frozen incumbent | — | 0.917083 | 0.940653 |
| Extra-training control | 0.916180 (epoch 5) | 0.917083 (epoch 0) | 0.940653 |
| Auxiliary boundary | 0.915748 (epoch 5) | 0.917083 (epoch 0) | 0.940653 |

The auxiliary training loss fell from 1.328848 to 0.708078, but this recipe did
not produce a validation-selected improvement. Equal retained boundary F1
reflects unchanged binary weights, not an outline-learning gain. The retained
auxiliary checkpoint contains the epoch-zero edge head; its quality is not
established. Final trained weights and per-epoch boundary F1 were not saved, so
this pilot cannot establish their outline accuracy separately from selection.
A longer paired study would need those diagnostics and repeated seeds.

See the [generated comparison](../runs/whu_boundary_2026-10-29/report.md),
[learning curves](../runs/whu_boundary_2026-10-29/learning_curves.png), and
[completion verification](../runs/whu_boundary_2026-10-29/completion_verification.json).
All three candidates were independently assessed on 1,036 validation tiles;
per-image TP/FP/FN/TN sums match their global results. No real test inference or
label changes were performed. Both original U-Net checkpoint hashes and all seven
registered training source hashes were verified unchanged.

Verification: 161 tests passed in each of the CPU and GPU environments, Ruff lint
passed, and changed pipeline/tests formatting passed. Native 512px/batch-4 CUDA
AMP optimization and synthetic auxiliary train/calibrate/save/load/evaluate/
predict checks passed. Three unrelated legacy scripts have pre-existing format
check failures; they were not changed. Synthetic scores are not scientific evidence.

## Annotation review

See [the annotation audit](WHU_ANNOTATION_AUDIT_2026-10-29.md) for a reproducible
train/validation review of shifted outlines, omissions and semantic uncertainty.
Equal numbers per coverage stratum do not represent their population proportions.
Targeted diagnostic cases are separate from the representative sample. No source
labels are changed. Dimensional and binary integrity do not establish correctness.

## Registered five-epoch experiment

This is a **single-seed exploratory pilot**, using all 4,736 training and 1,036
validation tiles at native 512 × 512 resolution with the official partitions.
Both arms retain epoch zero as a candidate and restart AdamW at learning rate
0.0001, weight decay 0.0001. They use batch size 4, seed 42, uniform sampling,
the existing standard synchronized geometric/photometric augmentation, positive
weight 1, CUDA AMP, clipping 1.0, and five epochs of cosine decay. Auxiliary-head
initialization preserves random state for matched sampling and augmentation.

- **Control:** binary BCE + soft Dice.
- **Boundary:** binary BCE + soft Dice, plus **0.2 × edge (BCE + soft Dice)**.
  An independent 1 × 1 head uses shared decoder features. Its target is a
  two-sided 3 × 3 morphological band derived after mask resizing/augmentation.
  Replicated padding avoids inventing edges along the tile frame. All-background
  and all-building tiles have no visible internal boundary.

Normal inference still returns one binary building logit; the auxiliary head is
used by the training forward pass. Existing binary checkpoints remain loadable.
Auxiliary checkpoint configuration records target definition, radius and weight.

Within each arm, select the epoch using validation global binary IoU at 0.50,
minimum improvement 0.0001; reload it and calibrate on the existing validation-only
0.15–0.85 threshold grid. Compare with the frozen incumbent using calibrated
validation IoU. A new arm must improve by more than 0.0001; exact ties favor
incumbent, then control. The original checkpoint is never overwritten.

Boundary F1 is a **diagnostic**, measured on the binary output's two-sided 3 × 3
boundary band with Chebyshev matching tolerance 2 native pixels. Precision and
recall pool matching counts across validation; empty denominators score zero.
It does not choose checkpoints or thresholds. Per-image pixel counts are saved.

The runner performs **no new test inference or selection**. The trainer's existing
integrity audit checks dimensions, encoding and duplicate imagery across all
partitions, generates previews, and hashes all frozen contents. Prior test-error
inspection informed the research direction. Independent geography and paired
repeat seeds are needed for confirmation. A short negative pilot does not rule
out a longer or differently weighted boundary experiment.

## Research scope

[Girard et al., CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/html/Girard_Polygonal_Building_Extraction_by_Frame_Field_Learning_CVPR_2021_paper.html)
report benefits from frame fields aligned with building contours and coupled
multitask objectives. This motivates contour-aware supervision. This simpler
edge head does not reproduce their frame field or polygonization method, or
establish a gain on WHU in advance. Its targets inherit source outline errors
and cannot resolve semantic ambiguity.

## Reproduction and evidence

```powershell
$env:OMP_NUM_THREADS = '4'
$env:MKL_NUM_THREADS = '4'
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
.venv-gpu/Scripts/python.exe -u -m roof_pipeline.boundary_experiment --data data/whu_official_prepared --incumbent runs/whu_refinement_2026-10-28/control/best.pt --out runs/whu_boundary_2026-10-29 --epochs 5 --batch-size 4 --workers 4 --device cuda --seed 42 --lr 0.0001 --boundary-weight 0.2 --boundary-radius 1
```

Use a new output directory each time. `registration.json` freezes the recipe
before fitting. `source_snapshot.zip`, `source_manifest.json` and
`environment_versions.json` record provenance. Live state is in
`workflow_status.json` and per-arm `progress.json`. Completed results are in
`report.md`, `comparison.json`, `selection.json`, per-arm histories and
`*_validation/` directories. Histories separate segmentation and edge losses.

## Three-class annotation gate

The review must establish repeated, credible container/trailer confusion and a
usable class policy before triggering a three-class pilot. An uncertain rectangle
does not establish object identity. If that gate is met, preregister independent
geographic groups with background, permanent building and container/trailer
classes; define greenhouse treatment and adjudicate uncertain objects. Preserve
the binary dataset and use explicit ignore regions rather than forced classes.
Report per-class confusion and binary building metrics for comparison. The
present work does not convert the binary dataset or train a three-class model.
