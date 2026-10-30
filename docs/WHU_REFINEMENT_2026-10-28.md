# WHU targeted refinement experiment — October 28, 2026

Status: implementation verified; the real matched experiment is being launched.
Results are written automatically to `runs/whu_refinement_2026-10-28/report.md`
and `comparison.json` after both arms and final selected-model evaluation finish.
Live state is in `workflow_status.json` and each arm's `progress.json`.

Verification: 116 tests passed, one existing historical-model test skipped;
the separate CPU suite passed all 103 tests with no dependency conflicts. Ruff
formatting/lint and a real synthetic CUDA/Windows two-arm workflow passed.

## Question and registered comparison

Can error-weighted training and moderate photometric variation improve the
baseline's missed textured/dark building masks and roof-like false positives?
The target is global pixel IoU **above 0.95**, measured honestly. It is not a
promised outcome or a reason to change the evaluation definition.

Baseline: U-Net / ResNet-34, official WHU geographic partitions,
4,736 train / 1,036 validation / 2,416 test. Baseline validation IoU is
0.916026 at 0.50; baseline test IoU is 0.892119. Initialization is the frozen
epoch-32 checkpoint from `runs/whu_r34_official_2026-10-28/best.pt`.

Two matched arms use the same architecture, 512px RGB input, batch size 4,
seed 42, four Windows workers, 20 epochs, CUDA mixed precision, BCE + Dice,
positive weight 1, AdamW restarted at learning rate 0.0001, weight decay 0.0001,
and cosine decay. The initial model is retained as an epoch-zero candidate in
each arm, so fine-tuning cannot silently replace it with a worse model.

1. **Targeted arm:** train-only hard-tile sampling and robust photometric variation.
2. **Control arm:** uniform sampling and the baseline augmentation recipe.

The targeted arm runs first. Comparing with the control distinguishes the
intervention's effect from extra optimization and restarting the learning rate.
This is a joint-intervention experiment; it does not isolate sampling from
photometric augmentation.

## Training-only sampling

The frozen baseline predicts only training tiles for mining. Each tile's error
fraction is `(FP + FN) / all_pixels`, at threshold 0.50. Sampling mixes 50%
uniform probability with 50% probability proportional to this error fraction.
Probability above five times uniform is redistributed to uncapped tiles.
Every training tile retains at least half the uniform probability, including
roof-free negatives. Sampling uses replacement for 4,736 draws per epoch.

The error scores, sampling probabilities, original checkpoint hash, dataset
content hash, and training IDs are saved as JSON and CSV. The trainer rejects
split, sample-ID, probability-policy, data-hash, or input-size mismatches.
Validation and test tiles are never eligible for mining or sampled training.

The general motivation is to spend more training effort on difficult examples,
as in [hard-example mining research](https://arxiv.org/abs/1604.03540). This
implementation is an offline tile-level adaptation; that object-detection
paper does not establish an expected WHU gain.

## Photometric intervention

Both arms retain synchronized horizontal/vertical flips and 90° rotations.
The control retains brightness and contrast factors 0.85–1.15 with probability
0.65 each. The targeted arm uses brightness 0.70–1.30, contrast 0.75–1.25
(probability 0.65 each), and gamma 0.70–1.40 (probability 0.50).
Photometric operations affect RGB only. No label interpolation or changes to
geographic partitions are introduced. Validation/test receive no augmentation.

## Selection and final measurement

Checkpoint selection within each arm uses validation global pixel IoU at 0.50,
with minimum improvement 0.0001. Threshold calibration uses the selected
checkpoint and validation only, following the baseline's 0.15–0.85 sweep.

Choose among the baseline, control, and targeted arm using validation IoU at
each model's validation-calibrated threshold, requiring more than 0.0001
improvement over baseline. This allows a sampling-induced threshold shift while
using the same fixed validation-only calibration policy for all candidates. Ties favor the
baseline, then control. Evaluate the selected frozen model/threshold once on
the unchanged test partition. No test-based threshold search, postprocessing,
checkpoint selection, relabeling, removal of difficult tiles, or repeated
attempts until 0.95 are part of this experiment.

**Interpretation:** baseline test errors have already been inspected and helped
motivate this intervention. Follow-up scores on that split are development
benchmark comparisons. A fresh independent geographic holdout is required for
a confirmatory generalization claim. The dataset remains a binary building
extent benchmark rather than a roof-material or instance segmentation task.

## Reproduction

```powershell
$env:OMP_NUM_THREADS = '4'
$env:MKL_NUM_THREADS = '4'
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
python -m roof_pipeline.refine --data data/whu_official_prepared --baseline runs/whu_r34_official_2026-10-28/best.pt --out runs/whu_refinement_2026-10-28 --epochs 20 --workers 4 --batch-size 4 --device cuda --seed 42 --lr 0.0001
```

The run directory contains the fixed protocol, source snapshot/hashes, progress,
training-only mining report, both arms' histories and checkpoints, and the
validation-selected comparison/test report. The original baseline is preserved.
