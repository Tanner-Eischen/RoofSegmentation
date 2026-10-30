# WHU DeepLabV3+ architecture experiment — October 29, 2026

Status: verification passed; the full 40-epoch GPU experiment is being launched.
The full 40-epoch run will save its final comparison to
`runs/whu_deeplab_r34_2026-10-29/report.md` and `comparison.json`.
Verification: 141 tests passed, one existing historical-model test skipped;
the separate CPU suite passed all 128 tests and its dependency check passed.
Native 512px/batch-4 GPU optimization and a complete synthetic DeepLab
train/calibrate/save/load/evaluate/predict workflow passed. Reloaded decoder
dilations were verified as 6/12/18.

## Registered experiment

The question is whether a DeepLabV3+ context/decoder architecture improves on the
U-Net / ResNet34 baseline. The target remains global building pixel IoU **above
0.95**. It is a target, not an expected or promised result.

- Dataset: the unchanged official WHU geographic partitions, 4,736 training,
  1,036 validation and 2,416 test tiles; no limited subset or reshuffling.
- Input: native 512 × 512 RGB with ImageNet normalization. Masks remain binary
  building extent labels, not building instances or roof-material classes.
- Architecture: `deeplabv3plus-resnet34`, ImageNet-pretrained encoder and fresh
  decoder. No U-Net warm start and no hard-example sampling report.
- DeepLab settings: encoder output stride 16, ASPP dilation rates **6/12/18**,
  ASPP dropout 0.5, one output logit channel and no output activation.
- Dilation rates are explicit because, at 512px with output stride 16, the
  deepest feature grid is 32 × 32. The installed SMP default rate-36 branch
  has no valid off-center taps at that grid size. This is a prior architecture
  configuration choice made before any real DeepLab fitting.
- Training: 40 epochs, batch size 4, seed 42, four Windows workers, CUDA AMP.
  WHU's training count divides by four, avoiding singleton batches in DeepLab's
  global-pooling BatchNorm branch.
- Optimizer/loss: AdamW, learning rate 0.0003, weight decay 0.0001, cosine decay,
  BCE + Dice, positive weight 1, gradient clipping 1.0.
- Augmentation: original synchronized flips and 90° rotations; brightness and
  contrast 0.85–1.15 with probability 0.65 each. Uniform shuffled sampling.
- Epoch budget: patience 40 permits all requested epochs. Keep the maximum
  validation global IoU at threshold 0.50, with minimum improvement 0.0001.
- Reload the selected checkpoint, then calibrate on validation only using the
  existing threshold grid 0.15–0.85 in steps of 0.05.
- Evaluate this preregistered architecture once on all test tiles after the
  checkpoint and threshold are frozen, regardless of its direction versus the
  baseline. No test-driven threshold/postprocessing or checkpoint selection.

The controlled recipe matches the original baseline except for the architecture
and its explicit context/decoder settings. Source/configuration hashes,
constructor options, data/checkpoint hashes and environment versions are saved.
The original 0.892119 test-IoU baseline is preserved.

## Validation and coverage diagnostics

After fitting, compare baseline and DeepLab on the same validation tiles at each
model's frozen validation-calibrated threshold. Report paired TP/FP/FN changes
and coverage slices: empty, positive coverage below 1%, 1–10%, and at least 10%.
For empty tiles, report false-positive pixel rate and number of tiles with any
predicted positives, rather than mean empty-tile IoU.

Also count training/validation tiles whose GT-positive RGB luminance mean is
below 0.30 on a 0–1 scale. This is a diagnostic brightness proxy, not a material
or building-type label. Coverage is area per tile, not individual building size.
Industrial roofs, greenhouses and containers have no separate source class IDs;
their semantic training coverage still needs manual review. These diagnostics
do not choose checkpoints or alter test labels.

This first run uses seed 42 for a direct recipe comparison. Multiple paired seeds
would be required to establish repeatability. Earlier test errors informed this
research direction, so follow-up scores are **development benchmark comparisons**.
A fresh independent geographic holdout is required for a confirmatory claim.

## Reproduction

```powershell
$env:OMP_NUM_THREADS = '4'
$env:MKL_NUM_THREADS = '4'
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
python -m roof_pipeline.train --data data/whu_official_prepared --run runs/whu_deeplab_r34_2026-10-29 --arch deeplabv3plus-resnet34 --pretrained --aspp-rates 6 12 18 --epochs 40 --batch-size 4 --size 512 --seed 42 --workers 4 --patience 40 --device cuda --lr 0.0003 --weight-decay 0.0001 --pos-weight 1 --min-delta 0.0001 --photometric standard
python -m roof_pipeline.evaluate --data data/whu_official_prepared --checkpoint runs/whu_deeplab_r34_2026-10-29/best.pt --out runs/whu_deeplab_r34_2026-10-29/test --device cuda
```

The saved `runner_source.py` in the experiment directory additionally performs
validation comparisons, frozen-input checks, coverage diagnostics and reporting.
Checkpoint rates are restored automatically by `roof_pipeline.models.load_model`;
legacy DeepLab checkpoints retain their original default rates.
