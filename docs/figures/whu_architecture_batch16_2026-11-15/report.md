# WHU architecture comparison: exploratory pilot (n=1 per model)

Seed 42; 40 epochs; common batch size 16; 512px inputs; unchanged publisher geographic partitions.
Primary: validation-best checkpoint, global pixel IoU at threshold 0.50 (earliest tied epoch).
Secondary: validation-only threshold calibration of that same checkpoint.

| Architecture | Selected epoch | Primary IoU @ 0.50 | Delta vs U-Net (IoU pp) | Secondary threshold | Secondary calibrated IoU |
| --- | ---: | ---: | ---: | ---: | ---: |
| unet-resnet34 | 34 | 0.913608 | +0.000 | 0.50 | 0.913608 |
| unetplusplus-resnet34 | 40 | 0.917648 | +0.404 | 0.50 | 0.917648 |
| segformer-b0 | 37 | 0.903176 | -1.043 | 0.50 | 0.903176 |

## Separate batch-16 profiling

Recorded epoch times include training and per-epoch validation and exclude startup audits, final calibration and checkpoint reassessment. Early U-Net training overlapped a subsequently cancelled batch-32 probe, so actual epoch times are not a clean speed comparison. Use the separate batch-16 runtime profile for throughput; short-probe rates are not end-to-end training rates.

Profile device: NVIDIA GeForce RTX 3090. Rates derive from median batch time, without confidence intervals.

| Architecture | Median train batch (s) | Train images/s | Median validation batch (s) | Validation images/s |
| --- | ---: | ---: | ---: | ---: |
| unet-resnet34 | 0.1493 | 107.2 | 0.0674 | 237.4 |
| unetplusplus-resnet34 | 0.3299 | 48.5 | 0.1699 | 94.2 |
| segformer-b0 | 0.5419 | 29.5 | 0.1003 | 159.5 |

## Interpretation and verification limits

- One seed per architecture (n=1, seed 42): seed variability and confidence intervals are not estimable.
- Validation-selected exploratory results on existing WHU geography; no new test inference or independent geographic confirmation.
- The primary score is global validation IoU at threshold 0.50; calibration is a secondary result on the same selected checkpoint and validation data.
- Architecture packages differ in encoder family, capacity and initialization; these comparisons do not isolate attention or any individual architectural mechanism.
- WHU masks describe binary building extent, not roof material, damage, pitch or individual roof instances.
- This utility verifies JSON consistency only; checkpoint tensors, dataset contents and remote execution must be verified separately by the training pipeline.

Curves: `learning_curves.png` (300 dpi) and `learning_curves.svg`. Circles mark the validation-selected epoch on both panels; they do not select the minimum training loss.

`summary.json` records the exact input JSON hashes, primary/secondary endpoints and observed epoch timing separately from the runtime profile.
