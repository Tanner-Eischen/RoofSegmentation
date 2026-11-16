# WHU architecture test follow-up

The official WHU test has already informed project development. This is a reused-test descriptive follow-up, not pristine holdout evidence or independent geographic confirmation.

All three original validation-best checkpoints; 2,416 test tiles each; threshold 0.50. No retraining, checkpoint reselection or threshold sweep.

| Model | Epoch | Validation IoU | Test IoU | Test Dice | Precision | Recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| U-Net / ResNet-34 | 34 | 0.913608 | 0.890162 | 0.941890 | 0.943658 | 0.940128 |
| U-Net++ / ResNet-34 | 40 | 0.917648 | 0.884568 | 0.938749 | 0.933765 | 0.943786 |
| SegFormer-B0 | 37 | 0.903176 | 0.881707 | 0.937135 | 0.939521 | 0.934761 |

Original training/validation artifacts and checkpoints verified unchanged. Per-image test counts reconcile with all aggregate metrics.

Inference used NVIDIA GeForce RTX 4060 Laptop GPU

CSV/JSON scores are fractions (0–1). The original validation selection remains the selection criterion; these test scores do not change it.

[Validation/test comparison figure](comparison.png) | [SVG](comparison.svg).

The per-model `inference_seconds` fields record the evaluator wall time, including checkpoint loading, full-dataset integrity checks, data loading and scoring. They are not pure-GPU inference latency or a controlled throughput benchmark.
