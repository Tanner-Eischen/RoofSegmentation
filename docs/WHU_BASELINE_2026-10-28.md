# WHU baseline — October 28, 2026

## Registered experiment

Status: all 40 epochs completed, the validation-selected checkpoint was reloaded
and calibrated, and frozen held-out evaluation completed on all 2,416 test tiles.

- Dataset: [WHU publisher aerial imagery and raster labels](https://gpcv.whu.edu.cn/data/building_dataset.html),
  using the original geographic partitions: 4,736 train, 1,036 validation,
  2,416 test tiles, at 512 × 512 pixels. No subset and no reshuffling.
- Target: binary building extent. This does not classify roof materials or damage.
- Model: U-Net with ImageNet-pretrained ResNet-34 encoder.
- Training: 40 epochs, batch size 4, 512px, seed 42, CUDA mixed precision.
- Optimizer: AdamW, learning rate 0.0003, weight decay 0.0001, cosine decay.
- Loss: BCE with logits plus soft Dice, positive weight 1.
- Augmentation: synchronized flips and 90° rotations; brightness/contrast on
  training images only. Validation and test receive no augmentation.
- Checkpoint: highest validation global pixel IoU at threshold 0.5, minimum
  improvement 0.0001. Patience 40 permits the requested full 40 epochs.
- Calibration: reload the selected checkpoint, then tune threshold on validation
  only over 0.15–0.85 in steps of 0.05.
- Test: evaluate the frozen checkpoint and threshold once on the 2,416 test tiles.
- Runtime: RTX 4060 Laptop GPU (8 GB), Python 3.10, PyTorch 2.5.1+cu121,
  torchvision 0.20.1+cu121, segmentation-models-pytorch 0.5.0.
- Loading: four Windows workers, persisted between epochs; four main CPU threads.

## Reproduction

The current machine's GPU environment is `.venv-gpu`. A fresh installation can
use another environment satisfying `requirements-roof-pipeline.txt` and a
compatible CUDA PyTorch build.

```powershell
$env:OMP_NUM_THREADS = '4'
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
python -m scripts.download_whu --source official --out data/raw/whu_official
python -m roof_pipeline.prepare whu --source data/raw/whu_official --out data/whu_official_prepared
python -m roof_pipeline.train --data data/whu_official_prepared --run runs/whu_r34_official_2026-10-28 --arch unet-resnet34 --pretrained --epochs 40 --batch-size 4 --size 512 --seed 42 --workers 4 --patience 40 --device cuda
python -m roof_pipeline.evaluate --data data/whu_official_prepared --checkpoint runs/whu_r34_official_2026-10-28/best.pt --out runs/whu_r34_official_2026-10-28/test --device cuda
```

Each run directory must be fresh. The trainer refuses to overwrite a run that
already contains a checkpoint. To reuse an existing official ZIP, pass
`--archive data/raw/whu_official.zip` to the downloader.

## Evidence and interpretation

The dataset audit rejects missing/nonbinary/misaligned masks, shared split groups,
and identical decoded image pixels across splits. The checkpoint binds the CSV,
dataset metadata, and every paired image/mask file by content fingerprint.
Lossless copies are detected; this is not a detector of near duplicates or
undocumented geographic overlap.

Source archive provenance, experiment arguments, code fingerprints, environment
versions, epoch history, selected checkpoint, validation calibration, held-out
metrics, and per-image results are retained with the dataset/run artifacts.
Large data and checkpoint artifacts are ignored by Git.

Synthetic checks demonstrate pipeline operation only. The interrupted HF mirror
download is not the dataset for this run. Its differently sized partitions have
unverified spatial provenance.

A WHU result measures this benchmark in its geographic domain. It does not
establish performance in Minnesota, on arbitrary satellite sensors, or on
building-instance boundaries. A separate external-region experiment is needed
for those claims. Earlier leaked-validation numbers are not a comparison baseline.


## Completed results

| Measure | Value |
| --- | ---: |
| Training epochs completed | 40 |
| Selected epoch | 32 |
| Validation IoU at 0.50 | 0.916026 |
| Validation-selected threshold | 0.50 |
| Validation IoU after calibration | 0.916026 |
| Held-out test images | 2416 |
| Held-out global pixel IoU | 0.892119 |
| Held-out Dice / F1 | 0.942984 |
| Held-out precision | 0.946375 |
| Held-out recall | 0.939617 |

The test threshold was frozen from validation. The per-image TP/FP/FN/TN sums
were checked against the aggregate JSON. Twelve qualitative examples were
exported after final evaluation: four low-IoU positive tiles, four high-IoU
positive tiles, and four other random tiles (seed 42). These examples did not
drive model selection. Single-image inference was checked against the matching
evaluation rows.

Training plus per-epoch validation took 102.7
minutes, excluding dataset preparation, startup audit, threshold calibration,
final test evaluation, and diagnostic rendering.

Archive SHA256: `62a4b4ef0fdfc98deee44b29fd0fe949554a928e1e9d38730fc45360668e2892`.

Dataset content SHA256: `538ac8569dcfd817b8557bc23ed18977de5dbba8d4b19719c82ab5ee4d1a308d`.

Checkpoint SHA256: `2c5757995dafaa206abac4e22e384cdef78b4d93ab9026975050307ac2d9c7c5`.

Local artifacts are under `runs/whu_r34_official_2026-10-28/`:

- `best.pt`, `experiment.json`, `source_manifest.json`, and `source_snapshot.zip`
- `environment_versions.json`, `verification.json`, and `history.json`
- `training_summary.json` and `test/test_metrics.json`
- `test/test_per_image.csv`
- `figures/learning_curve.png` and `figures/held_out_examples_1.png` through `3.png`

Verification before training: 54 tests passed and one existing backend test
requiring an unavailable historical model was skipped. The separate CPU pipeline
suite passed all 41 tests, and its dependency check passed. Ruff formatting and
lint passed. Windows CUDA training with multiprocessing and the pretrained
512px U-Net were exercised before the full run. Remote CI and container execution
were not run. The existing application's API adapter remains optional.
