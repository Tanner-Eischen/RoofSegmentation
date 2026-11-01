# RoofSegmentation: Validated Dataset + Training Upgrade

An independently executable pipeline integrated into `Tanner-Eischen/RoofSegmentation`.
It adds dataset preparation, training, and evaluation alongside the original app,
`train_v2.py`, and `evaluate.py`. The source package does not contain a trained
roof model. The included `smoke_data/` contains synthetic fixtures only.

The new commands use `roof_pipeline.*` rather than the legacy scripts. They do not
alter the existing React/FastAPI application. Use the new checkpoint in the existing
API only after checking that its architecture and checkpoint format are compatible.
**The new checkpoint format is not directly compatible with `api/inference.py`.**
To compare in that app, add an adapter or independently update inference. The new
`roof_pipeline.predict` command is already fully functional.

Experiment runners should save preregistered settings in `registration.json`.
The trainer owns `experiment.json` for its actual arguments and environment;
keeping these filenames separate preserves both records when a runner and the
trainer share the same output directory.

## Quick start (Windows PowerShell)

From repository root, with Python 3.10+ and a virtual environment (3.11 recommended
for a fresh installation; choose a version installed on your machine):

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
# For an NVIDIA GPU, install a compatible CUDA PyTorch wheel from pytorch.org first.
python -m pip install -r requirements-roof-pipeline.txt
```

If no external data is downloaded yet, first smoke-test with the small **synthetic**
image/mask fixtures included in `smoke_data/`:

```powershell
python -m roof_pipeline.prepare audit --root smoke_data
python -m roof_pipeline.train --data smoke_data --run runs/smoke --arch tiny --no-pretrained --epochs 1 --size 96 --batch-size 4 --device cpu
python -m roof_pipeline.evaluate --data smoke_data --checkpoint runs/smoke/best.pt --out runs/smoke/test --device cpu
python -m roof_pipeline.predict --checkpoint runs/smoke/best.pt --image smoke_data/images/synthetic_test_000.png --out runs/smoke/prediction --device cpu
python -m pytest tests/ -q
```

**Do not report the smoke data's IoU as scientific evidence.** Its sole purpose is to
prove the masks, training, checkpoint loading, prediction, and evaluation work.

## 1. Get REAL paired images + masks

**Option A: WHU (recommended first):**

- Publisher: https://gpcv.whu.edu.cn/data/building_dataset.html
- Convenient mirror: https://huggingface.co/datasets/giswqs/WHU-Building-Dataset
- The downloader defaults to the official publisher archive, preserves its
  4,736 / 1,036 / 2,416 geographic partitions, and records archive provenance.
  Its download is approximately 5.43 GB; allow additional room for extraction
  and the prepared dataset. Downloads can resume after interruption.
- To use the alternative mirror, explicitly pass `--source mirror`; its
  5,732 / 1,228 / 1,228 split has unverified spatial independence.
- Source includes paired 512x512 RGB building imagery/masks, spatially partitioned
  publisher data (the third-party mirror reports different split counts).
- Verify the actual distribution and license you choose. Do not remix the publisher
  train/val/test areas or combine test imagery into training.

```powershell
python -m scripts.download_whu --source official --out data/raw/whu
python -m roof_pipeline.prepare whu --source data/raw/whu --out data/whu_prepared
python -m roof_pipeline.prepare audit --root data/whu_prepared
```

For fast, **non-reportable** pipeline experiments, add `--limit-per-split 100` to the
import command. Remove that flag for the real experiment.

This conversion validates matching image/mask dimensions, normalizes binary masks
from 0/1/255 into 0/255, preserves the supplied partitions, and writes a manifest
with group identifiers. All non-black annotated pixels in WHU masks represent the
binary **building** target. Do not treat them as roof material or individual roofs.

**Option B: SpaceNet 2:** download imagery and corresponding building-label GeoJSON
from https://spacenet.ai/spacenet-buildings-dataset-v2/ . Prepare a CSV with
`image,vector,split,group` columns pointing to each GeoTIFF and WGS84 GeoJSON (see
`examples/geopairs/pairs_example.csv`). Make train/val/test disjoint by AOI or
spatial block. Then run:

```powershell
python -m roof_pipeline.prepare geopairs --pairs-csv data/my_pairs.csv --out data/spacenet_prepared --tile-size 512
python -m roof_pipeline.prepare audit --root data/spacenet_prepared
```

The importer rasterizes vector footprints with the image's affine transform,
normalizes RGB bands, chips the raster and writes a labeled mask for every chip.
It records source scene georeferencing separately. The GeoJSON must be in EPSG:4326.
For SpaceNet, use RGB product imagery, not arbitrary multispectral band ordering.

**Option C: label your own imagery in Labelme:** read `docs/LABELING_GUIDE.md`.

## 2. Train and select the model ON VALIDATION ONLY

Start with an ImageNet-initialized U-Net/ResNet-34 rather than a very large encoder.
The new trainer uses synced spatial image/mask augmentation, BCE+soft-Dice loss,
training-only class imbalance weights, early stopping, and checkpoint selection by
validation IoU at 0.50. It then reloads the **best checkpoint** to calibrate the
threshold using validation only.
The trainer also retains `final.pt` at fixed threshold 0.50 for controlled
endpoint comparisons; `best.pt` remains the validation-selected checkpoint.

```powershell
python -m roof_pipeline.train --data data/whu_prepared --run runs/whu_unet_r34 --arch unet-resnet34 --pretrained --epochs 40 --batch-size 4 --size 512 --workers 0 --seed 42
```

Model options: `unet-resnet34`, `unetplusplus-efficientnet-b0`,
`deeplabv3plus-resnet34`, and `tiny` (tiny is only for smoke tests).
Optional U-Net auxiliary edge supervision uses `--boundary-weight 0.2
--boundary-radius 1`, retaining binary building inference. See the
[matched boundary pilot and annotation review](docs/WHU_BOUNDARY_2026-10-29.md)
for the controlled runner, validation-only selection and limitations.
The [versioned adjudication workflow](docs/WHU_REVIEW_WORKFLOW_2026-10-29.md)
preserves original labels, supplies a reviewer worksheet and compares frozen
predictions with separately versioned review masks. AI proposals remain
provisional until domain adjudication is supplied.
The [provisional correction test](docs/WHU_CORRECTIONS_2026-10-29.md) compares
original versus reviewed training masks with an explicit frozen overlay, matched
control, saved final weights and unchanged validation targets.
For explicit DeepLab context settings, add `--aspp-rates 6 12 18`.
Rates are saved in the checkpoint and restored during inference; the flag is
rejected for other architectures. See the [DeepLab comparison protocol](docs/WHU_DEEPLAB_2026-10-29.md).
If GPU memory is insufficient, reduce batch size or use `--size 256` as a
separate, explicitly labeled experiment. Scaling whole tiles down can lose
roof-boundary detail, so 512 is preferred where possible.

## 3. Final held-out evaluation

For independent sampling/augmentation attribution, use the
[matched-seed 2×2 factorial study](ablation_study/README.md). It uses uniform
and hard-example sampling with replacement, fixed-budget final checkpoints,
three paired seeds, and unchanged original labels. The runner generates
validation-only contrasts, training curves, and matched prediction overlays.
The repeatedly inspected WHU test is excluded from this study.

For the matched follow-up experiment using training-only hard examples and
photometric augmentation, see [the refinement protocol](docs/WHU_REFINEMENT_2026-10-28.md).
`python -m roof_pipeline.refine` runs both the targeted arm and its extra-training
control, selects using validation, and evaluates the selected model. It preserves
the original baseline and records whether the measured IoU exceeds 0.95.

Freeze the model and training decisions first. Run once for a final, unbiased test.

```powershell
python -m roof_pipeline.evaluate --data data/whu_prepared --checkpoint runs/whu_unet_r34/best.pt --out runs/whu_unet_r34/test
```

The resulting `test_metrics.json` and `test_per_image.csv` contain the global
pixel IoU, Dice, precision, recall, F1, and sample-level diagnostics. The
checkpoint, manifest, dataset metadata, and image/mask content hashes must match.
Changed files at unchanged paths are rejected. Never choose the threshold on test.
Checkpoints predating the content fingerprint must be retrained before using this
frozen-data evaluator.

## 4. Visualize predictions / georeferenced inference

```powershell
python -m roof_pipeline.predict --checkpoint runs/whu_unet_r34/best.pt --image data/whu_prepared/images/whu_test_000000.png --out runs/whu_unet_r34/example
```

For a georeferenced RGB GeoTIFF with a known CRS:

```powershell
python -m roof_pipeline.predict --checkpoint runs/whu_unet_r34/best.pt --image data/geo/scenes/site01.tif --out runs/whu_unet_r34/geospatial --overlap 64
```

This writes the mask PNG, probability PNG, overlay, and, where georeferencing exists,
a mask GeoTIFF and EPSG:4326 polygon GeoJSON. **Area in m² is emitted only when
source CRS is projected in meters.** A PNG without transform has no defensible
square-meter area.

## Important scope and scientific limitations

- The provided files do **not** contain WHU / SpaceNet / AIRS original imagery or
  trained roof model weights. Downloads must be run separately.
- Do not use automatically predicted masks as independently verified labels.
- A model trained on WHU/AIRS labels learns building/roof extents, not roof
  materials, damage, 3D geometry, individual instance IDs, or cadastral boundaries.
- WHU is Christchurch-centered. A high WHU test IoU does **not** prove generalization
  to Minnesota, different sensors, or satellite imagery. Add an independent source
  before making broad claims.
- No arbitrary-address Google Maps imagery is included because of imagery-derived
  data licensing concerns. Use appropriately licensed data for trained models.
- The synthetic smoke model and app's former random demo values cannot establish IoU.
- Full-scene prediction reads the raster into memory. For enormous mosaics, use
  rasterio windowed inference as a subsequent optimization.
- Training determinism is best-effort on a fixed platform and seed. GPU kernels,
  libraries, and hardware revisions can produce differences.

## Files

```
roof_pipeline/
  prepare.py     WHU, Labelme, GeoJSON-to-raster image/mask preparation; leakage checks
  common.py      Manifest schema, mask verification, visual audits
  dataset.py     Paired image/mask loader and synchronized augmentation
  models.py      U-Net, U-Net++, DeepLabV3+, tiny smoke network
  metrics.py     Global pixel metrics and validation threshold calibration
  train.py       Best-checkpoint training and calibration
  evaluate.py    Locked test-set evaluation
  predict.py     Overlap-aware inference, PNG, GeoTIFF, GeoJSON
scripts/
  download_whu.py
  make_smoke_dataset.py
docs/
  LABELING_GUIDE.md
  DATASET_CHOICES.md
  EXPERIMENT_PROTOCOL.md
  INTEGRATION_NOTES.md
tests/
  test_roof_pipeline.py
smoke_data/
  images/, masks/, manifest.csv, dataset_meta.json
```
