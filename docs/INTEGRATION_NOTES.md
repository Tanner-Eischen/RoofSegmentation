# Integrating this upgrade into Tanner-Eischen/RoofSegmentation

## Integration approach

These additions coexist with current `train.py`, `train_v2.py`, `evaluate.py`,
`api/`, `app/`, and Docker. We intentionally did NOT overwrite the existing
sources because that could break the running demo or historical checkpoints.

The new default names are `python -m roof_pipeline.train`,
`python -m roof_pipeline.evaluate`, and `python -m roof_pipeline.predict`.

## Why not simply point the existing API at best.pt?

The new checkpoint schema is:

```
{
  "state_dict": model.state_dict(),
  "config": {"arch": "unet-resnet34", "image_size": 512, ...},
  "threshold": 0.5,
  "val_iou_at_0p5": ...
}
```

The existing `api/inference.py` assumes the older keys `model_state_dict` and
`args` and different architecture aliases. It would therefore NOT load this
checkpoint correctly. The new `predict.py` already handles the new format.
A future application PR can update the FastAPI service to accept both formats
with strict schema/version checks.

## Items to fix in the application when ready

1. **Separate demo and real results**: `app/src/api.js` currently returns
   randomly selected example scores from fake addresses. Never label them
   measured model confidence or IoU. Make sample runs deterministic and label
   them static demonstrations.
2. **Fix original Docker build**: root-level `geocode_satellite.py` and
   `inference.py` are actually under `api/`. The original Dockerfile copies
   missing paths and has no application healthcheck. The pipeline delivered
   here does not replace the original Dockerfile.
3. **Prohibit unsupported roof area**: the FastAPI endpoint returns `roof_area_px`
   and mean predicted positive probability as `confidence`; this is NOT
   calibrated prediction reliability. Do not convert px2 to m2 without valid
   geospatial transform and CRS.
4. **Input limits**: enforce file type/size, image dimensions and bounded
   batch sizes. Original job storage is in-memory and will not survive restart.
5. **Imagery licenses**: avoid relying on Google Maps-derived roof polygons
   and training/evaluation without confirming platform terms. The supplied
   training pipeline uses separately acquired, appropriately licensed data.
6. **Training-serving parity**: reuse preprocessing and checkpoint schema
   between `roof_pipeline.predict` and the final FastAPI integration.

## Suggested GitHub pull requests

PR 1: dataset/mask import, QC audit, source attribution and fixtures.
PR 2: reproducible training, locked threshold/test evaluation, CI smoke tests.
PR 3: checkpoint-compatible FastAPI inference, honest demo modes and Docker fixes.
PR 4: georeferenced export, portfolio case study with measured results.

## Smoke tests (performed during packaging)

- Synthetic fixtures generated from code.
- Pytest covers binary masks, Labelme polygon masks, leakage guard, paired WHU
  image/mask import, GeoJSON-to-GeoTIFF label rasterization and actual tiny
  PyTorch train/calibration/evaluation/inference.
- These are code quality checks, not evidence of roof segmentation accuracy.

### Optional backend application adapter now included

`integration_optional/api/roof_inference_compat.py` supports checkpoints written
by `roof_pipeline.train` and returns the existing FastAPI `load_model` and
`image_bytes_to_mask_and_polygons` call signatures. Copy it into your existing
`api/` directory, then change one import in `api/main.py`:

```python
# Replace the current `from inference import ...` line with:
from roof_inference_compat import image_bytes_to_mask_and_polygons, load_model
```

Set `MODEL_PATH` to a new `runs/whu_unet_r34/best.pt` and start the API as before.
This makes upload `/api/inference` use the newly trained checkpoint. It still
returns the old `confidence` field as mean positive probability, which is **not
calibrated accuracy**. The legacy loader is used only for old checkpoints, and
those files must be trusted due to pickle deserialization concerns.

The optional `integration_optional/Dockerfile.roof-pipeline` also fixes the
original missing-root-file COPY instructions, adds the new package, and uses a
Python HTTP healthcheck rather than assuming curl exists. Before using it:
copy the adapter into `api/`, change the import line, copy the Dockerfile to
repository root and mount the chosen checkpoint at `/app/models/best.pt`.
Container operation remains untested. Local CUDA training is checked separately
and recorded in the experiment artifacts.
