# Explicit NoData handling

Previously, training and evaluation decoded binary labels without a validity
mask, image loaders discarded alpha/raster validity, and the serving response
reported `valid_pixel_count` as the full image size. Missing pixels could
therefore contribute to predicted roof area, background counts, loss, and
threshold selection. This was a pipeline bug even when the input had no NoData.

## Validity policy

Validity is separate from the building class. Image pixels must be finite and
valid in every RGB band and any explicit alpha channel. Raster masks, declared
NoData, external `.msk`/`.aux.xml`, and PNG transparency are respected. Alpha 0
is invalid; alpha greater than 0 retains the original RGB without compositing.
Black and white RGB values alone never indicate missing data.
If an upstream export fills missing coverage with black but supplies no validity
metadata, add a documented alpha/GDAL mask before prediction or an explicit
prepared-data sidecar. The pipeline cannot distinguish that fill from valid
dark imagery using pixel color alone.

Labels retain class 0 = background and class 1/255 = building. Explicit label
NoData takes precedence, including a declared NoData value of 255. A declaration
that collides with a valid class must be checked against source provenance;
do not add `nodata=0` to ordinary background labels. Palette indices remain
class values rather than rendered palette colors for ordinary indexed masks.
One-bit `WhiteIsZero` TIFF labels are a distinct image encoding: their white
building pixels have raw index 0, and decoding preserves that photometric
interpretation. The official WHU source uses this encoding; treating its raw
indices as ordinary binary class codes would invert its labels. NoData validity
is determined from raw metadata before this class decoding.

Prepared data can include an optional `valid_mask` manifest field pointing to
`valid_masks/<sample_id>.png`, with 0 invalid and 1/255 valid. The effective
training mask intersects image, original label, optional replacement label,
and sidecar validity. All-valid legacy manifests and fingerprints remain
compatible. Sidecar bytes and external raster metadata companions are included
in new dataset fingerprints when present.

## Training and evaluation

The existing `(image, target)` dataset interface remains: targets are 0/1 with
`-1` for ignored pixels. Resize and geometric augmentation keep validity aligned.
Partial-validity resizing normalizes interpolation by valid weights; percentile
stretching and contrast statistics exclude missing pixels. Invalid model inputs
are neutral zero after ImageNet normalization.

BCE and Dice exclude ignored pixels. Fully invalid samples are removed before
the training forward pass and optimizer step, and a fully invalid training epoch
fails. Logs distinguish sampler draws, scored samples, and ignored samples.
Boundary supervision ignores neighborhoods crossing unknown pixels.
Architectures with batch-dependent normalization still require an adequate
number of scored samples per training batch; ignoring an entire sample can
reduce that number. The pipeline does not duplicate samples to hide this
architecture constraint.

Confusion counts, roof fractions, threshold calibration, and auxiliary boundary
diagnostics use valid pixels only. Fully invalid samples contribute zero scored
pixels, never true negatives; a fully invalid evaluation/calibration split fails.
Evaluation reports valid, invalid, and total pixel counts. Existing validation
checkpoint selection and frozen test thresholds are unchanged.

Annotation audit strata and diagnostic rescoring also exclude missing pixels.
The historical adjudication packet schema does not bind validity sidecars;
creating such a packet from a partially valid source now fails explicitly.
Supporting those exports requires a separately versioned review format rather
than silently discarding validity during file copying. Existing all-valid WHU
reviews remain compatible.

## Prediction, area, and serving

Sliding-window inference excludes invalid pixels from blending, masks, roof
area, coverage, probability summaries, and vector outputs. It skips fully
invalid chips and rejects a wholly invalid input. Edge padding is cropped to
the source extent. Both CLI and API emit a separate validity PNG; invalid
probabilities are zero placeholders and must be interpreted using that mask.

`valid_pixel_count` is the sum of valid source pixels, `invalid_pixel_count` is
the excluded count, and `total_pixel_count` is width times height. Coverage is
predicted valid roof pixels divided by valid pixels. GeoTIFF output uses a GDAL
validity mask with no class-0 NoData tag, preserving valid background. GeoJSON
polygonization excludes invalid pixels and retains holes. The API's legacy flat
polygon format cannot represent holes, so exteriors enclosing invalid pixels
are omitted and counted in `polygon_components_omitted_invalid`; the raster
mask remains authoritative.

The API accepts its existing PNG/JPEG image formats; georeferenced raster input
uses `roof_pipeline.predict`. Image-space uploads still have no physical area.

## Release migration

The preprocessing identity is now
`imagenet-rgb-validity-edgepad-tiled-v2`. Historical v1 release manifests are
rejected rather than claiming the old alpha/preprocessing policy. Generate a
fresh manifest for the same weights, then configure `ROOF_MODEL_RELEASE`:

```powershell
python -m roof_pipeline.release --checkpoint SAME.pt --out FRESH.validity-v2.release.json
```

Supply `--source-manifest ORIGINAL_TRAINING_SOURCE.json` when preserving an
existing training-source identity. Do not overwrite historical manifests or
reuse their runtime acceptance claims. No weight retraining or automatic model
download is part of this change. The runtime source identity now includes the
shared validity implementation.

## WHU impact audit

The [independent source audit](results/whu_nodata_audit_2026-11-18/README.md)
checks every original TIFF pair against the publisher ZIP and prepared PNGs,
including raster masks, alpha, nonfinite data, class/NoData collisions, source
filename mapping, and both image and label pixel parity. It also verifies
before/after identities of the source archive, prepared dataset, 95 registered
training artifacts, and published test/error packets. Read its per-split totals
and limits before interpreting impact; the existing study evidence is preserved.

The complete audit found **zero declared/alpha/mask/nonfinite invalid pixels**
and **zero image or normalized-label parity failures** in all 4,736 training,
1,036 validation, and 2,416 test pairs. All 2,146,435,072 source pixels are valid
under this policy. The archive, prepared dataset, 95 training artifacts, and
published test/error packets passed unchanged-byte checks.

Consequently, no WHU metric rerun is required for this declared-validity fix.
The three-model test IoUs remain 0.890162 (U-Net), 0.884568 (U-Net++), and
0.881707 (SegFormer-B0), with the original validation-best checkpoints and
threshold 0.50 preserved. The agricultural examples `whu_test_000046` and
`whu_test_001948` also have zero invalid pixels. Their U-Net++ false positives
are not explained by declared NoData. This audit does not establish annotation
completeness or detect undeclared missing coverage.

## Regression verification

Fixtures cover NoData sentinels, nonfinite TIFF data, alpha/transparency,
external mask/metadata files, valid black/white imagery, palette labels,
optional sidecars, fingerprint changes, synchronized geometry, valid-only
normalization, masked objectives/gradients, skipped invalid training samples,
calibration/counts, area, GeoTIFF masks, polygon holes, CLI/API artifact parity,
and release migration. All-valid fixtures check exact legacy resizing,
augmentation, loss arithmetic, fingerprints, and prediction compatibility.

Final local gates passed: **423 pipeline/ablation tests plus 13 subtests** in
the GPU Python environment, **49 API tests** in the supported Python 3.12 CPU
serving environment, and **12 frontend unit tests**. Python Ruff checks,
format checks for all changed/new Python files, frontend lint, the production
frontend build, and Git whitespace checks passed. Scientific model training
was not rerun; the CPU optimizer steps in regression tests use generated tiny
fixtures only.

These are local pipeline/serving tests and a source-data audit. They do not
retrain the scientific models or establish new Docker/browser/deployment
acceptance for the v2 preprocessing policy.
