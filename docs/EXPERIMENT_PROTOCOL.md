# Roof model experiment protocol

## Pre-registration

Record before training: dataset source and rights, geographic split, model,
seed, image size, augmentation policy, epoch limit, learning rate, primary
metric, and test protocol. Keep one experiment folder per run.

## Reference experiment A

- **Dataset:** WHU provided partitions, unmodified except strict masks 0/255.
- **Model:** U-Net with ImageNet-pretrained ResNet-34 encoder.
- **Input:** 512px RGB; ImageNet normalization.
- **Augmentation:** paired h/v flips, 90-degree rotations, restrained contrast
  and brightness adjustments on train ONLY.
- **Loss:** BCE with logits + soft Dice. Start `--pos-weight 1` and change
  class weighting only as a separate validation-driven experiment.
- **Optimizer:** AdamW, learning rate 3e-4, cosine decay, early stopping.
- **Checkpoint:** maximum global validation pixel IoU with threshold 0.5.
- **Threshold:** calibrate using that best checkpoint on validation pixels.
- **Test:** one evaluation using the calibrated threshold, after decisions freeze.

## Reference experiment B

Swap model to `unetplusplus-efficientnet-b0`; keep the exact same dataset,
split, image size, metrics, threshold selection, and seed. Compare validation
metric, wall time, VRAM and inference speed. Do not assume that the more
complex model will improve held-out IoU.

## Data experiments

1. **Mask QA:** visually inspect paired images and masks for image/mask shift,
   missing regions, tiny roof omissions, and label consistency. Fix real
   errors first.
2. **Negative sampling:** sample additional roof-free tiles for *training only*.
   Keep validation/test prevalence unchanged, or metrics will be incomparable.
3. **Resolution:** compare 512 input against a documented 256px experiment.
   Resizing to 256 reduces compute but discards small-roof boundary detail.
4. **Domain shift:** evaluate finalized WHU model against a separate SpaceNet
   or other region/sensor dataset, with a validated label mapping. Publish
   in-domain and out-of-domain metrics separately.
5. **Optional architectural comparison:** U-Net vs U-Net++ or DeepLabV3+.
   Keep all data and threshold policies fixed.
6. **Boundary refinement:** only after observing boundary-specific errors,
   consider a boundary-aware auxiliary loss or morphology. Morphological
   postprocessing must be selected on validation, not test.

## Evaluation requirements

Report global pixel IoU/Dice/precision/recall, per-scene IoU distributions,
small-vs-large-roof errors, empty-mask false positive frequency, and at least
12 qualitative examples: successes, misses, boundary errors, shadows/trees,
and industrial/residential mix. Avoid quoting pixel accuracy alone as it can
be dominated by the background class.

## Reproducibility checklist

- Manifest, dataset metadata, and every paired image/mask content fingerprint
  captured in checkpoint and verified before evaluation.
- Split group IDs and source provenance saved.
- No train/val/test group crossover or exact duplicate imagery.
- Every mask is binary and the same dimension as its image.
- Seed and model architecture recorded.
- Checkpoint and threshold selected ONLY from validation.
- Fixed test protocol, no test threshold sweep.
- Sample-level metrics and overlays exportable.
- `pip freeze > runs/whu_unet_r34/requirements_frozen.txt` saved after run.
- Document errors and deviations, including imperfect domain transfer.

## Interview and portfolio wording

Until there is a real held-out run: “Developed a reproducible roof/building
segmentation pipeline with paired geospatial labels, strict split validation,
model training, and threshold-calibrated evaluation.”

After a successful run, report the specific dataset, number of samples, held-out
IoU, Dice and domain limitations, directly from generated artifacts. Do **not**
claim the synthetic smoke run establishes model accuracy.
