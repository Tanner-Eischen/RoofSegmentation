# Pre-registered research protocol: matched-seed 2×2 training ablation

## Study claim and provenance

Project: `Tanner-Eischen/RoofSegmentation` (`roof_pipeline` implementation on current main branch as inspected October 31, 2026).

Primary question: Does training-error-weighted resampling and/or stronger photometric augmentation improve WHU aerial building-extent segmentation after a fixed fine-tuning budget? Is their interaction positive or negative?

This is **not** a new result yet. Existing project work includes real training/validation results, including a prior combined-intervention experiment; no outcomes are imputed to this planned factorial study.

## Dataset and sample unit

- **Primary:** Official WHU aerial building dataset, retain 4,736 train / 1,036 validation / 2,416 test *publisher spatial partitions*. Use the prepared official dataset already checked by the project's audit.
- All compared models use identical RGB/mask files, original partitions, content hashes and 512×512 preprocessing. No re-splitting, test mining, changes to labels, or use of synthetic smoke data.
- WHU labels indicate building extents, not independently verified rooftop polygons or materials.
- Prepared WHU manifest currently sets `group=whu_provided_val_partition` for every validation tile. That partition is one coarse geographic unit, not 1,036 independent geographic groups. A scene-cluster bootstrap or spatial cross-validation cannot be inferred from those group labels. If later implementing true spatial-block analysis, recover original tiling coordinates/georeferencing and validate them instead of treating sequential IDs as positions.
- Existing WHU test outcomes have already influenced project decisions. Do not call the previously inspected test a fresh confirmatory set.

## Hypotheses

- **H_A:** Error-weighted sampling (A1) changes fixed-epoch global validation IoU relative to uniform replacement (A0).
- **H_B:** Robust photometric augmentation (B1) changes fixed-epoch global validation IoU relative to standard augmentation (B0).
- **H_AB:** The effect of hard sampling depends on the augmentation condition (interaction, either direction).

No prespecified positive effect is assumed. Null or negative outcomes are valid and should be published in the case study.

## Four treatment arms

| Factor | A0/B0 | A1/B0 | A0/B1 | A1/B1 |
| --- | --- | --- | --- | --- |
| Sampling policy | Uniform replacement | Hard-example weighting replacement | Uniform replacement | Hard-example weighting replacement |
| Photometric | Standard | Standard | Robust | Robust |
| Architecture | U-Net/ResNet-34 | Same | Same | Same |
| Starting weights | Fixed baseline checkpoint | Same | Same | Same |
| Geographic partitions | Official WHU | Same | Same | Same |
| Image size | 512 | Same | Same | Same |
| Loss | BCE + soft Dice | Same | Same | Same |
| Optimizer | AdamW | Same | Same | Same |
| Epochs | 20 | Same | Same | Same |
| Threshold for PRIMARY metric | 0.50 | Same | Same | Same |

### Sampling factor A

- Mine TRAIN ONLY error fractions once with a frozen compatible baseline checkpoint. Per tile, error fraction is `(FP + FN) / pixels` at threshold 0.50, no augmentation.
- A1 uses existing mixture: 50% uniform plus 50% proportional to training error; cap probability at 5× uniform and maintain nonzero floor.
- **P0 correction required:** use weighted `WeightedRandomSampler(..., replacement=True, num_samples=len(train))` for A0 and A1. A0 weights all equal; A1 weights from the original mining report. This removes the replacement-policy confound. Each epoch contains the same number of training draws, but not necessarily the same number of *unique* images. That uniqueness difference is the treatment's natural consequence and should be logged.
- The supplied installation script adds `--uniform-replacement` without changing default behavior. Do not fabricate an all-zero-error hard-example report to simulate a uniform policy.

### Photometric factor B

- B0: current standard brightness and contrast factors of 0.85 to 1.15, each applied with probability 0.65.
- B1: brightness 0.70 to 1.30, contrast 0.75 to 1.25, and gamma 0.70 to 1.40 (as currently coded). Both use existing synchronized flips and 90-degree rotations.
- Validation receives no augmentation, including test-time flips.

### Frozen common settings

- Frozen 512-pixel U-Net/ResNet34 checkpoint trained on *identical* prepared WHU content; strict checkpoint/dataset fingerprint matching.
- Reinitialize the optimizer for each run; AdamW learning rate 1e-4, weight decay 1e-4, positive class weight 1, cosine schedule, clipping 1, CUDA AMP where supported.
- Seeds: 42, 1337, 2026, paired across four arms. Same checkpoint for every seed is intentional; seed variation reflects minibatches and augmentation stochasticity, not pretraining randomness.
- Fixed 20 epochs with patience set above the epoch limit. Run complete for an endpoint estimate. An interruption invalidates the arm until re-run from scratch under a new output directory.
- No boundary auxiliary head, provisional label overlays, TTA, test threshold optimization, or architecture changes during this experiment.
- Primary score is checkpoint `final.pt` on all geographic validation tiles with threshold 0.50. Evaluate `final.pt` separately, compare to `training_summary.json` to detect drift.
- `best.pt` is retained for secondary deployment selection, but is **not** substituted into the primary factorial effect estimate.

## Effects and analysis

Let `I_ab,s` be global validation pixel IoU at the final epoch for sampling `a∈{0,1}`, augmentation `b∈{0,1}`, seed `s`.

- Sampling main effect per seed: `A_s = [(I_10,s - I_00,s) + (I_11,s - I_01,s)] / 2`.
- Augmentation main effect per seed: `B_s = [(I_01,s - I_00,s) + (I_11,s - I_10,s)] / 2`.
- Interaction per seed: `AB_s = I_11,s - I_10,s - I_01,s + I_00,s`.
- Combined-vs-control difference: `I_11,s - I_00,s`.

Compute these *within each matched seed* before averaging effects across seeds. Report in IoU percentage points. Report all three individual contrasts, mean, range and standard deviation. Three seeds provide only limited uncertainty characterization; do not invent p-values or normal-theory confidence from 1,036 spatially adjacent images. For stronger statistical inference, preregister a larger number of independently seeded repetitions and independent geographic blocks.

Secondary diagnostics for subsequent code additions: per-block IoU once real independent blocks are recovered, boundary-F1 (fixed 2-pixel tolerance), missing-building/false-roof overlays, and inference latency/VRAM on a fixed device. The current runner intentionally does not claim to compute boundary-F1 or GPU cost statistics until instrumentation is present.

## Decision rule

This is a **factor attribution** study, not a model leaderboard. Every arm is reported regardless of gain or loss. If choosing a deployment configuration, choose only from held-out validation and state that this is exploratory. Never use the repeatedly consulted existing WHU test to decide which ablation to keep. Reserve a new frozen external dataset for confirmation.

Practical relevance: interpret effects relative to baseline errors, extra GPU compute, and failure modes. An improvement of 0.1 IoU percentage points may be swamped by seed variability or label uncertainty. Negative results can be equally valuable if robust augmentation increases false positives on bright roofs.

## Independent confirmation phase (separate deliverable)

1. Obtain imagery and labels for one or more independently acquired U.S. aerial regions, or a well-documented independent building extraction dataset (such as WHU-Mix held-out cities), with provenance and license. Do **not** treat AIRS-versus-WHU as independent geography by default: both rely on imagery from Christchurch, and label definitions differ. Also do not collect training screenshots from Google Maps imagery without a verified license permitting that use.
2. Preserve native pixel resolution and verify that label ontology matches WHU's *building extent* target. If evaluating actual rooftop boundary masks, treat it as a **new target/domain study**, not direct like-for-like IoU.
3. Select regions before inference and seal masks; no training, mining, threshold tuning, or feedback-driven relabeling from them.
4. Implement `evaluate_external` that permits a separately fingerprinted external dataset, records the model checkpoint hash, frozen threshold, exact imagery/mask versions, region IDs, and native resolution. Do **not** bypass existing same-dataset fingerprint guards.
5. After freezing the choice based on development validation, assess the selected model on the independent holdout once and report a distinct external metric with geographic coverage and error exemplars.

## Minimum publishable outputs

- Immutable study registration and source checkpoint hash.
- 12 full-run `experiment.json`, `history.json`, `final.pt`, `best.pt`, validation metrics, and logs.
- Four configurations × three seed comparison table with main effects and interactions.
- Learning curves (can be generated from each run's `history.json`) and side-by-side qualitative failure examples on validation tiles.
- A negative/null-result discussion and known data-label limitations.
- Optional new-region evaluation after the external-holdout phase is implemented.

## References

- WHU publisher data and geographic partition definitions: https://gpcv.whu.edu.cn/data/building_dataset.html
- Research demonstrating that spatial autocorrelation can inflate image segmentation validation results: https://doi.org/10.1016/j.ophoto.2022.100018
- Study repository and existing refinement protocol: https://github.com/Tanner-Eischen/RoofSegmentation/blob/main/docs/WHU_REFINEMENT_2026-10-28.md
- Segmentation Models PyTorch architectures: https://smp.readthedocs.io/en/latest/models.html
