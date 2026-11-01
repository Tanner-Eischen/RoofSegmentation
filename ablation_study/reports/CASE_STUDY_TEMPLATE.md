# What actually improves aerial building segmentation?

**Project:** RoofSegmentation | **Study:** Controlled hard-example sampling × photometric augmentation | **Status:** [PLANNED / COMPLETED]

## Research problem

An earlier matched experiment changed both training example sampling and photometric augmentation simultaneously. A performance difference could not be attributed to either change. I designed a 2×2 factorial study to estimate each effect independently, as well as their interaction.

## Data and validity

- Dataset and license: [ACTUAL SOURCE/RIGHTS]
- Geographic partition: [EXACT TRAIN/VAL/TEST PARTITIONS]
- Confirmed dataset SHA: [FROM REGISTRATION]
- Model and original checkpoint: [PATH + SHA]
- Limitations: buildings, not material/individual roof segmentation; existing WHU test previously inspected.

## Experimental controls

- [ ] All arms trained from the same frozen weights and loss, input and epoch budget.
- [ ] Uniform and hard samplers both draw with replacement.
- [ ] Three matched seeds ran to the full preregistered epoch endpoint.
- [ ] Validation masks and threshold remained unchanged.
- [ ] Data hashes verified; no test-driven tuning.

## Results

**Do not fill this table before actual training.**

| Treatment | Mean final val IoU | SD across seeds | Notes |
| --- | ---: | ---: | --- |
| Uniform / standard | [ ] | [ ] | Control |
| Hard / standard | [ ] | [ ] | Sampling-only |
| Uniform / robust | [ ] | [ ] | Augmentation-only |
| Hard / robust | [ ] | [ ] | Combined |

- Sampling effect: [ ] IoU percentage points
- Augmentation effect: [ ] IoU percentage points
- Interaction: [ ] IoU percentage points
- Mean final-vs-best checkpoint tradeoff: [ ]

## Visual analysis

[Insert registered learning curves, paired inference overlays, false-positive and false-negative examples, source annotations, test limitations. Keep sampled images representative; if targeting hard examples, separately identify them.]

## Interpretation and next step

[State what helped, what failed, variation between seeds, extra GPU/training cost, and why a single city's validation partition does not demonstrate geographic transfer.]

**External confirmation:** [NOT RUN / FRESH INDEPENDENT EVALUATION WITH EXACT DATA AND REGIONS]

## Resume-ready line

Designed and executed a matched-seed factorial ablation of [actual conditions] for aerial building segmentation, isolating sampling and augmentation effects under geographic validation, with reproducible data/checkpoint hashes and [measured results].

Do not publish this line as completed until training and analysis artifacts support it.
