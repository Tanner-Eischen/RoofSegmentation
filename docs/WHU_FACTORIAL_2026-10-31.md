# WHU factorial training ablation — October 31, 2026

Four arms isolate sampling (uniform versus frozen training-error weighting)
and augmentation (standard versus robust). Both sampling arms use replacement
for 4,736 draws per epoch. Actual unique-image coverage is logged in history.
See [the full protocol](../ablation_study/STUDY_PROTOCOL.md) and
[runner instructions](../ablation_study/README.md).

## Registered settings

- Original publisher partitions and masks: 4,736 training, 1,036 validation,
  2,416 test. No provisional correction overlays or auxiliary boundary head.
- U-Net/ResNet-34, 512 pixels, batch size 4, 20 full epochs per run.
- Seeds 42, 1337, 2026, paired across all four arms: 12 runs, 240 epochs total.
- All runs start from `runs/whu_r34_official_2026-10-28/best.pt`, the original
  epoch-32 baseline. Each run restarts AdamW at 0.0001, weight decay 0.0001,
  cosine schedule, BCE plus Dice, positive weight 1, and CUDA AMP.
- Fixed final-epoch checkpoint and threshold 0.50 are the primary endpoint.
  Validation-selected/calibrated `best.pt` remains a secondary artifact.
- Four Windows data workers and four CPU threads, on the local RTX 4060.

Dataset content SHA256:
`538ac8569dcfd817b8557bc23ed18977de5dbba8d4b19719c82ab5ee4d1a308d`.

## Execution and outputs

Study artifacts: `runs/whu_factorial_2026-10-31/` (Git-ignored).
`registration.json` freezes the actual settings, checkpoint/data/source hashes,
environment and run budget before mining. `source_snapshot.zip` retains code.
`workflow_status.json` and each run's `progress.json` record execution. The
launcher logs are in `runs/ablation_integration_validation/factorial.stdout.log`
and `factorial.stderr.log`. `mining_fingerprint.json` seals the single
training-only mining report.

After all runs complete, the runner verifies the endpoint checkpoints against
their per-image validation counts, then creates `analysis/report.md` and CSVs
containing every arm/seed score, paired effects, standard deviations and epoch
fitting/validation time. It also creates learning curves and fixed-seed matched
validation overlays under `analysis/figures/`. Artifact hashes and example IDs
are recorded. The case-study template remains unfilled until results exist.

The integration passed 242 tests and Ruff before launching real experiments.
Tests are workflow evidence; inspect the live artifacts for actual completion
and performance. No completed 12-run scientific outcome is claimed by this file.

## Interpretation

The experiment estimates fine-tuning effects conditional on one shared pretrained
checkpoint and the existing WHU validation geography. Three seeds characterize
limited optimization/augmentation variability. They do not measure independent
pretraining or geographic replication. The existing WHU test has already informed
development; this runner never uses it for inference or selection. Dataset hashes
and audits still read test files for integrity. A fresh geographic holdout remains
necessary for confirmatory generalization claims.
