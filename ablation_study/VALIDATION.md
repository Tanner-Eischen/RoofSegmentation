# Integrated verification — October 31, 2026

- The sampler control is integrated into `roof_pipeline/train.py`, with epoch draw/unique-image counts and unchanged legacy defaults.
- `python -m pytest tests ablation_study/tests -q`: 242 passed, including 13 unittest subtests.
- Ruff passed for the pipeline, tests, study, and existing CI Python scope.
- Tests include real tiny-model CPU training in legacy/uniform/hard modes and real CPU prediction-figure generation. Synthetic fixtures verify behavior, not WHU accuracy.
- Factorial numeric fixtures verify paired main effects, interaction, and seed variability. Regressions reject changed checkpoints/inputs, inconsistent factors, invalid metrics, and incomplete runs.
- Official WHU content fingerprint and publisher partition counts passed local preflight; CUDA and the RTX 4060 were detected.
- Integration alone did not establish WHU accuracy. The completed run is recorded
  below and under `runs/whu_factorial_2026-10-31/workflow_status.json`.
- Remote CI and browser/application UI checks were not performed. This work concerns the training pipeline; no UI access is required.

## Completed study — November 1, 2026

- All 12 runs finished their registered 20 epochs: 240 actual training epochs.
- Final checkpoint hashes, frozen inputs/factors, ordered histories, and all
  1,036 validation rows per run were checked against their aggregate counts.
- All originally registered code remained unchanged; a directory guard triggered
  by unrelated new serving modules was handled using the verified source snapshot.
- Both generated PNGs passed image decoding and original figure-hash checks.
- [Measured results and figures](../docs/WHU_FACTORIAL_2026-10-31.md) and the
  [portable packet](../docs/results/whu_factorial_2026-10-31/README.md) distinguish
  the primary validation analysis from any subsequent descriptive test reuse.

## Published results and supplemental test — November 2, 2026

- All 12 frozen final checkpoints were evaluated once at threshold 0.50 on all
  2,416 official test IDs; per-tile and aggregate metrics/counts were checked.
- The checkpoint/source/data identities remained unchanged. All 13 captured
  primary artifacts, including the original registration, were verified untouched.
- This is a separately frozen descriptive test follow-up, not a new confirmatory
  holdout. No test-based threshold, checkpoint, arm, seed, or deployment selection
  occurred. All endpoint scores were below the independently verified baseline cache.
- The published figures retain their original hashes. Portable result tables,
  confusion counts, histories and provenance have packet-level byte checksums.
- Current local checks: Ruff passed and 24 targeted regression tests passed,
  including 13 additional unittest subtests. Markdown targets and evidence
  aggregates were checked separately. Remote CI is not a claim of this record.
