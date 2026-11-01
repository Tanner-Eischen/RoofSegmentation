# Integrated verification — October 31, 2026

- The sampler control is integrated into `roof_pipeline/train.py`, with epoch draw/unique-image counts and unchanged legacy defaults.
- `python -m pytest tests ablation_study/tests -q`: 242 passed, including 13 unittest subtests.
- Ruff passed for the pipeline, tests, study, and existing CI Python scope.
- Tests include real tiny-model CPU training in legacy/uniform/hard modes and real CPU prediction-figure generation. Synthetic fixtures verify behavior, not WHU accuracy.
- Factorial numeric fixtures verify paired main effects, interaction, and seed variability. Regressions reject changed checkpoints/inputs, inconsistent factors, invalid metrics, and incomplete runs.
- Official WHU content fingerprint and publisher partition counts passed local preflight; CUDA and the RTX 4060 were detected.
- Real study progress is recorded under `runs/whu_factorial_2026-10-31/workflow_status.json`. No full-run result is claimed here.
- Remote CI and browser/application UI checks were not performed. This work concerns the training pipeline; no UI access is required.
