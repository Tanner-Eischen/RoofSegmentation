# RoofSegmentation: 2×2 training ablation study

**Status:** Integrated experiment runner and analysis. Real study status and measurements live under the chosen `runs/` output root; see `workflow_status.json`. No accuracy result is implied by integration or smoke tests.

**Target repository:** [Tanner-Eischen/RoofSegmentation](https://github.com/Tanner-Eischen/RoofSegmentation), specifically the newer `roof_pipeline` training and evaluation modules. Do **not** run this study through the older `train_v2.py`/`evaluate.py` scripts; those are a different checkpoint pipeline.

## Purpose

Your existing [WHU refinement](https://github.com/Tanner-Eischen/RoofSegmentation/blob/main/docs/WHU_REFINEMENT_2026-10-28.md) compared **hard-example sampling + robust photometric augmentation** against a **uniform-sampling + standard-augmentation** control. Because two factors changed together, that experiment cannot estimate either factor's independent effect or their interaction. This study does precisely that.

| Arm | Sampling | Photometric augmentation | Interpretation |
| --- | --- | --- | --- |
| `a0b0_uniform_standard` | Uniform with replacement | Standard | Matched control |
| `a1b0_hard_standard` | Training-error-weighted with replacement | Standard | Sampling treatment |
| `a0b1_uniform_robust` | Uniform with replacement | Robust | Augmentation treatment |
| `a1b1_hard_robust` | Training-error-weighted with replacement | Robust | Interaction/two-factor treatment |

**Sampling control:** This repository now includes `--uniform-replacement`. Default training still shuffles without replacement. The study explicitly uses replacement for both sampling policies, with identical numbers of draws per epoch, and logs actual unique-image counts. The original installation utility is retained for older compatible checkouts; do not apply it again here.

## Prerequisites

- Windows/PowerShell supported, Python 3.11 recommended, CUDA-capable PyTorch environment as in the repo's `README_UPGRADE.md`.
- Official WHU data already prepared, preferably at `data/whu_official_prepared/` with `manifest.csv` and real paired RGB/masks.
- A **frozen compatible** 512-pixel `unet-resnet34` checkpoint trained on the exact same dataset, such as `runs/whu_refinement_2026-10-28/control/best.pt`. The actual checkpoint path must exist locally and pass the repository's SHA checks.
- Execute commands from the root of your **RoofSegmentation** repository. The scripts in this package can live under `ablation_study/` there.
- There is **no** silent data download, API call, automatic test evaluation, or fallback to synthetic accuracy.

## 1. Verify the integrated repository

The package is installed at `ablation_study/`. Run from the repository root using your CUDA Python environment:

```powershell
# From RoofSegmentation repository root
.\.venv-gpu\Scripts\python.exe -m ruff check roof_pipeline tests ablation_study
.\.venv-gpu\Scripts\python.exe -m pytest tests ablation_study/tests -q
```

The study freezes source hashes/snapshot, full dataset contents, checkpoint,
mining report, settings, and CPU-thread environment. Changed inputs on resume
are rejected. A completed run is skipped only after checking its endpoint,
arguments, hashes, and per-image validation counts.

## 2. Review the complete design without training

```powershell
python .\ablation_study\scripts\run_factorial.py `
  --repo . `
  --data data/whu_official_prepared `
  --checkpoint runs/whu_r34_official_2026-10-28/best.pt `
  --out runs/whu_factorial_2026-10 `
  --dry-run
```

## 3. Run the 12-arm-seed training experiment

```powershell
$env:OMP_NUM_THREADS = '4'
$env:MKL_NUM_THREADS = '4'
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
.\.venv-gpu\Scripts\python.exe .\ablation_study\scripts\run_factorial.py `
  --repo . `
  --data data/whu_official_prepared `
  --checkpoint runs/whu_r34_official_2026-10-28/best.pt `
  --out runs/whu_factorial_2026-10 `
  --seeds 42,1337,2026 `
  --epochs 20 --batch-size 4 --workers 4 `
  --lr 0.0001 --device cuda
```

Use `--resume` with exactly the same configuration to skip fully completed arms. An interrupted arm is rejected rather than overwritten or silently resumed from a different training state. Use a new output root if the protocol changes.

`run_factorial.py` first mines **training-only** errors once, then runs four treatments per seed. All four treatments use with-replacement samplers, the same epoch budget, 512-pixel input, the same source checkpoint, and threshold 0.5 for the primary final-epoch metric. No test split is evaluated. Runs retain trainer `best.pt`, `final.pt`, history, validation detail and fingerprints.

## 4. Aggregate measurements

```powershell
python .\ablation_study\scripts\analyze_factorial.py --out runs/whu_factorial_2026-10
```

Generated under `runs/whu_factorial_2026-10/analysis/`:

- `report.md` with factual results and explicit limitations.
- `runs.csv`, one row per seed and treatment.
- `arms_summary.csv`, mean/range/SD of validation IoU.
- `contrasts_by_seed.csv`, paired sampling/augmentation/interaction contrasts.
- `contrasts_summary.csv`, effect magnitudes and across-seed variation.
- `figures/learning_curves.png` and matched validation overlays, with frozen
  sample selection and checkpoint/figure fingerprints.

Analysis and figures also run automatically after the final arm. To regenerate
figures explicitly:

```powershell
.\.venv-gpu\Scripts\python.exe .\ablation_study\scripts\plot_factorial.py --out runs/whu_factorial_2026-10 --device cuda
```

The analysis refuses mismatched checkpoints or partial runs, and does not create p-values from spatially correlated tiles.

## 5. What makes this a credible case study

Read `STUDY_PROTOCOL.md` for hypotheses, factor definitions, pre-specified endpoints, controls, interpretation, and external confirmation. Fill `reports/CASE_STUDY_TEMPLATE.md` **only with measurements from completed runs**.

**Dataset semantics:** WHU ground truths are binary *building extent* labels, not individually separated roof surfaces, material type, or damage. The legacy roof app name does not alter what the labels mean.

**Holdout limitation:** A WHU test score is not clean independent confirmation here because the project's earlier test results have already been inspected and helped motivate model changes. Use the geographically partitioned validation split for exploratory treatment comparisons, then acquire a genuinely new labeled geographic/sensor holdout for confirmation. Existing code binds checkpoint fingerprints to the training dataset, so independent-data evaluation requires an explicit new evaluation entry point with a frozen label mapping. Do not bypass hashes to force the existing evaluator to accept a different dataset.
