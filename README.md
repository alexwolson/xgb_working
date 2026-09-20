> **⚠️ Archived — superseded by [`alexwolson/steel-flow`](https://github.com/alexwolson/steel-flow).**
>
> The final code from this repository has been consolidated into the canonical
> `steel-flow` monorepo. This repository is kept read-only for historical reference.

# Project Overview

This repository provides a data processing and modelling pipeline for predicting target variables (`Count_EX1` or `Count_EX2`) from sensor-based time series data. The pipeline automates data extraction, cleaning, feature engineering, hyperparameter optimisation (Optuna), model training (XGBoost), evaluation, and visualisation (error distributions, SHAP explanations, confusion matrices).

The pipeline supports two prediction modes:
- **Regression** — predict a continuous count value directly
- **Binned classification** — predict an ordinal bin label (e.g. low / medium / high), trained as a multiclass XGBoost classifier

## Key Features

1. **Data Processing**
   - Processes raw `.xlsx` files from `data/raw/Organized_Data/`.
   - Extracts features and target values; caches preprocessed CSVs automatically.
   - Optional feature engineering: SEN geometry, clogging factors, mould position, lagged features, one-hot encoding.
   - Sheet-level train / validation / test split to preserve run independence.

2. **Hyperparameter Optimisation**
   - Optuna TPE sampler, seeded for reproducibility.
   - Regression mode: minimises MAE on the validation set.
   - Binned mode: minimises MAE of bin indices (ordinal-aware) on the validation set.
   - Studies persisted to `optuna/` (SQLite or JournalStorage for HPC multithreading).

3. **Model Training & Evaluation**
   - Regression: `XGBRegressor` — reports MAE, MAPE, R², RMSE.
   - Binned: `XGBClassifier` (`multi:softmax`) — reports accuracy, macro/weighted F1, Cohen's kappa, MAE of bin indices.
   - Models saved to `data/output/models/`.

4. **Visualisation & Explainability**
   - Regression: error histograms, SHAP beeswarm and bar plots, residual and actual-vs-predicted plots.
   - Binned: normalised confusion matrix heatmap.
   - All figures saved to `data/output/figures/`.

5. **Batch Experiment Runners**
   - `uv run replications` — runs the full regression experiment suite (`config/experiments/`).
   - `uv run experiments` — runs both the regression suite and the binning experiment suite (`config/experiments_binning/`), with a combined summary table.

6. **Predictions on New Data**
   - Applies trained model to new Excel files, writing predictions back row-by-row.
   - Binned models output label strings (`low` / `medium` / `high` for 3 bins, `bin_0` / `bin_1` / … for other N).

## Installation & Requirements

- **Python**: 3.10+
- **Package manager**: [uv](https://docs.astral.sh/uv/)

```bash
git clone https://github.com/alexwolson/xgb_working.git
cd xgb_working
uv sync
```

## Usage

All pipeline settings live in `config/base.toml`. Edit that file before running any command. To use an alternate config, pass its path as a positional argument.

### 1. Data Preparation

Place raw `.xlsx` files under `data/raw/Organized_Data/` in the appropriate subdirectory structure (e.g. `SEN06/`, `SEN07/`). CSVs are generated automatically on first run.

### 2. Configure your TOML

Open `config/base.toml` and set your experiment parameters:

```toml
[experiment]
targets = ["Count_EX1", "Count_EX2"]
study_name = "MyStudy"
objective = "reg:squarederror"

[data]
directory = "data/raw/Organized_Data"
discard_features = ["time[s]"]
onehot_encoding = true
sen_geometrical = true

[tune]
study_count = 100
tree_method = "gpu_hist"          # use "hist" on CPU-only machines
storage_path = "sqlite:///optuna/MyStudy.db"

[train]
subsample_shap = true

[predict]
config_file = "config/training_config_MyStudy.json"
data_directory = "New_Data"

# Optional — omit entirely to run in regression mode
[bins]
enabled = true
n_bins = 3
strategy = "equal_frequency"     # "equal_frequency" | "equal_width" | "log"
```

### 3. Hyperparameter Tuning

```bash
uv run tune                              # uses config/base.toml
uv run tune config/my_experiment.toml   # uses alternate config
```

### 4. Train Final Model

```bash
uv run train
uv run train config/my_experiment.toml
```

Saves the model to `data/output/models/` and the training config JSON to `config/`.

### 5. Predict on New Data

```bash
uv run predict
uv run predict config/my_experiment.toml
```

Writes predictions (continuous values or bin labels) back into the source Excel files.

### 6. Run Full Experiment Suites

```bash
# Regression experiments only (config/experiments/)
uv run replications

# Both regression + binning experiments
uv run experiments

# Skip a phase if already completed
uv run experiments --skip-replications
uv run experiments --skip-binning

# Force re-run completed experiments
uv run experiments --overwrite

# Use custom directories
uv run experiments --replications-dir config/experiments --binning-dir config/experiments_binning
```

Both runners skip experiments whose training config JSON already exists, and display a Rich summary table at the end.

## Binned Prediction Mode

When `[bins] enabled = true`, the pipeline trains a multiclass classifier instead of a regressor. The bin boundaries are computed from the **training data only** and saved in the training config JSON for use at prediction time.

Three binning strategies:

| Strategy | Description |
|----------|-------------|
| `equal_frequency` | Quantile-based — each bin contains approximately equal numbers of training samples |
| `equal_width` | Linearly spaced between training min and max — outer bins extend to ±∞ |
| `log` | Logarithmically spaced via log1p/expm1 — more resolution at small counts |

For 3 bins, predictions are labelled `low`, `medium`, `high`. For other values of `n_bins`, labels are `bin_0`, `bin_1`, ….

The Optuna tuning objective is MAE of bin indices (ordinal-aware: being two bins off is penalised twice as much as being one bin off), keeping the same `direction="minimize"` as regression mode.

## Results & Outputs

| Output | Location |
|--------|----------|
| Trained models | `data/output/models/<study_name>.json` |
| Training config JSON | `config/training_config_<study_name>.json` |
| Error histograms (regression) | `data/output/figures/error_histogram_<study_name>.pdf/png` |
| SHAP plots (regression) | `data/output/figures/shap_<study_name>.pdf/png` |
| Prediction error plots (regression) | `data/output/figures/error_plot_*_<study_name>.pdf/png` |
| Confusion matrix (binned) | `data/output/figures/confusion_matrix_<study_name>.pdf/png` |
| Optuna studies | `optuna/<study_name>.db` (or `optuna/optuna_<study_name>.log` for multithread) |

## Config Reference

**`[experiment]`**
- `targets`: Target variables to train. Default: `["Count_EX1"]`.
- `study_name`: Optuna study name prefix. Default: `"water_modelling"`.
- `objective`: XGBoost objective for regression (`"reg:squarederror"`, `"count:poisson"`, etc.). Ignored when `[bins] enabled = true`. Default: `"reg:squarederror"`.

**`[data]`**
- `directory`: Path to raw `.xlsx` files. Default: `"data/raw/Organized_Data"`.
- `discard_features`: Features to drop before training. Default: `[]`.
- `onehot_encoding`: One-hot encode categorical features. Default: `false`.
- `sen_geometrical`: Convert SEN codes to angle/depth features. Default: `false`.
- `clogging_factors`: Include clogging factor features. Default: `false`.
- `lagged_features`: Features to create lag columns for. Default: `[]`.
- `feature_lag_amount`: Number of lag steps. Default: `0`.
- `mould_position`: Include mould position as a feature. Default: `false`.
- `remove_rows`: Rows to drop from the end of each sheet. Default: `0`.
- `include_subdirs`: Restrict to matching subdirectory patterns. Default: `[]` (all).
- `exclude_subdirs`: Exclude matching subdirectory patterns. Default: `[]`.

**`[tune]`**
- `study_count`: Number of Optuna trials. Default: `1`.
- `tree_method`: XGBoost tree method (`"hist"`, `"gpu_hist"`). Default: `"gpu_hist"`.
- `storage_path`: Optuna study database URI. Default: `"sqlite:///optuna/water_modelling.db"`.
- `multithread`: Use JournalStorage for HPC multithreading. Default: `false`.

**`[train]`**
- `subsample_shap`: Subsample to 10% when computing SHAP values. Default: `false`. Has no effect in binned mode (SHAP is skipped).
- `config_file`: Output filename for training config JSON. Default: `""` (auto-derived as `training_config_<study_name>.json`).

**`[predict]`**
- `config_file`: Path to training config JSON produced by `uv run train`. Required.
- `data_directory`: Directory containing new Excel files. Default: `"New_Data"`.
- `piv_data`: Input is PIV format (iterates over 4 mould positions per row). Default: `false`.

**`[bins]`** *(optional — omit to run in regression mode)*
- `enabled`: Activate binned classification mode. Default: `false`.
- `n_bins`: Number of bins. Default: `3`.
- `strategy`: Binning strategy (`"equal_frequency"`, `"equal_width"`, `"log"`). Default: `"equal_frequency"`.

## Experiment Configs

Pre-configured experiment configs live in two directories:

- `config/experiments/` — 19 regression experiments covering a range of feature engineering choices (lag amounts, geometrical features, clogging factors, mould position, data subsets)
- `config/experiments_binning/` — 8 binning experiments: base feature set × all three strategies with N-sensitivity, and the best regression feature set × all three strategies

All experiment configs share the same Optuna database within their group (`optuna/experiments.db` or `optuna/experiments_binning.db`) to allow study reuse across runs.

## Notes

**GPU acceleration**: Set `tree_method = "gpu_hist"` for GPU machines; use `"hist"` on CPU-only.

**Data split**: Sheets are split alphabetically — last 20% to test, next 20% to validation, remainder to training. Entire experimental runs stay within one partition to prevent temporal leakage.

**CSV cache**: Preprocessed CSVs are cached in `data/raw/`. Delete them to force regeneration after changing feature engineering flags.

**MAPE with zero counts**: Zero-valued targets are excluded from MAPE. The count of excluded rows is reported as `mape_zero_excluded`.

**Study reuse**: Existing Optuna studies and trained models are reused unless removed. Delete `optuna/<study>.db` and `data/output/models/<study>.json` to start fresh.

---

# HPC Guide

### Setup

Clone the repository on your local machine. Prepare the dataset and add it to `data/raw/Organized_Data/`. Transfer the folder to `/home/` on the server using Globus.

Instructions below are for Beluga; some details may differ on other clusters.

### Remote shell

```bash
ssh <username>@beluga.alliancecan.ca
cd xgb_working
```

### Submit a job

```bash
sbatch scripts/hpc/study_job.sh    # hyperparameter tuning + training
sbatch scripts/hpc/predict_job.sh  # prediction on new data
```

Check parameters in the script before submitting.

### Monitor

```bash
sq                                         # find running job ID
tail -f bg<node>-<job>.out                 # live log feed
```

### Checkpointing

`study_job.sh` requests 1 day. Training all non-clogging data takes ~8 hours per target. If a job is interrupted, restarting will resume the Optuna study and add more trials. Reduce `study_count` in the config to end early — do not kill the process mid-run, as the training config JSON is written only on completion.

### Multithreading

Set `multithread = true` in `[tune]` to use `JournalStorage` instead of SQLite. Optuna log files are written to `optuna/`. Run `uv run train` with the same config after tuning to produce the training config JSON. `multithread_study_eval.sh` automates this sequence.

If you see DOS line-ending errors when submitting:
```bash
dos2unix scripts/hpc/study_job.sh
```
