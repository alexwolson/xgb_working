# Project Overview

This repository provides a data processing and modeling pipeline for predicting target variables (`Count_EX1` or `Count_EX2`) from sensor-based time series data. The code automates data extraction, cleaning, feature engineering, hyperparameter optimization (using Optuna), model training (XGBoost), evaluation, and generating visualizations (error distributions and SHAP explanations).

## Key Features

1. **Data Generation**:  
   - Processes raw `.xlsx` files located in `data/raw/Organized_Data/` directories.  
   - Extracts relevant features and target values to generate two files: `X.csv` (features) and `y.csv` (targets).

2. **Data Loading**:  
   - Loads pre-generated `X.csv` and `y.csv`.  
   - Handles optional one-hot encoding of categorical variables.
   - Allows discarding specific features.  
   - Splits data into train / validation / test sets at the **sheet level** — entire experimental runs go to one partition, preserving temporal independence.

3. **Hyperparameter Optimization with Optuna**:  
   - Creates or reuses an Optuna study stored in `water_modelling.db`.  
   - Optimizes XGBoost hyperparameters to minimize MAE on the **validation set**. The held-out test set is never seen during tuning.
   - Seeded TPE sampler (`seed=42`) for reproducible trial sequences.

4. **Model Training & Evaluation**:  
   - Trains an XGBoost model using the best-found hyperparameters.  
   - Saves the trained model to `data/output/models/`.  
   - Reports MAE, MAPE, R², and RMSE separately for train, validation, and held-out test sets.

5. **Visualization & Explainability**:  
   - Saves error distribution plots (error histograms) in `data/output/figures/`.  
   - Computes SHAP values to explain feature importance.  
   - Saves SHAP beeswarm and bar plots for insights into feature contributions.

6. **Predictions on Excel Sheets**
   - Outputs predictions row-by-row directly onto excel sheets found in the directory specified by `--data-directory`.

## Installation & Requirements

- **Python**: 3.10+
- **Package manager**: [uv](https://docs.astral.sh/uv/)

```bash
git clone https://github.com/alexwolson/steel_flow_sensor_prediction.git
cd steel_flow_sensor_prediction
uv sync
```

## Usage

All pipeline settings live in `config/base.toml`. Edit that file before running any command. To use an alternate config, pass its path as a positional argument.

### 1. Data Preparation

Place raw `.xlsx` files under `data/raw/Organized_Data/` in the appropriate subdirectory structure (e.g., `SEN06/`, `SEN07/`). CSVs are generated automatically on first run.

### 2. Configure `config/base.toml`

Open `config/base.toml` and set your experiment parameters. Key fields:

```toml
[experiment]
targets = ["Count_EX1", "Count_EX2"]   # target variables
study_name = "MyStudy"                  # Optuna study name prefix
objective = "reg:squarederror"          # XGBoost objective

[data]
directory = "data/raw/Organized_Data"
discard_features = ["time[s]"]
onehot_encoding = true
sen_geometrical = true

[tune]
study_count = 100                       # number of Optuna trials
tree_method = "gpu_hist"                # use "hist" on CPU-only machines

[train]
subsample_shap = true

[predict]
config_file = "config/training_config_MyStudy.json"
data_directory = "New_Data"
```

### 3. Hyperparameter Tuning

```bash
uv run tune                              # uses config/base.toml
uv run tune config/my_experiment.toml   # uses alternate config
```

### 4. Train Final Model

Run after tuning is complete:

```bash
uv run train
```

Saves the model to `data/output/models/` and the training config to `config/`.

### 5. Predict on New Data

Set `predict.config_file` and `predict.data_directory` in your TOML, then:

```bash
uv run predict
```

### Results & Outputs

1. **Models**: Saved in `data/output/models/` as `.json` files.
2. **Figures**: Error histograms and SHAP plots saved in `data/output/figures/`.
3. **Configs**: Training configuration JSON saved in `config/`.
4. **Optuna Study**: Results stored in the database at `tune.storage_path` (default: `water_modelling.db`).

### Config Reference

**`[experiment]`**
- `targets`: List of target variables (`"Count_EX1"`, `"Count_EX2"`). Default: `["Count_EX1"]`.
- `study_name`: Optuna study name prefix. Default: `"water_modelling"`.
- `objective`: XGBoost objective (`"reg:squarederror"`, `"count:poisson"`, `"reg:tweedie"`). Default: `"reg:squarederror"`.

**`[data]`**
- `directory`: Directory with raw data. Default: `"data/raw/Organized_Data"`.
- `discard_features`: List of features to discard. Default: `[]`.
- `onehot_encoding`: One-hot encode categorical features. Default: `false`.
- `sen_geometrical`: Convert SEN numbers to angle/depth features. Default: `false`.
- `clogging_factors`: Include clogging factors. Default: `false`.
- `lagged_features`: List of features to lag. Default: `[]`.
- `feature_lag_amount`: Number of lag steps. Default: `0`.
- `mould_position`: Include mould position as feature. Default: `false`.
- `remove_rows`: Rows to remove from end of each sheet. Default: `0`.

**`[tune]`**
- `study_count`: Number of Optuna trials. Default: `1`.
- `tree_method`: XGBoost tree method (`"hist"`, `"gpu_hist"`, etc.). Default: `"gpu_hist"`.
- `storage_path`: Optuna study storage path. Default: `"sqlite:///water_modelling.db"`.
- `multithread`: Use JournalStorage for HPC multithreading. Default: `false`.

**`[train]`**
- `subsample_shap`: Subsample SHAP values for faster computation. Default: `false`.
- `config_file`: Output filename for training config JSON (saved to `config/`). Default: `""` (auto-derived as `training_config_<study_name>.json`).

**`[predict]`**
- `config_file`: Path to training config JSON produced by `uv run train`. Required.
- `data_directory`: Directory with new Excel files. Default: `"New_Data"`.
- `piv_data`: Data is PIV format. Default: `false`.

## Notes

The script leverages GPU acceleration if available via tree_method='gpu_hist'. Adjust if needed.

Ensure that all directory paths exist or are writable before running.

When rerunning experiments, existing models and Optuna studies are reused unless removed or changed.

**Data split:** Data is split at the sheet level — entire experimental runs (sheets) are assigned to train, validation, or test. The split is alphabetical: the last 20% of sheets (by name) go to test, the next 20% to validation, and the remainder to training. This preserves run independence and prevents temporal leakage.

**CSV cache:** Preprocessed CSVs (`X.csv`, `y.csv`) are cached in `data/processed/`. The cache key includes a `v2_` prefix; if you have cached files from an older version of the pipeline, delete them to trigger regeneration.

**MAPE with zero counts:** When the target contains zero values, MAPE is undefined for those rows. The pipeline logs how many zero-target rows were excluded and reports this count as `mape_zero_excluded` in the metrics output.

# HPC Guide:

### Setup
Clone the repository on your local device.

Prepare the dataset and add it to `data/raw/Organized_Data/`.
Transfer the folder to the `/home/` directory on the chosen server using Globus.

Instructions below are for Beluga, some details might differ.

### Remote shell into the server
`ssh <username>@beluga.alliancecan.ca`
(make sure 2 factor authentication is on your account)

Enter the directory containing the repository:
`cd steel_flow_sensor_prediction`

### Submit the job
`sbatch scripts/hpc/study_job.sh` or `sbatch scripts/hpc/predict_job.sh` (CHECK THE PARAMETERS BEFORE RUNNING)
If `predict_job.sh` runs into issues, you can move the model, JSON file, and .db file (.log for multithreaded) to your own computer through Globus and run `uv run predict` locally instead.

### Monitor the job
`sq` to find the current running job ID
Wait for the status to become R (running)
`ls` to list all the files in the directory (you're looking for the .out file)
`tail -f bg<node number>-<job number>.out`
This will open a live feed of the job's output.

### Checkpointing
I put 1 day in `study_job.sh`, this might not be enough (training on all non-clogging data takes ~8 hours per target, 2 target variables). If a job runs out of time/is interrupted, restarting the script will try running another 100 trials. To end the trials early, change the `--study-count` to something lower, as interrupting it will prevent it from creating the `training_config` that allows you to predict using the model.
If you find that training two variables in one job is unfeasible, train them one at a time.

### Multithreading (not tested thoroughly)
Unfortunately, because Cedar and Graham are down for maintenance, proper multithreading is poorly implemented and might crash. This is because MySQL and PostgreSQL are only available on those clusters, and they support multi-thread write operations much better than SQLite. I got it briefly running with 2 cores. 

This led me to use a workaround: JournalStorage. This is selected through the `--multithread` parameter. Hopefully this is more stable. Additionally, run `uv run train` with the SAME parameters to train the best model and generate the training config JSON file. `multithread_study_eval.sh` is setup to do this. 

There's also sometimes an error saying:
```
sbatch: error: Batch script contains DOS line breaks (\r\n) 
sbatch: error: instead of expected UNIX line breaks (\n).
```

Run this:
`dos2unix foo.txt      # Replace foo.txt with the name of your file`

Note: I originally wrote the batch script to use a full day for training. However, it for sure takes less time. Change it to a few hours and test it. This will increase the job's priority in the queue.
