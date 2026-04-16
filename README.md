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
   - Splits data into training and testing sets.

3. **Hyperparameter Optimization with Optuna**:  
   - Creates or reuses an Optuna study stored in `water_modelling.db`.  
   - Optimizes XGBoost hyperparameters to minimize mean absolute error (MAE).

4. **Model Training & Evaluation**:  
   - Trains an XGBoost model using the best-found hyperparameters.  
   - Saves the trained model to `data/output/models/`.  
   - Evaluates model performance on the test set and logs the MAE.

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

### 1. Data Preparation

Place raw `.xlsx` files under `data/raw/Organized_Data/` in the appropriate subdirectory structure (e.g., `SEN06/`, `SEN07/`). CSVs are generated automatically on first run.

### 2. Hyperparameter Tuning

```bash
uv run tune --targets Count_EX1 Count_EX2 \
  --study-name MyStudy \
  --study-count 100 \
  --onehot-encoding \
  --sen-geometrical \
  --discard-features "time[s],SEN,L_wave_ht[mm]" \
  --data-directory data/raw/Organized_Data
```

### 3. Train Final Model

```bash
uv run train --targets Count_EX1 Count_EX2 \
  --study-name MyStudy \
  --onehot-encoding \
  --sen-geometrical \
  --discard-features "time[s],SEN,L_wave_ht[mm]" \
  --data-directory data/raw/Organized_Data \
  --subsample-shap
```

Saves the model to `data/output/models/` and the training config to `config/`.

### 4. Predict on New Data

```bash
uv run predict --config-file config/training_config_MyStudy.json \
  --data-directory New_Data
```

### Arguments

**`uv run tune` and `uv run train`**
- `--targets`: Space-separated target variables (`Count_EX1`, `Count_EX2`). Default: `Count_EX1`.
- `--study-name`: Optuna study name prefix. Default: `water_modelling`.
- `--study-count`: Number of Optuna trials. Default: `1`.
- `--onehot-encoding`: One-hot encode categorical features.
- `--sen-geometrical`: Convert SEN numbers to angle/depth features.
- `--clogging-factors`: Include clogging factors.
- `--discard-features`: Comma-separated features to discard.
- `--tree-method`: XGBoost tree method (`hist`, `gpu_hist`, etc.). Default: `gpu_hist`.
- `--data-directory`: Directory with raw data. Default: `data/raw/Organized_Data`.
- `--storage-path`: Optuna study storage path. Default: `sqlite:///water_modelling.db`.
- `--multithread`: Use JournalStorage for HPC multithreading.
- `--lagged-features`: Comma-separated features to lag.
- `--feature-lag-amount`: Number of lag steps.
- `--mould-position`: Include mould position as feature.
- `--remove-rows`: Rows to remove from the end of each sheet.

**`uv run train` only**
- `--subsample-shap`: Subsample SHAP values for faster computation.
- `--config-file`: Output filename for training config JSON (saved to `config/`).

**`uv run predict`**
- `--config-file`: Path to config JSON (e.g., `config/training_config_MyStudy.json`). Required.
- `--data-directory`: Directory with new Excel files. Default: `New_Data`.
- `--piv-data`: Data is PIV format.

### Results & Outputs

1. **Models**: Saved in `data/output/models/` as `.json` files.
2. **Figures**: Error histograms and SHAP plots saved in `data/output/figures/`.
3. **Configs**: Training configuration JSON saved in `config/`.
4. **Optuna Study**: Results stored in the database at `--storage-path` (default: `water_modelling.db`).

## Notes

The script leverages GPU acceleration if available via tree_method='gpu_hist'. Adjust if needed.

Ensure that all directory paths exist or are writable before running.

When rerunning experiments, existing models and Optuna studies are reused unless removed or changed.

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
`sbatch study_job.sh` or `sbatch predict_job.sh` (CHECK THE PARAMETERS BEFORE RUNNING)
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

This led me to use a workaround: JournalStorage. This is selected through the `--multithread` parameter. Hopefully this is more stable. Additionally, `uv run tune` must be run again with the SAME parameters and `--evaluate-only` to evaluate the model and build the training_config JSON file. `multithread_study_eval.sh` is setup to do this. 

There's also sometimes an error saying:
```
sbatch: error: Batch script contains DOS line breaks (\r\n) 
sbatch: error: instead of expected UNIX line breaks (\n).
```

Run this:
`dos2unix foo.txt      # Replace foo.txt with the name of your file`

Note: I originally wrote the batch script to use a full day for training. However, it for sure takes less time. Change it to a few hours and test it. This will increase the job's priority in the queue.
