# Project Overview

This repository provides a data processing and modeling pipeline for predicting target variables (`Count_EX1` or `Count_EX2`) from sensor-based time series data. The code automates data extraction, cleaning, feature engineering, hyperparameter optimization (using Optuna), model training (XGBoost), evaluation, and generating visualizations (error distributions and SHAP explanations).

## Key Features

1. **Data Generation**:  
   - Processes raw `.xlsx` files located in `Organized_Data/` directories.  
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
   - Saves the trained model to `xgb_models/`.  
   - Evaluates model performance on the test set and logs the MAE.

5. **Visualization & Explainability**:  
   - Saves error distribution plots (error histograms) in `figures/`.  
   - Computes SHAP values to explain feature importance.  
   - Saves SHAP beeswarm and bar plots for insights into feature contributions.

6. **Predictions on Excel Sheets**
   - Outputs predictions row-by-row directly onto excel sheets found in `New_Data`.

## Installation & Requirements

- **Python**: 3.8+ recommended
- **Dependencies**: Listed in both `requirements.txt` and `environment.yml`.

**Clone the repository**
```bash
git clone https://github.com/alexwolson/steel_flow_sensor_prediction.git
```

To set up the environment, you can either:

**Reccomended: Use uv**
```bash
uv sync
```

Make sure to start any future commands with `uv run`.

**Option 1: Use `requirements.txt` with pip**
```bash
python -m venv .venv
source .venv/Scripts/Activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**Option 2: Use `environment.yml` with conda**
```bash
conda env create -f environment.yml
conda activate water-modelling
```

## Usage
1. **Data Preparation**:  
   Ensure `Organized_Data` directories contain the necessary `.xlsx` files. If `X.csv` and `y.csv` are not present, they will be generated automatically.
2. **Run the Pipeline**:
    ```bash
    python run_study.py --targets Count_EX1 Count_EX2 \
   --study-name NewFeaturesTest_20250625 \
   --study_count 100 \
   --onehot-encoding \
   --sen-geometrical True \
   --clogging-factors False \
   --discard-features "time[s],SEN,L_wave_ht[mm]" \
   --data-directory Organized_Data/SEN10 \
   --subsample-shap
    ```
    
3. **Output Predictions**:
   ```bash
   python predict.py --config-file training_config_water_modelling.json \
   --data-directory New_Data
   ```

### Arguments:
**run_study.py**
- `--targets`: A space-separated list of target variables to predict (e.g., `Count_EX1`, `Count_EX2`). Default: `Count_EX1`.
- `--study-name`: Name of the Optuna study. Default: `water_modelling`.
- `--study_count`: Number of Optuna trials to run. Default: `1`.
- `--onehot-encoding`: Perform one-hot encoding of categorical variables. Default: Disabled.
- `--sen-geometrical`: Convert SEN numbers to geometrical features. Default: Disabled.
- `--clogging-factors`: If clogging factors are included in the dataset. Default: Disabled
- `--discard-features`: A comma-separated list of features to discard from the dataset (e.g., `SEN,L_wave_ht[mm]`).
- `--tree-method`: The `tree_method` parameter for XGBoost (`gpu_hist`, `hist`, etc.). Default: `gpu_hist`.
- `--data-directory`: Directory containing the raw data files. Default: `Organized_Data`.
- `--storage-path`: Path to the database for storing Optuna study results. Default: `sqlite:///water_modelling.db`.

**predict.py**
- `--config-file`: Path to the training configuration JSON file output by the training script (e.g., `training_config_water_modelling.json`). Required.
- `--data-directory` Directory containing new Excel files (will be scanned recursively). Default: `New_Data`.
  
### Results & Outputs:
1. **Models**:
   - Trained XGBoost models are saved in the `xgb_models/` directory as `.json` files.
   
2. **Figures**:
   - Error histograms and SHAP plots are saved in the `figures/` directory.  
     - Error histogram: Visualizes the distribution of prediction errors.  
     - SHAP Beeswarm Plot: Shows feature-level impact on predictions.  
     - SHAP Bar Plot: Summarized feature importance.

3. **Optuna Study**:
   - Study results (e.g., hyperparameters and trial metrics) are stored in the database specified by `--storage-path` (default: `water_modelling.db`).

4. **Predictions on Sheets**
   - Resulting models (selected by `--config-file` can be used to predict targets and output directly on sheets in the directory specified by `--data-directory`.

## Notes

The script leverages GPU acceleration if available via tree_method='gpu_hist'. Adjust if needed.

Ensure that all directory paths (Organized_Data/, xgb_models/, figures/) exist or are writable.

When rerunning experiments, existing models and Optuna studies are reused unless removed or changed.

# HPC Guide:

### Setup
Clone the repository on your local device.

Prepare the dataset and add it to the folder under `/data/`
Transfer the folder to the `/home/` directory on the chosen server using Globus

Instructions below are for Beluga, some details might differ

### Remote shell into the server
`ssh <username>@beluga.alliancecan.ca`
(make sure 2 factor authentication is on your account)

Enter the directory containing the repository:
`cd steel_flow_sensor_prediction`

### Submit the job
`sbatch study_job.sh` or `sbatch predict_job.sh` (CHECK THE PARAMETERS BEFORE RUNNING)
If `predict_job.sh` runs into issues, you can move the model, JSON file, and .db file (.log for multithreaded) to your own computer through Globus and run `start_predict.sh` locally instead.

### Monitor the job
`sq` to find the current running job ID
Wait for the status to become R (running)
`ls` to list all the files in the directory (you're looking for the .out file)
`tail -f bg<node number>-<job number>.out`
This will open a live feed of the job's output.

### Checkpointing
I put 1 day in `study_job.sh`, this might not be enough (training on all non-clogging data takes ~8 hours per target, 2 target variables). If a job runs out of time/is interrupted, restarting the script will try running another 100 trials. To end the trials early, change the `--study_count` to something lower, as interrupting it will prevent it from creating the `training_config` that allows you to predict using the model.
If you find that training two variables in one job is unfeasible, train them one at a time.

### Multithreading (not tested thoroughly)
Unfortunately, because Cedar and Graham are down for maintenance, proper multithreading is poorly implemented and might crash. This is because MySQL and PostgreSQL are only available on those clusters, and they support multi-thread write operations much better than SQLite. I got it briefly running with 2 cores. 

This led me to use a workaround: JournalStorage. This is selected through the `--multithread` parameter. Hopefully this is more stable. Additionally, run_study.py must be run again with the SAME parameters and `--evaluate-only` to evaluate the model and build the training_config JSON file. `multithread_study_eval.sh` is setup to do this. 

There's also sometimes an error saying:
```
sbatch: error: Batch script contains DOS line breaks (\r\n) 
sbatch: error: instead of expected UNIX line breaks (\n).
```

Run this:
`dos2unix foo.txt      # Replace foo.txt with the name of your file`

Note: I originally wrote the batch script to use a full day for training. However, it for sure takes less time. Change it to a few hours and test it. This will increase the job's priority in the queue.
