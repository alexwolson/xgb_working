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

**Option 1: Use `requirements.txt` with pip**
```bash
python -m venv .venv
source .venv/Scripts/Activate
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

Add the following files:
`study_job.sh`
```bash#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32000M
#SBATCH --time=0-20:0:0
#SBATCH --output=%N-%j.out

module load python/3.10
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index -r requirementscomputecan.txt

python run_study.py --targets Count_EX1 Count_EX2 \
--study-name ComputeCan_20250625_test1 \
--study_count 100 \
--onehot-encoding \
--sen-geometrical \
--discard-features "time[s]" \
--data-directory data \
--subsample-shap 
```

`predict_job.sh`
```bash#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32000M
#SBATCH --time=0-10:0:0
#SBATCH --output=%N-%j.out

module load python/3.10
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index -r requirementscomputecan.txt

python predict.py --config-file <TRAINING CONFIG FILE NAME> \
--data-directory data
```

`requirementscomputecan.txt`
```
alembic==1.14.0+computecanada
cloudpickle==3.1.1+computecanada
colorlog==6.9.0+computecanada
contourpy==1.3.1+computecanada
cycler==0.12.1+computecanada
et_xmlfile==2.0.0+computecanada
fonttools==4.56.0+computecanada
greenlet==3.1.1+computecanada
joblib==1.5.1+computecanada
kiwisolver==1.4.8+computecanada
llvmlite==0.44.0+computecanada
mako==1.3.10+computecanada
markdown_it_py==3.0.0+computecanada
MarkupSafe==2.1.5+computecanada
matplotlib==3.10.0+computecanada
mdurl==0.1.2+computecanada
numba==0.61.0+computecanada
numpy==2.1.1+computecanada
openpyxl==3.1.5+computecanada
optuna==4.1.0+computecanada
packaging==25.0+computecanada
pandas==2.2.3+computecanada
pillow==11.0.0+computecanada
pygments==2.19.2+computecanada
pyparsing==3.2.3+computecanada
python_dateutil==2.9.0.post0+computecanada
pytz==2025.2+computecanada
PyYAML==6.0.2+computecanada
rich==13.9.4+computecanada
scikit_learn==1.5.2+computecanada
scipy==1.14.1+computecanada
seaborn==0.13.2+computecanada
shap==0.46.0+computecanada
six==1.17.0+computecanada
slicer==0.0.8+computecanada
SQLAlchemy==2.0.35+computecanada
threadpoolctl==3.6.0+computecanada
tqdm==4.67.1+computecanada
```

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

### Monitor the job
`sq` to find the current running job ID
Wait for the status to become R (running)
`ls` to list all the files in the directory (you're looking for the .out file)
`tail -f bg<node number>-<job number>.out`
This will open a live feed of the job's output.

### Important Note
I put 20 hours in `study_job.sh`, this might not be enough (training on all non-clogging data takes ~8 hours per target, 2 target variables). If a job runs out of time/is interrupted, restarting the script will try running another 100 trials. To end the trials early, change the `--study_count` to something lower, as interrupting it will prevent it from creating the `training_config` that allows you to predict using the model.

