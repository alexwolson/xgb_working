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
   --study-name water_modelling \
   --study_count 100 \
   --onehot-encoding \
   --discard-features "time[s],SEN,L_wave_ht[mm]" \
   --data-directory Clogging_Data
   --subsample-shap
    ```
3. **Output Predictions**
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

Avoid column names that differ ONLY by non-alphanumeric characters.
