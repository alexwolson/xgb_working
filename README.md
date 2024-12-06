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

## Installation & Requirements

- **Python**: 3.8+ recommended
- **Dependencies**: Listed in both `requirements.txt` and `environment.yml`.
  
To set up the environment, you can either:

**Option 1: Use `requirements.txt` with pip**
```bash
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
    python run_study.py --targets Count_EX1 Count_EX1 \
   --study_name water_modelling \
   --study_count 100 \
   --onehot-encoding \
   --discard-features "SEN,L_wave_ht[mm]" \
   --subsample-shap
    ```
### Arguments:
- `--targets`: A space-separated list of target variables to predict (e.g., `Count_EX1`, `Count_EX2`). Default: `Count_EX1`.
- `--study-name`: Name of the Optuna study. Default: `water_modelling`.
- `--study_count`: Number of Optuna trials to run. Default: `1`.
- `--onehot-encoding`: Perform one-hot encoding of categorical variables. Default: Disabled.
- `--discard-features`: A comma-separated list of features to discard from the dataset (e.g., `SEN,L_wave_ht[mm]`).
- `--tree-method`: The `tree_method` parameter for XGBoost (`gpu_hist`, `hist`, etc.). Default: `gpu_hist`.
- `--data-directory`: Directory containing the raw data files. Default: `Organized_Data`.
- `--storage-path`: Path to the database for storing Optuna study results. Default: `sqlite:///water_modelling.db`.

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


## Notes

The script leverages GPU acceleration if available via tree_method='gpu_hist'. Adjust if needed.

Ensure that all directory paths (Organized_Data/, xgb_models/, figures/) exist or are writable.

When rerunning experiments, existing models and Optuna studies are reused unless removed or changed.