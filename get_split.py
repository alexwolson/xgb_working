import argparse
import json
import logging
import os
import re
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import matplotlib.pyplot as plt
import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
import pandas as pd
import seaborn as sns
import shap
import xgboost as xgb
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
    mean_squared_error,
    root_mean_squared_error,
    root_mean_squared_log_error,
    PredictionErrorDisplay,
)
from sklearn.model_selection import train_test_split

# Set up rich console for user-facing messages
console = Console()

# Configure logging to use rich's RichHandler
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console)],
)
logger = logging.getLogger(__name__)

def generate_csv_files(
    data_directory: str, preprocess_details:str, sen_geometrical: bool = False, clogging_factors: bool = False, mould_position: bool = False
) -> None:
    """
    Generate X.csv and y.csv from Excel files in the given data_directory.
    Checks for directory existence and empty file sets.
    """
    tabular_root = Path(data_directory)
    if not tabular_root.exists():
        logger.error(f"Data directory '{data_directory}' does not exist.")
        return

    logger.info(f"Generating X{preprocess_details}.csv and y{preprocess_details}.csv from raw data.")

    # Collect all Excel files
    excel_files = [
        file
        for folder in tabular_root.glob("**/*")
        if folder.is_dir()
        for file in folder.glob("**/*.xlsx")
    ]

    if not excel_files:
        logger.warning("No Excel files found. No CSVs generated.")
        return

    target_records = []
    feature_records = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(
            "[green]Processing Excel files...", total=len(excel_files)
        )

        for file in excel_files:
            try:
                excel_sheets = pd.read_excel(file, sheet_name=None)
            except Exception as e:
                logger.error(f"Failed to read {file}: {e}")
                progress.update(task, advance=1)
                continue

            for sheet_name, df in excel_sheets.items():
                # Check if 'Target' column exists
                if "Target" not in df.columns:
                    logger.warning(
                        f"'Target' column not found in sheet {sheet_name} of {file}. Skipping."
                    )
                    continue

                df.dropna(subset=["Target"], inplace=True)
                if df.empty:
                    logger.warning(
                        f"No valid rows after dropping NA 'Target' in {sheet_name} of {file}."
                    )
                    continue

                sheet_components = sheet_name.split("_")
                if len(sheet_components) < 4:
                    logger.warning(
                        f"Sheet name format unexpected: {sheet_name}. Skipping."
                    )
                    continue

                sheet_is_clogged = False
                if clogging_factors:
                    if len(sheet_components) == 4:
                        sheet_is_clogged = False
                    elif len(sheet_components) == 5:
                        sheet_is_clogged = True

                if sen_geometrical:
                    SEN_Geometry = {
                        "10": {"Angle": -15, "Depth": 40},
                        "09": {"Angle": 15, "Depth": 40},
                        "08": {"Angle": 0, "Depth": 20},
                        "07": {"Angle": -15, "Depth": 0},
                        "06": {"Angle": 15, "Depth": 0},
                    }
                    Angle = SEN_Geometry[sheet_components[0][3:5]]["Angle"]
                    Depth = SEN_Geometry[sheet_components[0][3:5]]["Depth"]

                if clogging_factors:
                    if sheet_is_clogged:
                        CF = sheet_components[1]
                    else:
                        CF = "0"

                for _, row in df.iterrows():
                    try:
                        feature_dict = {
                            #'SEN': sheet_components[0],
                            #'waterflow': sheet_components[1],
                            #'airflow': sheet_components[2],
                            "time[s]": row.iloc[0],
                            "AN_1_LL[m/s]": row.iloc[9],
                            "AN_2_LQ[m/s]": row.iloc[10],
                            "AN_3_RQ[m/s]": row.iloc[11],
                            "AN_4_RR[m/s]": row.iloc[12],
                            "ML_LL[mm]": row.iloc[17],
                            "ML_LQ[mm]": row.iloc[18],
                            "ML_RQ[mm]": row.iloc[19],
                            "ML_RR[mm]": row.iloc[20],
                            "L_wave_ht[mm]": row.iloc[21],
                            "R_wave_ht[mm]": row.iloc[22],
                        }

                        if sen_geometrical:
                            feature_dict["Angle"] = Angle
                            feature_dict["Depth"] = Depth
                        else:
                            feature_dict["SEN"] = sheet_components[0]


                        if clogging_factors and sheet_is_clogged:
                            feature_dict["CF"] = CF
                            feature_dict["waterflow"] = sheet_components[2]
                            feature_dict["airflow"] = sheet_components[3]
                            if mould_position:
                                feature_dict["mould_pos"] = sheet_components[4]
                        else:
                            feature_dict["waterflow"] = sheet_components[1]
                            feature_dict["airflow"] = sheet_components[2]
                            if mould_position:
                                feature_dict["mould_pos"] = sheet_components[3]
                            if clogging_factors:
                                feature_dict["CF"] = CF

                        feature_records.append(feature_dict)

                        target_records.append(
                            {
                                "label": sheet_name,
                                "time[s]": row.iloc[0],
                                "Count_EX1": row.iloc[24],
                                "Count_EX2": row.iloc[25],
                            }
                        )
                    except IndexError:
                        logger.error(
                            f"Row indexing failed for sheet {sheet_name} in {file}. Columns mismatch?"
                        )
                        continue

            progress.update(task, advance=1)

    if feature_records and target_records:
        X_df = pd.DataFrame(feature_records)
        y_df = pd.DataFrame(target_records)
        X_df.to_csv(f"{data_directory}/X{preprocess_details}.csv", index=False)
        y_df.to_csv(f"{data_directory}/y{preprocess_details}.csv", index=False)
        logger.info(f"Successfully generated X{preprocess_details}.csv and y{preprocess_details}.csv from raw data.")
    else:
        logger.warning(f"No data was extracted. X{preprocess_details}.csv and y{preprocess_details}.csv were not created.")

def load_data(
    target: str = "Count_EX1",
    onehot_encoding: bool = False,
    sen_geometrical: bool = False,
    clogging_factors: bool = False,
    discard_features: Optional[List[str]] = None,
    data_directory: str = "Organized_Data",
    feature_lag: int = 0,
    lagged_features: Optional[List[str]] = None,
    mould_position: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], Dict[str, List]]:
    """
    Load and preprocess the dataset for a given target variable.
    Returns training and test DataFrames, feature names, and, if onehot_encoding is True,
    a dictionary with the unique values for each categorical feature.
    """

    encoding_type = "OneHot" if onehot_encoding else "Categorical"
    sen_geometrical_type = "_Geometrical" if sen_geometrical else ""
    clogging_factors_type = "_Clogging" if clogging_factors else ""
    mould_position_type = "_Mould" if mould_position else "" 
    preprocess_details = (
        f"{encoding_type}{sen_geometrical_type}{clogging_factors_type}{mould_position_type}"
    )

    if (
        not Path(f"{data_directory}/X{preprocess_details}.csv").exists()
        or not Path(f"{data_directory}/y{preprocess_details}.csv").exists()
    ):
        logger.info(
            f"Generating X{preprocess_details}.csv and y{preprocess_details}.csv"
        )
        generate_csv_files(
            data_directory=data_directory,
            sen_geometrical=sen_geometrical,
            clogging_factors=clogging_factors,
            mould_position=mould_position,
            preprocess_details=preprocess_details
        )

    if (
        not Path(f"{data_directory}/X{preprocess_details}.csv").exists()
        or not Path(f"{data_directory}/y{preprocess_details}.csv").exists()
    ):
        logger.error(
            f"Failed to load data because X{preprocess_details}.csv or y{preprocess_details}.csv do not exist."
        )
        raise FileNotFoundError(
            f"X{preprocess_details}.csv or y{preprocess_details}.csv not found even after attempt to generate."
        )

    logger.info(f"Loading data for target: {target}")

    X_data = pd.read_csv(
        f"{data_directory}/X{preprocess_details}.csv", low_memory=False
    )
    y_all = pd.read_csv(f"{data_directory}/y{preprocess_details}.csv", low_memory=False)

    if target not in y_all.columns:
        logger.error(f"Target {target} not found in y.csv.")
        raise ValueError(f"Target column {target} does not exist in y.csv")

    y_data = y_all[target]

    logger.info("Formatting data types.")

    # Columns to be treated as categorical
    categorical_columns = ["SEN", "Angle", "Depth", "waterflow", "airflow", "CF", "mould_pos"]

    for cat_col in categorical_columns:
        if cat_col in X_data.columns:
            X_data[cat_col] = X_data[cat_col].astype("category")

    # Capture the unique categories before one-hot encoding
    onehot_values = {}
    for cat_feature in categorical_columns:
        if cat_feature in X_data.columns:
            onehot_values[cat_feature] = list(X_data[cat_feature].cat.categories)

    # Columns to be treated as floats
    float_columns = [
        "time[s]",
        "AN_1_LL[m/s]",
        "AN_2_LQ[m/s]",
        "AN_3_RQ[m/s]",
        "AN_4_RR[m/s]",
        "ML_LL[mm]",
        "ML_LQ[mm]",
        "ML_RQ[mm]",
        "ML_RR[mm]",
        "L_wave_ht[mm]",
        "R_wave_ht[mm]",
    ]
    for col in float_columns:
        if col in X_data.columns:
            X_data[col] = X_data[col].astype("float", errors="ignore")

    if onehot_encoding:
        logger.info("Applying one-hot encoding to categorical features.")
        for cat_feature in categorical_columns:
            if cat_feature in X_data.columns:
                X_data = pd.get_dummies(X_data, columns=[cat_feature])

    logger.info("Cleaning column names.")

    def clean_column_name(name: str) -> str:
        return re.sub(r"[^a-zA-Z0-9_]", "_", name)

    X_data.columns = [clean_column_name(col) for col in X_data.columns]

    cleaned_discard_features = []
    if discard_features:
        cleaned_discard_features = [clean_column_name(col) for col in discard_features]

    if cleaned_discard_features:
        logger.info(f"Discarding features: {cleaned_discard_features}")
        X_data.drop(columns=cleaned_discard_features, errors="ignore", inplace=True)

    # lag features
    cleaned_lagged_features = []
    if not feature_lag == 0:
        cleaned_lagged_features = [clean_column_name(col) for col in lagged_features]

    if cleaned_lagged_features:
        logger.info(f"Lagging features: {cleaned_lagged_features}")
        for feature in cleaned_lagged_features:
            for lag_amount in range(1, feature_lag+1):
                X_data[f"{feature}_lag{lag_amount}"] = X_data[feature].shift(lag_amount)
                X_data[feature] = X_data[feature].astype("float", errors="ignore")
                #logger.info(f"Column created: {feature}_lag{lag_amount}")

    features = X_data.columns.tolist()

    logger.info("Combining features and target.")
    combined_df = pd.concat([X_data, y_data], axis=1)
    combined_df.dropna(inplace=True)

    #logger.info(combined_df.head(5))

    if combined_df.empty:
        logger.error("No data available after merging and dropping NAs.")
        raise ValueError("No valid data available to train and test.")

    logger.info("Splitting data into training and test sets.")
    train_df, test_df = train_test_split(combined_df, test_size=0.2, random_state=42)

    # Add 'set' column for train/test indicator
    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df['set'] = 'train'
    test_df['set'] = 'test'

    # Combine back into full dataset with 'set' column, sorted by original index
    full_df = pd.concat([train_df, test_df]).sort_index()

    # Export combined data as single CSV file
    combined_file = f"{data_directory}/combined_{preprocess_details}_{target}.csv"
    full_df.to_csv(combined_file, index=False)
    logger.info(f"Exported combined data with train/test indicators to {combined_file}")

    logger.info(
        f"Data loaded successfully with {train_df.shape[0]} training samples and {test_df.shape[0]} test samples."
    )

    return train_df, test_df, features, onehot_values


def main():
    """
    Main entry point:
    - Parse command-line arguments only (no external config)
    - Load data
    - Run Optuna study
    - Evaluate model
    - Save training configuration as JSON
    """

    parser = argparse.ArgumentParser(
        description="Optimize XGBoost model for water data"
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        type=str,
        default=["Count_EX1"],
        help="Target variables",
        choices=["Count_EX1", "Count_EX2"],
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default="water_modelling",
        help="Optuna study name prefix",
    )
    parser.add_argument(
        "--study_count", type=int, default=1, help="Number of studies to run"
    )
    parser.add_argument(
        "--onehot-encoding",
        action="store_true",
        help="Use one-hot encoding for categorical features.",
    )
    parser.add_argument(
        "--sen-geometrical",
        action="store_true",
        help="Convert SEN numbers to geometrical features.",
    )
    parser.add_argument(
        "--clogging-factors",
        action="store_true",
        help="If clogging factors are included in the dataset.",
    )
    parser.add_argument(
        "--discard-features",
        type=str,
        default="",
        help='Comma-separated list of features to discard (e.g., "SEN,L_wave_ht[mm]")',
    )
    parser.add_argument(
        "--tree-method",
        type=str,
        choices=["auto", "exact", "approx", "hist", "gpu_hist"],
        default="gpu_hist",
        help="XGBoost tree method",
    )
    parser.add_argument(
        "--data-directory",
        type=str,
        default="Organized_Data",
        help="Directory where raw data is stored",
    )
    parser.add_argument(
        "--storage-path",
        type=str,
        default="sqlite:///water_modelling.db",
        help="Storage path for Optuna study results",
    )
    parser.add_argument(
        "--subsample-shap",
        action="store_true",
        help="Subsample SHAP values for faster computation",
        default=False,
    )
    parser.add_argument(
        "--config-file",
        type=str,
        default=None,
        help="Filename for saving the training configuration as JSON",
    )
    parser.add_argument(
        "--multithread",
        action="store_true",
        help="Use JournalStorage instead of SQLite, enabling multithreaded optimizing without MySQL and PostgreSQL.",
    )
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Only run evaluate_model and build the training config JSON. To be used after multithreaded training.",
    )
    parser.add_argument(
        "--lagged-features",
        type=str,
        default="",
        help="Comma separated list of input features to lag.",
    )
    parser.add_argument(
        "--feature-lag-amount",
        type=int,
        default=0,
        help="Positive int amount of lag for input features.",
    )
    parser.add_argument(
        "--mould-position",
        action="store_true",
        help="Use mould position (0-1,1-2,2-3,3-4) as a feature.",
    )

    args = parser.parse_args()

    # Generate config file name if not provided
    if args.config_file is None:
        args.config_file = f"training_config_{args.study_name}.json"

    discard_features = (
        [feat.strip() for feat in args.discard_features.split(",")]
        if args.discard_features
        else []
    )

    lagged_features = (
        [feat.strip() for feat in args.lagged_features.split(",")]
        if args.lagged_features
        else []
    )

    # List to store configuration details for each target
    training_configs = []

    for target in args.targets:
        train_df, test_df, features, onehot_values = load_data(
                        target=target,
                        onehot_encoding=args.onehot_encoding,
                        sen_geometrical=args.sen_geometrical,
                        clogging_factors=args.clogging_factors,
                        discard_features=discard_features,
                        data_directory=args.data_directory,
                        feature_lag=args.feature_lag_amount,
                        lagged_features=lagged_features,
                        mould_position=args.mould_position,
                    )