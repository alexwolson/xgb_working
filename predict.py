import argparse
import json
import logging
import re
from pathlib import Path

import pandas as pd
import xgboost as xgb
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    TaskID,
    TimeRemainingColumn,
)

# Set up rich logging for clear output
console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console)],
)
logger = logging.getLogger(__name__)


def clean_column_name(name: str) -> str:
    """Clean a column name (replace non-alphanumeric characters with underscores)."""
    return re.sub(r"[^a-zA-Z0-9_]", "_", name)


def predict_on_sheet(
    df: pd.DataFrame,
    file_name: str,
    sheet_name: str,
    model: xgb.XGBRegressor,
    model_name: str,
    onehot_encoding: bool,
    sen_geometrical: bool,
    clogging_factors: bool,
    drop_cols: list,
    feature_lag: int,
    lagged_features: list,
    onehot_values: dict = None,
) -> pd.DataFrame:
    """
    Process one sheet: for each row, extract features (using a mapping computed once per sheet)
    and add a new column (named after the model) with the prediction.

    If onehot_encoding is True, the provided onehot_values mapping is used to create dummy columns
    for categorical features ("SEN" and "waterflow"), ensuring that all possible dummy columns
    (even those that would be all zeros) are present.
    """
    # Use the sheet name to extract constant features
    sheet_components = sheet_name.split("_")
    if len(sheet_components) < 4:
        # Fallback: extract components from the file name
        file_name_components = file_name.split("_")
        if len(file_name_components) < 4:
            logger.error(
                f"Cannot extract components from file name '{file_name}' or sheet name '{sheet_name}'."
            )
            return df
        sheet_components = [
            file_name_components[1],  # SEN
            file_name_components[3].replace("mpm", "").replace(".", "_"),  # waterflow
            file_name_components[4].replace("LPM", ""),  # airflow
        ]

    # Define the expected column names (for values that vary by row)
    expected_cols = [
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

    # lag features
    cleaned_lagged_features = []
    if not feature_lag == 0:
        cleaned_lagged_features = [clean_column_name(col) for col in lagged_features]

    if cleaned_lagged_features:
        logger.info(f"Lagging features: {cleaned_lagged_features}")
        for feature in cleaned_lagged_features:
            for lag_amount in range(1, feature_lag+1):
                df[f"{feature}_lag{lag_amount}"] = df[feature].shift(lag_amount)
                expected_cols.append(f"{feature}_lag{lag_amount}")
                #logger.info(f"Column created: {feature}_lag{lag_amount}")

    # Compute the mapping from expected column names to actual DataFrame column names once per sheet
    actual_cols = {}
    for col in expected_cols:
        matches = [c for c in df.columns if str(c).strip().lower() == col.lower()]
        if not matches:
            logger.warning(
                f"Expected column '{col}' not found in sheet '{sheet_name}'."
            )
            logger.warning(f"Available columns: {df.columns.tolist()}")
            return df
        actual_cols[col] = matches[0]

    predictions = []
    # Process each row using the precomputed column mapping
    for idx, row in df.iterrows():
        try:
            if sen_geometrical:
                SEN_Geometry = {
                    "10": {"Angle": "-15", "Depth": "40"},
                    "09": {"Angle": "15", "Depth": "40"},
                    "08": {"Angle": "0", "Depth": "20"},
                    "07": {"Angle": "-15", "Depth": "0"},
                    "06": {"Angle": "15", "Depth": "0"},
                }
                Angle = SEN_Geometry[sheet_components[0][3:5]]["Angle"]
                Depth = SEN_Geometry[sheet_components[0][3:5]]["Depth"]

            if clogging_factors:
                if len(sheet_components) == 4:
                    CF = "0"
                elif len(sheet_components) == 5:
                    CF = sheet_components[1]

            # Build features dictionary using fixed features and values from the mapped columns
            features_dict = {}

            if sen_geometrical:
                features_dict["Angle"] = Angle
                features_dict["Depth"] = Depth
            else:
                features_dict["SEN"] = sheet_components[0]

            if clogging_factors:
                features_dict["CF"] = CF
                features_dict["waterflow"] = sheet_components[2]
                features_dict["airflow"] = sheet_components[3]
            else:
                features_dict["waterflow"] = sheet_components[1]
                features_dict["airflow"] = sheet_components[2]

            for col in expected_cols:
                features_dict[col] = row[actual_cols[col]]
        except Exception as e:
            logger.error(f"Error processing row {idx} in sheet '{sheet_name}': {e}")
            predictions.append(None)
            continue

        # Create a DataFrame for this single row and clean column names
        features_df = pd.DataFrame([features_dict])
        features_df.columns = [clean_column_name(col) for col in features_df.columns]

        # Convert numeric columns to proper data type
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
        float_columns = [clean_column_name(col) for col in float_columns]
        for col in float_columns:
            if col in features_df.columns:
                features_df[col] = pd.to_numeric(features_df[col], errors="coerce")

        # Apply one-hot encoding if enabled using training metadata
        if onehot_encoding and onehot_values is not None:
            for cat_feature in ["SEN", "Angle", "Depth", "waterflow", "airflow", "CF"]:
                if cat_feature in features_df.columns:
                    cat_val = features_df.at[0, cat_feature]
                    for possible_val in onehot_values.get(cat_feature, []):
                        dummy_col = clean_column_name(f"{cat_feature}_{possible_val}")
                        features_df[dummy_col] = (
                            1 if cat_val == str(possible_val) else 0
                        )
                    features_df.drop(columns=[cat_feature], inplace=True)

        # Drop specified columns, if any
        for col_to_drop in drop_cols:
            for column in list(features_df.columns):
                if clean_column_name(col_to_drop) == column:
                    features_df.drop(columns=[column], inplace=True)

        # Make prediction (assume single-row input)
        try:
            #print(features_df.to_string())
            pred = model.predict(features_df)[0]
        except Exception as e:
            logger.error(f"Prediction error in row {idx} of sheet '{sheet_name}': {e}")
            pred = None
        predictions.append(pred)

    # Add the predictions as a new column to the DataFrame
    df[model_name] = predictions
    return df


def process_excel_file(
    file_path: Path,
    model: xgb.XGBRegressor,
    model_name: str,
    onehot_encoding: bool,
    sen_geometrical: bool,
    clogging_factors: bool,
    drop_cols: list,
    feature_lag: int,
    lagged_features: list,
    onehot_values: dict = None,
    progress: "Progress" = None,
    progress_task: TaskID = None,
) -> None:
    """
    Open an Excel file, process each sheet to add predictions, update the global progress,
    and then overwrite the file with the new columns.
    """
    logger.info(f"Processing file: {file_path}")
    try:
        # Read all sheets; this returns a dict of DataFrames keyed by sheet name
        sheets = pd.read_excel(file_path, sheet_name=None)
    except Exception as e:
        logger.error(f"Error reading {file_path}: {e}")
        return

    file_name = file_path.stem
    updated_sheets = {}

    for sheet_name, df in sheets.items():
        updated_df = predict_on_sheet(
            df,
            file_name,
            sheet_name,
            model,
            model_name,
            onehot_encoding,
            sen_geometrical,
            clogging_factors,
            drop_cols,
            feature_lag,
            lagged_features,
            onehot_values,
        )
        updated_sheets[sheet_name] = updated_df
        # Update the global progress bar after processing each sheet
        if progress is not None and progress_task is not None:
            progress.update(progress_task, advance=1)

    # Overwrite the existing Excel file with updated sheets
    try:
        with pd.ExcelWriter(file_path, engine="openpyxl", mode="w") as writer:
            for sheet_name, updated_df in updated_sheets.items():
                updated_df.to_excel(writer, sheet_name=sheet_name, index=False)
        logger.info(f"Updated file saved: {file_path}")
    except Exception as e:
        logger.error(f"Error saving updated file {file_path}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Apply a trained model (via JSON config) to new raw data for row-by-row predictions"
    )
    parser.add_argument(
        "--config-file",
        type=str,
        required=True,
        help="Path to the training configuration JSON file output by the training script",
    )
    parser.add_argument(
        "--data-directory",
        type=str,
        default="New_Data",
        help="Directory containing new Excel files (will be scanned recursively)",
    )
    args = parser.parse_args()

    # Load the training configuration from JSON
    try:
        with open(args.config_file, "r") as f:
            training_configs = json.load(f)
    except Exception as e:
        logger.error(f"Error loading config file {args.config_file}: {e}")
        return

    # Ensure we have a list of configurations (one per target/model)
    if not isinstance(training_configs, list):
        training_configs = [training_configs]

    data_dir = Path(args.data_directory)
    if not data_dir.exists():
        logger.error(f"Data directory {args.data_directory} does not exist.")
        return

    excel_files = list(data_dir.glob("**/*.xlsx"))
    if not excel_files:
        logger.warning(f"No Excel files found in {args.data_directory}.")
        return

    # Process each configuration entry from the JSON file
    for config in training_configs:
        model_file = config["model_save_location"]
        model_name = config["study_name"]
        onehot_encoding = config.get("onehot_encoding", False)
        sen_geometrical = config.get("sen_geometrical", False)
        clogging_factors = config.get("clogging_factors", False)
        drop_cols = config.get("discard_features", [])
        onehot_values = config.get("onehot_values", {}) if onehot_encoding else {}
        feature_lag = config.get("feature_lag")
        lagged_features = config.get("lagged_features")

        logger.info(f"Loading model from {model_file} for configuration: {model_name}")
        model = xgb.XGBRegressor(enable_categorical=not onehot_encoding)
        try:
            model.load_model(model_file)
        except Exception as e:
            logger.error(f"Error loading model from {model_file}: {e}")
            continue

        # Calculate total number of sheets across all Excel files
        total_sheets = 0
        for file_path in excel_files:
            try:
                xls = pd.ExcelFile(file_path)
                total_sheets += len(xls.sheet_names)
            except Exception as e:
                logger.error(f"Error reading file {file_path} for progress count: {e}")

        # Create a global progress bar for processing sheets in all files
        with Progress(
            SpinnerColumn(),
            BarColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            progress_task = progress.add_task(
                f"Processing sheets for model {model_name}", total=total_sheets
            )
            # Process each Excel file with the loaded model and configuration
            for file_path in excel_files:
                process_excel_file(
                    file_path,
                    model,
                    model_name,
                    onehot_encoding,
                    sen_geometrical,
                    clogging_factors,
                    drop_cols,
                    feature_lag,
                    lagged_features,
                    onehot_values,
                    progress,
                    progress_task,
                )
        logger.info(f"Predictions added using model {model_name} from {model_file}")

    logger.info("All files processed successfully.")


if __name__ == "__main__":
    main()
