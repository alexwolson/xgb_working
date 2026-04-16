import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import xgboost as xgb
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskID, TextColumn, TimeElapsedColumn, TimeRemainingColumn

from steel_flow.data import clean_column_name

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

SEN_GEOMETRY_STR: Dict[str, Dict[str, str]] = {
    "10": {"Angle": "-15", "Depth": "40"},
    "09": {"Angle": "15", "Depth": "40"},
    "08": {"Angle": "0", "Depth": "20"},
    "07": {"Angle": "-15", "Depth": "0"},
    "06": {"Angle": "15", "Depth": "0"},
}

FLOAT_COLUMNS = [
    "time[s]", "AN_1_LL[m/s]", "AN_2_LQ[m/s]", "AN_3_RQ[m/s]", "AN_4_RR[m/s]",
    "ML_LL[mm]", "ML_LQ[mm]", "ML_RQ[mm]", "ML_RR[mm]", "L_wave_ht[mm]", "R_wave_ht[mm]",
]

CATEGORICAL_FEATURES = ["SEN", "Angle", "Depth", "waterflow", "airflow", "CF", "mould_pos"]


def predict_on_sheet(
    df: pd.DataFrame,
    file_name: str,
    sheet_name: str,
    model: xgb.XGBRegressor,
    model_name: str,
    onehot_encoding: bool,
    sen_geometrical: bool,
    clogging_factors: bool,
    drop_cols: List[str],
    feature_lag: int,
    lagged_features: List[str],
    all_sheets: list,
    onehot_values: Optional[Dict] = None,
    mould_position: bool = False,
    piv_data: bool = False,
) -> pd.DataFrame:
    """
    Add a prediction column to a single sheet DataFrame.
    Returns updated DataFrame with model_name column appended.
    """
    if not piv_data:
        sheet_components = sheet_name.split("_")
        sheet_is_clogged = False
        if clogging_factors:
            sheet_is_clogged = len(sheet_components) == 5
    else:
        sheet_components = list(all_sheets)[1].split("_")
        sheet_is_clogged = False
        if clogging_factors and "Clog" in sheet_components[1]:
            sheet_is_clogged = True
            sheet_components[2] = sheet_components[2][:3]
            sheet_components[3] = sheet_components[3][:1]
        sheet_components[2] = sheet_components[2][:3]
        sheet_components[3] = sheet_components[3][:1]

    if len(sheet_components) < 4:
        file_name_components = file_name.split("_")
        if len(file_name_components) < 4:
            logger.error(f"Cannot extract components from '{file_name}' or '{sheet_name}'.")
            return df
        sheet_components = [
            file_name_components[1],
            file_name_components[3].replace("mpm", "").replace(".", "_"),
            file_name_components[4].replace("LPM", ""),
        ]

    expected_cols = list(FLOAT_COLUMNS)
    nice_df = df.copy()

    if feature_lag != 0:
        for feature in lagged_features:
            for lag_amount in range(1, feature_lag + 1):
                df[f"{feature}_lag{lag_amount}"] = df[feature].shift(lag_amount)
                expected_cols.append(f"{feature}_lag{lag_amount}")

    actual_cols = {}
    for col in expected_cols:
        matches = [c for c in df.columns if str(c).strip().lower() == col.lower()]
        if not matches:
            logger.warning(f"Expected column '{col}' not found in sheet '{sheet_name}'.")
            logger.warning(f"Available columns: {df.columns.tolist()}")
            return df
        actual_cols[col] = matches[0]

    predictions = []
    piv_preds = {"0-1": [], "1-2": [], "2-3": [], "3-4": []}

    for idx, row in df.iterrows():
        try:
            if sen_geometrical:
                sen_key = sheet_components[0][3:5]
                if sen_key not in SEN_GEOMETRY_STR:
                    logger.warning(f"Unknown SEN key '{sen_key}' in sheet '{sheet_name}'. Skipping row {idx}.")
                    predictions.append(None)
                    continue
                angle = SEN_GEOMETRY_STR[sen_key]["Angle"]
                depth = SEN_GEOMETRY_STR[sen_key]["Depth"]

            cf = "0"
            if clogging_factors and sheet_is_clogged:
                cf = sheet_components[1]

            features_dict: Dict = {}
            if sen_geometrical:
                features_dict["Angle"] = angle
                features_dict["Depth"] = depth
            else:
                features_dict["SEN"] = sheet_components[0]

            if clogging_factors and sheet_is_clogged:
                features_dict["CF"] = cf
                features_dict["waterflow"] = sheet_components[2]
                features_dict["airflow"] = sheet_components[3]
                if mould_position and not piv_data:
                    features_dict["mould_pos"] = sheet_components[4]
            else:
                features_dict["waterflow"] = sheet_components[1]
                features_dict["airflow"] = sheet_components[2]
                if mould_position and not piv_data:
                    features_dict["mould_pos"] = sheet_components[3]
                if clogging_factors:
                    features_dict["CF"] = cf

            for col in expected_cols:
                features_dict[col] = row[actual_cols[col]]

        except Exception as e:
            logger.error(f"Error processing row {idx} in sheet '{sheet_name}': {e}")
            predictions.append(None)
            continue

        mld_iterations = ["0-1", "1-2", "2-3", "3-4"] if piv_data else [features_dict.get("mould_pos", "0-1")]
        sum_preds = []

        for mld_iter in mld_iterations:
            features_dict["mould_pos"] = mld_iter
            features_df = pd.DataFrame([features_dict])
            features_df.columns = [clean_column_name(col) for col in features_df.columns]

            clean_float_cols = [clean_column_name(col) for col in FLOAT_COLUMNS]
            for col in clean_float_cols:
                if col in features_df.columns:
                    features_df[col] = pd.to_numeric(features_df[col], errors="coerce")

            if onehot_encoding and onehot_values:
                for cat_feature in CATEGORICAL_FEATURES:
                    if cat_feature in features_df.columns:
                        cat_val = features_df.at[0, cat_feature]
                        for possible_val in onehot_values.get(cat_feature, []):
                            dummy_col = clean_column_name(f"{cat_feature}_{possible_val}")
                            features_df[dummy_col] = 1 if cat_val == str(possible_val) else 0
                        features_df.drop(columns=[cat_feature], inplace=True)

            for col_to_drop in drop_cols:
                for column in list(features_df.columns):
                    if clean_column_name(col_to_drop) == column:
                        features_df.drop(columns=[column], inplace=True)

            correct_cols = model.get_booster().feature_names
            features_df = features_df[correct_cols]

            try:
                pred = model.predict(features_df)[0]
            except Exception as e:
                logger.error(f"Prediction error in row {idx} of sheet '{sheet_name}': {e}")
                pred = None

            if piv_data:
                sum_preds.append(pred)
                piv_preds[mld_iter].append(pred)
            else:
                predictions.append(pred)

        if piv_data:
            predictions.append(sum(sum_preds))

    nice_df[model_name] = predictions
    if piv_data:
        for mld_iter in ["0-1", "1-2", "2-3", "3-4"]:
            nice_df[f"{model_name} mld {mld_iter}"] = piv_preds[mld_iter]
    return nice_df


def process_excel_file(
    file_path: Path,
    model: xgb.XGBRegressor,
    model_name: str,
    onehot_encoding: bool,
    sen_geometrical: bool,
    clogging_factors: bool,
    drop_cols: List[str],
    feature_lag: int,
    lagged_features: List[str],
    onehot_values: Optional[Dict] = None,
    mould_position: bool = False,
    progress: Optional[Progress] = None,
    progress_task: Optional[TaskID] = None,
    piv_data: bool = False,
) -> None:
    """Process all sheets in an Excel file, writing predictions back to the file."""
    logger.info(f"Processing file: {file_path}")
    try:
        sheets = pd.read_excel(file_path, sheet_name=None)
    except Exception as e:
        logger.error(f"Error reading {file_path}: {e}")
        return

    file_name = file_path.stem
    updated_sheets = {}
    all_sheets = sheets.keys()

    for sheet_name, df in sheets.items():
        updated_df = predict_on_sheet(
            df, file_name, sheet_name, model, model_name,
            onehot_encoding, sen_geometrical, clogging_factors,
            drop_cols, feature_lag, lagged_features, all_sheets,
            onehot_values, mould_position, piv_data,
        )
        updated_sheets[sheet_name] = updated_df
        if progress is not None and progress_task is not None:
            progress.update(progress_task, advance=1)

    try:
        with pd.ExcelWriter(file_path, engine="openpyxl", mode="w") as writer:
            for sheet_name, updated_df in updated_sheets.items():
                updated_df.to_excel(writer, sheet_name=sheet_name, index=False)
        logger.info(f"Updated file saved: {file_path}")
    except Exception as e:
        logger.error(f"Error saving updated file {file_path}: {e}")
