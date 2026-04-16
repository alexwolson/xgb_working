import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

SEN_GEOMETRY: Dict[str, Dict[str, int]] = {
    "10": {"Angle": -15, "Depth": 40},
    "09": {"Angle": 15, "Depth": 40},
    "08": {"Angle": 0, "Depth": 20},
    "07": {"Angle": -15, "Depth": 0},
    "06": {"Angle": 15, "Depth": 0},
}

CATEGORICAL_COLUMNS = ["SEN", "Angle", "Depth", "waterflow", "airflow", "CF", "mould_pos"]

FLOAT_COLUMNS = [
    "time[s]", "AN_1_LL[m/s]", "AN_2_LQ[m/s]", "AN_3_RQ[m/s]", "AN_4_RR[m/s]",
    "ML_LL[mm]", "ML_LQ[mm]", "ML_RQ[mm]", "ML_RR[mm]", "L_wave_ht[mm]", "R_wave_ht[mm]",
]


def clean_column_name(name: str) -> str:
    """Replace non-alphanumeric characters with underscores."""
    return re.sub(r"[^a-zA-Z0-9_]", "_", name)


def _sheet_split(
    df: pd.DataFrame,
    test_frac: float = 0.2,
    val_frac: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split *df* into (train, val, test) partitions at the sheet level.

    Sheets are sorted alphabetically for reproducibility. The last ``test_frac``
    of sheets become test; the preceding ``val_frac`` of the remainder become
    validation; the rest are training.  Raises ``ValueError`` if fewer than 3
    distinct sheets are present.
    """
    unique_sheets = sorted(df["label"].unique())
    n = len(unique_sheets)
    if n < 3:
        raise ValueError(
            f"Need at least 3 distinct sheets for a train/val/test split, found {n}."
        )
    n_test = max(1, round(n * test_frac))
    n_val = max(1, round((n - n_test) * val_frac))
    if n - n_test - n_val < 1:
        raise ValueError(
            f"With {n} sheets, n_test={n_test} and n_val={n_val} leave no training data."
        )
    test_sheets = set(unique_sheets[-n_test:])
    val_sheets = set(unique_sheets[-(n_test + n_val):-n_test])
    train_sheets = set(unique_sheets[:-(n_test + n_val)])
    return (
        df[df["label"].isin(train_sheets)].copy(),
        df[df["label"].isin(val_sheets)].copy(),
        df[df["label"].isin(test_sheets)].copy(),
    )


def generate_csv_files(
    data_directory: str,
    preprocess_details: str,
    sen_geometrical: bool = False,
    clogging_factors: bool = False,
    mould_position: bool = False,
    remove_rows: int = 0,
) -> None:
    """Generate X<suffix>.csv and y<suffix>.csv from Excel files in data_directory."""
    tabular_root = Path(data_directory)
    if not tabular_root.exists():
        logger.error(f"Data directory '{data_directory}' does not exist.")
        return

    logger.info(f"Generating X{preprocess_details}.csv and y{preprocess_details}.csv from raw data.")

    excel_files = list(tabular_root.glob("**/*.xlsx"))

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
    ) as progress:
        task = progress.add_task("[green]Processing Excel files...", total=len(excel_files))

        for file in excel_files:
            try:
                excel_sheets = pd.read_excel(file, sheet_name=None)
            except Exception as e:
                logger.error(f"Failed to read {file}: {e}")
                progress.update(task, advance=1)
                continue

            for sheet_name, df in excel_sheets.items():
                if "Target" not in df.columns:
                    logger.warning(f"'Target' column not found in sheet {sheet_name} of {file}. Skipping.")
                    continue

                df = df.dropna(subset=["Target"])
                if df.empty:
                    logger.warning(f"No valid rows after dropping NA 'Target' in {sheet_name} of {file}.")
                    continue

                if remove_rows != 0:
                    df = df[:-remove_rows]
                if df.empty:
                    logger.warning(f"No valid rows after dropping {remove_rows} rows in {sheet_name} of {file}.")
                    continue

                sheet_components = sheet_name.split("_")
                if len(sheet_components) < 4:
                    logger.warning(f"Sheet name format unexpected: {sheet_name}. Skipping.")
                    continue

                sheet_is_clogged = False
                if clogging_factors:
                    sheet_is_clogged = len(sheet_components) == 5

                if sen_geometrical:
                    sen_key = sheet_components[0][3:5]
                    if sen_key not in SEN_GEOMETRY:
                        logger.warning(f"Unknown SEN key '{sen_key}' in sheet {sheet_name}. Skipping.")
                        continue
                    angle = SEN_GEOMETRY[sen_key]["Angle"]
                    depth = SEN_GEOMETRY[sen_key]["Depth"]

                cf = "0"
                if clogging_factors and sheet_is_clogged:
                    cf = sheet_components[1]

                for _, row in df.iterrows():
                    try:
                        feature_dict = {
                            "label": sheet_name,
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
                            feature_dict["Angle"] = angle
                            feature_dict["Depth"] = depth
                        else:
                            feature_dict["SEN"] = sheet_components[0]

                        if clogging_factors and sheet_is_clogged:
                            feature_dict["CF"] = cf
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
                                feature_dict["CF"] = cf

                        feature_records.append(feature_dict)
                        target_records.append({
                            "label": sheet_name,
                            "time[s]": row.iloc[0],
                            "Count_EX1": row.iloc[24],
                            "Count_EX2": row.iloc[25],
                        })
                    except IndexError:
                        logger.error(f"Row indexing failed for sheet {sheet_name} in {file}.")
                        continue

            progress.update(task, advance=1)

    if feature_records and target_records:
        X_df = pd.DataFrame(feature_records)
        y_df = pd.DataFrame(target_records)
        X_df.to_csv(f"{data_directory}/X{preprocess_details}.csv", index=False)
        y_df.to_csv(f"{data_directory}/y{preprocess_details}.csv", index=False)
        logger.info(f"Successfully generated X{preprocess_details}.csv and y{preprocess_details}.csv.")
    else:
        logger.warning("No data extracted. CSVs not created.")


def load_data(
    target: str = "Count_EX1",
    onehot_encoding: bool = False,
    sen_geometrical: bool = False,
    clogging_factors: bool = False,
    discard_features: Optional[List[str]] = None,
    data_directory: str = "data/raw/Organized_Data",
    feature_lag: int = 0,
    lagged_features: Optional[List[str]] = None,
    mould_position: bool = False,
    remove_rows: int = 0,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str], Dict[str, List]]:
    """
    Load and preprocess dataset for a given target variable.

    Returns (train_df, val_df, test_df, feature_names, onehot_values).
    All three DataFrames contain feature columns, the target column, a 'set' column,
    and a 'label' column (sheet name, excluded from feature_names).
    val_df is for HPO evaluation; test_df is the held-out evaluation set.
    Callers index features with df[features] and target with df[target].
    """
    encoding_type = "OneHot" if onehot_encoding else "Categorical"
    preprocess_details = (
        f"v2_{encoding_type}"
        f"{'_Geometrical' if sen_geometrical else ''}"
        f"{'_Clogging' if clogging_factors else ''}"
        f"{'_Mould' if mould_position else ''}"
        f"{f'_removed{remove_rows}' if remove_rows != 0 else ''}"
    )

    x_path = Path(f"{data_directory}/X{preprocess_details}.csv")
    y_path = Path(f"{data_directory}/y{preprocess_details}.csv")

    if not x_path.exists() or not y_path.exists():
        logger.info(f"Generating {x_path.name} and {y_path.name}")
        generate_csv_files(
            data_directory=data_directory,
            sen_geometrical=sen_geometrical,
            clogging_factors=clogging_factors,
            mould_position=mould_position,
            preprocess_details=preprocess_details,
            remove_rows=remove_rows,
        )

    if not x_path.exists() or not y_path.exists():
        raise FileNotFoundError(f"{x_path.name} or {y_path.name} not found after generation attempt.")

    logger.info(f"Loading data for target: {target}")
    X_data = pd.read_csv(x_path, low_memory=False)
    y_all = pd.read_csv(y_path, low_memory=False)

    if target not in y_all.columns:
        raise ValueError(f"Target column '{target}' does not exist in {y_path.name}")

    y_data = y_all[target]

    for cat_col in CATEGORICAL_COLUMNS:
        if cat_col in X_data.columns:
            X_data[cat_col] = X_data[cat_col].astype("category")

    onehot_values: Dict[str, List] = {}
    for cat_feature in CATEGORICAL_COLUMNS:
        if cat_feature in X_data.columns:
            onehot_values[cat_feature] = list(X_data[cat_feature].cat.categories)

    for col in FLOAT_COLUMNS:
        if col in X_data.columns:
            X_data[col] = X_data[col].astype("float", errors="ignore")

    if onehot_encoding:
        logger.info("Applying one-hot encoding.")
        for cat_feature in CATEGORICAL_COLUMNS:
            if cat_feature in X_data.columns:
                X_data = pd.get_dummies(X_data, columns=[cat_feature])

    X_data.columns = [clean_column_name(col) for col in X_data.columns]

    if discard_features:
        cleaned = [clean_column_name(f) for f in discard_features]
        cleaned = [c for c in cleaned if c != "label"]  # protect the sheet identifier
        logger.info(f"Discarding features: {cleaned}")
        X_data.drop(columns=cleaned, errors="ignore", inplace=True)

    if feature_lag != 0 and lagged_features:
        cleaned_lag = [clean_column_name(f) for f in lagged_features]
        logger.info(f"Lagging features per sheet: {cleaned_lag}")

        def _apply_sheet_lags(group: pd.DataFrame) -> pd.DataFrame:
            group = group.copy()
            for feature in cleaned_lag:
                for lag_amount in range(1, feature_lag + 1):
                    group[f"{feature}_lag{lag_amount}"] = group[feature].shift(lag_amount)
                group[feature] = group[feature].astype("float", errors="ignore")
            return group

        label_series = X_data["label"]
        X_data = X_data.groupby("label", group_keys=False).apply(
            _apply_sheet_lags, include_groups=False
        )
        X_data["label"] = label_series

    features = [c for c in X_data.columns.tolist() if c != "label"]

    combined_df = pd.concat([X_data, y_data], axis=1)
    combined_df.dropna(inplace=True)

    if combined_df.empty:
        raise ValueError("No valid data available after merging and dropping NAs.")

    train_df, val_df, test_df = _sheet_split(combined_df)
    train_df["set"] = "train"
    val_df["set"] = "val"
    test_df["set"] = "test"

    full_df = pd.concat([train_df, val_df, test_df]).sort_index()
    combined_file = f"{data_directory}/combined_{preprocess_details}_{target}.csv"
    full_df.to_csv(combined_file, index=False)
    logger.info(f"Exported combined data to {combined_file}")
    logger.info(
        f"Data loaded: {len(train_df)} train ({train_df['label'].nunique()} sheets), "
        f"{len(val_df)} val ({val_df['label'].nunique()} sheets), "
        f"{len(test_df)} test ({test_df['label'].nunique()} sheets)."
    )

    return train_df, val_df, test_df, features, onehot_values
