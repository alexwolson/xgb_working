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
from sklearn.metrics import mean_absolute_error
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
    data_directory: str, sen_geometrical: bool = False, clogging_factors: bool = False
) -> None:
    """
    Generate X.csv and y.csv from Excel files in the given data_directory.
    Checks for directory existence and empty file sets.
    """
    tabular_root = Path(data_directory)
    if not tabular_root.exists():
        logger.error(f"Data directory '{data_directory}' does not exist.")
        return

    logger.info("Generating X.csv and y.csv from raw data.")

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
                    if len(sheet_components) == 4:
                        CF = "0"
                    elif len(sheet_components) == 5:
                        CF = sheet_components[1]

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

                        if clogging_factors:
                            feature_dict["CF"] = CF
                            feature_dict["waterflow"] = sheet_components[2]
                            feature_dict["airflow"] = sheet_components[3]
                        else:
                            feature_dict["waterflow"] = sheet_components[1]
                            feature_dict["airflow"] = sheet_components[2]

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
        X_df.to_csv(f"{data_directory}/X.csv", index=False)
        y_df.to_csv(f"{data_directory}/y.csv", index=False)
        logger.info("Successfully generated X.csv and y.csv from raw data.")
    else:
        logger.warning("No data was extracted. X.csv and y.csv were not created.")


def load_data(
    target: str = "Count_EX1",
    onehot_encoding: bool = False,
    sen_geometrical: bool = False,
    clogging_factors: bool = False,
    discard_features: Optional[List[str]] = None,
    data_directory: str = "Organized_Data",
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], Dict[str, List]]:
    """
    Load and preprocess the dataset for a given target variable.
    Returns training and test DataFrames, feature names, and, if onehot_encoding is True,
    a dictionary with the unique values for each categorical feature.
    """

    if (
        not Path(f"{data_directory}/X.csv").exists()
        or not Path(f"{data_directory}/y.csv").exists()
    ):
        logger.info("Generating X.csv and Y.csv")
        generate_csv_files(
            data_directory=data_directory,
            sen_geometrical=sen_geometrical,
            clogging_factors=clogging_factors,
        )

    if (
        not Path(f"{data_directory}/X.csv").exists()
        or not Path(f"{data_directory}/y.csv").exists()
    ):
        logger.error("Failed to load data because X.csv or y.csv do not exist.")
        raise FileNotFoundError(
            "X.csv or y.csv not found even after attempt to generate."
        )

    logger.info(f"Loading data for target: {target}")

    X_data = pd.read_csv(f"{data_directory}/X.csv", low_memory=False)
    y_all = pd.read_csv(f"{data_directory}/y.csv", low_memory=False)

    if target not in y_all.columns:
        logger.error(f"Target {target} not found in y.csv.")
        raise ValueError(f"Target column {target} does not exist in y.csv")

    y_data = y_all[target]

    logger.info("Formatting data types.")

    # Columns to be treated as categorical
    categorical_columns = ["SEN", "Angle", "Depth", "waterflow", "airflow", "CF"]

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

    features = X_data.columns.tolist()

    logger.info("Combining features and target.")
    combined_df = pd.concat([X_data, y_data], axis=1)
    combined_df.dropna(inplace=True)

    if combined_df.empty:
        logger.error("No data available after merging and dropping NAs.")
        raise ValueError("No valid data available to train and test.")

    logger.info("Splitting data into training and test sets.")
    train_df, test_df = train_test_split(combined_df, test_size=0.2, random_state=42)

    logger.info(
        f"Data loaded successfully with {train_df.shape[0]} training samples and {test_df.shape[0]} test samples."
    )

    return train_df, test_df, features, onehot_values


def run_optuna_study(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: List[str],
    target: str,
    study_name: str,
    study_count: int = 1,
    onehot_encoding: bool = False,
    tree_method: str = "gpu_hist",
    storage_path: str = "sqlite:///water_modelling.db",
    multithread: bool = False,
) -> optuna.Study:
    """
    Run an Optuna study to optimize XGBoost hyperparameters for the given dataset.
    """

    logger.info(f"Starting Optuna study: {study_name}")

    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective": "reg:squarederror",
            "eval_metric": "mae",
            "booster": "gbtree",
            "verbosity": 0,
            "tree_method": tree_method,
            "grow_policy": trial.suggest_categorical(
                "grow_policy", ["depthwise", "lossguide"]
            ),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 1.0, log=True),
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 1e3, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 1e3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000, log=True),
        }

        model = xgb.XGBRegressor(**params, enable_categorical=not onehot_encoding)
        model.fit(train_df[features], train_df[target])
        preds = model.predict(test_df[features])
        mae = mean_absolute_error(test_df[target], preds)
        return mae

    if multithread:
        storage = optuna.storages.JournalStorage(
            optuna.storages.JournalFileBackend("optuna_journal_storage.log")
        )
    else:
        storage = storage_path

    study = optuna.create_study(
        direction="minimize",
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
    )

    logger.info("Optimizing hyperparameters with Optuna.")

    study.optimize(
        objective, n_trials=study_count, gc_after_trial=True, show_progress_bar=True
    )

    logger.info(f"Best trial for {study_name}: {study.best_trial.number}")
    logger.info(f"Best value (MAE): {study.best_value:.4f}")
    logger.info(f"Best params: {study.best_params}")

    return study


def evaluate_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: List[str],
    target: str,
    study_name: str,
    best_params: Dict[str, float],
    onehot_encoding: bool = False,
    subsample_shap: Optional[bool] = False,
) -> None:
    """
    Evaluate the model using the best parameters found by the Optuna study.
    Generate predictions, compute errors, and create plots (error histogram, SHAP plots).
    """

    logger.info(f"Evaluating model for study: {study_name}")

    X_train = train_df[features]
    y_train = train_df[target]

    X_test = test_df[features]
    y_test = test_df[target]

    os.makedirs("xgb_models", exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    model_path = Path(f"xgb_models/{study_name}.json")

    if model_path.exists():
        logger.info(f"Loading model from file: {model_path}")
        model = xgb.XGBRegressor(enable_categorical=not onehot_encoding)
        model.load_model(str(model_path))
    else:
        logger.info(f"Training model for {study_name}")
        model = xgb.XGBRegressor(**best_params, enable_categorical=not onehot_encoding)
        model.fit(X_train, y_train)
        model.save_model(str(model_path))
        logger.info(f"Model saved to {model_path}")

    logger.info("Generating predictions on test set.")
    predictions = model.predict(X_test)
    errors = abs(y_test - predictions)
    mean_error = errors.mean()
    logger.info(f"Mean Absolute Error on test set: {mean_error:.4f}")

    logger.info("Generating error histogram.")
    sns.histplot(errors, bins=50, kde=True, stat="density")
    plt.title(f"Histogram of Errors for {study_name}")
    plt.xlabel("Absolute Error")
    plt.ylabel("Density")
    plt.savefig(f"figures/error_histogram_{study_name}.pdf")
    plt.close()

    logger.info("Calculating SHAP values.")
    if subsample_shap:
        explainer = shap.Explainer(model, X_train.sample(frac=0.1).astype("float64"))
        shap_values = explainer(X_test[: len(X_test) // 10])
    else:
        explainer = shap.Explainer(model, X_train.astype("float64"))
        shap_values = explainer(X_test)

    logger.info("Generating SHAP beeswarm plot.")
    plt.figure()
    shap.plots.beeswarm(shap_values, show=False, max_display=30)
    plt.title(f"SHAP Beeswarm Plot for {study_name}")
    plt.tight_layout()
    plt.subplots_adjust(left=0.3)
    plt.savefig(f"figures/shap_{study_name}.pdf")
    plt.close()

    logger.info("Generating SHAP bar plot.")
    plt.figure()
    shap.plots.bar(shap_values, show=False, max_display=30)
    plt.title(f"SHAP Bar Plot for {study_name}")
    plt.tight_layout()
    plt.subplots_adjust(left=0.3)
    plt.savefig(f"figures/shap_bar_{study_name}.pdf")
    plt.close()


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

    args = parser.parse_args()

    # Generate config file name if not provided
    if args.config_file is None:
        args.config_file = f"training_config_{args.study_name}.json"

    discard_features = (
        [feat.strip() for feat in args.discard_features.split(",")]
        if args.discard_features
        else []
    )

    # List to store configuration details for each target
    training_configs = []

    for target in args.targets:
        encoding_type = "OneHot" if args.onehot_encoding else "Categorical"
        sen_geometrical_type = "_Geometrical" if args.sen_geometrical else ""
        clogging_factors_type = "_Clogging" if args.clogging_factors else ""
        full_study_name = f"{args.study_name}_{target}_{encoding_type}{sen_geometrical_type}{clogging_factors_type}"

        if not args.evaluate_only:
            logger.info(
                f"Starting experiment for target: {target} - study: {full_study_name}"
            )

            # load_data now returns onehot_values as an additional element
            train_df, test_df, features, onehot_values = load_data(
                target=target,
                onehot_encoding=args.onehot_encoding,
                sen_geometrical=args.sen_geometrical,
                clogging_factors=args.clogging_factors,
                discard_features=discard_features,
                data_directory=args.data_directory,
            )

            study = run_optuna_study(
                train_df=train_df,
                test_df=test_df,
                features=features,
                target=target,
                study_name=full_study_name,
                study_count=args.study_count,
                onehot_encoding=args.onehot_encoding,
                tree_method=args.tree_method,
                storage_path=args.storage_path,
                multithread=args.multithread,
            )

            best_trial = study.best_trial
            best_params = best_trial.params
        else:
            if args.multithread:
                storage = JournalStorage(
                    JournalFileBackend("optuna_journal_storage.log")
                )
            else:
                storage = args.storage_path

            study = optuna.load_study(study_name=full_study_name, storage=storage)

            evaluate_model(
                train_df=train_df,
                test_df=test_df,
                features=features,
                target=target,
                study_name=full_study_name,
                best_params=best_params,
                onehot_encoding=args.onehot_encoding,
                subsample_shap=args.subsample_shap,
            )

            # Build the configuration details for this run
            config = {
                "target": target,
                "onehot_encoding": args.onehot_encoding,
                "sen_geometrical": args.sen_geometrical,
                "clogging_factors": args.clogging_factors,
                "discard_features": discard_features,
                "data_directory": args.data_directory,
                "model_save_location": f"xgb_models/{full_study_name}.json",
                "tree_method": args.tree_method,
                "study_name": full_study_name,
                "storage_path": args.storage_path,
                "multithread": args.multithread,
                "study_count": args.study_count,
            }
            if args.onehot_encoding:
                config["onehot_values"] = onehot_values

            training_configs.append(config)

        # Save the accumulated training configuration to a JSON file
        with open(args.config_file, "w") as f:
            json.dump(training_configs, f, indent=4)
        logger.info(f"Training configuration saved to {args.config_file}")

        logger.info("All experiments completed successfully.")


if __name__ == "__main__":
    main()
