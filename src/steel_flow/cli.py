import argparse
import json
import logging
from pathlib import Path

import optuna
import pandas as pd
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
import xgboost as xgb
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn

from steel_flow import data as data_mod
from steel_flow import evaluate as eval_mod
from steel_flow import predict as predict_mod
from steel_flow import train as train_mod
from steel_flow import tune as tune_mod

console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console)],
)
logger = logging.getLogger(__name__)


def _common_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments shared by tune and train commands."""
    parser.add_argument("--targets", nargs="+", type=str, default=["Count_EX1"],
                        choices=["Count_EX1", "Count_EX2"], help="Target variables")
    parser.add_argument("--study-name", type=str, default="water_modelling", help="Optuna study name prefix")
    parser.add_argument("--study-count", type=int, default=1, help="Number of Optuna trials")
    parser.add_argument("--onehot-encoding", action="store_true", help="One-hot encode categorical features")
    parser.add_argument("--sen-geometrical", action="store_true", help="Convert SEN to geometrical features")
    parser.add_argument("--clogging-factors", action="store_true", help="Include clogging factors")
    parser.add_argument("--discard-features", type=str, default="",
                        help='Comma-separated features to discard (e.g., "SEN,L_wave_ht[mm]")')
    parser.add_argument("--tree-method", type=str, default="gpu_hist",
                        choices=["auto", "exact", "approx", "hist", "gpu_hist"])
    parser.add_argument("--data-directory", type=str, default="data/raw/Organized_Data")
    parser.add_argument("--storage-path", type=str, default="sqlite:///water_modelling.db")
    parser.add_argument("--multithread", action="store_true",
                        help="Use JournalStorage for HPC multithreading")
    parser.add_argument("--lagged-features", type=str, default="",
                        help="Comma-separated features to lag")
    parser.add_argument("--feature-lag-amount", type=int, default=0)
    parser.add_argument("--mould-position", action="store_true", help="Include mould position as feature")
    parser.add_argument("--remove-rows", type=int, default=0, help="Rows to remove from end of each sheet")


def _build_study_name(args: argparse.Namespace, target: str) -> str:
    """Build the full Optuna study name from CLI arguments."""
    encoding_type = "OneHot" if args.onehot_encoding else "Categorical"
    return (
        f"{args.study_name}_{target}_{encoding_type}"
        f"{'_Geometrical' if args.sen_geometrical else ''}"
        f"{'_Clogging' if args.clogging_factors else ''}"
        f"_{args.feature_lag_amount}Lag"
        f"{'_Mould' if args.mould_position else ''}"
    )


def tune() -> None:
    """Entry point for `uv run tune`. Runs Optuna hyperparameter search."""
    parser = argparse.ArgumentParser(description="Run Optuna hyperparameter search for XGBoost")
    _common_args(parser)
    args = parser.parse_args()

    discard_features = [f.strip() for f in args.discard_features.split(",") if f.strip()]
    lagged_features = [f.strip() for f in args.lagged_features.split(",") if f.strip()]

    for target in args.targets:
        full_study_name = _build_study_name(args, target)
        logger.info(f"Starting tune for target: {target} — study: {full_study_name}")

        train_df, test_df, features, _ = data_mod.load_data(
            target=target,
            onehot_encoding=args.onehot_encoding,
            sen_geometrical=args.sen_geometrical,
            clogging_factors=args.clogging_factors,
            discard_features=discard_features,
            data_directory=args.data_directory,
            feature_lag=args.feature_lag_amount,
            lagged_features=lagged_features,
            mould_position=args.mould_position,
            remove_rows=args.remove_rows,
        )

        tune_mod.run_study(
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

    logger.info("Tuning complete.")


def train() -> None:
    """Entry point for `uv run train`. Loads best Optuna params, trains final model, evaluates."""
    parser = argparse.ArgumentParser(description="Train final model using best Optuna params and evaluate")
    _common_args(parser)
    parser.add_argument("--subsample-shap", action="store_true")
    parser.add_argument("--config-file", type=str, default=None,
                        help="Output filename for training config JSON (default: training_config_<study-name>.json)")
    args = parser.parse_args()

    if args.config_file is None:
        args.config_file = f"training_config_{args.study_name}.json"

    discard_features = [f.strip() for f in args.discard_features.split(",") if f.strip()]
    lagged_features = [f.strip() for f in args.lagged_features.split(",") if f.strip()]
    training_configs = []

    for target in args.targets:
        full_study_name = _build_study_name(args, target)
        logger.info(f"Training for target: {target} — study: {full_study_name}")

        storage = (
            JournalStorage(JournalFileBackend(f"optuna_{full_study_name}.log"))
            if args.multithread
            else args.storage_path
        )
        study = optuna.load_study(study_name=full_study_name, storage=storage)
        best_params = study.best_trial.params

        train_df, test_df, features, onehot_values = data_mod.load_data(
            target=target,
            onehot_encoding=args.onehot_encoding,
            sen_geometrical=args.sen_geometrical,
            clogging_factors=args.clogging_factors,
            discard_features=discard_features,
            data_directory=args.data_directory,
            feature_lag=args.feature_lag_amount,
            lagged_features=lagged_features,
            mould_position=args.mould_position,
            remove_rows=args.remove_rows,
        )

        model = train_mod.train_model(
            train_df=train_df,
            features=features,
            target=target,
            study_name=full_study_name,
            best_params=best_params,
            onehot_encoding=args.onehot_encoding,
        )

        X_train = train_df[features]
        y_train = train_df[target]
        X_test = test_df[features]
        y_test = test_df[target]

        logger.info("Evaluating on test set.")
        test_metrics = eval_mod.compute_metrics(model, X_test, y_test)
        logger.info("Evaluating on train set.")
        train_metrics = eval_mod.compute_metrics(model, X_train, y_train)

        eval_mod.plot_error_histogram(model, X_test, y_test, full_study_name)
        eval_mod.plot_shap(model, X_train, X_test, full_study_name, args.subsample_shap)
        eval_mod.plot_prediction_error(model, X_test, y_test, full_study_name)

        config = {
            "target": target,
            "onehot_encoding": args.onehot_encoding,
            "sen_geometrical": args.sen_geometrical,
            "clogging_factors": args.clogging_factors,
            "discard_features": discard_features,
            "data_directory": args.data_directory,
            "model_save_location": f"data/output/models/{full_study_name}.json",
            "tree_method": args.tree_method,
            "study_name": full_study_name,
            "storage_path": args.storage_path,
            "multithread": args.multithread,
            "study_count": args.study_count,
            "feature_lag": args.feature_lag_amount,
            "lagged_features": lagged_features,
            "mould_position": args.mould_position,
            "remove_rows": args.remove_rows,
            "test": test_metrics,
            "train": train_metrics,
        }
        if args.onehot_encoding:
            config["onehot_values"] = onehot_values
        training_configs.append(config)

    train_mod.save_training_config(training_configs, args.config_file)
    logger.info("Training complete.")


def predict() -> None:
    """Entry point for `uv run predict`. Applies trained models to new Excel files."""
    parser = argparse.ArgumentParser(
        description="Apply trained model(s) from a config JSON to new Excel files"
    )
    parser.add_argument("--config-file", type=str, required=True,
                        help="Path to training config JSON produced by `uv run train` (e.g., config/training_config_myStudy.json)")
    parser.add_argument("--data-directory", type=str, default="New_Data",
                        help="Directory containing new Excel files (scanned recursively)")
    parser.add_argument("--piv-data", action="store_true", help="Data is PIV format")
    args = parser.parse_args()

    try:
        with open(args.config_file, "r") as f:
            training_configs = json.load(f)
    except Exception as e:
        logger.error(f"Error loading config file {args.config_file}: {e}")
        return

    if not isinstance(training_configs, list):
        training_configs = [training_configs]

    data_dir = Path(args.data_directory)
    if not data_dir.exists():
        logger.error(f"Data directory '{args.data_directory}' does not exist.")
        return

    excel_files = list(data_dir.glob("**/*.xlsx"))
    if not excel_files:
        logger.warning(f"No Excel files found in {args.data_directory}.")
        return

    for config in training_configs:
        model_file = config["model_save_location"]
        model_name = config["study_name"]
        onehot_encoding = config.get("onehot_encoding", False)
        sen_geometrical = config.get("sen_geometrical", False)
        clogging_factors = config.get("clogging_factors", False)
        drop_cols = config.get("discard_features", [])
        onehot_values = config.get("onehot_values", {}) if onehot_encoding else {}
        feature_lag = config.get("feature_lag", 0)
        lagged_features = config.get("lagged_features", [])
        mould_position = config.get("mould_position", False)

        logger.info(f"Loading model from {model_file}")
        model = xgb.XGBRegressor(enable_categorical=not onehot_encoding)
        try:
            model.load_model(model_file)
        except Exception as e:
            logger.error(f"Error loading model from {model_file}: {e}")
            continue

        total_sheets = sum(len(pd.ExcelFile(fp).sheet_names) for fp in excel_files)

        with Progress(
            SpinnerColumn(), BarColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(), TimeRemainingColumn(), console=console,
        ) as progress:
            progress_task = progress.add_task(
                f"Processing sheets for {model_name}", total=total_sheets
            )
            for file_path in excel_files:
                predict_mod.process_excel_file(
                    file_path, model, model_name, onehot_encoding, sen_geometrical,
                    clogging_factors, drop_cols, feature_lag, lagged_features,
                    onehot_values, mould_position, progress, progress_task, args.piv_data,
                )

        logger.info(f"Predictions added using model {model_name}")

    logger.info("All files processed.")
