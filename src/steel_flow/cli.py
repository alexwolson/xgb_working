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
from steel_flow import wandb_utils
from steel_flow.config import load_config

console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console)],
)
logger = logging.getLogger(__name__)


def _build_study_name(cfg, target: str) -> str:
    e, d = cfg.experiment, cfg.data
    encoding_type = "OneHot" if d.onehot_encoding else "Categorical"
    return (
        f"{e.study_name}_{target}_{encoding_type}"
        f"{'_Geometrical' if d.sen_geometrical else ''}"
        f"{'_Clogging' if d.clogging_factors else ''}"
        f"_{d.feature_lag_amount}Lag"
        f"{'_Mould' if d.mould_position else ''}"
    )


def tune() -> None:
    """Entry point for `uv run tune [config_path]`."""
    cfg = load_config()
    e, d, t, w = cfg.experiment, cfg.data, cfg.tune, cfg.wandb

    for target in e.targets:
        full_study_name = _build_study_name(cfg, target)
        logger.info(f"Starting tune for target: {target} — study: {full_study_name}")

        train_df, val_df, _test_df, features, _ = data_mod.load_data(
            target=target,
            onehot_encoding=d.onehot_encoding,
            sen_geometrical=d.sen_geometrical,
            clogging_factors=d.clogging_factors,
            discard_features=d.discard_features,
            data_directory=d.directory,
            feature_lag=d.feature_lag_amount,
            lagged_features=d.lagged_features,
            mould_position=d.mould_position,
            remove_rows=d.remove_rows,
            include_subdirs=d.include_subdirs,
            exclude_subdirs=d.exclude_subdirs,
        )

        if w.enabled:
            run = wandb_utils.init_run(cfg, target, full_study_name)
            wandb_utils.save_run_id(full_study_name, run.id)
            trial_callbacks = [wandb_utils.make_trial_callback(run)]
        else:
            run = None
            trial_callbacks = []

        tune_mod.run_study(
            train_df=train_df,
            val_df=val_df,
            features=features,
            target=target,
            study_name=full_study_name,
            study_count=t.study_count,
            onehot_encoding=d.onehot_encoding,
            tree_method=t.tree_method,
            storage_path=t.storage_path,
            multithread=t.multithread,
            seed=42,
            objective=e.objective,
            wandb_callbacks=trial_callbacks,
        )

        if run:
            run.finish()

    logger.info("Tuning complete.")


def train() -> None:
    """Entry point for `uv run train [config_path]`."""
    cfg = load_config()
    e, d, t, tr, w = cfg.experiment, cfg.data, cfg.tune, cfg.train, cfg.wandb

    config_file = tr.config_file or f"training_config_{e.study_name}.json"
    training_configs = []

    for target in e.targets:
        full_study_name = _build_study_name(cfg, target)
        logger.info(f"Training for target: {target} — study: {full_study_name}")

        if w.enabled:
            run_id = wandb_utils.load_run_id(full_study_name)
            run = wandb_utils.resume_run(run_id, cfg, target, full_study_name)
        else:
            run = None

        storage = (
            JournalStorage(JournalFileBackend(f"optuna_{full_study_name}.log"))
            if t.multithread
            else t.storage_path
        )
        study = optuna.load_study(study_name=full_study_name, storage=storage)
        best_params = study.best_trial.params

        if run:
            run.config.update(
                {f"best_params/{k}": v for k, v in best_params.items()},
                allow_val_change=True,
            )

        train_df, val_df, test_df, features, onehot_values = data_mod.load_data(
            target=target,
            onehot_encoding=d.onehot_encoding,
            sen_geometrical=d.sen_geometrical,
            clogging_factors=d.clogging_factors,
            discard_features=d.discard_features,
            data_directory=d.directory,
            feature_lag=d.feature_lag_amount,
            lagged_features=d.lagged_features,
            mould_position=d.mould_position,
            remove_rows=d.remove_rows,
        )

        model = train_mod.train_model(
            train_df=train_df,
            features=features,
            target=target,
            study_name=full_study_name,
            best_params=best_params,
            onehot_encoding=d.onehot_encoding,
            objective=e.objective,
        )

        X_train, y_train = train_df[features], train_df[target]
        X_val, y_val = val_df[features], val_df[target]
        X_test, y_test = test_df[features], test_df[target]

        logger.info("Evaluating on held-out test set (never seen during HPO).")
        test_metrics = eval_mod.compute_metrics(model, X_test, y_test)
        logger.info("Evaluating on validation set.")
        val_metrics = eval_mod.compute_metrics(model, X_val, y_val)
        logger.info("Evaluating on train set.")
        train_metrics = eval_mod.compute_metrics(model, X_train, y_train)

        eval_mod.plot_error_histogram(model, X_test, y_test, full_study_name)
        eval_mod.plot_shap(model, X_train, X_test, full_study_name, tr.subsample_shap)
        eval_mod.plot_prediction_error(model, X_test, y_test, full_study_name)

        if run:
            wandb_utils.log_final_metrics(run, train_metrics, val_metrics, test_metrics)
            wandb_utils.log_plots(run, full_study_name)
            wandb_utils.log_model_artifact(run, full_study_name)
            run.finish()

        config = {
            "target": target,
            "onehot_encoding": d.onehot_encoding,
            "sen_geometrical": d.sen_geometrical,
            "clogging_factors": d.clogging_factors,
            "discard_features": d.discard_features,
            "data_directory": d.directory,
            "model_save_location": f"data/output/models/{full_study_name}.json",
            "tree_method": t.tree_method,
            "objective": e.objective,
            "study_name": full_study_name,
            "storage_path": t.storage_path,
            "multithread": t.multithread,
            "study_count": t.study_count,
            "feature_lag": d.feature_lag_amount,
            "lagged_features": d.lagged_features,
            "mould_position": d.mould_position,
            "remove_rows": d.remove_rows,
            "test": test_metrics,
            "val": val_metrics,
            "train": train_metrics,
        }
        if d.onehot_encoding:
            config["onehot_values"] = onehot_values
        training_configs.append(config)

    train_mod.save_training_config(training_configs, config_file)
    logger.info("Training complete.")


def predict() -> None:
    """Entry point for `uv run predict [config_path]`."""
    cfg = load_config()
    p = cfg.predict

    try:
        with open(p.config_file, "r") as f:
            training_configs = json.load(f)
    except Exception as e:
        logger.error(f"Error loading config file {p.config_file}: {e}")
        return

    if not isinstance(training_configs, list):
        training_configs = [training_configs]

    data_dir = Path(p.data_directory)
    if not data_dir.exists():
        logger.error(f"Data directory '{p.data_directory}' does not exist.")
        return

    excel_files = list(data_dir.glob("**/*.xlsx"))
    if not excel_files:
        logger.warning(f"No Excel files found in {p.data_directory}.")
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
                    onehot_values, mould_position, progress, progress_task, p.piv_data,
                )

        logger.info(f"Predictions added using model {model_name}")

    logger.info("All files processed.")
