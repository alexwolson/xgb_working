import logging
from typing import Dict, List

import optuna
import pandas as pd
import xgboost as xgb
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from rich.console import Console
from rich.logging import RichHandler
from sklearn.metrics import mean_absolute_error

console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console)],
)
logger = logging.getLogger(__name__)


def run_study(
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
    """Run an Optuna hyperparameter search for XGBoost. Returns the completed study."""
    logger.info(f"Starting Optuna study: {study_name}")

    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective": "reg:squarederror",
            "eval_metric": "mae",
            "booster": "gbtree",
            "verbosity": 0,
            "tree_method": tree_method,
            "grow_policy": trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"]),
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
        return mean_absolute_error(test_df[target], preds)

    storage = (
        JournalStorage(JournalFileBackend(f"optuna_{study_name}.log"))
        if multithread
        else storage_path
    )

    study = optuna.create_study(
        direction="minimize",
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=study_count, gc_after_trial=True, show_progress_bar=True)

    logger.info(f"Best trial: {study.best_trial.number}, MAE: {study.best_value:.4f}")
    logger.info(f"Best params: {study.best_params}")
    return study
