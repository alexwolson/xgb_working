import logging
from typing import List

import optuna
import pandas as pd
import xgboost as xgb
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from sklearn.metrics import mean_absolute_error

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def run_study(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    features: List[str],
    target: str,
    study_name: str,
    study_count: int = 1,
    onehot_encoding: bool = False,
    tree_method: str = "gpu_hist",
    storage_path: str = "sqlite:///water_modelling.db",
    multithread: bool = False,
    seed: int = 42,
    objective: str = "reg:squarederror",
) -> optuna.Study:
    """Run an Optuna hyperparameter search for XGBoost. Returns the completed study.

    Args:
        train_df: Training data (features + target).
        val_df: Validation data used for the Optuna objective. Never the held-out test set.
        features: Column names to use as model inputs.
        target: Target column name.
        study_name: Optuna study name (used as DB key).
        study_count: Number of Optuna trials to run.
        onehot_encoding: Whether features are one-hot encoded.
        tree_method: XGBoost tree construction method.
        storage_path: SQLite URI or path for Optuna storage.
        multithread: Use JournalStorage instead of SQLite.
        seed: Random seed for the TPE sampler.
        objective: XGBoost objective function (e.g. 'reg:squarederror', 'count:poisson').

    Returns:
        optuna.Study: The completed study object with best_trial and best_value populated.
    """
    logger.info(f"Starting Optuna study: {study_name}")

    def objective_fn(trial: optuna.Trial) -> float:
        params = {
            "objective": objective,
            "booster": "gbtree",
            "verbosity": 0,
            "tree_method": tree_method,
            "grow_policy": trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"]),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 1.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 1e3, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 1e3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000, log=True),
        }
        model = xgb.XGBRegressor(**params, enable_categorical=not onehot_encoding)
        model.fit(train_df[features], train_df[target])
        preds = model.predict(val_df[features])
        return mean_absolute_error(val_df[target], preds)

    storage = (
        JournalStorage(JournalFileBackend(f"optuna_{study_name}.log"))
        if multithread
        else storage_path
    )

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(
        direction="minimize",
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        sampler=sampler,
    )
    study.optimize(objective_fn, n_trials=study_count, gc_after_trial=True, show_progress_bar=True)

    logger.info(f"Best trial: {study.best_trial.number}, val MAE: {study.best_value:.4f}")
    logger.info(f"Best params: {study.best_params}")
    return study
