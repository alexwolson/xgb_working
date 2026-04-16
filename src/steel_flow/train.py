import json
import logging
from pathlib import Path
from typing import Dict, List

import pandas as pd
import xgboost as xgb

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def train_model(
    train_df: pd.DataFrame,
    features: List[str],
    target: str,
    study_name: str,
    best_params: Dict,
    onehot_encoding: bool = False,
    objective: str = "reg:squarederror",
    models_dir: str = "data/output/models",
) -> xgb.XGBRegressor:
    """
    Train (or load cached) XGBoost model with given hyperparameters.
    Saves model to models_dir/<study_name>.json. Returns the model.
    """
    Path(models_dir).mkdir(parents=True, exist_ok=True)
    model_path = Path(models_dir) / f"{study_name}.json"

    if model_path.exists():
        logger.info(f"Loading cached model from {model_path}")
        model = xgb.XGBRegressor(enable_categorical=not onehot_encoding)
        model.load_model(str(model_path))
    else:
        logger.info(f"Training model for {study_name} (objective={objective})")
        model = xgb.XGBRegressor(**best_params, objective=objective, enable_categorical=not onehot_encoding)
        model.fit(train_df[features], train_df[target])
        model.save_model(str(model_path))
        logger.info(f"Model saved to {model_path}")

    return model


def save_training_config(
    configs: List[Dict],
    config_file: str,
    config_dir: str = "config",
) -> None:
    """Write training config list to config_dir/config_file as JSON."""
    Path(config_dir).mkdir(parents=True, exist_ok=True)
    config_path = Path(config_dir) / config_file
    with open(config_path, "w") as f:
        json.dump(configs, f, indent=4)
    logger.info(f"Training configuration saved to {config_path}")
