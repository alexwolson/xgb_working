import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import shap
import xgboost as xgb
from sklearn.metrics import (
    PredictionErrorDisplay,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
    root_mean_squared_error,
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def compute_metrics(
    model: xgb.XGBRegressor,
    X: pd.DataFrame,
    y: pd.Series,
) -> dict:
    """
    Compute regression metrics for model predictions on (X, y).
    Returns dict with keys: mean_absolute_error, mean_absolute_percent_error,
    r_squared, mean_squared_error, root_mean_squared_error.
    """
    predictions = model.predict(X)
    metrics = {
        "mean_absolute_error": float(mean_absolute_error(y, predictions)),
        "mean_absolute_percent_error": float(mean_absolute_percentage_error(y, predictions) * 100),
        "r_squared": float(r2_score(y, predictions)),
        "mean_squared_error": float(mean_squared_error(y, predictions)),
        "root_mean_squared_error": float(root_mean_squared_error(y, predictions)),
    }
    logger.info(f"MAE: {metrics['mean_absolute_error']:.4f}")
    logger.info(f"MAPE: {metrics['mean_absolute_percent_error']:.4f}%")
    logger.info(f"R²: {metrics['r_squared']:.4f}")
    logger.info(f"RMSE: {metrics['root_mean_squared_error']:.4f}")
    return metrics


def plot_error_histogram(
    model: xgb.XGBRegressor,
    X: pd.DataFrame,
    y: pd.Series,
    study_name: str,
    figures_dir: str = "data/output/figures",
) -> None:
    """Save error histogram PDF to figures_dir."""
    Path(figures_dir).mkdir(parents=True, exist_ok=True)
    predictions = model.predict(X)
    errors = abs(y - predictions)
    sns.histplot(errors, bins=50, kde=True, stat="density")
    plt.title(f"Histogram of Errors for {study_name}")
    plt.xlabel("Absolute Error")
    plt.ylabel("Density")
    plt.savefig(f"{figures_dir}/error_histogram_{study_name}.pdf")
    plt.close()
    logger.info(f"Error histogram saved to {figures_dir}/error_histogram_{study_name}.pdf")


def plot_shap(
    model: xgb.XGBRegressor,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    study_name: str,
    subsample_shap: bool = False,
    figures_dir: str = "data/output/figures",
) -> None:
    """Save SHAP beeswarm and bar PDFs to figures_dir."""
    Path(figures_dir).mkdir(parents=True, exist_ok=True)
    logger.info("Calculating SHAP values.")
    if subsample_shap:
        explainer = shap.Explainer(model, X_train.sample(frac=0.1).astype("float64"))
        shap_values = explainer(X_test[: len(X_test) // 10].astype("float64"))
    else:
        explainer = shap.Explainer(model, X_train.astype("float64"))
        shap_values = explainer(X_test.astype("float64"))

    plt.figure()
    shap.plots.beeswarm(shap_values, show=False, max_display=100)
    plt.title(f"SHAP Beeswarm Plot for {study_name}")
    plt.tight_layout()
    plt.subplots_adjust(left=0.3)
    plt.savefig(f"{figures_dir}/shap_{study_name}.pdf")
    plt.close()

    plt.figure()
    shap.plots.bar(shap_values, show=False, max_display=100)
    plt.title(f"SHAP Bar Plot for {study_name}")
    plt.tight_layout()
    plt.subplots_adjust(left=0.3)
    plt.savefig(f"{figures_dir}/shap_bar_{study_name}.pdf")
    plt.close()
    logger.info(f"SHAP plots saved to {figures_dir}/")


def plot_prediction_error(
    model: xgb.XGBRegressor,
    X: pd.DataFrame,
    y: pd.Series,
    study_name: str,
    figures_dir: str = "data/output/figures",
) -> None:
    """Save residuals and actual-vs-predicted PDFs to figures_dir."""
    Path(figures_dir).mkdir(parents=True, exist_ok=True)
    predictions = model.predict(X)

    plt.figure()
    PredictionErrorDisplay.from_predictions(y, predictions, subsample=0.1)
    plt.title(f"Prediction Error Plot for {study_name}")
    plt.tight_layout()
    plt.subplots_adjust(left=0.3)
    plt.savefig(f"{figures_dir}/error_plot_residuals_{study_name}.pdf")
    plt.close()

    plt.figure()
    PredictionErrorDisplay.from_predictions(y, predictions, kind="actual_vs_predicted", subsample=0.1)
    plt.title(f"Prediction Error Plot for {study_name}")
    plt.tight_layout()
    plt.subplots_adjust(left=0.3)
    plt.savefig(f"{figures_dir}/error_plot_actuals_{study_name}.pdf")
    plt.close()
    logger.info(f"Prediction error plots saved to {figures_dir}/")
