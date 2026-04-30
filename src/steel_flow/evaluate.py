import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import xgboost as xgb
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PredictionErrorDisplay,
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
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
    r_squared, mean_squared_error, root_mean_squared_error, mape_zero_excluded.
    """
    predictions = model.predict(X)
    n_zeros = int((y == 0).sum())
    if n_zeros > 0:
        logger.warning(
            f"MAPE: {n_zeros} of {len(y)} ({100 * n_zeros / len(y):.1f}%) "
            "true values are zero and are excluded from sklearn's MAPE calculation."
        )
    metrics = {
        "mean_absolute_error": float(mean_absolute_error(y, predictions)),
        "mean_absolute_percent_error": float(mean_absolute_percentage_error(y, predictions) * 100),
        "r_squared": float(r2_score(y, predictions)),
        "mean_squared_error": float(mean_squared_error(y, predictions)),
        "root_mean_squared_error": float(root_mean_squared_error(y, predictions)),
        "mape_zero_excluded": n_zeros,
    }
    logger.info(f"MAE: {metrics['mean_absolute_error']:.4f}")
    logger.info(f"MAPE: {metrics['mean_absolute_percent_error']:.4f}% (zeros excluded: {n_zeros})")
    logger.info(f"R²: {metrics['r_squared']:.4f}")
    logger.info(f"RMSE: {metrics['root_mean_squared_error']:.4f}")
    return metrics


def compute_classification_metrics(
    model: xgb.XGBClassifier,
    X: pd.DataFrame,
    y: pd.Series,
) -> dict:
    """Compute classification metrics for a binned model.

    Returns accuracy, macro/weighted F1, precision, recall, Cohen's kappa,
    and MAE-of-bins (mean absolute distance between true and predicted bin index).
    """
    predictions = model.predict(X)
    metrics = {
        "accuracy": float(accuracy_score(y, predictions)),
        "f1_macro": float(f1_score(y, predictions, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y, predictions, average="weighted", zero_division=0)),
        "precision_macro": float(precision_score(y, predictions, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y, predictions, average="macro", zero_division=0)),
        "cohen_kappa": float(cohen_kappa_score(y, predictions)),
        "mae_bins": float(mean_absolute_error(y, predictions)),
    }
    logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"F1 (macro): {metrics['f1_macro']:.4f}  F1 (weighted): {metrics['f1_weighted']:.4f}")
    logger.info(f"Cohen's kappa: {metrics['cohen_kappa']:.4f}")
    logger.info(f"MAE (bins): {metrics['mae_bins']:.4f}")
    return metrics


def plot_confusion_matrix(
    model: xgb.XGBClassifier,
    X: pd.DataFrame,
    y: pd.Series,
    n_bins: int,
    study_name: str,
    figures_dir: str = "data/output/figures",
) -> None:
    """Save a normalised confusion matrix heatmap PDF/PNG to figures_dir."""
    Path(figures_dir).mkdir(parents=True, exist_ok=True)
    predictions = model.predict(X)
    labels = (
        ["low", "medium", "high"]
        if n_bins == 3
        else [f"bin_{i}" for i in range(n_bins)]
    )
    cm = confusion_matrix(y, predictions, normalize="true")
    fig, ax = plt.subplots(figsize=(max(4, n_bins), max(4, n_bins)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, colorbar=True, values_format=".2f")
    ax.set_title(f"Confusion Matrix (normalised) — {study_name}")
    plt.tight_layout()
    plt.savefig(f"{figures_dir}/confusion_matrix_{study_name}.pdf")
    plt.savefig(f"{figures_dir}/confusion_matrix_{study_name}.png", dpi=150)
    plt.close()
    logger.info(f"Confusion matrix saved to {figures_dir}/confusion_matrix_{study_name}.pdf")


def plot_error_histogram(
    model: xgb.XGBRegressor,
    X: pd.DataFrame,
    y: pd.Series,
    study_name: str,
    figures_dir: str = "data/output/figures",
) -> None:
    """Save a two-panel error histogram PDF to figures_dir.

    Left panel: absolute errors. Right panel: signed errors (true − predicted)
    with a dashed zero line to reveal systematic bias.
    """
    Path(figures_dir).mkdir(parents=True, exist_ok=True)
    predictions = model.predict(X)
    signed_errors = y.values - predictions
    abs_errors = np.abs(signed_errors)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    sns.histplot(abs_errors, bins=50, kde=True, stat="density", ax=axes[0])
    axes[0].set_title(f"Absolute Error — {study_name}")
    axes[0].set_xlabel("Absolute Error")
    axes[0].set_ylabel("Density")

    sns.histplot(signed_errors, bins=50, kde=True, stat="density", ax=axes[1])
    axes[1].axvline(0, color="red", linestyle="--", alpha=0.7, label="zero")
    axes[1].set_title(f"Signed Error (true − predicted) — {study_name}")
    axes[1].set_xlabel("Signed Error")
    axes[1].set_ylabel("Density")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(f"{figures_dir}/error_histogram_{study_name}.pdf")
    plt.savefig(f"{figures_dir}/error_histogram_{study_name}.png", dpi=150)
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
    """Save SHAP beeswarm and bar PDFs to figures_dir.

    When subsample_shap=True, uses a random 10 % sample of both train (for the
    SHAP background) and test (for explanations), with random_state=42 for
    reproducibility. Sample sizes are shown in plot titles.
    """
    Path(figures_dir).mkdir(parents=True, exist_ok=True)
    logger.info("Calculating SHAP values.")

    if subsample_shap:
        explain_set = X_test.sample(frac=0.1, random_state=42)
    else:
        explain_set = X_test

    n_explain = len(explain_set)
    n_test_total = len(X_test)
    sample_note = f"n={n_explain}" if not subsample_shap else f"n={n_explain} of {n_test_total}"

    # shap.TreeExplainer is incompatible with XGBoost 3.x (base_score format changed).
    # Use XGBoost's native pred_contribs instead, then wrap in shap.Explanation.
    dm = xgb.DMatrix(explain_set, enable_categorical=True)
    raw_contribs = model.get_booster().predict(dm, pred_contribs=True)
    # raw_contribs: (n_samples, n_features + 1) — last col is bias/base_score

    data_numeric = explain_set.copy()
    for col in data_numeric.select_dtypes("category").columns:
        data_numeric[col] = data_numeric[col].cat.codes

    shap_values = shap.Explanation(
        values=raw_contribs[:, :-1],
        base_values=raw_contribs[:, -1],
        data=data_numeric.values,
        feature_names=explain_set.columns.tolist(),
    )

    plt.figure()
    shap.plots.beeswarm(shap_values, show=False, max_display=100)
    plt.title(f"SHAP Beeswarm — {study_name} ({sample_note})")
    plt.tight_layout()
    plt.subplots_adjust(left=0.3)
    plt.savefig(f"{figures_dir}/shap_{study_name}.pdf")
    plt.savefig(f"{figures_dir}/shap_{study_name}.png", dpi=150)
    plt.close()

    plt.figure()
    shap.plots.bar(shap_values, show=False, max_display=100)
    plt.title(f"SHAP Bar — {study_name} ({sample_note})")
    plt.tight_layout()
    plt.subplots_adjust(left=0.3)
    plt.savefig(f"{figures_dir}/shap_bar_{study_name}.pdf")
    plt.savefig(f"{figures_dir}/shap_bar_{study_name}.png", dpi=150)
    plt.close()
    logger.info(f"SHAP plots saved to {figures_dir}/ ({sample_note})")


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
    plt.savefig(f"{figures_dir}/error_plot_residuals_{study_name}.png", dpi=150)
    plt.close()

    plt.figure()
    PredictionErrorDisplay.from_predictions(y, predictions, kind="actual_vs_predicted", subsample=0.1)
    plt.title(f"Prediction Error Plot for {study_name}")
    plt.tight_layout()
    plt.subplots_adjust(left=0.3)
    plt.savefig(f"{figures_dir}/error_plot_actuals_{study_name}.pdf")
    plt.savefig(f"{figures_dir}/error_plot_actuals_{study_name}.png", dpi=150)
    plt.close()
    logger.info(f"Prediction error plots saved to {figures_dir}/")
