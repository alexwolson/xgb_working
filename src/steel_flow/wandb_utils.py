from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

_RUN_ID_FILE = Path("data/output/wandb_run_ids.json")


def _load_run_ids() -> dict:
    if _RUN_ID_FILE.exists():
        return json.loads(_RUN_ID_FILE.read_text())
    return {}


def _save_run_ids(run_ids: dict) -> None:
    _RUN_ID_FILE.parent.mkdir(parents=True, exist_ok=True)
    _RUN_ID_FILE.write_text(json.dumps(run_ids, indent=2))


def save_run_id(full_study_name: str, run_id: str) -> None:
    run_ids = _load_run_ids()
    run_ids[full_study_name] = run_id
    _save_run_ids(run_ids)


def load_run_id(full_study_name: str) -> Optional[str]:
    return _load_run_ids().get(full_study_name)


def _build_tags(cfg: SimpleNamespace, target: str) -> list[str]:
    d = cfg.data
    tags = [target, Path(d.directory).name]
    if d.sen_geometrical:
        tags.append("geo")
    if d.clogging_factors:
        tags.append("clogging")
    if d.mould_position:
        tags.append("mould")
    if d.feature_lag_amount > 0:
        tags.append(f"lag{d.feature_lag_amount}")
    if d.onehot_encoding:
        tags.append("onehot")
    return tags


def _build_config_dict(cfg: SimpleNamespace, target: str) -> dict:
    e, d, t = cfg.experiment, cfg.data, cfg.tune
    return {
        "study_name": e.study_name,
        "target": target,
        "objective": e.objective,
        "data.directory": d.directory,
        "data.onehot_encoding": d.onehot_encoding,
        "data.sen_geometrical": d.sen_geometrical,
        "data.clogging_factors": d.clogging_factors,
        "data.mould_position": d.mould_position,
        "data.discard_features": d.discard_features,
        "data.lagged_features": d.lagged_features,
        "data.feature_lag_amount": d.feature_lag_amount,
        "data.remove_rows": d.remove_rows,
        "tune.study_count": t.study_count,
        "tune.tree_method": t.tree_method,
        "tune.multithread": t.multithread,
    }


def init_run(cfg: SimpleNamespace, target: str, full_study_name: str):
    """Create a new W&B run for the tune phase."""
    import wandb
    w = cfg.wandb
    return wandb.init(
        project=w.project,
        entity=w.entity or None,
        name=full_study_name,
        group=cfg.experiment.study_name,
        tags=_build_tags(cfg, target),
        config=_build_config_dict(cfg, target),
        resume="allow",
    )


def resume_run(run_id: Optional[str], cfg: SimpleNamespace, target: str, full_study_name: str):
    """Resume the W&B run created during tuning. Falls back to a new run if no ID found."""
    import wandb
    w = cfg.wandb
    if run_id:
        return wandb.init(
            project=w.project,
            entity=w.entity or None,
            id=run_id,
            resume="must",
        )
    logger.warning(f"No W&B run ID for {full_study_name} — creating new run.")
    return init_run(cfg, target, full_study_name)


def make_trial_callback(run):
    """Return an Optuna callback that logs each trial as a W&B step."""
    def _callback(study: object, trial: object) -> None:
        if trial.value is None:
            return
        run.log(
            {
                "trial/mae": trial.value,
                "trial/best_mae": study.best_value,
                **{f"trial/params/{k}": v for k, v in trial.params.items()},
            },
            step=trial.number,
        )

    return _callback


def log_final_metrics(
    run,
    train_metrics: dict,
    val_metrics: dict,
    test_metrics: dict,
) -> None:
    """Write train/val/test metrics to the run summary."""
    for split, metrics in [("train", train_metrics), ("val", val_metrics), ("test", test_metrics)]:
        for name, value in metrics.items():
            run.summary[f"{split}/{name}"] = value


def log_plots(run, study_name: str, figures_dir: str = "data/output/figures") -> None:
    """Upload PNG plots to W&B. evaluate.py saves PNGs alongside PDFs."""
    import wandb
    base = Path(figures_dir)
    candidates = {
        "plots/error_histogram": f"error_histogram_{study_name}.png",
        "plots/shap_beeswarm": f"shap_{study_name}.png",
        "plots/shap_bar": f"shap_bar_{study_name}.png",
        "plots/error_residuals": f"error_plot_residuals_{study_name}.png",
        "plots/error_actuals": f"error_plot_actuals_{study_name}.png",
    }
    images = {
        key: wandb.Image(str(base / filename))
        for key, filename in candidates.items()
        if (base / filename).exists()
    }
    if images:
        run.log(images)


def log_model_artifact(run, study_name: str, models_dir: str = "data/output/models") -> None:
    """Upload the trained XGBoost model as a W&B artifact."""
    import wandb
    model_path = Path(models_dir) / f"{study_name}.json"
    if not model_path.exists():
        logger.warning(f"Model file not found for artifact upload: {model_path}")
        return
    artifact = wandb.Artifact(name=study_name, type="model")
    artifact.add_file(str(model_path))
    run.log_artifact(artifact)
