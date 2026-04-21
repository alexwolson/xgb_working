# Config TOML Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace all CLI argument parsing with a `config/base.toml` file; commands accept one optional positional arg (config path) and nothing else.

**Architecture:** A new `config.py` module reads a TOML file and returns a `SimpleNamespace` of sections. `cli.py` is rewritten to call `load_config()` instead of argparse. Pipeline internals (`data.py`, `tune.py`, `train.py`, `evaluate.py`, `predict.py`) are untouched.

**Tech Stack:** Python 3.10+, `tomllib` (stdlib ≥3.11) / `tomli` (fallback for 3.10), `uv` for dependency management.

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `config/base.toml` | Create | Default experiment configuration |
| `src/steel_flow/config.py` | Create | TOML loader + validator → `SimpleNamespace` |
| `src/steel_flow/cli.py` | Rewrite | Read config, call pipeline (no argparse) |
| `pyproject.toml` | Modify | Add `tomli` dependency |
| `tests/test_config.py` | Create | Unit tests for config loader |
| `scripts/hpc/study_job.sh` | Modify | Remove CLI flags |
| `scripts/hpc/multithread_study_train.sh` | Modify | Remove CLI flags |
| `scripts/hpc/multithread_study_eval.sh` | Modify | Remove CLI flags |
| `scripts/hpc/predict_job.sh` | Modify | Remove CLI flags |
| `scripts/hpc/piv_pred.sh` | Modify | Remove CLI flags |
| `README.md` | Modify | Rewrite usage section |

---

## Task 1: Add `tomli` dependency and create `config/base.toml`

**Files:**
- Modify: `pyproject.toml`
- Create: `config/base.toml`

- [ ] **Step 1: Add `tomli` to dependencies**

Run:
```bash
uv add tomli
```

Expected: `pyproject.toml` gains `"tomli"` in `dependencies`, `uv.lock` is updated.

- [ ] **Step 2: Create `config/base.toml` with default values**

Create `config/base.toml` with this exact content:

```toml
[experiment]
targets = ["Count_EX1"]
study_name = "water_modelling"
objective = "reg:squarederror"

[data]
directory = "data/raw/Organized_Data"
discard_features = []
onehot_encoding = false
sen_geometrical = false
clogging_factors = false
lagged_features = []
feature_lag_amount = 0
mould_position = false
remove_rows = 0

[tune]
study_count = 1
tree_method = "gpu_hist"
storage_path = "sqlite:///water_modelling.db"
multithread = false

[train]
subsample_shap = false
config_file = ""

[predict]
config_file = "config/training_config_water_modelling.json"
data_directory = "New_Data"
piv_data = false
```

Note: `train.config_file = ""` means auto-derive as `training_config_<study_name>.json`. `predict.config_file` is the path to the JSON output from `uv run train`.

- [ ] **Step 3: Verify TOML is valid**

Run:
```bash
uv run python -c "import tomllib; tomllib.load(open('config/base.toml', 'rb'))"
```

Expected: no output, exit code 0.

- [ ] **Step 4: Commit**

```bash
git add config/base.toml pyproject.toml uv.lock
git commit -m "feat: add tomli dependency and config/base.toml"
```

---

## Task 2: Create `src/steel_flow/config.py` (TDD)

**Files:**
- Create: `tests/test_config.py`
- Create: `src/steel_flow/config.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_config.py`:

```python
import sys
import pytest
from pathlib import Path


def _write_valid_toml(tmp_path: Path) -> Path:
    content = """
[experiment]
targets = ["Count_EX1"]
study_name = "test_study"
objective = "reg:squarederror"

[data]
directory = "data/raw"
discard_features = []
onehot_encoding = false
sen_geometrical = false
clogging_factors = false
lagged_features = []
feature_lag_amount = 0
mould_position = false
remove_rows = 0

[tune]
study_count = 5
tree_method = "hist"
storage_path = "sqlite:///test.db"
multithread = false

[train]
subsample_shap = true
config_file = ""

[predict]
config_file = "config/training_config_test.json"
data_directory = "New_Data"
piv_data = false
"""
    p = tmp_path / "test_config.toml"
    p.write_text(content)
    return p


def test_valid_toml_loads_correctly(tmp_path):
    from steel_flow.config import load_config
    cfg = load_config(str(_write_valid_toml(tmp_path)))
    assert cfg.experiment.study_name == "test_study"
    assert cfg.experiment.targets == ["Count_EX1"]
    assert cfg.data.onehot_encoding is False
    assert cfg.tune.study_count == 5
    assert cfg.train.subsample_shap is True
    assert cfg.predict.data_directory == "New_Data"


def test_missing_section_raises_system_exit(tmp_path):
    # TOML missing [data], [tune], [train], [predict]
    content = """
[experiment]
targets = ["Count_EX1"]
study_name = "test_study"
objective = "reg:squarederror"
"""
    p = tmp_path / "bad_config.toml"
    p.write_text(content)
    from steel_flow.config import load_config
    with pytest.raises(SystemExit) as exc_info:
        load_config(str(p))
    assert "data" in str(exc_info.value)


def test_default_path_used_when_no_arg(monkeypatch, tmp_path):
    import steel_flow.config as cfg_mod
    monkeypatch.setattr(sys, "argv", ["cmd"])
    monkeypatch.setattr(cfg_mod, "DEFAULT_CONFIG_PATH", str(tmp_path / "nonexistent.toml"))
    from steel_flow.config import load_config
    with pytest.raises(SystemExit) as exc_info:
        load_config()
    assert "nonexistent.toml" in str(exc_info.value)
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
uv run pytest tests/test_config.py -v
```

Expected: `ImportError` or `ModuleNotFoundError` — `steel_flow.config` does not exist yet.

- [ ] **Step 3: Implement `src/steel_flow/config.py`**

Create `src/steel_flow/config.py`:

```python
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # type: ignore[no-redef]

DEFAULT_CONFIG_PATH = "config/base.toml"

_REQUIRED: dict[str, list[str]] = {
    "experiment": ["targets", "study_name", "objective"],
    "data": [
        "directory", "discard_features", "onehot_encoding", "sen_geometrical",
        "clogging_factors", "lagged_features", "feature_lag_amount",
        "mould_position", "remove_rows",
    ],
    "tune": ["study_count", "tree_method", "storage_path", "multithread"],
    "train": ["subsample_shap", "config_file"],
    "predict": ["config_file", "data_directory", "piv_data"],
}


def load_config(path: str | None = None) -> SimpleNamespace:
    if path is None:
        path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CONFIG_PATH

    config_path = Path(path)
    if not config_path.exists():
        sys.exit(f"Config file not found: {config_path}")

    with open(config_path, "rb") as f:
        raw = tomllib.load(f)

    for section, keys in _REQUIRED.items():
        if section not in raw:
            sys.exit(f"Config missing required section: [{section}]")
        for key in keys:
            if key not in raw[section]:
                sys.exit(f"Config [{section}] missing required key: {key}")

    return SimpleNamespace(**{
        section: SimpleNamespace(**raw[section])
        for section in _REQUIRED
    })
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
uv run pytest tests/test_config.py -v
```

Expected: 3 tests pass.

- [ ] **Step 5: Run full test suite to confirm no regressions**

```bash
uv run pytest -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/steel_flow/config.py tests/test_config.py
git commit -m "feat: add config.py TOML loader with validation"
```

---

## Task 3: Rewrite `src/steel_flow/cli.py`

**Files:**
- Modify: `src/steel_flow/cli.py`

No new unit tests — cli.py wires together pipeline calls. Correctness is verified by running the pipeline.

- [ ] **Step 1: Replace `cli.py` entirely**

Replace the full contents of `src/steel_flow/cli.py` with:

```python
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
    e, d, t = cfg.experiment, cfg.data, cfg.tune

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
        )

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
        )

    logger.info("Tuning complete.")


def train() -> None:
    """Entry point for `uv run train [config_path]`."""
    cfg = load_config()
    e, d, t, tr = cfg.experiment, cfg.data, cfg.tune, cfg.train

    config_file = tr.config_file or f"training_config_{e.study_name}.json"
    training_configs = []

    for target in e.targets:
        full_study_name = _build_study_name(cfg, target)
        logger.info(f"Training for target: {target} — study: {full_study_name}")

        storage = (
            JournalStorage(JournalFileBackend(f"optuna_{full_study_name}.log"))
            if t.multithread
            else t.storage_path
        )
        study = optuna.load_study(study_name=full_study_name, storage=storage)
        best_params = study.best_trial.params

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
```

- [ ] **Step 2: Run full test suite**

```bash
uv run pytest -v
```

Expected: all tests pass (no tests directly import cli.py, so this is a regression check on `test_data.py` and `test_config.py`).

- [ ] **Step 3: Smoke-test tune entry point with config file**

This requires real data. Run with a small trial count to confirm end-to-end:

```bash
uv run tune config/base.toml
```

Expected: runs 1 Optuna trial (default `study_count = 1`), logs train/val study results, exits cleanly.

- [ ] **Step 4: Commit**

```bash
git add src/steel_flow/cli.py
git commit -m "feat: rewrite cli.py to use config.toml instead of argparse"
```

---

## Task 4: Update HPC scripts

**Files:**
- Modify: `scripts/hpc/study_job.sh`
- Modify: `scripts/hpc/multithread_study_train.sh`
- Modify: `scripts/hpc/multithread_study_eval.sh`
- Modify: `scripts/hpc/predict_job.sh`
- Modify: `scripts/hpc/piv_pred.sh`

All CLI flags are removed. Each script passes only the config path.

- [ ] **Step 1: Update `scripts/hpc/study_job.sh`**

Replace the `uv run tune ...` line (currently lines 13–20) with:

```bash
uv run tune config/base.toml
```

The full file after edit:

```bash
#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32000M
#SBATCH --time=1-0:0:0
#SBATCH --output=%N-%j.out

module load python/3.10
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index -r requirements-computecan.txt

uv run tune config/base.toml
```

- [ ] **Step 2: Update `scripts/hpc/multithread_study_train.sh`**

Replace the `uv run train ...` block (currently lines 15–22) with:

```bash
uv run train config/base.toml
```

The full file after edit:

```bash
#!/bin/bash
#SBATCH --cpus-per-task=1
#SBATCH --mem=32000M
#SBATCH --time=1-0:0:0
#SBATCH --output=%N-%j.out

module load python/3.10
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

pip install --no-index --upgrade pip
pip install --no-index -r requirements-computecan.txt

uv run train config/base.toml
```

- [ ] **Step 3: Update `scripts/hpc/multithread_study_eval.sh`**

Replace the `srun uv run tune ...` block (currently lines 15–24) with:

```bash
srun uv run tune config/base.toml
```

The full file after edit:

```bash
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=3G
#SBATCH --ntasks=10
#SBATCH --time=1-0:0:0
#SBATCH --output=%N-%j.out

module load python/3.10

virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

pip install --no-index --upgrade pip
pip install --no-index -r requirements-computecan.txt

srun uv run tune config/base.toml
```

- [ ] **Step 4: Update `scripts/hpc/predict_job.sh`**

Replace the `uv run predict ...` block with:

```bash
uv run predict config/base.toml
```

The full file after edit:

```bash
#!/bin/bash
#SBATCH --cpus-per-task=1
#SBATCH --mem=32000M
#SBATCH --time=1-0:0:0
#SBATCH --output=%N-%j.out

module load python/3.10

virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

pip install --no-index --upgrade pip
pip install --no-index -r requirements-computecan.txt

uv run predict config/base.toml
```

- [ ] **Step 5: Update `scripts/hpc/piv_pred.sh`**

Replace the `uv run predict ...` line with:

```bash
uv run predict config/base.toml
```

The full file after edit:

```bash
uv run predict config/base.toml
```

- [ ] **Step 6: Commit**

```bash
git add scripts/hpc/
git commit -m "feat: update HPC scripts to use config/base.toml"
```

---

## Task 5: Update README

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Replace the Usage section**

In `README.md`, find the `## Usage` section (lines 47–113) and replace it with:

```markdown
## Usage

All pipeline settings live in `config/base.toml`. Edit that file before running any command. To use an alternate config, pass its path as a positional argument.

### 1. Data Preparation

Place raw `.xlsx` files under `data/raw/Organized_Data/` in the appropriate subdirectory structure (e.g., `SEN06/`, `SEN07/`). CSVs are generated automatically on first run.

### 2. Configure `config/base.toml`

Open `config/base.toml` and set your experiment parameters. Key fields:

```toml
[experiment]
targets = ["Count_EX1", "Count_EX2"]   # target variables
study_name = "MyStudy"                  # Optuna study name prefix
objective = "reg:squarederror"          # XGBoost objective

[data]
directory = "data/raw/Organized_Data"
discard_features = ["time[s]"]
onehot_encoding = true
sen_geometrical = true

[tune]
study_count = 100                       # number of Optuna trials
tree_method = "gpu_hist"                # use "hist" on CPU-only machines

[train]
subsample_shap = true

[predict]
config_file = "config/training_config_MyStudy.json"
data_directory = "New_Data"
```

### 3. Hyperparameter Tuning

```bash
uv run tune                              # uses config/base.toml
uv run tune config/my_experiment.toml   # uses alternate config
```

### 4. Train Final Model

```bash
uv run train
```

Saves the model to `data/output/models/` and the training config to `config/`.

### 5. Predict on New Data

Set `predict.config_file` and `predict.data_directory` in your TOML, then:

```bash
uv run predict
```

### Config Reference

**`[experiment]`**
- `targets`: List of target variables (`"Count_EX1"`, `"Count_EX2"`). Default: `["Count_EX1"]`.
- `study_name`: Optuna study name prefix. Default: `"water_modelling"`.
- `objective`: XGBoost objective (`"reg:squarederror"`, `"count:poisson"`, `"reg:tweedie"`). Default: `"reg:squarederror"`.

**`[data]`**
- `directory`: Directory with raw data. Default: `"data/raw/Organized_Data"`.
- `discard_features`: List of features to discard. Default: `[]`.
- `onehot_encoding`: One-hot encode categorical features. Default: `false`.
- `sen_geometrical`: Convert SEN numbers to angle/depth features. Default: `false`.
- `clogging_factors`: Include clogging factors. Default: `false`.
- `lagged_features`: List of features to lag. Default: `[]`.
- `feature_lag_amount`: Number of lag steps. Default: `0`.
- `mould_position`: Include mould position as feature. Default: `false`.
- `remove_rows`: Rows to remove from end of each sheet. Default: `0`.

**`[tune]`**
- `study_count`: Number of Optuna trials. Default: `1`.
- `tree_method`: XGBoost tree method (`"hist"`, `"gpu_hist"`, etc.). Default: `"gpu_hist"`.
- `storage_path`: Optuna study storage path. Default: `"sqlite:///water_modelling.db"`.
- `multithread`: Use JournalStorage for HPC multithreading. Default: `false`.

**`[train]`**
- `subsample_shap`: Subsample SHAP values for faster computation. Default: `false`.
- `config_file`: Output filename for training config JSON (saved to `config/`). Default: `""` (auto-derived as `training_config_<study_name>.json`).

**`[predict]`**
- `config_file`: Path to training config JSON produced by `uv run train`. Required.
- `data_directory`: Directory with new Excel files. Default: `"New_Data"`.
- `piv_data`: Data is PIV format. Default: `false`.
```

- [ ] **Step 2: Verify the file looks correct**

```bash
uv run python -c "import pathlib; print(pathlib.Path('README.md').read_text()[1500:3500])"
```

Expected: the new Usage section appears with TOML examples and the Config Reference table.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: update README usage section for config.toml refactor"
```
