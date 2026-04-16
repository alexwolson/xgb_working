# Full Restructure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restructure the repo into a proper installable Python package with `uv run tune`, `uv run train`, and `uv run predict` entry points, clean directory layout, and a single authoritative `pyproject.toml`.

**Architecture:** Business logic moves from three root-level scripts into `src/steel_flow/` (data, tune, train, evaluate, predict modules). A `cli.py` module wires argparse to those modules and is registered as the `[project.scripts]` entry points. Data files, figures, and models live under `data/` (gitignored). Shell scripts split into `scripts/local/` and `scripts/hpc/`.

**Tech Stack:** Python 3.10+, uv, XGBoost, Optuna, SHAP, scikit-learn, pandas, seaborn, matplotlib, rich, openpyxl.

---

## File Map

| Action | Path |
|--------|------|
| Delete | `rewerwrew.py`, `wergbtqwbertbwre.py`, `run_study_old.py`, `start.txt`, `requirements.txt`, `environment.yml` |
| Move → delete | `requirementscomputecan.txt` → `scripts/hpc/requirements-computecan.txt` |
| Move (physical, not git-tracked) | `Organized_Data/` → `data/raw/Organized_Data/`, `figures/` → `data/output/figures/` |
| Move (git-tracked) | `start_study.sh`, `start_predict.sh` → `scripts/local/` |
| Move (git-tracked) | `study_job.sh`, `predict_job.sh`, `multithread_*.sh`, `piv_pred.sh` → `scripts/hpc/` |
| Move (git-tracked) | `training_config_20250328.json` → `config/` |
| Modify | `pyproject.toml` — add all deps, `[project.scripts]`, `[build-system]` |
| Modify | `.gitignore` — remove old path entries, ensure `data/` is covered |
| Create | `src/steel_flow/__init__.py` |
| Create | `src/steel_flow/data.py` — `clean_column_name`, `generate_csv_files`, `load_data` |
| Create | `src/steel_flow/tune.py` — `run_study` |
| Create | `src/steel_flow/train.py` — `train_model`, `save_training_config` |
| Create | `src/steel_flow/evaluate.py` — `compute_metrics`, `plot_error_histogram`, `plot_shap`, `plot_prediction_error` |
| Create | `src/steel_flow/predict.py` — `predict_on_sheet`, `process_excel_file` |
| Create | `src/steel_flow/cli.py` — `tune`, `train`, `predict` entry points |
| Create | `tests/test_data.py`, `tests/test_imports.py` |
| Delete | `run_study.py`, `predict.py`, `get_split.py` (after package is working) |
| Update | `README.md`, `scripts/local/start_study.sh`, `scripts/local/start_predict.sh` |

---

## Task 1: Create branch and delete dead files

**Files:**
- Delete: `rewerwrew.py`, `wergbtqwbertbwre.py`, `run_study_old.py`, `start.txt`, `requirements.txt`, `environment.yml`

- [ ] **Step 1: Create the cleanup branch**

```bash
git checkout -b cleanup/full-restructure
```

- [ ] **Step 2: Delete dead files**

```bash
git rm rewerwrew.py wergbtqwbertbwre.py run_study_old.py start.txt requirements.txt environment.yml
```

- [ ] **Step 3: Verify deletions**

```bash
git status
```
Expected: 6 files staged for deletion, nothing else modified.

- [ ] **Step 4: Commit**

```bash
git commit -m "chore: delete dead files (empty stubs, backup, scratch notes, old dep files)"
```

---

## Task 2: Reorganize directory layout

**Files:**
- Create: `data/raw/.gitkeep`, `data/output/figures/.gitkeep`, `data/output/models/.gitkeep`
- Create: `config/`
- Move (git): `training_config_20250328.json` → `config/training_config_20250328.json`
- Move (git): 6 shell scripts → `scripts/local/` and `scripts/hpc/`
- Move (git): `requirementscomputecan.txt` → `scripts/hpc/requirements-computecan.txt`
- Move (physical): `Organized_Data/` → `data/raw/Organized_Data/`
- Move (physical): `figures/` → `data/output/figures/`
- Modify: `.gitignore`

- [ ] **Step 1: Create new directory skeleton with gitkeep files**

```bash
mkdir -p data/raw data/output/figures data/output/models config scripts/local scripts/hpc
touch data/raw/.gitkeep data/output/figures/.gitkeep data/output/models/.gitkeep
```

- [ ] **Step 2: Move git-tracked files**

```bash
git mv training_config_20250328.json config/training_config_20250328.json
git mv start_study.sh scripts/local/start_study.sh
git mv start_predict.sh scripts/local/start_predict.sh
git mv study_job.sh scripts/hpc/study_job.sh
git mv predict_job.sh scripts/hpc/predict_job.sh
git mv multithread_job.sh scripts/hpc/multithread_job.sh
git mv multithread_study_train.sh scripts/hpc/multithread_study_train.sh
git mv multithread_study_eval.sh scripts/hpc/multithread_study_eval.sh
git mv piv_pred.sh scripts/hpc/piv_pred.sh
git mv requirementscomputecan.txt scripts/hpc/requirements-computecan.txt
```

- [ ] **Step 3: Move physically-untracked data directories (if they exist on disk)**

```bash
# Only run these if the directories exist locally
[ -d Organized_Data ] && mv Organized_Data data/raw/Organized_Data
[ -d figures ] && mv figures/* data/output/figures/ && rmdir figures
```

- [ ] **Step 4: Update .gitignore**

Open `.gitignore`. Replace the top block (lines 1–5) that reads:
```
.idea
figures/
data/
xgb_models
pivdata
cloggedData/
```
with:
```
.idea
data/
xgb_models/
pivdata/
cloggedData/
/New_Data/
/Clogging_Data/
```

Also remove these lines further down in the file (they're now covered by `data/`):
```
/New_Data/
/Clogging_Data/
/Organized_Data/
/water_modelling.db
/X.csv
/y.csv
training_config*
```

And add below them:
```
/water_modelling.db
/X.csv
/y.csv
training_config*
*.log
```

- [ ] **Step 5: Stage and verify**

```bash
git add -A
git status
```
Expected: shell scripts moved, config file moved, requirementscomputecan.txt renamed, .gitignore modified, .gitkeep files added.

- [ ] **Step 6: Commit**

```bash
git commit -m "chore: reorganize directory layout into data/, config/, scripts/"
```

---

## Task 3: Fix pyproject.toml

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Rewrite pyproject.toml**

Replace the entire contents of `pyproject.toml` with:

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "steel-flow-sensor-prediction"
version = "0.1.0"
description = "XGBoost pipeline for predicting steel flow sensor targets"
readme = "README.md"
requires-python = ">=3.10"
dependencies = [
    "matplotlib",
    "rich",
    "xgboost",
    "optuna",
    "shap",
    "scikit-learn",
    "pandas",
    "openpyxl",
    "seaborn",
]

[project.scripts]
tune = "steel_flow.cli:tune"
train = "steel_flow.cli:train"
predict = "steel_flow.cli:predict"

[tool.hatch.build.targets.wheel]
packages = ["src/steel_flow"]
```

- [ ] **Step 2: Run uv sync to verify deps resolve**

```bash
uv sync
```
Expected: uv resolves and installs all packages without errors. This will fail at the end because `steel_flow` package doesn't exist yet — that's fine. If you see a `ModuleNotFoundError` for `steel_flow`, that's expected. Dependency resolution itself should succeed.

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "chore: fix pyproject.toml — add all deps and uv run entry points"
```

---

## Task 4: Write failing tests for the new package

**Files:**
- Create: `tests/__init__.py`
- Create: `tests/test_imports.py`
- Create: `tests/test_data.py`

- [ ] **Step 1: Create tests directory**

```bash
mkdir -p tests
touch tests/__init__.py
```

- [ ] **Step 2: Write import tests**

Create `tests/test_imports.py`:

```python
def test_data_module_importable():
    from steel_flow.data import generate_csv_files, load_data, clean_column_name
    assert callable(generate_csv_files)
    assert callable(load_data)
    assert callable(clean_column_name)


def test_tune_module_importable():
    from steel_flow.tune import run_study
    assert callable(run_study)


def test_train_module_importable():
    from steel_flow.train import train_model, save_training_config
    assert callable(train_model)
    assert callable(save_training_config)


def test_evaluate_module_importable():
    from steel_flow.evaluate import compute_metrics, plot_error_histogram, plot_shap, plot_prediction_error
    assert callable(compute_metrics)
    assert callable(plot_error_histogram)
    assert callable(plot_shap)
    assert callable(plot_prediction_error)


def test_predict_module_importable():
    from steel_flow.predict import predict_on_sheet, process_excel_file
    assert callable(predict_on_sheet)
    assert callable(process_excel_file)


def test_cli_module_importable():
    from steel_flow.cli import tune, train, predict
    assert callable(tune)
    assert callable(train)
    assert callable(predict)
```

- [ ] **Step 3: Write unit tests for clean_column_name**

Create `tests/test_data.py`:

```python
def test_clean_column_name_brackets():
    from steel_flow.data import clean_column_name
    assert clean_column_name("time[s]") == "time_s_"


def test_clean_column_name_slash():
    from steel_flow.data import clean_column_name
    assert clean_column_name("AN_1_LL[m/s]") == "AN_1_LL_m_s_"


def test_clean_column_name_already_clean():
    from steel_flow.data import clean_column_name
    assert clean_column_name("already_clean") == "already_clean"


def test_clean_column_name_spaces():
    from steel_flow.data import clean_column_name
    assert clean_column_name("has space") == "has_space"
```

- [ ] **Step 4: Run tests to confirm they fail**

```bash
uv run pytest tests/ -v
```
Expected: `ModuleNotFoundError: No module named 'steel_flow'` — all tests fail. This is correct.

---

## Task 5: Create the package skeleton and data.py

**Files:**
- Create: `src/steel_flow/__init__.py`
- Create: `src/steel_flow/data.py`

- [ ] **Step 1: Create package init**

```bash
mkdir -p src/steel_flow
```

Create `src/steel_flow/__init__.py` with content:
```python
```
(empty file)

- [ ] **Step 2: Create src/steel_flow/data.py**

Create `src/steel_flow/data.py` with the following content (migrated and reorganized from `run_study.py`):

```python
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from sklearn.model_selection import train_test_split

console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console)],
)
logger = logging.getLogger(__name__)

SEN_GEOMETRY: Dict[str, Dict[str, int]] = {
    "10": {"Angle": -15, "Depth": 40},
    "09": {"Angle": 15, "Depth": 40},
    "08": {"Angle": 0, "Depth": 20},
    "07": {"Angle": -15, "Depth": 0},
    "06": {"Angle": 15, "Depth": 0},
}

CATEGORICAL_COLUMNS = ["SEN", "Angle", "Depth", "waterflow", "airflow", "CF", "mould_pos"]

FLOAT_COLUMNS = [
    "time[s]", "AN_1_LL[m/s]", "AN_2_LQ[m/s]", "AN_3_RQ[m/s]", "AN_4_RR[m/s]",
    "ML_LL[mm]", "ML_LQ[mm]", "ML_RQ[mm]", "ML_RR[mm]", "L_wave_ht[mm]", "R_wave_ht[mm]",
]


def clean_column_name(name: str) -> str:
    """Replace non-alphanumeric characters with underscores."""
    return re.sub(r"[^a-zA-Z0-9_]", "_", name)


def generate_csv_files(
    data_directory: str,
    preprocess_details: str,
    sen_geometrical: bool = False,
    clogging_factors: bool = False,
    mould_position: bool = False,
    remove_rows: int = 0,
) -> None:
    """Generate X<suffix>.csv and y<suffix>.csv from Excel files in data_directory."""
    tabular_root = Path(data_directory)
    if not tabular_root.exists():
        logger.error(f"Data directory '{data_directory}' does not exist.")
        return

    logger.info(f"Generating X{preprocess_details}.csv and y{preprocess_details}.csv from raw data.")

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
        task = progress.add_task("[green]Processing Excel files...", total=len(excel_files))

        for file in excel_files:
            try:
                excel_sheets = pd.read_excel(file, sheet_name=None)
            except Exception as e:
                logger.error(f"Failed to read {file}: {e}")
                progress.update(task, advance=1)
                continue

            for sheet_name, df in excel_sheets.items():
                if "Target" not in df.columns:
                    logger.warning(f"'Target' column not found in sheet {sheet_name} of {file}. Skipping.")
                    continue

                df.dropna(subset=["Target"], inplace=True)
                if df.empty:
                    logger.warning(f"No valid rows after dropping NA 'Target' in {sheet_name} of {file}.")
                    continue

                if remove_rows != 0:
                    df = df[:-remove_rows]
                if df.empty:
                    logger.warning(f"No valid rows after dropping {remove_rows} rows in {sheet_name} of {file}.")
                    continue

                sheet_components = sheet_name.split("_")
                if len(sheet_components) < 4:
                    logger.warning(f"Sheet name format unexpected: {sheet_name}. Skipping.")
                    continue

                sheet_is_clogged = False
                if clogging_factors:
                    sheet_is_clogged = len(sheet_components) == 5

                if sen_geometrical:
                    sen_key = sheet_components[0][3:5]
                    angle = SEN_GEOMETRY[sen_key]["Angle"]
                    depth = SEN_GEOMETRY[sen_key]["Depth"]

                cf = "0"
                if clogging_factors and sheet_is_clogged:
                    cf = sheet_components[1]

                for _, row in df.iterrows():
                    try:
                        feature_dict = {
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
                            feature_dict["Angle"] = angle
                            feature_dict["Depth"] = depth
                        else:
                            feature_dict["SEN"] = sheet_components[0]

                        if clogging_factors and sheet_is_clogged:
                            feature_dict["CF"] = cf
                            feature_dict["waterflow"] = sheet_components[2]
                            feature_dict["airflow"] = sheet_components[3]
                            if mould_position:
                                feature_dict["mould_pos"] = sheet_components[4]
                        else:
                            feature_dict["waterflow"] = sheet_components[1]
                            feature_dict["airflow"] = sheet_components[2]
                            if mould_position:
                                feature_dict["mould_pos"] = sheet_components[3]
                            if clogging_factors:
                                feature_dict["CF"] = cf

                        feature_records.append(feature_dict)
                        target_records.append({
                            "label": sheet_name,
                            "time[s]": row.iloc[0],
                            "Count_EX1": row.iloc[24],
                            "Count_EX2": row.iloc[25],
                        })
                    except IndexError:
                        logger.error(f"Row indexing failed for sheet {sheet_name} in {file}.")
                        continue

            progress.update(task, advance=1)

    if feature_records and target_records:
        X_df = pd.DataFrame(feature_records)
        y_df = pd.DataFrame(target_records)
        X_df.to_csv(f"{data_directory}/X{preprocess_details}.csv", index=False)
        y_df.to_csv(f"{data_directory}/y{preprocess_details}.csv", index=False)
        logger.info(f"Successfully generated X{preprocess_details}.csv and y{preprocess_details}.csv.")
    else:
        logger.warning(f"No data extracted. CSVs not created.")


def load_data(
    target: str = "Count_EX1",
    onehot_encoding: bool = False,
    sen_geometrical: bool = False,
    clogging_factors: bool = False,
    discard_features: Optional[List[str]] = None,
    data_directory: str = "data/raw/Organized_Data",
    feature_lag: int = 0,
    lagged_features: Optional[List[str]] = None,
    mould_position: bool = False,
    remove_rows: int = 0,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], Dict[str, List]]:
    """
    Load and preprocess dataset for a given target variable.
    Returns (train_df, test_df, feature_names, onehot_values).
    """
    encoding_type = "OneHot" if onehot_encoding else "Categorical"
    preprocess_details = (
        f"{encoding_type}"
        f"{'_Geometrical' if sen_geometrical else ''}"
        f"{'_Clogging' if clogging_factors else ''}"
        f"{'_Mould' if mould_position else ''}"
        f"{f'_removed{remove_rows}' if remove_rows != 0 else ''}"
    )

    x_path = Path(f"{data_directory}/X{preprocess_details}.csv")
    y_path = Path(f"{data_directory}/y{preprocess_details}.csv")

    if not x_path.exists() or not y_path.exists():
        logger.info(f"Generating {x_path.name} and {y_path.name}")
        generate_csv_files(
            data_directory=data_directory,
            sen_geometrical=sen_geometrical,
            clogging_factors=clogging_factors,
            mould_position=mould_position,
            preprocess_details=preprocess_details,
            remove_rows=remove_rows,
        )

    if not x_path.exists() or not y_path.exists():
        raise FileNotFoundError(f"{x_path.name} or {y_path.name} not found after generation attempt.")

    logger.info(f"Loading data for target: {target}")
    X_data = pd.read_csv(x_path, low_memory=False)
    y_all = pd.read_csv(y_path, low_memory=False)

    if target not in y_all.columns:
        raise ValueError(f"Target column '{target}' does not exist in {y_path.name}")

    y_data = y_all[target]

    for cat_col in CATEGORICAL_COLUMNS:
        if cat_col in X_data.columns:
            X_data[cat_col] = X_data[cat_col].astype("category")

    onehot_values: Dict[str, List] = {}
    for cat_feature in CATEGORICAL_COLUMNS:
        if cat_feature in X_data.columns:
            onehot_values[cat_feature] = list(X_data[cat_feature].cat.categories)

    for col in FLOAT_COLUMNS:
        if col in X_data.columns:
            X_data[col] = X_data[col].astype("float", errors="ignore")

    if onehot_encoding:
        logger.info("Applying one-hot encoding.")
        for cat_feature in CATEGORICAL_COLUMNS:
            if cat_feature in X_data.columns:
                X_data = pd.get_dummies(X_data, columns=[cat_feature])

    X_data.columns = [clean_column_name(col) for col in X_data.columns]

    if discard_features:
        cleaned = [clean_column_name(f) for f in discard_features]
        logger.info(f"Discarding features: {cleaned}")
        X_data.drop(columns=cleaned, errors="ignore", inplace=True)

    if feature_lag != 0 and lagged_features:
        cleaned_lag = [clean_column_name(f) for f in lagged_features]
        logger.info(f"Lagging features: {cleaned_lag}")
        for feature in cleaned_lag:
            for lag_amount in range(1, feature_lag + 1):
                X_data[f"{feature}_lag{lag_amount}"] = X_data[feature].shift(lag_amount)
            X_data[feature] = X_data[feature].astype("float", errors="ignore")

    features = X_data.columns.tolist()

    combined_df = pd.concat([X_data, y_data], axis=1)
    combined_df.dropna(inplace=True)

    if combined_df.empty:
        raise ValueError("No valid data available after merging and dropping NAs.")

    train_df, test_df = train_test_split(combined_df, test_size=0.2, random_state=42)
    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df["set"] = "train"
    test_df["set"] = "test"

    full_df = pd.concat([train_df, test_df]).sort_index()
    combined_file = f"{data_directory}/combined_{preprocess_details}_{target}.csv"
    full_df.to_csv(combined_file, index=False)
    logger.info(f"Exported combined data to {combined_file}")
    logger.info(f"Data loaded: {train_df.shape[0]} train, {test_df.shape[0]} test samples.")

    return train_df, test_df, features, onehot_values
```

- [ ] **Step 3: Run import and unit tests**

```bash
uv run pytest tests/test_imports.py::test_data_module_importable tests/test_data.py -v
```
Expected: `test_data_module_importable` PASSES, all `test_data.py` tests PASS. Other import tests still fail.

- [ ] **Step 4: Commit**

```bash
git add src/ tests/
git commit -m "feat: add steel_flow package skeleton and data.py"
```

---

## Task 6: Implement tune.py

**Files:**
- Create: `src/steel_flow/tune.py`

- [ ] **Step 1: Create src/steel_flow/tune.py**

```python
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
```

- [ ] **Step 2: Run tune import test**

```bash
uv run pytest tests/test_imports.py::test_tune_module_importable -v
```
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add src/steel_flow/tune.py
git commit -m "feat: add tune.py — Optuna hyperparameter search"
```

---

## Task 7: Implement train.py and evaluate.py

**Files:**
- Create: `src/steel_flow/train.py`
- Create: `src/steel_flow/evaluate.py`

- [ ] **Step 1: Create src/steel_flow/train.py**

```python
import json
import logging
from pathlib import Path
from typing import Dict, List

import pandas as pd
import xgboost as xgb
from rich.console import Console
from rich.logging import RichHandler

console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console)],
)
logger = logging.getLogger(__name__)


def train_model(
    train_df: pd.DataFrame,
    features: List[str],
    target: str,
    study_name: str,
    best_params: Dict,
    onehot_encoding: bool = False,
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
        logger.info(f"Training model for {study_name}")
        model = xgb.XGBRegressor(**best_params, enable_categorical=not onehot_encoding)
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
```

- [ ] **Step 2: Create src/steel_flow/evaluate.py**

```python
import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import shap
import xgboost as xgb
from rich.console import Console
from rich.logging import RichHandler
from sklearn.metrics import (
    PredictionErrorDisplay,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
    root_mean_squared_error,
)

console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console)],
)
logger = logging.getLogger(__name__)


def compute_metrics(
    model: xgb.XGBRegressor,
    X: pd.DataFrame,
    y: pd.Series,
) -> dict:
    """
    Compute regression metrics for model predictions on (X, y).
    Returns dict with keys: mae, mape, r2, mse, rmse.
    """
    predictions = model.predict(X)
    errors = abs(y - predictions)
    metrics = {
        "mean_absolute_error": float(errors.mean()),
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
        shap_values = explainer(X_test[: len(X_test) // 10])
    else:
        explainer = shap.Explainer(model, X_train.astype("float64"))
        shap_values = explainer(X_test)

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
```

- [ ] **Step 3: Run train and evaluate import tests**

```bash
uv run pytest tests/test_imports.py::test_train_module_importable tests/test_imports.py::test_evaluate_module_importable -v
```
Expected: Both PASS.

- [ ] **Step 4: Commit**

```bash
git add src/steel_flow/train.py src/steel_flow/evaluate.py
git commit -m "feat: add train.py and evaluate.py"
```

---

## Task 8: Implement predict.py

**Files:**
- Create: `src/steel_flow/predict.py`

- [ ] **Step 1: Create src/steel_flow/predict.py**

Migrate `predict_on_sheet` and `process_excel_file` from the old `predict.py`, updating the import of `clean_column_name` to come from `steel_flow.data`:

```python
import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import xgboost as xgb
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskID, TextColumn, TimeElapsedColumn, TimeRemainingColumn

from steel_flow.data import clean_column_name

console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console)],
)
logger = logging.getLogger(__name__)

SEN_GEOMETRY_STR: Dict[str, Dict[str, str]] = {
    "10": {"Angle": "-15", "Depth": "40"},
    "09": {"Angle": "15", "Depth": "40"},
    "08": {"Angle": "0", "Depth": "20"},
    "07": {"Angle": "-15", "Depth": "0"},
    "06": {"Angle": "15", "Depth": "0"},
}

FLOAT_COLUMNS = [
    "time[s]", "AN_1_LL[m/s]", "AN_2_LQ[m/s]", "AN_3_RQ[m/s]", "AN_4_RR[m/s]",
    "ML_LL[mm]", "ML_LQ[mm]", "ML_RQ[mm]", "ML_RR[mm]", "L_wave_ht[mm]", "R_wave_ht[mm]",
]

CATEGORICAL_FEATURES = ["SEN", "Angle", "Depth", "waterflow", "airflow", "CF", "mould_pos"]


def predict_on_sheet(
    df: pd.DataFrame,
    file_name: str,
    sheet_name: str,
    model: xgb.XGBRegressor,
    model_name: str,
    onehot_encoding: bool,
    sen_geometrical: bool,
    clogging_factors: bool,
    drop_cols: List[str],
    feature_lag: int,
    lagged_features: List[str],
    all_sheets: list,
    onehot_values: Optional[Dict] = None,
    mould_position: bool = False,
    piv_data: bool = False,
) -> pd.DataFrame:
    """
    Add a prediction column to a single sheet DataFrame.
    Returns updated DataFrame with model_name column appended.
    """
    if not piv_data:
        sheet_components = sheet_name.split("_")
        sheet_is_clogged = False
        if clogging_factors:
            sheet_is_clogged = len(sheet_components) == 5
    else:
        sheet_components = list(all_sheets)[1].split("_")
        sheet_is_clogged = False
        if clogging_factors and "Clog" in sheet_components[1]:
            sheet_is_clogged = True
            sheet_components[2] = sheet_components[2][:3]
            sheet_components[3] = sheet_components[3][:1]
        sheet_components[2] = sheet_components[2][:3]
        sheet_components[3] = sheet_components[3][:1]

    if len(sheet_components) < 4:
        file_name_components = file_name.split("_")
        if len(file_name_components) < 4:
            logger.error(f"Cannot extract components from '{file_name}' or '{sheet_name}'.")
            return df
        sheet_components = [
            file_name_components[1],
            file_name_components[3].replace("mpm", "").replace(".", "_"),
            file_name_components[4].replace("LPM", ""),
        ]

    expected_cols = list(FLOAT_COLUMNS)
    nice_df = df.copy()

    if feature_lag != 0:
        for feature in lagged_features:
            for lag_amount in range(1, feature_lag + 1):
                df[f"{feature}_lag{lag_amount}"] = df[feature].shift(lag_amount)
                expected_cols.append(f"{feature}_lag{lag_amount}")

    actual_cols = {}
    for col in expected_cols:
        matches = [c for c in df.columns if str(c).strip().lower() == col.lower()]
        if not matches:
            logger.warning(f"Expected column '{col}' not found in sheet '{sheet_name}'.")
            logger.warning(f"Available columns: {df.columns.tolist()}")
            return df
        actual_cols[col] = matches[0]

    predictions = []
    piv_preds = {"0-1": [], "1-2": [], "2-3": [], "3-4": []}

    for idx, row in df.iterrows():
        try:
            if sen_geometrical:
                sen_key = sheet_components[0][3:5]
                angle = SEN_GEOMETRY_STR[sen_key]["Angle"]
                depth = SEN_GEOMETRY_STR[sen_key]["Depth"]

            cf = "0"
            if clogging_factors and sheet_is_clogged:
                cf = sheet_components[1]

            features_dict: Dict = {}
            if sen_geometrical:
                features_dict["Angle"] = angle
                features_dict["Depth"] = depth
            else:
                features_dict["SEN"] = sheet_components[0]

            if clogging_factors and sheet_is_clogged:
                features_dict["CF"] = cf
                features_dict["waterflow"] = sheet_components[2]
                features_dict["airflow"] = sheet_components[3]
                if mould_position and not piv_data:
                    features_dict["mould_pos"] = sheet_components[4]
            else:
                features_dict["waterflow"] = sheet_components[1]
                features_dict["airflow"] = sheet_components[2]
                if mould_position and not piv_data:
                    features_dict["mould_pos"] = sheet_components[3]
                if clogging_factors:
                    features_dict["CF"] = cf

            for col in expected_cols:
                features_dict[col] = row[actual_cols[col]]

        except Exception as e:
            logger.error(f"Error processing row {idx} in sheet '{sheet_name}': {e}")
            predictions.append(None)
            continue

        mld_iterations = ["0-1", "1-2", "2-3", "3-4"] if piv_data else [features_dict.get("mould_pos", "0-1")]
        sum_preds = []

        for mld_iter in mld_iterations:
            features_dict["mould_pos"] = mld_iter
            features_df = pd.DataFrame([features_dict])
            features_df.columns = [clean_column_name(col) for col in features_df.columns]

            clean_float_cols = [clean_column_name(col) for col in FLOAT_COLUMNS]
            for col in clean_float_cols:
                if col in features_df.columns:
                    features_df[col] = pd.to_numeric(features_df[col], errors="coerce")

            if onehot_encoding and onehot_values:
                for cat_feature in CATEGORICAL_FEATURES:
                    if cat_feature in features_df.columns:
                        cat_val = features_df.at[0, cat_feature]
                        for possible_val in onehot_values.get(cat_feature, []):
                            dummy_col = clean_column_name(f"{cat_feature}_{possible_val}")
                            features_df[dummy_col] = 1 if cat_val == str(possible_val) else 0
                        features_df.drop(columns=[cat_feature], inplace=True)

            for col_to_drop in drop_cols:
                for column in list(features_df.columns):
                    if clean_column_name(col_to_drop) == column:
                        features_df.drop(columns=[column], inplace=True)

            correct_cols = model.get_booster().feature_names
            features_df = features_df[correct_cols]

            try:
                pred = model.predict(features_df)[0]
            except Exception as e:
                logger.error(f"Prediction error in row {idx} of sheet '{sheet_name}': {e}")
                pred = None

            if piv_data:
                sum_preds.append(pred)
                piv_preds[mld_iter].append(pred)
            else:
                predictions.append(pred)

        if piv_data:
            predictions.append(sum(sum_preds))

    nice_df[model_name] = predictions
    if piv_data:
        for mld_iter in ["0-1", "1-2", "2-3", "3-4"]:
            nice_df[f"{model_name} mld {mld_iter}"] = piv_preds[mld_iter]
    return nice_df


def process_excel_file(
    file_path: Path,
    model: xgb.XGBRegressor,
    model_name: str,
    onehot_encoding: bool,
    sen_geometrical: bool,
    clogging_factors: bool,
    drop_cols: List[str],
    feature_lag: int,
    lagged_features: List[str],
    onehot_values: Optional[Dict] = None,
    mould_position: bool = False,
    progress: Optional[Progress] = None,
    progress_task: Optional[TaskID] = None,
    piv_data: bool = False,
) -> None:
    """Process all sheets in an Excel file, writing predictions back to the file."""
    logger.info(f"Processing file: {file_path}")
    try:
        sheets = pd.read_excel(file_path, sheet_name=None)
    except Exception as e:
        logger.error(f"Error reading {file_path}: {e}")
        return

    file_name = file_path.stem
    updated_sheets = {}
    all_sheets = sheets.keys()

    for sheet_name, df in sheets.items():
        updated_df = predict_on_sheet(
            df, file_name, sheet_name, model, model_name,
            onehot_encoding, sen_geometrical, clogging_factors,
            drop_cols, feature_lag, lagged_features, all_sheets,
            onehot_values, mould_position, piv_data,
        )
        updated_sheets[sheet_name] = updated_df
        if progress is not None and progress_task is not None:
            progress.update(progress_task, advance=1)

    try:
        with pd.ExcelWriter(file_path, engine="openpyxl", mode="w") as writer:
            for sheet_name, updated_df in updated_sheets.items():
                updated_df.to_excel(writer, sheet_name=sheet_name, index=False)
        logger.info(f"Updated file saved: {file_path}")
    except Exception as e:
        logger.error(f"Error saving updated file {file_path}: {e}")
```

- [ ] **Step 2: Run predict import test**

```bash
uv run pytest tests/test_imports.py::test_predict_module_importable -v
```
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add src/steel_flow/predict.py
git commit -m "feat: add predict.py — sheet-level prediction logic"
```

---

## Task 9: Implement cli.py

**Files:**
- Create: `src/steel_flow/cli.py`

- [ ] **Step 1: Create src/steel_flow/cli.py**

```python
import argparse
import json
import logging
from pathlib import Path

import optuna
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
    parser.add_argument("--study_count", type=int, default=1, help="Number of Optuna trials")
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


def tune() -> None:
    """Entry point for `uv run tune`. Runs Optuna hyperparameter search."""
    parser = argparse.ArgumentParser(description="Run Optuna hyperparameter search for XGBoost")
    _common_args(parser)
    parser.add_argument("--subsample-shap", action="store_true", default=False)
    args = parser.parse_args()

    discard_features = [f.strip() for f in args.discard_features.split(",") if f.strip()]
    lagged_features = [f.strip() for f in args.lagged_features.split(",") if f.strip()]

    for target in args.targets:
        encoding_type = "OneHot" if args.onehot_encoding else "Categorical"
        full_study_name = (
            f"{args.study_name}_{target}_{encoding_type}"
            f"{'_Geometrical' if args.sen_geometrical else ''}"
            f"{'_Clogging' if args.clogging_factors else ''}"
            f"_{args.feature_lag_amount}Lag"
            f"{'_Mould' if args.mould_position else ''}"
        )
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
    parser.add_argument("--subsample-shap", action="store_true", default=False)
    parser.add_argument("--config-file", type=str, default=None,
                        help="Output filename for training config JSON (default: training_config_<study-name>.json)")
    args = parser.parse_args()

    if args.config_file is None:
        args.config_file = f"training_config_{args.study_name}.json"

    discard_features = [f.strip() for f in args.discard_features.split(",") if f.strip()]
    lagged_features = [f.strip() for f in args.lagged_features.split(",") if f.strip()]
    training_configs = []

    for target in args.targets:
        encoding_type = "OneHot" if args.onehot_encoding else "Categorical"
        full_study_name = (
            f"{args.study_name}_{target}_{encoding_type}"
            f"{'_Geometrical' if args.sen_geometrical else ''}"
            f"{'_Clogging' if args.clogging_factors else ''}"
            f"_{args.feature_lag_amount}Lag"
            f"{'_Mould' if args.mould_position else ''}"
        )
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
                        help="Path to training config JSON produced by `uv run train`")
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
```

- [ ] **Step 2: Run the full test suite**

```bash
uv run pytest tests/ -v
```
Expected: All tests PASS (imports + clean_column_name unit tests).

- [ ] **Step 3: Smoke-test the entry points parse args correctly**

```bash
uv run tune --help
uv run train --help
uv run predict --help
```
Expected: Each prints its argparse help text without errors.

- [ ] **Step 4: Commit**

```bash
git add src/steel_flow/cli.py
git commit -m "feat: add cli.py — tune, train, predict entry points"
```

---

## Task 10: Delete old CLI files and update shell scripts

**Files:**
- Delete: `run_study.py`, `predict.py`, `get_split.py`
- Modify: `scripts/local/start_study.sh`, `scripts/local/start_predict.sh`

- [ ] **Step 1: Delete old root-level scripts**

```bash
git rm run_study.py predict.py get_split.py
```

- [ ] **Step 2: Update scripts/local/start_study.sh**

Replace the contents of `scripts/local/start_study.sh` with:

```bash
uv run tune --targets Count_EX1 Count_EX2 \
--study-name cftest0812-3 \
--study_count 100 \
--onehot-encoding \
--sen-geometrical \
--discard-features "time[s]" \
--data-directory data/raw/Organized_Data \
--tree-method gpu_hist \
--lagged-features "AN_1_LL[m/s],AN_2_LQ[m/s],AN_3_RQ[m/s],AN_4_RR[m/s],ML_LL[mm],ML_LQ[mm],ML_RQ[mm],ML_RR[mm]" \
--feature-lag-amount 10 \
--mould-position \
--clogging-factors
```

- [ ] **Step 3: Update scripts/local/start_predict.sh**

Open `scripts/local/start_predict.sh`. Replace any `uv run predict.py` calls with `uv run predict`, and update `--config-file` paths to point at `config/` directory.

Example (adapt to actual content):
```bash
uv run predict --config-file config/training_config_20250328.json \
--data-directory New_Data
```

- [ ] **Step 4: Update HPC scripts that reference run_study.py**

In each of `scripts/hpc/study_job.sh`, `scripts/hpc/multithread_study_train.sh`, `scripts/hpc/multithread_study_eval.sh`, replace `python run_study.py` or `uv run run_study.py` with the appropriate `uv run tune` or `uv run train` invocation. Replace `python predict.py` or `uv run predict.py` with `uv run predict`.

- [ ] **Step 5: Verify no references to old scripts remain**

```bash
grep -r "run_study.py\|predict.py\|get_split.py" scripts/ --include="*.sh"
```
Expected: No matches.

- [ ] **Step 6: Run full test suite one more time**

```bash
uv run pytest tests/ -v
```
Expected: All PASS.

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "chore: delete old root-level scripts, update shell scripts to use uv run entry points"
```

---

## Task 11: Update README and final verification

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update the Installation section in README.md**

Replace the **Installation & Requirements** section with:

```markdown
## Installation & Requirements

- **Python**: 3.10+
- **Package manager**: [uv](https://docs.astral.sh/uv/)

```bash
git clone https://github.com/alexwolson/steel_flow_sensor_prediction.git
cd steel_flow_sensor_prediction
uv sync
```
```

- [ ] **Step 2: Update the Usage section in README.md**

Replace the **Usage** section with:

```markdown
## Usage

### 1. Data Preparation

Place raw `.xlsx` files under `data/raw/Organized_Data/` in the appropriate subdirectory structure. The CSVs will be generated automatically on first run.

### 2. Hyperparameter Tuning

```bash
uv run tune --targets Count_EX1 Count_EX2 \
  --study-name MyStudy \
  --study_count 100 \
  --onehot-encoding \
  --sen-geometrical \
  --discard-features "time[s],SEN,L_wave_ht[mm]" \
  --data-directory data/raw/Organized_Data \
  --subsample-shap
```

### 3. Train Final Model

```bash
uv run train --targets Count_EX1 Count_EX2 \
  --study-name MyStudy \
  --onehot-encoding \
  --sen-geometrical \
  --discard-features "time[s],SEN,L_wave_ht[mm]" \
  --data-directory data/raw/Organized_Data \
  --subsample-shap
```

This saves the model to `data/output/models/` and the training config to `config/`.

### 4. Predict on New Data

```bash
uv run predict --config-file config/training_config_MyStudy.json \
  --data-directory New_Data
```
```

- [ ] **Step 3: Update the directory structure section in README.md**

Replace the directory description to match the new layout from the spec.

- [ ] **Step 4: Verify root directory is clean**

```bash
ls
```
Expected: `config/  data/  docs/  pyproject.toml  README.md  scripts/  src/  uv.lock`

No Python files, no shell scripts, no txt files at root level.

- [ ] **Step 5: Run final test suite**

```bash
uv run pytest tests/ -v
```
Expected: All PASS.

- [ ] **Step 6: Commit**

```bash
git add README.md
git commit -m "docs: update README for new entry points and directory layout"
```

---

## Success Checklist

- [ ] `uv run tune --help` prints usage without errors
- [ ] `uv run train --help` prints usage without errors
- [ ] `uv run predict --help` prints usage without errors
- [ ] `uv sync` installs all deps from `pyproject.toml` alone (no `requirements.txt` needed)
- [ ] `uv run pytest tests/ -v` — all tests pass
- [ ] Root directory contains only: `config/`, `data/`, `docs/`, `pyproject.toml`, `README.md`, `scripts/`, `src/`, `uv.lock`
- [ ] No business logic lives in root-level Python files
- [ ] `git log --oneline` shows 8 clean commits on `cleanup/full-restructure`
