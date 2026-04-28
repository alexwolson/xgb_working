# Methodology Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix three critical data leakage and temporal integrity issues in the XGBoost prediction pipeline, plus associated model search-space and evaluation improvements, as documented in `docs/methodology-review-2026-04-16.md`.

**Architecture:** All changes are confined to existing modules — no new files. The data pipeline gets a `label` column (sheet identifier) threaded through from CSV generation to splitting, enabling per-sheet lag computation and sheet-level train/val/test partitioning. The `run_study` objective function switches from `test_df` to a new `val_df` so the held-out test set is never seen during hyperparameter selection. Downstream changes in `train.py`, `cli.py`, and `evaluate.py` are mechanical updates to match the new signatures.

**Tech Stack:** Python 3.10+, XGBoost, Optuna, pandas, scikit-learn, seaborn, matplotlib, shap, numpy, pytest, uv

---

## File Map

| File | Change |
|---|---|
| `src/steel_flow/data.py` | Add `label` to feature CSV; per-sheet lags via `groupby`; new `_sheet_split` helper; `load_data` returns `(train_df, val_df, test_df, features, onehot_values)` |
| `src/steel_flow/tune.py` | Accept `val_df` instead of `test_df`; seeded sampler; add `subsample`/`colsample_bytree` to search space; accept `objective` param |
| `src/steel_flow/train.py` | Accept `objective` param and pass to `XGBRegressor` |
| `src/steel_flow/cli.py` | Unpack three-way split; add `--objective` flag; pass `val_df` to `run_study`; report val metrics separately |
| `src/steel_flow/evaluate.py` | Log zero-count MAPE coverage; add signed-error panel; SHAP random subsampling |
| `tests/test_data.py` | New tests for `_sheet_split` and per-sheet lag isolation |

---

## Task 1: Add `label` column to feature CSV and fix per-sheet lag computation

**Files:**
- Modify: `src/steel_flow/data.py`
- Modify: `tests/test_data.py`

### Background

`generate_csv_files` currently creates feature records without any sheet identifier. When lags are applied later in `load_data`, the `shift()` operation crosses sheet boundaries — the lag at the first row of Sheet B is filled from Sheet A's last row, which is physically meaningless (they are unrelated experiments).

This task:
1. Adds `"label": sheet_name` to each feature record so the CSV carries a sheet identifier.
2. Bumps the preprocessing format to `v2_*` so existing cached CSVs are bypassed and regenerated automatically.
3. Rewrites the lag computation in `load_data` to use `groupby("label")` so lags reset at every sheet boundary.

---

- [ ] **Step 1: Write failing tests for per-sheet lag isolation**

Add to `tests/test_data.py`:

```python
import pandas as pd
import numpy as np


class TestPerSheetLag:
    def test_lag_does_not_cross_sheet_boundary(self):
        """First row of sheet_B must have NaN lag, not sheet_A's last value."""
        df = pd.DataFrame({
            "label": ["sheet_A", "sheet_A", "sheet_A",
                       "sheet_B", "sheet_B", "sheet_B"],
            "feature": [10.0, 20.0, 30.0, 100.0, 200.0, 300.0],
        })

        def apply_lag(group: pd.DataFrame) -> pd.DataFrame:
            group = group.copy()
            group["feature_lag1"] = group["feature"].shift(1)
            return group

        result = df.groupby("label", group_keys=False).apply(apply_lag)
        sheet_b = result[result["label"] == "sheet_B"].reset_index(drop=True)
        assert pd.isna(sheet_b.loc[0, "feature_lag1"]), (
            "Expected NaN at first row of sheet_B; got "
            f"{sheet_b.loc[0, 'feature_lag1']} (leaked from sheet_A)"
        )

    def test_lag_values_correct_within_sheet(self):
        """Lag-1 value at row i should equal the feature value at row i-1."""
        df = pd.DataFrame({
            "label": ["sheet_A", "sheet_A", "sheet_A"],
            "feature": [10.0, 20.0, 30.0],
        })

        def apply_lag(group: pd.DataFrame) -> pd.DataFrame:
            group = group.copy()
            group["feature_lag1"] = group["feature"].shift(1)
            return group

        result = df.groupby("label", group_keys=False).apply(apply_lag).reset_index(drop=True)
        assert pd.isna(result.loc[0, "feature_lag1"])
        assert result.loc[1, "feature_lag1"] == 10.0
        assert result.loc[2, "feature_lag1"] == 20.0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_data.py::TestPerSheetLag -v
```

Expected: both PASS already (they test pure pandas — the groupby logic is what we're about to add to `load_data`, and these tests validate the approach directly). If they pass here, the concept is correct; the regression prevention happens in the integration.

- [ ] **Step 3: Add `label` to feature records in `generate_csv_files`**

In `src/steel_flow/data.py` at line ~114, inside the `for _, row in df.iterrows()` loop:

```python
# BEFORE:
feature_dict = {
    "time[s]": row.iloc[0],
    "AN_1_LL[m/s]": row.iloc[9],

# AFTER:
feature_dict = {
    "label": sheet_name,
    "time[s]": row.iloc[0],
    "AN_1_LL[m/s]": row.iloc[9],
```

- [ ] **Step 4: Bump format version in `preprocess_details` to invalidate old cached CSVs**

In `src/steel_flow/data.py` at line ~190, inside `load_data`:

```python
# BEFORE:
preprocess_details = (
    f"{encoding_type}"
    f"{'_Geometrical' if sen_geometrical else ''}"
    f"{'_Clogging' if clogging_factors else ''}"
    f"{'_Mould' if mould_position else ''}"
    f"{f'_removed{remove_rows}' if remove_rows != 0 else ''}"
)

# AFTER:
preprocess_details = (
    f"v2_{encoding_type}"
    f"{'_Geometrical' if sen_geometrical else ''}"
    f"{'_Clogging' if clogging_factors else ''}"
    f"{'_Mould' if mould_position else ''}"
    f"{f'_removed{remove_rows}' if remove_rows != 0 else ''}"
)
```

- [ ] **Step 5: Replace the lag computation block in `load_data` with per-sheet groupby**

In `src/steel_flow/data.py`, replace lines ~251–257:

```python
# BEFORE:
if feature_lag != 0 and lagged_features:
    cleaned_lag = [clean_column_name(f) for f in lagged_features]
    logger.info(f"Lagging features: {cleaned_lag}")
    for feature in cleaned_lag:
        for lag_amount in range(1, feature_lag + 1):
            X_data[f"{feature}_lag{lag_amount}"] = X_data[feature].shift(lag_amount)
        X_data[feature] = X_data[feature].astype("float", errors="ignore")

# AFTER:
if feature_lag != 0 and lagged_features:
    cleaned_lag = [clean_column_name(f) for f in lagged_features]
    logger.info(f"Lagging features per sheet: {cleaned_lag}")

    def _apply_sheet_lags(group: pd.DataFrame) -> pd.DataFrame:
        group = group.copy()
        for feature in cleaned_lag:
            for lag_amount in range(1, feature_lag + 1):
                group[f"{feature}_lag{lag_amount}"] = group[feature].shift(lag_amount)
            group[feature] = group[feature].astype("float", errors="ignore")
        return group

    X_data = X_data.groupby("label", group_keys=False).apply(_apply_sheet_lags)
```

- [ ] **Step 6: Exclude `label` from the features list**

In `src/steel_flow/data.py` at line ~259, change:

```python
# BEFORE:
features = X_data.columns.tolist()

# AFTER:
features = [c for c in X_data.columns.tolist() if c != "label"]
```

- [ ] **Step 7: Run all tests**

```bash
uv run pytest tests/ -v
```

Expected: 10 passed (existing tests unaffected; lag tests pass).

- [ ] **Step 8: Commit**

```bash
git add src/steel_flow/data.py tests/test_data.py
git commit -m "fix: add label column to feature CSV; compute lags per sheet to prevent cross-sheet contamination"
```

---

## Task 2: Sheet-level train / validation / test split

**Files:**
- Modify: `src/steel_flow/data.py`
- Modify: `tests/test_data.py`

### Background

The current split uses `sklearn.model_selection.train_test_split` with `shuffle=True`, which randomly assigns rows to train and test ignoring temporal order. Because the data consists of independent experimental runs (sheets), the correct unit of splitting is the sheet: entire runs go to one partition. This task replaces the two-way random split with a three-way sheet-level split, returning `(train_df, val_df, test_df)`. The validation set is used by Optuna (Task 3); the test set remains held out until final evaluation (Task 5).

The split is alphabetical by sheet name for reproducibility. Approximately 64 % train / 16 % val / 20 % test.

---

- [ ] **Step 1: Write failing tests for `_sheet_split`**

Add to `tests/test_data.py`:

```python
from steel_flow.data import _sheet_split


class TestSheetSplit:
    def _make_df(self, sheets, rows_per_sheet=10):
        return pd.DataFrame({
            "label": sum(([s] * rows_per_sheet for s in sheets), []),
            "x": range(len(sheets) * rows_per_sheet),
        })

    def test_no_overlap_between_partitions(self):
        df = self._make_df(["A", "B", "C", "D", "E"])
        train, val, test = _sheet_split(df)
        assert not (set(train["label"]) & set(val["label"])), "train/val overlap"
        assert not (set(train["label"]) & set(test["label"])), "train/test overlap"
        assert not (set(val["label"]) & set(test["label"])), "val/test overlap"

    def test_all_sheets_assigned(self):
        sheets = ["A", "B", "C", "D", "E"]
        df = self._make_df(sheets)
        train, val, test = _sheet_split(df)
        assigned = set(train["label"]) | set(val["label"]) | set(test["label"])
        assert assigned == set(sheets)

    def test_raises_with_fewer_than_three_sheets(self):
        df = self._make_df(["A", "B"])
        with pytest.raises(ValueError, match="at least 3"):
            _sheet_split(df)

    def test_test_set_contains_last_sheets_alphabetically(self):
        """With 5 sheets A–E, n_test=1, test should be {E}."""
        df = self._make_df(["C", "A", "E", "B", "D"])  # unsorted input
        _, _, test = _sheet_split(df, test_frac=0.2)
        assert set(test["label"]) == {"E"}
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_data.py::TestSheetSplit -v
```

Expected: ImportError — `_sheet_split` does not exist yet.

- [ ] **Step 3: Add `_sheet_split` helper above `load_data` in `data.py`**

Add after the `clean_column_name` function (around line 32) and before `generate_csv_files`:

```python
def _sheet_split(
    df: pd.DataFrame,
    test_frac: float = 0.2,
    val_frac: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split *df* into (train, val, test) partitions at the sheet level.

    Sheets are sorted alphabetically for reproducibility. The last ``test_frac``
    of sheets become test; the preceding ``val_frac`` of the remainder become
    validation; the rest are training.  Raises ``ValueError`` if fewer than 3
    distinct sheets are present.
    """
    unique_sheets = sorted(df["label"].unique())
    n = len(unique_sheets)
    if n < 3:
        raise ValueError(
            f"Need at least 3 distinct sheets for a train/val/test split, found {n}."
        )
    n_test = max(1, round(n * test_frac))
    n_val = max(1, round((n - n_test) * val_frac))
    if n - n_test - n_val < 1:
        raise ValueError(
            f"With {n} sheets, n_test={n_test} and n_val={n_val} leave no training data."
        )
    test_sheets = set(unique_sheets[-n_test:])
    val_sheets = set(unique_sheets[-(n_test + n_val):-n_test])
    train_sheets = set(unique_sheets[:-(n_test + n_val)])
    return (
        df[df["label"].isin(train_sheets)].copy(),
        df[df["label"].isin(val_sheets)].copy(),
        df[df["label"].isin(test_sheets)].copy(),
    )
```

- [ ] **Step 4: Run `TestSheetSplit` to verify tests now pass**

```bash
uv run pytest tests/test_data.py::TestSheetSplit -v
```

Expected: 4 passed.

- [ ] **Step 5: Replace the random split in `load_data` with `_sheet_split`**

In `src/steel_flow/data.py`, remove the `train_test_split` import (line ~8):

```python
# DELETE this line:
from sklearn.model_selection import train_test_split
```

Then replace the split block at line ~267:

```python
# BEFORE:
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

# AFTER:
train_df, val_df, test_df = _sheet_split(combined_df)
train_df["set"] = "train"
val_df["set"] = "val"
test_df["set"] = "test"

full_df = pd.concat([train_df, val_df, test_df]).sort_index()
combined_file = f"{data_directory}/combined_{preprocess_details}_{target}.csv"
full_df.to_csv(combined_file, index=False)
logger.info(f"Exported combined data to {combined_file}")
logger.info(
    f"Data loaded: {len(train_df)} train ({train_df['label'].nunique()} sheets), "
    f"{len(val_df)} val ({val_df['label'].nunique()} sheets), "
    f"{len(test_df)} test ({test_df['label'].nunique()} sheets)."
)

return train_df, val_df, test_df, features, onehot_values
```

- [ ] **Step 6: Update the `load_data` return type annotation**

Change the function signature docstring and return type hint at line ~182:

```python
# BEFORE:
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], Dict[str, List]]:
    """
    Load and preprocess dataset for a given target variable.

    Returns (train_df, test_df, feature_names, onehot_values).
    Both DataFrames contain feature columns, the target column, and a 'set' column.
    Callers should index features with train_df[features] and target with train_df[target].
    """

# AFTER:
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str], Dict[str, List]]:
    """
    Load and preprocess dataset for a given target variable.

    Returns (train_df, val_df, test_df, feature_names, onehot_values).
    All three DataFrames contain feature columns, the target column, and a 'set' column.
    Callers index features with df[features] and target with df[target].
    val_df is for HPO evaluation; test_df is the held-out evaluation set.
    """
```

- [ ] **Step 7: Run all tests**

```bash
uv run pytest tests/ -v
```

Expected: 12 passed (4 new sheet-split tests + 2 lag tests + 6 existing).

- [ ] **Step 8: Commit**

```bash
git add src/steel_flow/data.py tests/test_data.py
git commit -m "fix: replace random row-level split with sheet-level train/val/test partitioning"
```

---

## Task 3: Update `tune.py` — validation set, seeded sampler, expanded search space, objective

**Files:**
- Modify: `src/steel_flow/tune.py`

### Background

The Optuna objective currently evaluates on `test_df`, making the reported test metrics optimistic (HPO leakage). This task:
1. Replaces `test_df` with `val_df` in `run_study`.
2. Seeds the TPE sampler for reproducibility.
3. Adds `subsample` and `colsample_bytree` to the search space (they were fixed at 1.0).
4. Adds an `objective` parameter so callers can specify `count:poisson` or `reg:tweedie`.

Note: the `objective` function name conflicts with the XGBoost hyperparameter name. The inner function is renamed `objective_fn` to avoid shadowing.

---

- [ ] **Step 1: Replace the full `run_study` function in `src/steel_flow/tune.py`**

```python
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
        val_df: Validation data used for the Optuna objective. Never the test set.
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
```

- [ ] **Step 2: Run all tests**

```bash
uv run pytest tests/ -v
```

Expected: 12 passed (the module import test for `tune` still passes; the signature change is not tested by existing tests).

- [ ] **Step 3: Commit**

```bash
git add src/steel_flow/tune.py
git commit -m "fix: tune.py uses validation set for HPO objective; seed sampler; add subsample/colsample_bytree to search space; add objective param"
```

---

## Task 4: Update `train.py` — accept and apply `objective` parameter

**Files:**
- Modify: `src/steel_flow/train.py`

### Background

`train_model` currently passes `best_params` (from `study.best_trial.params`) directly to `XGBRegressor`. The best params contain only the `suggest_*` parameters from the Optuna trial — not fixed params like `objective`, which is set separately in the CLI. Without this change, the final trained model would always use `reg:squarederror` regardless of `--objective`.

---

- [ ] **Step 1: Update `train_model` in `src/steel_flow/train.py`**

```python
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
        model = xgb.XGBRegressor(
            **best_params,
            objective=objective,
            enable_categorical=not onehot_encoding,
        )
        model.fit(train_df[features], train_df[target])
        model.save_model(str(model_path))
        logger.info(f"Model saved to {model_path}")

    return model
```

- [ ] **Step 2: Run all tests**

```bash
uv run pytest tests/ -v
```

Expected: 12 passed.

- [ ] **Step 3: Commit**

```bash
git add src/steel_flow/train.py
git commit -m "fix: train_model accepts objective param and passes it to XGBRegressor"
```

---

## Task 5: Update `cli.py` — thread `val_df`, add `--objective`, report val metrics

**Files:**
- Modify: `src/steel_flow/cli.py`

### Background

`cli.py` is the coordinator. It calls `load_data`, which now returns three DataFrames, and calls `run_study`, which now expects `val_df`. This task updates both `tune()` and `train()` entry points, adds the `--objective` flag, and ensures `train()` reports validation and test metrics separately (with a clear label that test metrics are from the held-out set).

---

- [ ] **Step 1: Add `--objective` to `_common_args`**

In `src/steel_flow/cli.py`, inside `_common_args`, after the `--remove-rows` argument:

```python
parser.add_argument(
    "--objective",
    type=str,
    default="reg:squarederror",
    choices=["reg:squarederror", "count:poisson", "reg:tweedie"],
    help=(
        "XGBoost objective function. Default: reg:squarederror. "
        "Use count:poisson for non-negative integer count targets."
    ),
)
```

- [ ] **Step 2: Update `tune()` to unpack three DataFrames and pass `val_df` to `run_study`**

Replace the `tune()` body in `src/steel_flow/cli.py`:

```python
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

        train_df, val_df, test_df, features, _ = data_mod.load_data(
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
            val_df=val_df,
            features=features,
            target=target,
            study_name=full_study_name,
            study_count=args.study_count,
            onehot_encoding=args.onehot_encoding,
            tree_method=args.tree_method,
            storage_path=args.storage_path,
            multithread=args.multithread,
            seed=42,
            objective=args.objective,
        )

    logger.info("Tuning complete.")
```

- [ ] **Step 3: Update `train()` to unpack three DataFrames, pass `objective`, and report val/test separately**

Replace the `train()` body in `src/steel_flow/cli.py`:

```python
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

        train_df, val_df, test_df, features, onehot_values = data_mod.load_data(
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
            objective=args.objective,
        )

        X_train = train_df[features]
        y_train = train_df[target]
        X_val = val_df[features]
        y_val = val_df[target]
        X_test = test_df[features]
        y_test = test_df[target]

        logger.info("Evaluating on held-out test set (never seen during HPO).")
        test_metrics = eval_mod.compute_metrics(model, X_test, y_test)
        logger.info("Evaluating on validation set.")
        val_metrics = eval_mod.compute_metrics(model, X_val, y_val)
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
            "objective": args.objective,
            "study_name": full_study_name,
            "storage_path": args.storage_path,
            "multithread": args.multithread,
            "study_count": args.study_count,
            "feature_lag": args.feature_lag_amount,
            "lagged_features": lagged_features,
            "mould_position": args.mould_position,
            "remove_rows": args.remove_rows,
            "test": test_metrics,
            "val": val_metrics,
            "train": train_metrics,
        }
        if args.onehot_encoding:
            config["onehot_values"] = onehot_values
        training_configs.append(config)

    train_mod.save_training_config(training_configs, args.config_file)
    logger.info("Training complete.")
```

- [ ] **Step 4: Run all tests**

```bash
uv run pytest tests/ -v
```

Expected: 12 passed.

- [ ] **Step 5: Commit**

```bash
git add src/steel_flow/cli.py
git commit -m "fix: cli.py threads val_df through tune/train; adds --objective flag; reports val and test metrics separately"
```

---

## Task 6: Fix `evaluate.py` — MAPE zero warning, signed error histogram, SHAP random subsampling

**Files:**
- Modify: `src/steel_flow/evaluate.py`

### Background

Three independent improvements:

1. **MAPE zero coverage**: When true count values are 0, sklearn silently excludes them from MAPE. Log how many rows are excluded so users know when the MAPE figure is computed on a subset.

2. **Signed error histogram**: The existing histogram shows only absolute errors — systematic over- or under-prediction is invisible. Show a two-panel figure: absolute errors (existing) and signed errors `(true − predicted)` with a zero-line.

3. **SHAP random subsampling**: The existing subsampled path slices `X_test[:len//10]` (first N rows, not random). Replace with `.sample(frac=0.1, random_state=42)` and add the sample size to the plot title.

---

- [ ] **Step 1: Add `import numpy as np` to `evaluate.py`**

In `src/steel_flow/evaluate.py`, line 1:

```python
import logging
import numpy as np          # ADD THIS
from pathlib import Path
```

- [ ] **Step 2: Update `compute_metrics` to log zero-count MAPE coverage**

Replace `compute_metrics` in `src/steel_flow/evaluate.py`:

```python
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
```

- [ ] **Step 3: Update `plot_error_histogram` to show signed errors alongside absolute errors**

Replace `plot_error_histogram` in `src/steel_flow/evaluate.py`:

```python
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
    plt.close()
    logger.info(f"Error histogram saved to {figures_dir}/error_histogram_{study_name}.pdf")
```

- [ ] **Step 4: Update `plot_shap` to use random subsampling and add sample size to titles**

Replace `plot_shap` in `src/steel_flow/evaluate.py`:

```python
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
        background = X_train.sample(frac=0.1, random_state=42).astype("float64")
        explain_set = X_test.sample(frac=0.1, random_state=42).astype("float64")
    else:
        background = X_train.astype("float64")
        explain_set = X_test.astype("float64")

    n_explain = len(explain_set)
    n_test_total = len(X_test)
    sample_note = f"n={n_explain}" if not subsample_shap else f"n={n_explain} of {n_test_total}"

    explainer = shap.Explainer(model, background)
    shap_values = explainer(explain_set)

    plt.figure()
    shap.plots.beeswarm(shap_values, show=False, max_display=100)
    plt.title(f"SHAP Beeswarm — {study_name} ({sample_note})")
    plt.tight_layout()
    plt.subplots_adjust(left=0.3)
    plt.savefig(f"{figures_dir}/shap_{study_name}.pdf")
    plt.close()

    plt.figure()
    shap.plots.bar(shap_values, show=False, max_display=100)
    plt.title(f"SHAP Bar — {study_name} ({sample_note})")
    plt.tight_layout()
    plt.subplots_adjust(left=0.3)
    plt.savefig(f"{figures_dir}/shap_bar_{study_name}.pdf")
    plt.close()
    logger.info(f"SHAP plots saved to {figures_dir}/ ({sample_note})")
```

- [ ] **Step 5: Run all tests**

```bash
uv run pytest tests/ -v
```

Expected: 12 passed.

- [ ] **Step 6: Commit**

```bash
git add src/steel_flow/evaluate.py
git commit -m "fix: evaluate.py logs MAPE zero-count exclusions; two-panel error histogram with signed errors; SHAP uses random subsample"
```

---

## Self-Review

**Spec coverage check (against `docs/methodology-review-2026-04-16.md`):**

| Issue | Severity | Task |
|---|---|---|
| Test set used for HPO | Critical | Task 3 (val_df in objective) + Task 5 (cli.py) |
| Random split of time-series | Critical | Task 2 (sheet-level split) |
| Lag features cross sheet boundaries | Critical | Task 1 (per-sheet lags) |
| Fixed subsample/colsample_bytree | Important | Task 3 (added to search space) |
| Count data with MSE objective | Important | Task 3–5 (--objective flag) |
| MAPE undefined at zero | Important | Task 6 (compute_metrics warning) |
| Optuna not seeded | Minor | Task 3 (TPESampler seed) |
| SHAP subsampling non-random | Minor | Task 6 (sample()) |
| Signed error histogram | Minor | Task 6 (two-panel plot) |
| remove_rows documentation | Minor | Not in scope — text-only change, add to README separately |

All 9 code-level issues covered. `remove_rows` documentation is a README edit deferred to a separate commit.

**Placeholder scan:** No TBD, TODO, or incomplete steps found.

**Type consistency check:**
- `_sheet_split` returns `Tuple[DataFrame, DataFrame, DataFrame]` — used consistently as `train_df, val_df, test_df = _sheet_split(...)` in Tasks 2 and 5.
- `load_data` return type updated to `Tuple[DataFrame, DataFrame, DataFrame, List[str], Dict]` — unpacked as `train_df, val_df, test_df, features, onehot_values` in Tasks 5 (tune and train).
- `run_study` gains `val_df` param — caller in Task 5 passes `val_df=val_df`. ✓
- `train_model` gains `objective` param — caller in Task 5 passes `objective=args.objective`. ✓
- `compute_metrics` return dict gains `mape_zero_excluded` key — callers in Task 5 store the whole dict in config. ✓
