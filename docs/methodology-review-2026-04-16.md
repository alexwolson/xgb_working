# Methodology Review
_Date: 2026-04-16_

## Overview

This document presents a critical methodology review of the steel flow sensor prediction pipeline. The pipeline uses XGBoost to predict bubble counts (`Count_EX1`, `Count_EX2`) from time-series sensor readings extracted from Excel sheets. The review covers experimental design, data processing, model selection, evaluation validity, and reproducibility. Issues are grouped by severity: **Critical**, **Important**, and **Minor**.

---

## Pipeline Summary

The pipeline operates in three stages:

1. **Data preparation** (`data.py`): Raw `.xlsx` files are parsed sheet by sheet. Each sheet represents one experimental condition (identified by the sheet name, e.g., `SEN06_100_50`). Features are extracted by column index, categorical metadata is derived from the sheet name, and all sheets are concatenated into a single `X.csv`/`y.csv`.

2. **Hyperparameter tuning** (`tune.py`, `uv run tune`): An Optuna study runs N trials. Each trial trains an XGBoost model on `train_df` and evaluates MAE on `test_df`. The trial with the lowest MAE is selected as the best.

3. **Final training and evaluation** (`train.py`, `evaluate.py`, `uv run train`): The best hyperparameters from the Optuna study are used to train a final model on `train_df`. That model is evaluated on `test_df` and metrics (MAE, MAPE, R², RMSE) are reported.

The train/test split is performed in `data.py` as a single random 80/20 split using `sklearn.model_selection.train_test_split` with `random_state=42` and default `shuffle=True`.

---

## Critical Issues

### 1. Data Leakage: Test Set Used for Hyperparameter Selection

**Location:** `src/steel_flow/tune.py:47–49`, `src/steel_flow/cli.py:165`

**The problem:**

During hyperparameter tuning, every Optuna trial trains on `train_df` and computes its objective score on `test_df`:

```python
# tune.py — inside the Optuna objective
model.fit(train_df[features], train_df[target])
preds = model.predict(test_df[features])
return mean_absolute_error(test_df[target], preds)
```

The Optuna study then selects the trial with the lowest test-set MAE as the "best" trial. After tuning, `uv run train` loads those best hyperparameters and evaluates the final model on the same `test_df`:

```python
# cli.py — inside train()
test_metrics = eval_mod.compute_metrics(model, X_test, y_test)
```

This is a form of **data leakage via model selection**. The test set has been used to guide the choice of hyperparameters across potentially hundreds of trials. The test set is therefore no longer an independent measure of generalization — the best trial was, by definition, the one that happened to perform best on it. The reported test metrics will be **optimistically biased** relative to true out-of-sample performance. The degree of bias grows with the number of Optuna trials.

**Why it matters:**

When the pipeline reports, for example, MAE = 3.2 on the test set, it is not reporting the expected MAE on new, unseen data. It is reporting the MAE of the single best configuration chosen from N configurations evaluated on that data. This overstates model quality and can mislead decisions about deployment readiness.

**Correct approach — three-way split:**

Introduce a held-out test set that is never seen during tuning:

```
All data
├── Train (64%)     → Optuna trains on this
├── Validation (16%) → Optuna evaluates on this (replaces test_df in objective)
└── Test (20%)      → Evaluated once, after final model is selected
```

Modify `tune.py` to accept `val_df` instead of `test_df` and evaluate on validation. Keep `test_df` separate. Report final metrics from the held-out test set only.

**Alternative — cross-validation within training:**

Use k-fold cross-validation inside the Optuna objective, operating entirely on `train_df`. This avoids the need for a separate validation split but is more computationally expensive:

```python
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, train_df[features], train_df[target],
                         cv=5, scoring="neg_mean_absolute_error")
return -scores.mean()
```

Either approach eliminates the leakage and produces unbiased test-set estimates.

---

### 2. Random Splitting of Time-Series Data

**Location:** `src/steel_flow/data.py:267`

**The problem:**

The data is sensor time-series — each sheet contains time-ordered rows with a `time[s]` column. The train/test split uses a random shuffle:

```python
train_df, test_df = train_test_split(combined_df, test_size=0.2, random_state=42)
```

`train_test_split` with default `shuffle=True` randomly assigns rows to train and test without regard to their temporal order. This means:

- Time points from the middle or end of a recording can appear in training while earlier points are in the test set.
- The model can "see" what happens in the future (relative to some test points) during training.
- If lagged features are used (see Issue 3 below), the lags for a test-set row may have been constructed from values that are in the training set.

For time-series data, **the independence assumption of a random split is violated**. Adjacent time points are strongly correlated, so train/test split must respect the temporal ordering of the data.

**Why it matters:**

A model trained and evaluated with a random split on time-series data will almost always show better performance than it achieves in deployment, where it genuinely predicts future observations from past ones. The model may be learning temporal autocorrelation patterns (a current reading predicts the next reading) rather than the physical relationship between process parameters and bubble counts.

**Correct approach:**

Sort data by time before splitting and take the final 20% of time as the test set:

```python
combined_df = combined_df.sort_values("time_s")  # cleaned column name
n = len(combined_df)
train_df = combined_df.iloc[:int(n * 0.8)].copy()
test_df = combined_df.iloc[int(n * 0.8):].copy()
```

If multiple independent experimental runs (sheets) are present, the split should ideally be at the run level — entire runs go to train or test — rather than within runs. This better reflects deployment conditions, where the model is applied to a completely new experimental run.

---

### 3. Lag Features Computed Across Sheet Boundaries and Before Splitting

**Location:** `src/steel_flow/data.py:251–258`

**The problem:**

When lag features are requested, they are computed on the full concatenated `X_data` DataFrame before the train/test split:

```python
# data.py — inside load_data()
for feature in cleaned_lag:
    for lag_amount in range(1, feature_lag + 1):
        X_data[f"{feature}_lag{lag_amount}"] = X_data[feature].shift(lag_amount)
```

This creates two distinct problems:

**Problem A — Cross-sheet contamination:** All sheets are concatenated into one DataFrame. `shift()` operates on the concatenated index without any reset between sheets. The lag at the first row of Sheet B is filled with the last value of Sheet A. Sheet A and Sheet B represent different experimental conditions (different SEN type, flow rates) and are physically unrelated recordings. Lag values spanning sheet boundaries are meaningless.

**Problem B — Post-split lag leakage:** After `shift()` is applied globally, a random train/test split is performed. A test-set row at position `t` may have a `lag1` value that was drawn from position `t-1`, which may be in the training set — this is correct. But position `t-1` could also be in the test set and `t` in training, meaning training uses a value that is "from the future" relative to the test set's timeline. With a random split, these relationships are scrambled unpredictably.

**Correct approach:**

Compute lag features per-sheet, before concatenation, and reset the index between sheets:

```python
sheet_dfs = []
for sheet_name, df in all_sheets.items():
    df = df.copy()
    for feature in lagged_features:
        for lag in range(1, feature_lag + 1):
            df[f"{feature}_lag{lag}"] = df[feature].shift(lag)
    df = df.dropna()  # remove rows with NaN lags at the start of each sheet
    sheet_dfs.append(df)
combined = pd.concat(sheet_dfs, ignore_index=True)
```

This confines lags to within each recording and eliminates cross-sheet contamination.

---

## Important Issues

### 4. Stochastic Regularization Parameters Excluded from Search Space

**Location:** `src/steel_flow/tune.py:39–40`

```python
"subsample": 1.0,
"colsample_bytree": 1.0,
```

These two parameters control the fraction of training rows and features sampled per tree, respectively. They are fixed at 1.0 (no subsampling) and are not included in the Optuna search space. Both are among the most impactful XGBoost regularization parameters:

- `subsample < 1.0` introduces row-level stochasticity that reduces variance and is especially effective on correlated data (like time-series).
- `colsample_bytree < 1.0` introduces feature-level stochasticity and can reduce overfitting to noisy features.

With both fixed at 1.0, the hyperparameter search has fewer degrees of freedom and may find hyperparameter combinations that overfit to the training/validation set in ways that subsampling would otherwise prevent.

**Recommendation:** Include both in the Optuna search space, e.g.:

```python
"subsample": trial.suggest_float("subsample", 0.5, 1.0),
"colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
```

---

### 5. Count Data Modeled with Mean Squared Error Objective

**Location:** `src/steel_flow/tune.py:32`

```python
"objective": "reg:squarederror",
```

The prediction targets (`Count_EX1`, `Count_EX2`) are named as counts, which are non-negative integers. Squared error regression makes no assumptions about the target distribution and can produce negative predictions, which are physically impossible for count data.

For count targets, more appropriate XGBoost objectives include:

- **`count:poisson`**: Models the target as Poisson-distributed (non-negative, mean = variance). Well-suited to counts.
- **`reg:tweedie`**: A generalization of Poisson/gamma regression; useful when the variance-mean relationship differs from pure Poisson.

Using a misspecified objective does not necessarily prevent the model from fitting well, but it means the optimization criterion (squared error) does not match the statistical structure of the outcome. It can lead to worse calibration and predictions that need to be clipped to zero post-hoc.

**Recommendation:** Try `count:poisson` as an alternative objective and compare holdout performance. If Count_EX1 and Count_EX2 exhibit zero-inflation, investigate whether a two-stage model (classifier for zero vs. non-zero, then regressor on non-zero values) is warranted.

---

### 6. MAPE is Undefined or Distorted When True Counts Are Zero

**Location:** `src/steel_flow/evaluate.py:35`

```python
"mean_absolute_percent_error": float(mean_absolute_percentage_error(y, predictions) * 100),
```

Mean Absolute Percentage Error divides by the true value:

```
MAPE = (1/n) * Σ |y_i - ŷ_i| / |y_i|
```

When `y_i = 0`, this term is undefined. Sklearn's implementation handles this by masking zero-true-value rows (skipping them from the average). If a significant fraction of observations have a true count of zero, the reported MAPE:

1. Is computed on a different subset of data than MAE and RMSE.
2. Can be heavily influenced by rows where the true count is very small (e.g., 1 or 2), making tiny absolute errors appear as large percentages.

**Recommendation:** Report what fraction of zero-target rows exist and note whether they are excluded from MAPE. Consider replacing MAPE with a more robust alternative such as Mean Absolute Scaled Error (MASE) or simply rely on MAE and RMSE, which do not have this singularity.

---

### 7. Optuna Hyperparameter Search Is Not Seeded

**Location:** `src/steel_flow/tune.py:57–63`

```python
study = optuna.create_study(
    direction="minimize",
    study_name=study_name,
    storage=storage,
    load_if_exists=True,
)
study.optimize(objective, n_trials=study_count, ...)
```

No sampler with a fixed random seed is provided. The default TPE sampler draws from its own internal RNG, which is seeded from system entropy at creation time. Two runs of `uv run tune` with identical arguments can produce different best hyperparameter sets.

This affects reproducibility: the training config JSON produced by `uv run train` depends on which Optuna study was loaded, and that study's outcome varies by run.

**Recommendation:**

```python
sampler = optuna.samplers.TPESampler(seed=42)
study = optuna.create_study(..., sampler=sampler)
```

Note that `load_if_exists=True` means existing study trials are reused — the seed primarily matters for the *first* creation of a study. Document this behavior clearly.

---

## Minor Issues

### 8. SHAP Subsampling Reduces Stability Without Documentation

**Location:** `src/steel_flow/evaluate.py:79–80`

When `--subsample-shap` is used:

```python
explainer = shap.Explainer(model, X_train.sample(frac=0.1).astype("float64"))
shap_values = explainer(X_test[: len(X_test) // 10].astype("float64"))
```

Both the background dataset (10% of training) and the explained set (first 10% of test) are heavily subsampled. On small datasets this can produce noisy, run-to-run-variable SHAP plots. The 10% subsample is also taken from the start of the test set (`[:len//10]`), which is not a random sample and may not be representative if the test set has any ordering.

**Recommendation:** Use `X_test.sample(frac=0.1)` instead of `X_test[:len//10]`. Add the sample sizes to the plot titles so readers can assess reliability.

---

### 9. Error Histogram Uses Absolute Errors Only

**Location:** `src/steel_flow/evaluate.py:57`

```python
errors = abs(y - predictions)
```

Absolute errors are always non-negative and symmetric. The histogram therefore cannot reveal systematic bias: a model that always over-predicts by 5 would show the same distribution as one that always under-predicts by 5. The `plot_prediction_error` function's residuals plot (also produced) addresses this, but the error histogram is the more prominent output in the figures directory.

**Recommendation:** Optionally plot signed errors (`y - predictions`) alongside or instead of absolute errors to make systematic bias immediately visible.

---

### 10. `remove_rows` Is a Domain Assumption Without Documentation

**Location:** `src/steel_flow/data.py:85–86`

```python
if remove_rows != 0:
    df = df[:-remove_rows]
```

This silently removes the last N rows of every sheet. The intended use case is presumably to strip artifact-affected rows at the end of an experimental recording. However, nothing in the code or parameter documentation explains when and why these rows should be removed, what value is appropriate, or how it interacts with lag features (the lagged values for rows near the removed region may still be computed from the discarded rows if lags were computed per-sheet after this step).

**Recommendation:** Document the physical justification for row removal in the README and CLI help text. Add a warning if `remove_rows >= len(df)` for a sheet (currently this produces an empty DataFrame with a generic warning, but the cause is not clear to the user).

---

## Evaluation of Reported Metrics

Given the critical issues above, the metrics reported by `uv run train` should be interpreted with the following caveats:

| Metric | Reported as | Likely actual |
|---|---|---|
| Test MAE | Unbiased generalization error | **Optimistic** (inflated by HPO leakage) |
| Test R² | Proportion of variance explained on new data | **Optimistic** |
| Train MAE | Lower bound on fit quality | Valid |
| Train/test gap | Indicator of overfitting | **Unreliable** (test not truly held out) |

Until the three critical issues are addressed, the test-set metrics should be treated as **training-set performance on a held-out shard**, not as an estimate of deployment performance.

---

## Recommended Priority Order

| # | Issue | Severity | Effort |
|---|---|---|---|
| 1 | Introduce validation split; use test set only for final reporting | Critical | Medium |
| 2 | Replace random split with temporal split (per-run if possible) | Critical | Low |
| 3 | Compute lag features per-sheet before concatenation | Critical | Medium |
| 4 | Add `subsample` and `colsample_bytree` to Optuna search space | Important | Trivial |
| 5 | Evaluate `count:poisson` objective as alternative | Important | Low |
| 6 | Handle/document MAPE behavior with zero-count targets | Important | Low |
| 7 | Seed Optuna sampler for reproducibility | Minor | Trivial |
| 8 | Use random subsample for SHAP; add sample size to plot titles | Minor | Trivial |
| 9 | Plot signed errors in error histogram | Minor | Trivial |
| 10 | Document `remove_rows` domain assumption | Minor | Trivial |
