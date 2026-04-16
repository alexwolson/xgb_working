# Full Restructure — Design Spec
_Date: 2026-04-16_

## Goal

Perform a complete cleanup pass of the repository: delete dead files, reorganize the directory layout, consolidate dependencies, and extract the business logic into a proper installable Python package with `uv run` entry points.

---

## Directory Layout

```
steel_flow_sensor_prediction/
├── src/
│   └── steel_flow/
│       ├── __init__.py
│       ├── cli.py           # Entry points: tune(), train(), predict()
│       ├── data.py          # CSV generation, loading, feature engineering
│       ├── tune.py          # Optuna hyperparameter search
│       ├── train.py         # Train final model with best params + evaluate
│       └── evaluate.py      # Metrics, SHAP plots, error histograms
├── scripts/
│   ├── local/
│   │   ├── start_study.sh
│   │   └── start_predict.sh
│   └── hpc/
│       ├── study_job.sh
│       ├── predict_job.sh
│       ├── multithread_job.sh
│       ├── multithread_study_train.sh
│       ├── multithread_study_eval.sh
│       ├── piv_pred.sh
│       └── requirements-computecan.txt
├── config/
│   └── training_config_20250328.json
├── data/
│   ├── raw/
│   │   └── Organized_Data/        # existing SEN06–SEN10 dirs
│   └── output/
│       ├── figures/               # error histograms, SHAP plots
│       └── models/                # trained XGBoost .json files
├── docs/
│   └── superpowers/
│       └── specs/
├── pyproject.toml                 # single dep source of truth + [project.scripts]
├── uv.lock
└── README.md
```

### Files deleted
- `rewerwrew.py` — empty scratch file
- `wergbtqwbertbwre.py` — empty scratch file
- `run_study_old.py` — stale backup of run_study.py
- `start.txt` — scratch run-command notes
- `requirements.txt` — superseded by pyproject.toml
- `requirementscomputecan.txt` — moved to scripts/hpc/
- `environment.yml` — uv is the recommended install path
- `run_study.py` — logic moves into src/steel_flow/
- `predict.py` — logic moves into src/steel_flow/
- `get_split.py` — absorbed into src/steel_flow/data.py

### .gitignore additions
- `data/` — raw Excel files and generated outputs are not tracked

---

## Module Responsibilities

### `data.py`
- `generate_csv_files()` — processes raw `.xlsx` files → `X<suffix>.csv` / `y<suffix>.csv`
- `load_data()` — loads CSVs, applies one-hot encoding, discards specified features
- `get_split()` — train/test split (logic absorbed from `get_split.py`)
- Feature engineering helpers: SEN geometrical conversion, clogging factor handling, lag features, mould position

### `tune.py`
- `run_study()` — creates or resumes an Optuna study, runs N trials
- `objective()` — inner function defining the XGBoost hyperparameter search space
- Supports SQLite storage (default) and JournalStorage backend (HPC multithreading)

### `train.py`
- `train_model()` — trains XGBoost with a given set of hyperparameters, saves model to `data/output/models/`
- `save_training_config()` — writes training config JSON to `config/`

### `evaluate.py`
- `compute_metrics()` — MAE, MAPE, R², RMSE
- `plot_error_histogram()` — saves error distribution plots to `data/output/figures/`
- `plot_shap()` — saves SHAP beeswarm and bar plots to `data/output/figures/`

### `cli.py`
- `tune()` — argument parsing + calls `tune.run_study()`; entry point for `uv run tune`
- `train()` — argument parsing + calls `train.train_model()` + `evaluate.*`; entry point for `uv run train`
- `predict()` — argument parsing + runs predictions on new Excel files; entry point for `uv run predict`

---

## Dependency Management

`pyproject.toml` becomes the single source of truth. All deps currently used in code are added explicitly (without pinned versions):

- `xgboost`
- `optuna`
- `shap`
- `scikit-learn`
- `pandas`
- `openpyxl`
- `seaborn`
- `matplotlib` (already present)
- `rich` (already present)

`[project.scripts]` section maps entry points:
```toml
[project.scripts]
tune = "steel_flow.cli:tune"
train = "steel_flow.cli:train"
predict = "steel_flow.cli:predict"
```

`scripts/hpc/requirements-computecan.txt` is kept as-is for Compute Canada environments that require pip-based installs.

---

## Commit Sequence

Each commit leaves the repo in a runnable state.

1. **Delete dead files** — `rewerwrew.py`, `wergbtqwbertbwre.py`, `run_study_old.py`, `start.txt`, `requirements.txt`, `requirementscomputecan.txt`, `environment.yml`
2. **Reorganize directories** — move data, figures, models, config, and scripts into new layout; update `.gitignore`; add `.gitkeep` files to `data/raw/`, `data/output/figures/`, `data/output/models/` so the empty dirs survive a fresh clone
3. **Fix `pyproject.toml`** — add missing deps, add `[project.scripts]`
4. **Extract `src/steel_flow/` package** — migrate logic from `run_study.py`, `predict.py`, `get_split.py` into package modules
5. **Delete old CLI files** — remove `run_study.py`, `predict.py`, `get_split.py`
6. **Update README** — reflect new commands and directory layout

---

## Success Criteria

- `uv run tune`, `uv run train`, `uv run predict` all work with equivalent arguments to the old scripts
- No business logic lives outside `src/steel_flow/`
- Root directory contains only: `src/`, `scripts/`, `config/`, `data/`, `docs/`, `pyproject.toml`, `uv.lock`, `README.md`
- `uv sync` installs all required dependencies from `pyproject.toml` alone
