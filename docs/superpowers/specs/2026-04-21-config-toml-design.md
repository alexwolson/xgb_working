# Config TOML Refactor Design

## Goal

Replace all CLI argument parsing in `tune`, `train`, and `predict` with a single `config/base.toml` file. The only remaining "argument" each command accepts is an optional positional path to an alternate TOML config.

## Architecture

Three layers:

1. **`config/base.toml`** — the canonical config file, checked into version control, containing all settings with sensible defaults.
2. **`src/steel_flow/config.py`** — config loader: reads TOML, validates required keys, returns a `SimpleNamespace` of sections.
3. **`src/steel_flow/cli.py`** — entry points call `load_config()`, read from the returned namespace, and pass values through to the existing pipeline functions unchanged.

The pipeline functions (`data.load_data`, `tune.run_study`, `train.train_model`, `evaluate.*`, `predict.*`) are **not modified** — only `cli.py` changes how it sources its inputs.

## TOML Structure

`config/base.toml`:

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
config_file = ""  # auto-derived as training_config_<study_name>.json if empty

[predict]
config_file = "config/training_config_water_modelling.json"
data_directory = "New_Data"
piv_data = false
```

All five sections are required. Missing sections or missing required keys within them raise `SystemExit` with a descriptive error message.

## Config Loader (`src/steel_flow/config.py`)

```python
try:
    import tomllib
except ImportError:
    import tomli as tomllib  # Python 3.10 fallback

def load_config(path: str | None = None) -> SimpleNamespace:
    ...
```

- If `path` is `None`, reads `sys.argv[1]` if provided, else defaults to `config/base.toml`.
- Parses TOML, validates all required top-level sections exist.
- Returns a `SimpleNamespace` where each attribute corresponds to a TOML section, itself a `SimpleNamespace`:
  - `cfg.experiment.targets`, `cfg.data.directory`, `cfg.tune.study_count`, etc.

Required sections: `experiment`, `data`, `tune`, `train`, `predict`.

Required keys per section:

| Section | Required keys |
|---|---|
| `experiment` | `targets`, `study_name`, `objective` |
| `data` | `directory`, `discard_features`, `onehot_encoding`, `sen_geometrical`, `clogging_factors`, `lagged_features`, `feature_lag_amount`, `mould_position`, `remove_rows` |
| `tune` | `study_count`, `tree_method`, `storage_path`, `multithread` |
| `train` | `subsample_shap`, `config_file` |
| `predict` | `config_file`, `data_directory`, `piv_data` |

## CLI Changes (`src/steel_flow/cli.py`)

- Remove all `argparse` imports and argument definitions.
- Remove `_common_args()`.
- Rewrite `_build_study_name(cfg, target)` to accept a `SimpleNamespace` instead of `argparse.Namespace`.
- Each entry point calls `load_config()` at the top and reads from the returned namespace.
- `sys.argv[1]` (optional config path) is the only user input; no flags, no `--` options.

Example invocations after the refactor:

```bash
uv run tune                              # uses config/base.toml
uv run tune config/my_experiment.toml   # uses alternate config
uv run train config/my_experiment.toml
uv run predict config/my_experiment.toml
```

## HPC Scripts

All `scripts/hpc/*.sh` files are updated to remove CLI flags. Each script becomes:

```bash
uv run tune config/base.toml
```

(or the relevant command). The TOML file holds all parameters previously passed as flags.

## Dependency Change

Add `tomli` to `pyproject.toml` dependencies. Used as a fallback for Python 3.10 (which lacks `tomllib` in stdlib). Python 3.11+ uses the stdlib version via `try/except ImportError`.

## Testing (`tests/test_config.py`)

Three tests:

1. **Valid TOML loads correctly** — write a minimal valid TOML to a temp file, call `load_config(path)`, assert key values are accessible via dot notation.
2. **Missing section raises SystemExit** — write a TOML missing one required section, assert `SystemExit` is raised with a message containing the section name.
3. **Default path used when no arg given** — patch `sys.argv` to `["cmd"]` (no positional arg), assert `load_config()` attempts to open `config/base.toml`.

## Files Changed

| File | Action |
|---|---|
| `src/steel_flow/config.py` | Create |
| `src/steel_flow/cli.py` | Modify |
| `config/base.toml` | Create |
| `pyproject.toml` | Add `tomli` dependency |
| `scripts/hpc/study_job.sh` | Remove CLI flags |
| `scripts/hpc/multithread_job.sh` | Remove CLI flags |
| `scripts/hpc/multithread_study_eval.sh` | Remove CLI flags |
| `scripts/hpc/multithread_study_train.sh` | Remove CLI flags |
| `scripts/hpc/predict_job.sh` | Remove CLI flags |
| `scripts/hpc/piv_pred.sh` | Remove CLI flags |
| `tests/test_config.py` | Create |
| `README.md` | Update usage section |

## Out of Scope

- Changes to `data.py`, `tune.py`, `train.py`, `evaluate.py`, `predict.py` — pipeline internals are untouched.
- Config validation beyond section/key presence (e.g., valid objective values) — left to the pipeline functions as before.
- Multiple config file merging or environment variable overrides.
