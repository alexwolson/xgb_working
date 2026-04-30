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
        "mould_position", "remove_rows", "include_subdirs", "exclude_subdirs",
    ],
    "tune": ["study_count", "tree_method", "storage_path", "multithread"],
    "train": ["subsample_shap", "config_file"],
    "predict": ["config_file", "data_directory", "piv_data"],
    "wandb": ["project", "entity", "enabled"],
}

_VALID_BIN_STRATEGIES = {"equal_frequency", "equal_width", "log"}
_BINS_DEFAULTS = {"enabled": False, "n_bins": 3, "strategy": "equal_frequency"}


def load_config(path: str | None = None) -> SimpleNamespace:
    if path is None:
        path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CONFIG_PATH

    config_path = Path(path)
    if not config_path.exists():
        sys.exit(f"Config file not found: {config_path}")

    try:
        with open(config_path, "rb") as f:
            raw = tomllib.load(f)
    except Exception as e:
        sys.exit(f"Failed to parse config file {config_path}: {e}")

    for section, keys in _REQUIRED.items():
        if section not in raw:
            sys.exit(f"Config missing required section: [{section}]")
        for key in keys:
            if key not in raw[section]:
                sys.exit(f"Config [{section}] missing required key: {key}")

    cfg = SimpleNamespace(**{
        section: SimpleNamespace(**raw[section])
        for section in _REQUIRED
    })

    bins_raw = {**_BINS_DEFAULTS, **raw.get("bins", {})}
    if bins_raw["strategy"] not in _VALID_BIN_STRATEGIES:
        sys.exit(
            f"Config [bins] strategy must be one of {sorted(_VALID_BIN_STRATEGIES)}, "
            f"got: {bins_raw['strategy']!r}"
        )
    cfg.bins = SimpleNamespace(**bins_raw)

    return cfg
