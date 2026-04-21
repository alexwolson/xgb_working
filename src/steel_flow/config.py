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
