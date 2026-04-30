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
include_subdirs = []
exclude_subdirs = []

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

[wandb]
project = "test-project"
entity = ""
enabled = false
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


def test_bins_defaults_when_section_absent(tmp_path):
    from steel_flow.config import load_config
    cfg = load_config(str(_write_valid_toml(tmp_path)))
    assert cfg.bins.enabled is False
    assert cfg.bins.n_bins == 3
    assert cfg.bins.strategy == "equal_frequency"


def test_bins_section_overrides_defaults(tmp_path):
    from steel_flow.config import load_config
    base = _write_valid_toml(tmp_path).read_text()
    p = tmp_path / "bins_config.toml"
    p.write_text(base + "\n[bins]\nenabled = true\nn_bins = 5\nstrategy = \"log\"\n")
    cfg = load_config(str(p))
    assert cfg.bins.enabled is True
    assert cfg.bins.n_bins == 5
    assert cfg.bins.strategy == "log"


def test_invalid_bins_strategy_raises_system_exit(tmp_path):
    from steel_flow.config import load_config
    base = _write_valid_toml(tmp_path).read_text()
    p = tmp_path / "bad_bins.toml"
    p.write_text(base + "\n[bins]\nenabled = true\nn_bins = 3\nstrategy = \"bad_strategy\"\n")
    with pytest.raises(SystemExit) as exc_info:
        load_config(str(p))
    assert "bad_strategy" in str(exc_info.value)


def test_missing_key_raises_system_exit(tmp_path):
    # [experiment] is missing the required "objective" key
    content = """
[experiment]
targets = ["Count_EX1"]
study_name = "test_study"

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
include_subdirs = []
exclude_subdirs = []

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

[wandb]
project = "test-project"
entity = ""
enabled = false
"""
    p = tmp_path / "missing_key.toml"
    p.write_text(content)
    from steel_flow.config import load_config
    with pytest.raises(SystemExit) as exc_info:
        load_config(str(p))
    assert "objective" in str(exc_info.value)
