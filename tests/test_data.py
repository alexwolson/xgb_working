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
