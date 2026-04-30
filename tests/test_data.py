import pytest
import pandas as pd
import numpy as np

from steel_flow.data import _sheet_split


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

        labels = df["label"].values
        result = df.groupby("label", group_keys=False).apply(apply_lag, include_groups=False)
        result["label"] = labels
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

        labels = df["label"].values
        result = df.groupby("label", group_keys=False).apply(apply_lag, include_groups=False)
        result["label"] = labels
        result = result.reset_index(drop=True)
        assert pd.isna(result.loc[0, "feature_lag1"])
        assert result.loc[1, "feature_lag1"] == 10.0
        assert result.loc[2, "feature_lag1"] == 20.0


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
        """With 5 sheets A-E, n_test=1, test should be {E}."""
        df = self._make_df(["C", "A", "E", "B", "D"])  # unsorted input
        _, _, test = _sheet_split(df, test_frac=0.2)
        assert set(test["label"]) == {"E"}

    def test_approximate_split_ratios(self):
        """With 10 sheets and default fractions (0.2/0.2), test gets ~2 sheets, val ~2, train ~6."""
        df = self._make_df([f"sheet_{i:02d}" for i in range(10)])
        train, val, test = _sheet_split(df)
        assert len(set(test["label"])) == 2, f"Expected 2 test sheets, got {len(set(test['label']))}"
        assert len(set(val["label"])) == 2, f"Expected 2 val sheets, got {len(set(val['label']))}"
        assert len(set(train["label"])) == 6, f"Expected 6 train sheets, got {len(set(train['label']))}"

    def test_minimum_three_sheets_works(self):
        """Exactly 3 sheets should split into 1 train, 1 val, 1 test."""
        df = self._make_df(["A", "B", "C"])
        train, val, test = _sheet_split(df)
        assert len(set(train["label"])) == 1
        assert len(set(val["label"])) == 1
        assert len(set(test["label"])) == 1
        assert set(train["label"]) | set(val["label"]) | set(test["label"]) == {"A", "B", "C"}


class TestComputeBinEdges:
    def _y(self):
        return pd.Series(range(1, 101), dtype=float)

    def test_equal_frequency_edge_count(self):
        from steel_flow.data import compute_bin_edges
        edges = compute_bin_edges(self._y(), n_bins=4, strategy="equal_frequency")
        assert len(edges) == 3

    def test_equal_width_edge_count(self):
        from steel_flow.data import compute_bin_edges
        edges = compute_bin_edges(self._y(), n_bins=5, strategy="equal_width")
        assert len(edges) == 4

    def test_log_edge_count(self):
        from steel_flow.data import compute_bin_edges
        edges = compute_bin_edges(self._y(), n_bins=3, strategy="log")
        assert len(edges) == 2

    def test_edges_are_monotonically_increasing(self):
        from steel_flow.data import compute_bin_edges
        for strategy in ("equal_frequency", "equal_width", "log"):
            edges = compute_bin_edges(self._y(), n_bins=4, strategy=strategy)
            assert all(a < b for a, b in zip(edges, edges[1:])), strategy

    def test_invalid_strategy_raises(self):
        from steel_flow.data import compute_bin_edges
        with pytest.raises(ValueError, match="Unknown binning strategy"):
            compute_bin_edges(self._y(), n_bins=3, strategy="bad")

    def test_edges_are_json_serializable(self):
        import json
        from steel_flow.data import compute_bin_edges
        edges = compute_bin_edges(self._y(), n_bins=3, strategy="equal_frequency")
        json.dumps(edges)


class TestApplyBins:
    def _y(self):
        return pd.Series([0.0, 5.0, 10.0, 15.0, 20.0])

    def test_all_values_assigned(self):
        from steel_flow.data import apply_bins
        result = apply_bins(self._y(), edges_inner=[8.0, 14.0])
        assert result.notna().all()

    def test_bin_labels_in_range(self):
        from steel_flow.data import apply_bins
        result = apply_bins(self._y(), edges_inner=[8.0, 14.0])
        assert set(result).issubset({0, 1, 2})

    def test_out_of_training_range_clamps_to_outermost_bin(self):
        from steel_flow.data import apply_bins
        y = pd.Series([-100.0, 0.0, 5.0, 10.0, 999.0])
        result = apply_bins(y, edges_inner=[3.0, 7.0])
        assert result.iloc[0] == 0
        assert result.iloc[-1] == 2

    def test_three_bins_three_labels(self):
        from steel_flow.data import compute_bin_edges, apply_bins
        y = pd.Series(range(1, 100), dtype=float)
        edges = compute_bin_edges(y, n_bins=3, strategy="equal_frequency")
        result = apply_bins(y, edges)
        assert set(result) == {0, 1, 2}
