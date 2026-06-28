import pandas as pd

from src.data.clean import (
    clean_pipeline,
    detect_outliers_iqr,
    drop_duplicate_rows,
    impute_missing_values,
    select_modeling_columns,
)


def _valid_row(overrides=None):
    row = {"Moisture": 5.20, "Volatile_matter": 31.10, "Fixed_Carbon": 40.70, "Std.Ash": 23.00, "GCV": 22.84477109}
    if overrides:
        row.update(overrides)
    return row


def test_drop_duplicate_rows_removes_exact_duplicates():
    df = pd.DataFrame([_valid_row(), _valid_row(), _valid_row({"GCV": 15.0})])
    deduped, n_removed = drop_duplicate_rows(df)
    assert len(deduped) == 2
    assert n_removed == 1


def test_select_modeling_columns_drops_ultimate_analysis_columns():
    df = pd.DataFrame([{**_valid_row(), "Hydrogen": 4.9, "Carbon": 57.9}])
    modeling_df = select_modeling_columns(df)
    assert list(modeling_df.columns) == ["Moisture", "Volatile_matter", "Fixed_Carbon", "Std.Ash", "GCV"]


def test_impute_missing_values_fills_with_median():
    df = pd.DataFrame([_valid_row(), _valid_row({"Moisture": None}), _valid_row({"Moisture": 7.0})])
    imputed_df, counts = impute_missing_values(df)
    assert counts["Moisture"] == 1
    assert imputed_df["Moisture"].isnull().sum() == 0


def test_detect_outliers_iqr_flags_extreme_value():
    rows = [_valid_row({"GCV": 22.0 + i * 0.1}) for i in range(20)]
    rows.append(_valid_row({"GCV": 100.0}))
    df = pd.DataFrame(rows)
    report = detect_outliers_iqr(df)
    assert report["GCV"]["count"] >= 1


def test_clean_pipeline_removes_duplicates_and_invalid_rows():
    rows = [_valid_row(), _valid_row(), _valid_row({"Moisture": -1.0, "Std.Ash": 30.90})]
    df = pd.DataFrame(rows)
    clean_df, report = clean_pipeline(df)
    assert report["duplicates_removed"] == 1
    assert report["rows_failed_validation"] == 1
    assert len(clean_df) == 1
    assert list(clean_df.columns) == ["Moisture", "Volatile_matter", "Fixed_Carbon", "Std.Ash", "GCV"]
