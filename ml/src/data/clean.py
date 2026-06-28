import pandas as pd

from src.config import FEATURE_COLUMNS, TARGET_COLUMN
from src.data.schema import validate_dataframe

MODELING_COLUMNS = FEATURE_COLUMNS + [TARGET_COLUMN]


def drop_duplicate_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    deduped = df.drop_duplicates().reset_index(drop=True)
    return deduped, len(df) - len(deduped)


def select_modeling_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df[MODELING_COLUMNS].copy()


def impute_missing_values(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = df.copy()
    imputed_counts = {}
    for column in MODELING_COLUMNS:
        missing = int(df[column].isnull().sum())
        imputed_counts[column] = missing
        if missing > 0:
            df[column] = df[column].fillna(df[column].median())
    return df, imputed_counts


def detect_outliers_iqr(df: pd.DataFrame) -> dict:
    report = {}
    for column in MODELING_COLUMNS:
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        count = int(((df[column] < lower_bound) | (df[column] > upper_bound)).sum())
        report[column] = {
            "lower_bound": float(lower_bound),
            "upper_bound": float(upper_bound),
            "count": count,
        }
    return report


def clean_pipeline(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    valid_df, validation_errors = validate_dataframe(raw_df)
    deduped_df, n_duplicates = drop_duplicate_rows(valid_df)
    modeling_df = select_modeling_columns(deduped_df)
    imputed_df, imputed_counts = impute_missing_values(modeling_df)
    outliers = detect_outliers_iqr(imputed_df)

    report = {
        "raw_row_count": int(len(raw_df)),
        "rows_failed_validation": len(validation_errors),
        "validation_errors": validation_errors,
        "duplicates_removed": n_duplicates,
        "rows_after_cleaning": int(len(imputed_df)),
        "missing_values_imputed": imputed_counts,
        "outliers": outliers,
    }
    return imputed_df, report
