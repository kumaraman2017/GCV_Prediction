import pandas as pd
import pytest
from pydantic import ValidationError

from src.data.schema import CoalRecord, validate_dataframe


def test_valid_record_passes():
    record = CoalRecord.model_validate({
        "Moisture": 5.20, "Volatile_matter": 31.10, "Fixed_Carbon": 40.70,
        "Std.Ash": 23.00, "GCV": 22.84477109,
    })
    assert record.moisture == 5.20


def test_negative_value_rejected():
    with pytest.raises(ValidationError):
        CoalRecord.model_validate({
            "Moisture": -1.0, "Volatile_matter": 31.10, "Fixed_Carbon": 40.70,
            "Std.Ash": 30.90, "GCV": 22.84477109,
        })


def test_value_over_100_rejected():
    with pytest.raises(ValidationError):
        CoalRecord.model_validate({
            "Moisture": 5.20, "Volatile_matter": 31.10, "Fixed_Carbon": 40.70,
            "Std.Ash": 123.00, "GCV": 22.84477109,
        })


def test_proximate_sum_must_be_close_to_100():
    with pytest.raises(ValidationError):
        CoalRecord.model_validate({
            "Moisture": 5.20, "Volatile_matter": 31.10, "Fixed_Carbon": 40.70,
            "Std.Ash": 10.00, "GCV": 22.84477109,
        })


def test_validate_dataframe_splits_valid_and_invalid_rows():
    df = pd.DataFrame([
        {"Moisture": 5.20, "Volatile_matter": 31.10, "Fixed_Carbon": 40.70, "Std.Ash": 23.00, "GCV": 22.84477109},
        {"Moisture": -1.0, "Volatile_matter": 31.10, "Fixed_Carbon": 40.70, "Std.Ash": 30.90, "GCV": 22.84477109},
        {"Moisture": 10.70, "Volatile_matter": 28.00, "Fixed_Carbon": 30.00, "Std.Ash": 31.30, "GCV": 15.49101361},
    ])
    valid_df, errors = validate_dataframe(df)
    assert len(valid_df) == 2
    assert len(errors) == 1
    assert errors[0]["index"] == 1
