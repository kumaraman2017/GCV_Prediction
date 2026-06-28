import pandas as pd

from src.config import CLEAN_DATA_PATH, DATA_QUALITY_REPORT_PATH


def test_run_clean_produces_expected_outputs():
    assert CLEAN_DATA_PATH.exists()
    assert DATA_QUALITY_REPORT_PATH.exists()

    clean_df = pd.read_csv(CLEAN_DATA_PATH)
    assert len(clean_df) == 4513
    assert list(clean_df.columns) == ["Moisture", "Volatile_matter", "Fixed_Carbon", "Std.Ash", "GCV"]
