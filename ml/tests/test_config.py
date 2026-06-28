from src.config import (
    CLEAN_DATA_PATH,
    DATA_QUALITY_REPORT_PATH,
    FEATURE_COLUMNS,
    MODELS_DIR,
    RAW_DATA_PATH,
    TARGET_COLUMN,
)


def test_raw_data_path_points_to_existing_file():
    assert RAW_DATA_PATH.exists()
    assert RAW_DATA_PATH.name == "coal_all.csv"


def test_feature_and_target_columns_are_defined():
    assert FEATURE_COLUMNS == ["Moisture", "Volatile_matter", "Fixed_Carbon", "Std.Ash"]
    assert TARGET_COLUMN == "GCV"


def test_output_paths_are_under_expected_directories():
    assert CLEAN_DATA_PATH.parent.name == "processed"
    assert DATA_QUALITY_REPORT_PATH.parent.name == "processed"
    assert MODELS_DIR.name == "models"
