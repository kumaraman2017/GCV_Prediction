from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "coal_all.csv"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
CLEAN_DATA_PATH = PROCESSED_DATA_DIR / "coal_clean.csv"
DATA_QUALITY_REPORT_PATH = PROCESSED_DATA_DIR / "data_quality_report.json"

MODELS_DIR = PROJECT_ROOT / "models"
BEST_MODEL_PATH = MODELS_DIR / "best_model.joblib"
MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.json"
MODEL_COMPARISON_PATH = MODELS_DIR / "model_comparison.json"
SHAP_GLOBAL_PATH = MODELS_DIR / "shap_global.json"

FEATURE_COLUMNS = ["Moisture", "Volatile_matter", "Fixed_Carbon", "Std.Ash"]
TARGET_COLUMN = "GCV"
RANDOM_SEED = 42
TEST_SIZE = 0.2
PROXIMATE_SUM_TOLERANCE = 0.5
