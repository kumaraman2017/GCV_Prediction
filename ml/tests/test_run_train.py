import json

import joblib

from src.config import BEST_MODEL_PATH, MODEL_COMPARISON_PATH, MODEL_METADATA_PATH


def test_run_train_persists_best_model_bundle():
    artifact = joblib.load(BEST_MODEL_PATH)
    required_keys = {
        "estimator", "scaler", "feature_columns", "target_column",
        "confidence_method", "is_tree_based", "X_train", "y_train",
        "residual_std", "residuals_train", "feature_ranges",
    }
    assert set(artifact.keys()) == required_keys
    assert artifact["feature_columns"] == ["Moisture", "Volatile_matter", "Fixed_Carbon", "Std.Ash"]
    assert artifact["confidence_method"] in {"bagging_ensemble", "single_tree", "knn_residual"}


def test_run_train_artifact_arrays_are_consistent():
    artifact = joblib.load(BEST_MODEL_PATH)
    assert len(artifact["X_train"]) == len(artifact["y_train"]) == len(artifact["residuals_train"])

    for feature in artifact["feature_columns"]:
        low, high = artifact["feature_ranges"][feature]
        assert low <= high


def test_run_train_persists_metadata_and_comparison():
    metadata = json.loads(MODEL_METADATA_PATH.read_text())
    assert "model_name" in metadata
    assert set(metadata["metrics"].keys()) == {"rmse", "mae", "r2"}

    comparison = json.loads(MODEL_COMPARISON_PATH.read_text())
    assert metadata["model_name"] in comparison
    assert len(comparison) >= 7
