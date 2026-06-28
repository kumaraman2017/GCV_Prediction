import json
from datetime import datetime, timezone

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    BEST_MODEL_PATH,
    CLEAN_DATA_PATH,
    FEATURE_COLUMNS,
    MODEL_COMPARISON_PATH,
    MODEL_METADATA_PATH,
    MODELS_DIR,
    RANDOM_SEED,
    TARGET_COLUMN,
    TEST_SIZE,
)
from src.models.registry import build_registry
from src.models.train import select_best, train_and_tune


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(CLEAN_DATA_PATH)

    X = df[FEATURE_COLUMNS].to_numpy()
    y = df[TARGET_COLUMN].to_numpy()
    strata = pd.qcut(df[TARGET_COLUMN], q=4, labels=False)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=strata
    )

    registry = build_registry(random_seed=RANDOM_SEED)
    trained_models = [
        train_and_tune(spec, X_train, y_train, X_test, y_test, random_seed=RANDOM_SEED)
        for spec in registry
    ]
    best = select_best(trained_models)

    X_train_for_best = X_train
    if best.scaler is not None:
        X_train_for_best = best.scaler.transform(X_train)
    residuals_train = y_train - best.estimator.predict(X_train_for_best)

    feature_ranges = {
        column: (float(X_train[:, i].min()), float(X_train[:, i].max()))
        for i, column in enumerate(FEATURE_COLUMNS)
    }

    artifact = {
        "estimator": best.estimator,
        "scaler": best.scaler,
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "confidence_method": best.spec.confidence_method,
        "is_tree_based": best.spec.is_tree_based,
        "X_train": X_train_for_best,
        "y_train": y_train,
        "residual_std": float(residuals_train.std()),
        "residuals_train": residuals_train,
        "feature_ranges": feature_ranges,
    }
    joblib.dump(artifact, BEST_MODEL_PATH)

    metadata = {
        "model_name": best.spec.name,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "metrics": best.metrics,
    }
    MODEL_METADATA_PATH.write_text(json.dumps(metadata, indent=2))

    comparison = {trained.spec.name: trained.metrics for trained in trained_models}
    MODEL_COMPARISON_PATH.write_text(json.dumps(comparison, indent=2))

    print(f"Best model: {best.spec.name} (RMSE={best.metrics['rmse']:.3f}, R2={best.metrics['r2']:.3f})")
    print(f"Saved {BEST_MODEL_PATH}, {MODEL_METADATA_PATH}, {MODEL_COMPARISON_PATH}")


if __name__ == "__main__":
    main()
