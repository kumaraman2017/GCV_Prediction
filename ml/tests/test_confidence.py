import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from src.models.confidence import _extrapolation_penalty, compute_confidence

FEATURE_COLUMNS = ["Moisture", "Volatile_matter", "Fixed_Carbon", "Std.Ash"]


def _synthetic_training_data(n=100, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.uniform(5, 40, size=(n, 4))
    y = X[:, 0] * 0.3 + X[:, 1] * 0.5 - X[:, 2] * 0.2 + rng.normal(0, 0.5, size=n)
    return X, y


def _feature_ranges(X: np.ndarray) -> dict:
    return {
        column: (float(X[:, i].min()), float(X[:, i].max()))
        for i, column in enumerate(FEATURE_COLUMNS)
    }


def _sample_input(X: np.ndarray) -> dict:
    row = X[0]
    return {column: float(row[i]) for i, column in enumerate(FEATURE_COLUMNS)}


def _base_artifact(estimator, X, y, method):
    residuals = y - estimator.predict(X)
    return {
        "feature_columns": FEATURE_COLUMNS,
        "scaler": None,
        "confidence_method": method,
        "estimator": estimator,
        "X_train": X,
        "y_train": y,
        "residuals_train": residuals,
        "residual_std": float(residuals.std()),
        "feature_ranges": _feature_ranges(X),
    }


def test_bagging_ensemble_confidence_in_valid_range():
    X, y = _synthetic_training_data()
    estimator = RandomForestRegressor(n_estimators=20, random_state=42).fit(X, y)
    artifact = _base_artifact(estimator, X, y, "bagging_ensemble")
    score = compute_confidence(artifact, _sample_input(X))
    assert 0.0 <= score <= 100.0


def test_single_tree_confidence_in_valid_range():
    X, y = _synthetic_training_data()
    estimator = DecisionTreeRegressor(max_depth=4, random_state=42).fit(X, y)
    artifact = _base_artifact(estimator, X, y, "single_tree")
    score = compute_confidence(artifact, _sample_input(X))
    assert 0.0 <= score <= 100.0


def test_knn_residual_confidence_in_valid_range():
    X, y = _synthetic_training_data()
    estimator = LinearRegression().fit(X, y)
    artifact = _base_artifact(estimator, X, y, "knn_residual")
    score = compute_confidence(artifact, _sample_input(X))
    assert 0.0 <= score <= 100.0


def test_extrapolation_penalty_is_one_when_in_range():
    feature_ranges = {col: (0.0, 50.0) for col in FEATURE_COLUMNS}
    raw_input = {"Moisture": 10.0, "Volatile_matter": 20.0, "Fixed_Carbon": 30.0, "Std.Ash": 25.0}
    assert _extrapolation_penalty(raw_input, feature_ranges) == 1.0


def test_extrapolation_penalty_below_one_when_out_of_range():
    feature_ranges = {col: (0.0, 50.0) for col in FEATURE_COLUMNS}
    raw_input = {"Moisture": 100.0, "Volatile_matter": 20.0, "Fixed_Carbon": 30.0, "Std.Ash": 25.0}
    penalty = _extrapolation_penalty(raw_input, feature_ranges)
    assert 0.5 <= penalty < 1.0
