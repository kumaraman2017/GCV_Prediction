import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from src.models.confidence import _extrapolation_penalty, compute_confidence, compute_confidence_interval

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


def test_confidence_interval_contains_prediction():
    X, y = _synthetic_training_data()
    estimator = RandomForestRegressor(n_estimators=20, random_state=42).fit(X, y)
    artifact = _base_artifact(estimator, X, y, "bagging_ensemble")
    raw_input = _sample_input(X)
    prediction = float(estimator.predict(X[0].reshape(1, -1))[0])

    low, high = compute_confidence_interval(artifact, raw_input, prediction)
    assert low <= prediction <= high


def test_confidence_interval_widens_for_out_of_range_input():
    X, y = _synthetic_training_data()
    estimator = RandomForestRegressor(n_estimators=20, random_state=42).fit(X, y)
    artifact = _base_artifact(estimator, X, y, "bagging_ensemble")
    prediction = float(estimator.predict(X[0].reshape(1, -1))[0])

    in_range_input = _sample_input(X)
    out_of_range_input = dict(in_range_input)
    out_of_range_input["Moisture"] = artifact["feature_ranges"]["Moisture"][1] + 50.0

    in_range_low, in_range_high = compute_confidence_interval(artifact, in_range_input, prediction)
    out_of_range_low, out_of_range_high = compute_confidence_interval(artifact, out_of_range_input, prediction)

    assert (out_of_range_high - out_of_range_low) > (in_range_high - in_range_low)


def test_confidence_interval_larger_z_widens_interval():
    X, y = _synthetic_training_data()
    estimator = LinearRegression().fit(X, y)
    artifact = _base_artifact(estimator, X, y, "knn_residual")
    raw_input = _sample_input(X)
    prediction = float(estimator.predict(X[0].reshape(1, -1))[0])

    narrow_low, narrow_high = compute_confidence_interval(artifact, raw_input, prediction, z=1.0)
    wide_low, wide_high = compute_confidence_interval(artifact, raw_input, prediction, z=2.0)

    assert (wide_high - wide_low) > (narrow_high - narrow_low)
