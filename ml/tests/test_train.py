import numpy as np

from src.models.registry import ModelSpec, build_registry
from src.models.train import TrainedModel, select_best, train_and_tune


def _synthetic_data(n=200, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.uniform(0, 50, size=(n, 4))
    y = X[:, 0] * 0.3 + X[:, 1] * 0.5 - X[:, 2] * 0.2 + rng.normal(0, 1, size=n)
    return X[:150], y[:150], X[150:], y[150:]


def test_train_and_tune_decision_tree_produces_metrics():
    X_train, y_train, X_test, y_test = _synthetic_data()
    spec = next(s for s in build_registry(random_seed=42) if s.name == "decision_tree")
    trained = train_and_tune(spec, X_train, y_train, X_test, y_test, random_seed=42)
    assert isinstance(trained, TrainedModel)
    assert trained.metrics["rmse"] >= 0
    assert trained.metrics["mae"] >= 0
    assert trained.metrics["r2"] <= 1.0
    assert trained.scaler is None


def test_train_and_tune_applies_scaler_for_linear_models():
    X_train, y_train, X_test, y_test = _synthetic_data()
    spec = next(s for s in build_registry(random_seed=42) if s.name == "linear_regression")
    trained = train_and_tune(spec, X_train, y_train, X_test, y_test, random_seed=42)
    assert trained.scaler is not None
    assert trained.metrics["rmse"] >= 0


def test_select_best_picks_lowest_rmse():
    fake_spec = ModelSpec("fake", lambda: None, {}, False, False, "knn_residual")
    good = TrainedModel(spec=fake_spec, estimator=None, scaler=None, metrics={"rmse": 1.0, "mae": 1.0, "r2": 0.9})
    bad = TrainedModel(spec=fake_spec, estimator=None, scaler=None, metrics={"rmse": 5.0, "mae": 3.0, "r2": 0.5})
    assert select_best([good, bad]) is good
