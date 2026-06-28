from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler

from src.models.registry import ModelSpec


@dataclass
class TrainedModel:
    spec: ModelSpec
    estimator: Any
    scaler: Optional[StandardScaler]
    metrics: dict


def train_and_tune(
    spec: ModelSpec,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    random_seed: int,
) -> TrainedModel:
    scaler = None
    X_train_used, X_test_used = X_train, X_test

    if spec.needs_scaling:
        scaler = StandardScaler().fit(X_train)
        X_train_used = scaler.transform(X_train)
        X_test_used = scaler.transform(X_test)

    if spec.param_distributions:
        search = RandomizedSearchCV(
            spec.estimator_factory(),
            param_distributions=spec.param_distributions,
            n_iter=10,
            cv=5,
            scoring="neg_root_mean_squared_error",
            random_state=random_seed,
            n_jobs=-1,
        )
        search.fit(X_train_used, y_train)
        estimator = search.best_estimator_
    else:
        estimator = spec.estimator_factory()
        estimator.fit(X_train_used, y_train)

    predictions = estimator.predict(X_test_used)
    metrics = {
        "rmse": float(root_mean_squared_error(y_test, predictions)),
        "mae": float(mean_absolute_error(y_test, predictions)),
        "r2": float(r2_score(y_test, predictions)),
    }
    return TrainedModel(spec=spec, estimator=estimator, scaler=scaler, metrics=metrics)


def select_best(trained_models: list[TrainedModel]) -> TrainedModel:
    return min(trained_models, key=lambda trained: trained.metrics["rmse"])
