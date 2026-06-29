import numpy as np

Z_90 = 1.645  # two-sided normal z-score for a ~90% interval


def _to_feature_vector(raw_input: dict, feature_columns: list[str]) -> np.ndarray:
    return np.array([raw_input[column] for column in feature_columns], dtype=float)


def _transform_for_model(raw_vector: np.ndarray, scaler) -> np.ndarray:
    if scaler is None:
        return raw_vector
    return scaler.transform(raw_vector.reshape(1, -1))[0]


def _bagging_ensemble_spread(estimator, X_row: np.ndarray) -> float:
    tree_predictions = np.array([tree.predict(X_row.reshape(1, -1))[0] for tree in estimator.estimators_])
    return float(tree_predictions.std())


def _single_tree_spread(estimator, X_train: np.ndarray, residuals_train: np.ndarray, X_row: np.ndarray, fallback: float) -> float:
    leaf_id = estimator.apply(X_row.reshape(1, -1))[0]
    train_leaf_ids = estimator.apply(X_train)
    mask = train_leaf_ids == leaf_id
    local_residuals = residuals_train[mask] if mask.sum() > 1 else np.array([fallback])
    return float(np.abs(local_residuals).std()) if len(local_residuals) > 1 else float(np.abs(local_residuals).mean())


def _knn_residual_spread(X_train: np.ndarray, residuals_train: np.ndarray, X_row: np.ndarray, k: int = 15) -> float:
    distances = np.linalg.norm(X_train - X_row, axis=1)
    neighbor_count = min(k, len(distances))
    nearest_idx = np.argsort(distances)[:neighbor_count]
    local_residuals = residuals_train[nearest_idx]
    return float(np.abs(local_residuals).std()) if neighbor_count > 1 else float(np.abs(local_residuals).mean())


def compute_spread(artifact: dict, X_row: np.ndarray) -> float:
    method = artifact["confidence_method"]
    if method == "bagging_ensemble":
        return _bagging_ensemble_spread(artifact["estimator"], X_row)
    if method == "single_tree":
        return _single_tree_spread(
            artifact["estimator"], artifact["X_train"], artifact["residuals_train"], X_row, artifact["residual_std"]
        )
    if method == "knn_residual":
        return _knn_residual_spread(artifact["X_train"], artifact["residuals_train"], X_row)
    raise ValueError(f"Unknown confidence_method: {method}")


def _extrapolation_penalty(raw_input: dict, feature_ranges: dict) -> float:
    penalty = 1.0
    for feature, value in raw_input.items():
        low, high = feature_ranges[feature]
        span = high - low
        if value < low:
            distance = low - value
        elif value > high:
            distance = value - high
        else:
            continue
        penalty *= max(0.5, 1.0 - (distance / span))
    return penalty


def compute_confidence(artifact: dict, raw_input: dict) -> float:
    feature_columns = artifact["feature_columns"]
    raw_vector = _to_feature_vector(raw_input, feature_columns)
    X_row = _transform_for_model(raw_vector, artifact["scaler"])

    spread = compute_spread(artifact, X_row)
    residual_std = artifact["residual_std"]
    base_score = 100.0 if residual_std <= 0 else 100.0 * float(np.exp(-spread / residual_std))

    penalty = _extrapolation_penalty(raw_input, artifact["feature_ranges"])
    return max(0.0, min(100.0, base_score * penalty))


def compute_confidence_interval(artifact: dict, raw_input: dict, prediction: float, z: float = Z_90) -> tuple[float, float]:
    """Prediction interval combining ensemble/local spread (epistemic) with the
    model's test-set RMSE (irreducible error), widened when inputs extrapolate
    beyond the training data's observed range."""
    feature_columns = artifact["feature_columns"]
    raw_vector = _to_feature_vector(raw_input, feature_columns)
    X_row = _transform_for_model(raw_vector, artifact["scaler"])

    spread = compute_spread(artifact, X_row)
    residual_std = artifact["residual_std"]
    total_uncertainty = float(np.sqrt(spread**2 + residual_std**2))

    penalty = max(_extrapolation_penalty(raw_input, artifact["feature_ranges"]), 1e-6)
    half_width = z * total_uncertainty / penalty

    return prediction - half_width, prediction + half_width
