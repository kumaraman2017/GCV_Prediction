import numpy as np


def _to_feature_vector(raw_input: dict, feature_columns: list[str]) -> np.ndarray:
    return np.array([raw_input[column] for column in feature_columns], dtype=float)


def _transform_for_model(raw_vector: np.ndarray, scaler) -> np.ndarray:
    if scaler is None:
        return raw_vector
    return scaler.transform(raw_vector.reshape(1, -1))[0]


def _bagging_ensemble_confidence(estimator, X_row: np.ndarray, residual_std: float) -> float:
    tree_predictions = np.array([tree.predict(X_row.reshape(1, -1))[0] for tree in estimator.estimators_])
    spread = float(tree_predictions.std())
    if residual_std <= 0:
        return 100.0
    return 100.0 * float(np.exp(-spread / residual_std))


def _single_tree_confidence(estimator, X_train: np.ndarray, residuals_train: np.ndarray, X_row: np.ndarray, residual_std: float) -> float:
    leaf_id = estimator.apply(X_row.reshape(1, -1))[0]
    train_leaf_ids = estimator.apply(X_train)
    mask = train_leaf_ids == leaf_id
    local_residuals = residuals_train[mask] if mask.sum() > 1 else np.array([residual_std])
    local_spread = float(np.abs(local_residuals).std()) if len(local_residuals) > 1 else float(np.abs(local_residuals).mean())
    if residual_std <= 0:
        return 100.0
    return 100.0 * float(np.exp(-local_spread / residual_std))


def _knn_residual_confidence(X_train: np.ndarray, residuals_train: np.ndarray, X_row: np.ndarray, residual_std: float, k: int = 15) -> float:
    distances = np.linalg.norm(X_train - X_row, axis=1)
    neighbor_count = min(k, len(distances))
    nearest_idx = np.argsort(distances)[:neighbor_count]
    local_spread = float(np.abs(residuals_train[nearest_idx]).std()) if neighbor_count > 1 else float(np.abs(residuals_train[nearest_idx]).mean())
    if residual_std <= 0:
        return 100.0
    return 100.0 * float(np.exp(-local_spread / residual_std))


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

    method = artifact["confidence_method"]
    residual_std = artifact["residual_std"]

    if method == "bagging_ensemble":
        base_score = _bagging_ensemble_confidence(artifact["estimator"], X_row, residual_std)
    elif method == "single_tree":
        base_score = _single_tree_confidence(
            artifact["estimator"], artifact["X_train"], artifact["residuals_train"], X_row, residual_std
        )
    elif method == "knn_residual":
        base_score = _knn_residual_confidence(
            artifact["X_train"], artifact["residuals_train"], X_row, residual_std
        )
    else:
        raise ValueError(f"Unknown confidence_method: {method}")

    penalty = _extrapolation_penalty(raw_input, artifact["feature_ranges"])
    return max(0.0, min(100.0, base_score * penalty))
