from typing import Any

import numpy as np
import shap


def build_explainer(is_tree_based: bool, estimator: Any, background: np.ndarray):
    if is_tree_based:
        return shap.TreeExplainer(estimator)
    return shap.LinearExplainer(estimator, background)


def _expected_value(explainer) -> float:
    base_value = explainer.expected_value
    if isinstance(base_value, (list, np.ndarray)):
        return float(np.asarray(base_value).flatten()[0])
    return float(base_value)


def global_feature_importance(explainer, X_sample: np.ndarray, feature_names: list[str]) -> dict:
    shap_values = np.asarray(explainer.shap_values(X_sample))
    mean_abs = np.abs(shap_values).mean(axis=0)
    return {name: float(value) for name, value in zip(feature_names, mean_abs)}


def summary_plot_data(explainer, X_sample: np.ndarray, raw_X_sample: np.ndarray, feature_names: list[str]) -> list[dict]:
    shap_values = np.asarray(explainer.shap_values(X_sample))
    records = []
    for row_idx in range(X_sample.shape[0]):
        for col_idx, name in enumerate(feature_names):
            records.append({
                "feature": name,
                "feature_value": float(raw_X_sample[row_idx, col_idx]),
                "shap_value": float(shap_values[row_idx, col_idx]),
            })
    return records


def explain_single_prediction(explainer, X_row: np.ndarray, feature_names: list[str], raw_values: dict) -> dict:
    shap_values = np.asarray(explainer.shap_values(X_row.reshape(1, -1)))[0]
    base_value = _expected_value(explainer)
    contributions = [
        {"feature": name, "value": raw_values[name], "shap_value": float(shap_values[idx])}
        for idx, name in enumerate(feature_names)
    ]
    contributions.sort(key=lambda c: abs(c["shap_value"]), reverse=True)
    return {
        "base_value": base_value,
        "contributions": contributions,
        "prediction": base_value + float(shap_values.sum()),
    }


def generate_explanation_sentence(contributions: list[dict], top_n: int = 3, unit: str = "MJ/kg") -> str:
    increasing = [c for c in contributions if c["shap_value"] > 0][:top_n]
    decreasing = [c for c in contributions if c["shap_value"] < 0][:top_n]

    parts = []
    if increasing:
        described = ", ".join(f"{c['feature']} ({c['shap_value']:+.2f} {unit})" for c in increasing)
        parts.append(f"{described} increased the predicted GCV")
    if decreasing:
        described = ", ".join(f"{c['feature']} ({c['shap_value']:+.2f} {unit})" for c in decreasing)
        parts.append(f"{described} decreased it")

    if not parts:
        return "No feature had a significant effect on this prediction."
    return ", while ".join(parts) + "."
