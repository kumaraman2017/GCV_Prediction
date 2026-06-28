import numpy as np
from sklearn.tree import DecisionTreeRegressor

from src.explain.shap_explain import (
    build_explainer,
    explain_single_prediction,
    generate_explanation_sentence,
    global_feature_importance,
    summary_plot_data,
)

FEATURE_COLUMNS = ["Moisture", "Volatile_matter", "Fixed_Carbon", "Std.Ash"]


def _fit_model():
    rng = np.random.RandomState(0)
    X = rng.uniform(5, 40, size=(100, 4))
    y = X[:, 0] * 0.3 + X[:, 1] * 0.5 - X[:, 2] * 0.2 + X[:, 3] * 0.1
    model = DecisionTreeRegressor(max_depth=4, random_state=42).fit(X, y)
    return model, X, y


def test_shap_values_sum_to_prediction_minus_base_value():
    model, X, y = _fit_model()
    explainer = build_explainer(is_tree_based=True, estimator=model, background=X)
    raw_values = {name: float(X[0, idx]) for idx, name in enumerate(FEATURE_COLUMNS)}

    result = explain_single_prediction(explainer, X[0], FEATURE_COLUMNS, raw_values)
    actual_prediction = float(model.predict(X[0].reshape(1, -1))[0])

    shap_sum = sum(c["shap_value"] for c in result["contributions"])
    assert abs(result["base_value"] + shap_sum - actual_prediction) < 1e-4
    assert abs(result["prediction"] - actual_prediction) < 1e-4


def test_global_feature_importance_has_all_features():
    model, X, y = _fit_model()
    explainer = build_explainer(is_tree_based=True, estimator=model, background=X)
    importance = global_feature_importance(explainer, X, FEATURE_COLUMNS)
    assert set(importance.keys()) == set(FEATURE_COLUMNS)
    assert all(value >= 0 for value in importance.values())


def test_summary_plot_data_has_expected_row_count():
    model, X, y = _fit_model()
    explainer = build_explainer(is_tree_based=True, estimator=model, background=X)
    records = summary_plot_data(explainer, X, X, FEATURE_COLUMNS)
    assert len(records) == X.shape[0] * len(FEATURE_COLUMNS)


def test_generate_explanation_sentence_mentions_increasing_and_decreasing_features():
    contributions = [
        {"feature": "Fixed_Carbon", "value": 40.0, "shap_value": 2.3},
        {"feature": "Std.Ash", "value": 23.0, "shap_value": -3.0},
        {"feature": "Moisture", "value": 5.0, "shap_value": 0.01},
    ]
    sentence = generate_explanation_sentence(contributions)
    assert "Fixed_Carbon" in sentence
    assert "Std.Ash" in sentence
    assert "increased" in sentence
    assert "decreased" in sentence
