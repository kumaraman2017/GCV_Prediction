import json

from src.config import FEATURE_COLUMNS, SHAP_GLOBAL_PATH


def test_run_explain_persists_global_feature_importance():
    output = json.loads(SHAP_GLOBAL_PATH.read_text())
    assert set(output["global_feature_importance"].keys()) == set(FEATURE_COLUMNS)
    assert len(output["summary_plot_sample"]) > 0
    assert set(output["summary_plot_sample"][0].keys()) == {"feature", "feature_value", "shap_value"}
