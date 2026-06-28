import json

import joblib
import pandas as pd

from src.config import BEST_MODEL_PATH, CLEAN_DATA_PATH, FEATURE_COLUMNS, SHAP_GLOBAL_PATH
from src.explain.shap_explain import build_explainer, global_feature_importance, summary_plot_data

SAMPLE_SIZE = 200


def main() -> None:
    artifact = joblib.load(BEST_MODEL_PATH)
    df = pd.read_csv(CLEAN_DATA_PATH)

    sample_df = df.sample(n=min(SAMPLE_SIZE, len(df)), random_state=42)
    raw_X_sample = sample_df[FEATURE_COLUMNS].to_numpy()

    X_sample = raw_X_sample
    if artifact["scaler"] is not None:
        X_sample = artifact["scaler"].transform(raw_X_sample)

    explainer = build_explainer(artifact["is_tree_based"], artifact["estimator"], artifact["X_train"])

    importance = global_feature_importance(explainer, X_sample, FEATURE_COLUMNS)
    summary_data = summary_plot_data(explainer, X_sample, raw_X_sample, FEATURE_COLUMNS)

    output = {
        "global_feature_importance": importance,
        "summary_plot_sample": summary_data,
    }
    SHAP_GLOBAL_PATH.write_text(json.dumps(output, indent=2))
    print(f"Wrote {SHAP_GLOBAL_PATH} ({len(summary_data)} summary-plot records)")


if __name__ == "__main__":
    main()
