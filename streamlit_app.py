import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
ML_DIR = APP_DIR / "ml"
sys.path.insert(0, str(ML_DIR))

from src.config import CLEAN_DATA_PATH, FEATURE_COLUMNS, MODEL_METADATA_PATH  # noqa: E402
from src.explain.shap_explain import (  # noqa: E402
    build_explainer,
    explain_single_prediction,
    generate_explanation_sentence,
)
from src.models.confidence import compute_confidence  # noqa: E402

BEST_MODEL_PATH = APP_DIR / "models" / "best_model.joblib"

KCAL_PER_MJ = 1000 / 4.1868  # GCV is predicted in MJ/kg; coal industry conventionally reports kcal/kg

DEFAULTS = {
    "Moisture": 5.20,
    "Volatile_matter": 31.10,
    "Fixed_Carbon": 40.70,
    "Std.Ash": 23.00,
}

st.set_page_config(page_title="Coal GCV Predictor", page_icon="⛏️", layout="centered")


@st.cache_resource
def load_artifact():
    return joblib.load(BEST_MODEL_PATH)


@st.cache_resource
def load_explainer(_artifact):
    return build_explainer(_artifact["is_tree_based"], _artifact["estimator"], _artifact["X_train"])


@st.cache_data
def load_clean_dataset():
    return pd.read_csv(CLEAN_DATA_PATH)


@st.cache_data
def load_metadata():
    return json.loads(MODEL_METADATA_PATH.read_text())


def reset_inputs():
    for column, value in DEFAULTS.items():
        st.session_state[column] = value


def load_example():
    df = load_clean_dataset()
    row = df.sample(n=1).iloc[0]
    for column in FEATURE_COLUMNS:
        st.session_state[column] = float(row[column])


for column, value in DEFAULTS.items():
    st.session_state.setdefault(column, value)

st.title("Coal GCV Predictor")
st.caption(
    "Predicts Gross Calorific Value (GCV) from proximate analysis "
    "(Moisture, Volatile Matter, Fixed Carbon, Ash), with a confidence "
    "score and a SHAP-based explanation of the prediction."
)

input_col1, input_col2 = st.columns(2)
with input_col1:
    st.number_input("Moisture (%)", min_value=0.0, max_value=100.0, step=0.1, key="Moisture")
    st.number_input("Volatile Matter (%)", min_value=0.0, max_value=100.0, step=0.1, key="Volatile_matter")
with input_col2:
    st.number_input("Fixed Carbon (%)", min_value=0.0, max_value=100.0, step=0.1, key="Fixed_Carbon")
    st.number_input("Ash (%)", min_value=0.0, max_value=100.0, step=0.1, key="Std.Ash")

raw_input = {column: float(st.session_state[column]) for column in FEATURE_COLUMNS}
total = sum(raw_input.values())
if abs(total - 100.0) <= 0.5:
    st.caption(f"Sum of inputs: {total:.1f}% ✅")
else:
    st.caption(f"Sum of inputs: {total:.1f}% ⚠️ — proximate analysis values normally sum to ~100%.")

button_col1, button_col2, button_col3 = st.columns(3)
predict_clicked = button_col1.button("Predict", type="primary", use_container_width=True)
button_col2.button("Reset", on_click=reset_inputs, use_container_width=True)
button_col3.button("Load example", on_click=load_example, use_container_width=True)

if predict_clicked:
    artifact = load_artifact()

    feature_vector = np.array([raw_input[column] for column in FEATURE_COLUMNS], dtype=float)
    model_input = feature_vector
    if artifact["scaler"] is not None:
        model_input = artifact["scaler"].transform(feature_vector.reshape(1, -1))[0]

    prediction_mj = float(artifact["estimator"].predict(model_input.reshape(1, -1))[0])
    prediction_kcal = prediction_mj * KCAL_PER_MJ
    confidence = compute_confidence(artifact, raw_input)

    explainer = load_explainer(artifact)
    explanation = explain_single_prediction(explainer, model_input, FEATURE_COLUMNS, raw_input)
    contributions_kcal = [
        {**contribution, "shap_value": contribution["shap_value"] * KCAL_PER_MJ}
        for contribution in explanation["contributions"]
    ]
    sentence = generate_explanation_sentence(contributions_kcal, unit="kcal/kg")

    metadata = load_metadata()
    rmse_kcal = metadata["metrics"]["rmse"] * KCAL_PER_MJ

    st.divider()
    result_col1, result_col2, result_col3 = st.columns(3)
    result_col1.metric("Predicted GCV", f"{prediction_kcal:,.0f} kcal/kg")
    result_col2.metric("Confidence", f"{confidence:.0f}%")
    result_col3.metric(
        "Model RMSE",
        f"± {rmse_kcal:,.0f} kcal/kg",
        help=f"Typical error of the {metadata['model_name']} model on held-out test data — not specific to this prediction.",
    )

    st.subheader("Why this prediction?")
    st.write(sentence)

    contributions_df = pd.DataFrame(contributions_kcal).set_index("feature")["shap_value"]
    st.bar_chart(contributions_df)
