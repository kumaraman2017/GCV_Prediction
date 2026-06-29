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
sys.path.insert(0, str(APP_DIR))

from src.config import CLEAN_DATA_PATH, FEATURE_COLUMNS, MODEL_METADATA_PATH  # noqa: E402
from src.explain.shap_explain import (  # noqa: E402
    build_explainer,
    explain_single_prediction,
    generate_explanation_sentence,
)
from src.models.confidence import compute_confidence_interval  # noqa: E402

from ui.charts import confidence_interval_chart, shap_waterfall  # noqa: E402
from ui.components import (  # noqa: E402
    render_feature_input,
    render_footer,
    render_header,
    render_kpi_card,
    render_sum_pill,
)
from ui.styles import inject_global_css  # noqa: E402
from ui.theme import get_theme  # noqa: E402

BEST_MODEL_PATH = APP_DIR / "models" / "best_model.joblib"
REPO_URL = "https://github.com/kumaraman2017/GCV_Prediction"

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

dark_mode = st.session_state.get("dark_mode_toggle", True)
theme = get_theme("dark" if dark_mode else "light")
inject_global_css(theme)

render_header(theme)

st.markdown('<p class="section-heading">Proximate Analysis</p>', unsafe_allow_html=True)
slider_col1, slider_col2 = st.columns(2)
with slider_col1:
    render_feature_input("💧", "Moisture", "Moisture")
    render_feature_input("💨", "Volatile Matter", "Volatile_matter")
with slider_col2:
    render_feature_input("⚡", "Fixed Carbon", "Fixed_Carbon")
    render_feature_input("🪨", "Ash", "Std.Ash")

raw_input = {column: float(st.session_state[column]) for column in FEATURE_COLUMNS}
total = sum(raw_input.values())
render_sum_pill(total, theme)

with st.container(key="card-actions"):
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
    interval_low_mj, interval_high_mj = compute_confidence_interval(artifact, raw_input, prediction_mj)
    interval_low_kcal = interval_low_mj * KCAL_PER_MJ
    interval_high_kcal = interval_high_mj * KCAL_PER_MJ

    explainer = load_explainer(artifact)
    explanation = explain_single_prediction(explainer, model_input, FEATURE_COLUMNS, raw_input)
    contributions_kcal = [
        {**contribution, "shap_value": contribution["shap_value"] * KCAL_PER_MJ}
        for contribution in explanation["contributions"]
    ]
    sentence = generate_explanation_sentence(contributions_kcal, unit="kcal/kg")

    metadata = load_metadata()
    rmse_kcal = metadata["metrics"]["rmse"] * KCAL_PER_MJ

    kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
    with kpi_col1:
        render_kpi_card("🔥", "Predicted GCV", f"{prediction_kcal:,.0f}", "kcal/kg")
    with kpi_col2:
        with st.container(key="card-gauge"):
            st.markdown('<div class="slider-label">90% Confidence Interval</div>', unsafe_allow_html=True)
            interval_fig = confidence_interval_chart(
                interval_low_kcal, interval_high_kcal, prediction_kcal, "kcal/kg", theme
            )
            st.plotly_chart(interval_fig, use_container_width=True, config={"displayModeBar": False})
    with kpi_col3:
        render_kpi_card("📐", "Model RMSE", f"± {rmse_kcal:,.0f}", f"{metadata['model_name']}, test set")

    with st.container(key="card-explain"):
        st.markdown('<p class="section-heading">Why this prediction?</p>', unsafe_allow_html=True)
        st.markdown(f'<p class="explain-sentence">{sentence}</p>', unsafe_allow_html=True)
        base_value_kcal = explanation["base_value"] * KCAL_PER_MJ
        fig = shap_waterfall(contributions_kcal, base_value_kcal, prediction_kcal, "kcal/kg", theme)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

render_footer(REPO_URL)
