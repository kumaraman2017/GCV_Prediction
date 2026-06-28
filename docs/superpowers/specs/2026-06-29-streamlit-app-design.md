# Streamlit Demo App — Design

**Goal:** A focused, deployable Streamlit MVP for the Coal GCV model: predict + confidence + SHAP explanation, on one page. Reuses the existing trained model and ML pipeline modules directly — no backend.

## Scope

- One page, four inputs (Moisture, Volatile_matter, Fixed_Carbon, Std.Ash), Predict/Reset/Load-example buttons.
- Outputs: predicted GCV, confidence score (via `ml/src/models/confidence.py`'s `compute_confidence`), SHAP per-prediction contribution chart + plain-language sentence (via `ml/src/explain/shap_explain.py`'s `explain_single_prediction` / `generate_explanation_sentence`).
- Out of scope (per the original spec's later sub-projects, not rebuilt here): Analytics, Model Performance, Dataset Explorer, Prediction History, About pages; FastAPI backend; React frontend.

## Files

- `streamlit_app.py` (repo root) — the app.
- `requirements.txt` (repo root) — deployment-only deps, pinned to locally-installed versions for joblib-unpickle compatibility: `streamlit`, `pandas==3.0.4`, `numpy==2.4.6`, `scikit-learn==1.9.0`, `joblib==1.5.3`, `shap==0.51.0`, `matplotlib`.
- `.python-version` (repo root) — `3.11`, for Streamlit Community Cloud's Python pin.

## Reuse strategy

`streamlit_app.py` inserts `ml/` onto `sys.path` (same pattern already used by `ml/notebooks/build_eda_notebook.py`) and imports `src.config`, `src.models.confidence.compute_confidence`, `src.explain.shap_explain.build_explainer/explain_single_prediction/generate_explanation_sentence` directly — no duplicated logic.

## Prediction flow

1. Load `models/best_model.joblib` once (`st.cache_resource`).
2. On Predict: build raw feature vector → apply `artifact["scaler"]` if present → `artifact["estimator"].predict(...)` → predicted GCV.
3. `compute_confidence(artifact, raw_input)` → confidence score.
4. Build a SHAP explainer via `build_explainer(artifact["is_tree_based"], artifact["estimator"], artifact["X_train"])` (cached), call `explain_single_prediction` → bar chart of the four contributions + `generate_explanation_sentence`.

## Deployment

Push to GitHub (already the case); user deploys via share.streamlit.io pointing at `streamlit_app.py` on `master` — requires their account, not something I can do directly.
