# Coal GCV Prediction

Predicts Coal Gross Calorific Value (GCV) from proximate analysis (Moisture, Volatile Matter, Fixed Carbon, Ash), with a per-prediction confidence score and SHAP-based explainability.

## Live demo

A Streamlit app (`streamlit_app.py`) serves the trained model directly — no backend required.

Run locally:
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

To deploy on [Streamlit Community Cloud](https://share.streamlit.io): connect this repo, set the main file path to `streamlit_app.py`, and deploy.

## ML pipeline

The model itself — data cleaning, training/tuning across ~10 regression models, confidence scoring, and SHAP explainability — lives under [`ml/`](ml/). See [`ml/README.md`](ml/README.md) for how to retrain it.

## Project layout

- `data/` — raw and cleaned datasets
- `ml/` — training pipeline, tests, EDA notebook
- `models/` — persisted trained model and metrics
- `streamlit_app.py` — the demo app
- `docs/` — design specs and implementation plans
