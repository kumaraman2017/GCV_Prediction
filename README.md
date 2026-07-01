<div align="center">

# ⛏️ Coal GCV Predictor

**Predict the Gross Calorific Value of coal from proximate analysis — with explainable AI and a 90% statistical confidence interval.**

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://gcvprediction-z2gkxweznrnkrqp6fgvwfd.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![SHAP](https://img.shields.io/badge/SHAP-XAI-29B5E8?style=for-the-badge)](https://shap.readthedocs.io/)

</div>

---

## What it does

Coal laboratories report four proximate-analysis measurements for every sample — **Moisture**, **Volatile Matter**, **Fixed Carbon**, and **Ash**. From these four numbers alone, this app predicts the coal's **Gross Calorific Value (GCV)** in kcal/kg, displays a **90% confidence interval** around the prediction, and explains exactly *why* the model made that prediction using **SHAP feature attribution**.

> **Project context:** Final-year engineering capstone — a production-grade ML pipeline with an interactive web front-end, end-to-end from raw CSV to live cloud deployment.

---

## Table of Contents

- [Live Demo](#live-demo)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [ML Pipeline](#ml-pipeline)
- [Model Results](#model-results)
- [Run Locally](#run-locally)
- [Retrain the Model](#retrain-the-model)
- [Tests](#tests)

---

## Live Demo

**[gcvprediction-z2gkxweznrnkrqp6fgvwfd.streamlit.app](https://gcvprediction-z2gkxweznrnkrqp6fgvwfd.streamlit.app/)**

The app is deployed on Streamlit Community Cloud and requires no login. Drag the sliders, hit **Predict**, and the model responds instantly with:
- Predicted GCV (kcal/kg)
- A 90% confidence interval chart
- A SHAP waterfall explaining each feature's contribution

---

## Features

| Feature | Details |
|---|---|
| **Custom slider UI** | Glass-card widgets with +/− step buttons, gradient slider track, real-time sum validation |
| **Dark / Light theme** | One-click toggle; glassmorphism design with Space Grotesk font |
| **Statistical confidence** | 90% CI = ensemble spread ⊕ model RMSE, widened for out-of-distribution inputs |
| **SHAP explainability** | Per-prediction waterfall chart showing each feature's push up/down in kcal/kg |
| **Natural-language summary** | Auto-generated sentence naming the top positive and negative drivers |
| **Load example** | Randomly pulls a real row from the cleaned dataset for quick testing |
| **Model RMSE display** | Shows test-set error alongside every prediction for honest uncertainty communication |

---

## Tech Stack

| Layer | Library / Tool |
|---|---|
| ML training | scikit-learn 1.9, ExtraTreesRegressor (winner of 10-model comparison) |
| Explainability | SHAP 0.51 |
| Data | pandas 3.0, NumPy 2.4 |
| Web app | Streamlit ≥ 1.38 |
| Charts | Plotly ≥ 5.24 |
| Persistence | joblib |
| Testing | pytest (44 tests, 100% pass) |
| Deployment | Streamlit Community Cloud |
| Language | Python 3.11 |

---

## Project Structure

```
gcv_coal_project/
│
├── streamlit_app.py          # Main Streamlit app entry point
├── requirements.txt          # Runtime deps for Streamlit Cloud
├── .python-version           # 3.11 (pinned for Streamlit Cloud)
│
├── ui/                       # Front-end layer
│   ├── theme.py              # Dark/light theme dataclasses
│   ├── styles.py             # Glassmorphism CSS injection
│   ├── components.py         # render_feature_input(), KPI cards, header, footer
│   └── charts.py             # confidence_interval_chart(), shap_waterfall()
│
├── ml/                       # Self-contained ML pipeline
│   ├── run_pipeline.py       # Runs clean → train → explain in sequence
│   ├── run_clean.py          # Data validation and cleaning
│   ├── run_train.py          # Model training, selection, and artifact export
│   ├── run_explain.py        # SHAP global feature importance
│   ├── requirements.txt      # ML-specific deps (includes dev/test tools)
│   │
│   ├── src/
│   │   ├── config.py         # Paths and constants (FEATURE_COLUMNS, TARGET_COLUMN, …)
│   │   ├── data/
│   │   │   ├── schema.py     # Pydantic-style row validation
│   │   │   └── clean.py      # Proximate-sum constraint, outlier removal
│   │   ├── models/
│   │   │   ├── registry.py   # All candidate models with hyperparameter grids
│   │   │   ├── train.py      # CV + GridSearch, RMSE-based selection
│   │   │   └── confidence.py # Ensemble spread, kNN-residual, extrapolation penalty
│   │   └── explain/
│   │       └── shap_explain.py  # TreeExplainer / LinearExplainer, waterfall data
│   │
│   ├── tests/                # 12 test modules, 44 tests
│   └── notebooks/            # Exploratory data analysis
│
├── data/
│   ├── raw/coal_all.csv      # Original dataset
│   └── processed/            # Cleaned CSV + data quality report (git-ignored)
│
└── models/                   # Trained artefacts (git-tracked)
    ├── best_model.joblib     # 11-key artifact bundle
    ├── model_metadata.json   # Winner name + metrics
    ├── model_comparison.json # All 10 models' RMSE/MAE/R²
    └── shap_global.json      # Global feature importance
```

---

## ML Pipeline

The pipeline runs three sequential stages:

```
data/raw/coal_all.csv
        │
        ▼
 1. run_clean.py
    ├─ Validates Moisture + VM + FC + Ash ≈ 100% (±0.5%)
    ├─ Removes rows with missing or physically impossible values
    └─ Writes data/processed/coal_clean.csv
        │
        ▼
 2. run_train.py
    ├─ Trains ~10 regression models (Linear, Ridge, Lasso, SVR,
    │   KNN, DecisionTree, RandomForest, GradientBoosting,
    │   ExtraTrees, XGBoost)
    ├─ Tunes each with GridSearchCV (5-fold, RMSE scorer)
    ├─ Selects best by hold-out test RMSE
    └─ Writes models/best_model.joblib (artifact bundle)
        │
        ▼
 3. run_explain.py
    ├─ Builds SHAP explainer (TreeExplainer or LinearExplainer)
    └─ Writes models/shap_global.json
```

### Artifact bundle

`best_model.joblib` is a single dict containing everything the app needs at inference time:

| Key | Description |
|---|---|
| `estimator` | Fitted sklearn estimator |
| `scaler` | Fitted StandardScaler or `None` |
| `feature_columns` | `["Moisture", "Volatile_matter", "Fixed_Carbon", "Std.Ash"]` |
| `target_column` | `"GCV"` |
| `confidence_method` | `"bagging_ensemble"` / `"single_tree"` / `"knn_residual"` |
| `is_tree_based` | Boolean — selects SHAP explainer type |
| `X_train` | Training set features (scaled if applicable) |
| `y_train` | Training set targets |
| `residuals_train` | Per-sample training residuals |
| `residual_std` | Test RMSE — used as uncertainty baseline |
| `feature_ranges` | `{feature: (min, max)}` — extrapolation detection |

### Confidence interval

The 90% CI is derived as:

```
half_width = Z₉₀ × √(spread² + residual_std²) / extrapolation_penalty

where:
  Z₉₀             = 1.645
  spread           = ensemble std (bagging) | leaf-local std (tree) | kNN-local residual std
  residual_std     = test RMSE of the winning model
  extrapolation_penalty ∈ (0, 1] — shrinks to <1 when any input falls outside training ranges
```

---

## Model Results

Ten regression models were evaluated on an 80/20 train/test split:

| Rank | Model | Test RMSE (MJ/kg) | R² |
|---|---|---|---|
| **1** | **ExtraTreesRegressor** | **0.6425** | **0.990** |
| 2 | RandomForestRegressor | ~0.70 | ~0.988 |
| 3 | GradientBoostingRegressor | ~0.75 | ~0.986 |
| 4+ | SVR, KNN, Ridge, … | higher | lower |

> RMSE of 0.6425 MJ/kg ≈ **±153 kcal/kg** on the test set.

---

## Run Locally

```bash
# 1. Clone
git clone https://github.com/kumaraman2017/GCV_Prediction.git
cd GCV_Prediction

# 2. Install runtime dependencies
pip install -r requirements.txt

# 3. Launch the app
streamlit run streamlit_app.py
```

The app opens at `http://localhost:8501`. The trained model is already committed to the repo under `models/` — no retraining needed to run the app.

---

## Retrain the Model

```bash
# 1. Create the ML virtual environment
cd ml
py -3.11 -m venv ../.venv
../.venv/Scripts/python.exe -m pip install -r requirements.txt

# 2. Run the full pipeline (clean → train → explain)
../.venv/Scripts/python.exe run_pipeline.py
```

Output artefacts are written to `models/` and `data/processed/`. The Streamlit app picks up the new model on next load (it calls `st.cache_resource`, so restart the server or clear cache if already running).

---

## Tests

```bash
cd ml
../.venv/Scripts/python.exe -m pytest
```

**44 tests across 12 modules** — all passing in ~2 minutes (SHAP JIT warmup dominates).

| Module | What it covers |
|---|---|
| `test_config.py` | Path resolution, constant types |
| `test_schema.py` | Row-level validation rules |
| `test_clean.py` | Proximate-sum constraint, missing-value handling |
| `test_run_clean.py` | End-to-end cleaning output files |
| `test_registry.py` | All models instantiate without error |
| `test_train.py` | CV selection, artifact structure |
| `test_run_train.py` | Persisted artifact completeness and array consistency |
| `test_confidence.py` | CI bounds, out-of-range widening, Z-scaling, unknown method error |
| `test_shap_explain.py` | SHAP additivity for tree and linear explainers |
| `test_run_explain.py` | SHAP global output file |
| `test_run_pipeline.py` | Full pipeline smoke test |
| `test_eda_notebook.py` | EDA notebook executes without error |

---

<div align="center">

Made with Python · scikit-learn · SHAP · Streamlit

</div>
