# ML Pipeline Design — Coal GCV Prediction

**Sub-project 1 of 5** in the Coal GCV Prediction platform (ML Pipeline → Backend API → Frontend Dashboard → Docker Packaging → Docs).

## Goal

Produce a trained, evaluated, and explainable regression model that predicts Coal Gross Calorific Value (GCV) from four proximate-analysis inputs, persisted to disk for the backend API to load.

## Dataset

`data/raw/coal_all.csv` — 4,540 rows, columns: `Moisture, Volatile_matter, Fixed_Carbon, Std.Ash, Hydrogen, Carbon, Nitrogen, Oxygen, Sulfur, GCV`.

Verified properties:
- No missing values.
- 27 exact duplicate rows.
- `Moisture + Volatile_matter + Fixed_Carbon + Std.Ash == 100` exactly for all rows (standard proximate-analysis convention) — used as a domain validation rule.
- Ultimate-analysis columns (`Hydrogen, Carbon, Nitrogen, Oxygen, Sulfur`) exist in the raw data but are **excluded** from model inputs by decision — only the four proximate features are used, matching the project's stated UI/input spec.

## Repository layout (established now, used by all later sub-projects)

```
gcv_coal_project/
├── data/
│   ├── raw/coal_all.csv
│   └── processed/              # cleaned dataset + data quality report
├── ml/
│   ├── src/                    # pipeline source (data, models, explain)
│   ├── notebooks/               # 01_eda.ipynb
│   └── tests/                  # pytest suite for this sub-project
├── models/                     # persisted joblib + json artifacts
├── backend/                    # sub-project 2 (FastAPI)
├── frontend/                   # sub-project 3 (React)
└── docs/
```

## 1. Data validation & preprocessing (`ml/src/data/`)

- Pydantic schema validates the four input columns + GCV are numeric, non-negative, each ≤ 100.
- Domain rule: reject rows where `Moisture+Volatile_matter+Fixed_Carbon+Std.Ash` deviates from 100 by more than ±0.5; log to data quality report.
- Drop the 27 exact duplicate rows.
- Drop the 5 ultimate-analysis columns — not used as model inputs.
- Missing-value handling: per-feature median imputation (defensive; current data has none, but required for the pipeline to be production-ready against new data).
- Outlier analysis: IQR-based detection per feature, counts and bounds written to the data quality report for visibility. Statistical outliers are **not auto-removed** — they represent legitimate coal types (e.g. lignite vs. anthracite) rather than data errors.
- Outputs: `data/processed/coal_clean.csv` (5 columns only — the 4 proximate features + GCV; ultimate-analysis columns are fully dropped, not merely unused), `data/processed/data_quality_report.json`.

## 2. Exploratory Data Analysis (`ml/notebooks/01_eda.ipynb`)

Feature distributions, correlation heatmap, GCV vs. each feature scatter plots, and coal-type clustering observations. Findings feed the Analytics dashboard page in the frontend sub-project.

## 3. Modeling

**Split:** 80/20 train/test, fixed random seed, stratified by GCV quartile.

**Scaling:** `StandardScaler` fit on train only; applied for Linear Regression, Ridge, Lasso. Tree-based models (Decision Tree, Random Forest, Extra Trees, Gradient Boosting, XGBoost, CatBoost, LightGBM) trained unscaled. The scaler is persisted bundled with any model that needs it.

**Models trained:** Linear Regression, Ridge, Lasso, Decision Tree, Random Forest, Extra Trees, Gradient Boosting, XGBoost, CatBoost, LightGBM (LightGBM skipped with a logged warning if it cannot be installed in the environment — pipeline must not fail because of this).

**Hyperparameter tuning:** `RandomizedSearchCV`, 5-fold CV, fixed seed, per-model parameter distributions. Chosen over exhaustive `GridSearchCV` as the practical compromise across 10 candidate models.

**Model selection:** All tuned models scored on the held-out test set via RMSE, MAE, R². Best model = lowest test RMSE (primary criterion); MAE/R² reported as supporting metrics. Full comparison table persisted for the Model Performance dashboard page.

**Persistence:** winning model (+ scaler if applicable) saved as `models/best_model.joblib` via joblib.

## 4. Confidence score

Per-prediction confidence (0–100%), computed as:

1. **Base uncertainty:**
   - If best model is an ensemble (RF/Extra Trees/GBM/XGBoost/CatBoost/LightGBM): std-dev of predictions across individual trees/estimators, normalized against that model's training-residual std.
   - If best model is linear (Linear/Ridge/Lasso): classical linear-regression prediction-interval formula (residual standard error scaled by leverage).
   - If best model is a single Decision Tree: training-residual-distribution percentile of the leaf the input falls into.
2. **Extrapolation penalty:** score is capped downward if any input feature falls outside that feature's observed min/max range in the training data.

Implemented in `ml/src/models/confidence.py`, model-type-dispatched, with unit tests covering all three branches.

## 5. Explainable AI (`ml/src/explain/`)

- `shap.TreeExplainer` if best model is tree-based; `shap.LinearExplainer` if linear — no `KernelExplainer` needed since both model families have exact, fast explainers available.
- **Global feature importance:** mean |SHAP value| per feature → `models/shap_global.json`.
- **Summary plot data:** per-sample SHAP values for a 200-row representative sample of the test set → same file, consumed by frontend as an interactive Plotly beeswarm/strip chart (no static images).
- **Waterfall / force plot:** computed on-demand per prediction request by the backend (not precomputed), returned as ordered SHAP contributions from base value to final prediction. Waterfall and force plot are treated as the same underlying data (per-prediction SHAP contributions), rendered as a single interactive Plotly waterfall — a literal force-plot HTML/JS embed doesn't fit the React/Plotly stack and would be redundant with the waterfall.
- **Plain-language explanation:** templated sentence generated from the SHAP values, e.g. "High Fixed Carbon (+2.3 MJ/kg) and low Moisture (+1.1 MJ/kg) increased the predicted GCV, while high Ash (-3.0 MJ/kg) decreased it."

## 6. Persisted artifacts (consumed by backend in sub-project 2)

```
models/
├── best_model.joblib          # winning model (+ scaler if needed)
├── model_metadata.json        # model name, version, training date, feature order
├── model_comparison.json      # all trained models' RMSE/MAE/R²
├── shap_global.json           # global importance + summary-plot sample
└── data_quality_report.json
```

## 7. Testing (`ml/tests/`, pytest)

- Data validation: proximate-sum rule enforcement, duplicate removal, missing-value imputation.
- Model training: pipeline runs end-to-end on a small fixture sample and beats a naive mean-prediction baseline.
- Confidence score: all three model-type branches produce scores in [0, 100].
- SHAP correctness: explanation values sum to `prediction − base_value` within floating-point tolerance.

## Out of scope for this sub-project

- FastAPI serving (sub-project 2).
- Frontend rendering of any charts (sub-project 3) — this sub-project only produces the data those charts will consume.
- Docker packaging, CI/CD, cloud deployment (explicitly descoped per project owner — this is a college project, kept simple and explainable, not a DevOps showcase).
