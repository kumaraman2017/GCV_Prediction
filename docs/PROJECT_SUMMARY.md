# Coal GCV Predictor — Complete Project Summary & Interview Playbook

> Repository: `GCV_Prediction` (local folder `gcv_coal_project`) · Owner: kumaraman2017 · Live demo: https://gcvprediction-z2gkxweznrnkrqp6fgvwfd.streamlit.app/

**A note on scope before you read further:** this document was written against the *actual code in the repository*, not the originally imagined 5-sub-project vision. The original design spec (`docs/superpowers/specs/2026-06-28-ml-pipeline-design.md`) planned a `ML Pipeline → Backend API (FastAPI) → Frontend Dashboard (React) → Docker → Docs` pipeline. **Only the ML pipeline was built as originally planned; the Backend API and React frontend were deliberately replaced with a single-file Streamlit app** (see `docs/superpowers/specs/2026-06-29-streamlit-app-design.md`). Docker and CI/CD were explicitly descoped by the project owner: *"this is a college project, kept simple and explainable, not a DevOps showcase."*

Because of this, several sections requested in a generic "full-stack project summary" template (React components, Node/Express routes, JWT auth, MongoDB/Redis, Docker, CI/CD) **do not exist in this codebase**. Rather than inventing them, every such section below explicitly states "Not present" and explains what exists instead. Everything else is described exactly as implemented, with file:line references.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [One-Minute Elevator Pitch](#2-one-minute-elevator-pitch)
3. [Architecture](#3-architecture)
4. [Tech Stack](#4-tech-stack)
5. [Folder-by-Folder Explanation](#5-folder-by-folder-explanation)
6. [File-by-File Summary](#6-file-by-file-summary)
7. [Complete Backend Flow](#7-complete-backend-flow-training-time)
8. [Complete Frontend Flow](#8-complete-frontend-flow-inference-time)
9. [Database Design](#9-database-design)
10. [APIs](#10-apis)
11. [Important Algorithms](#11-important-algorithms)
12. [Business Logic](#12-business-logic)
13. [Important Components](#13-important-components)
14. [State Management](#14-state-management)
15. [Third-Party Services](#15-third-party-services)
16. [Authentication Flow](#16-authentication-flow)
17. [Error Handling](#17-error-handling)
18. [Performance Optimizations](#18-performance-optimizations)
19. [Security](#19-security)
20. [Deployment](#20-deployment)
21. [Configuration Files](#21-configuration-files)
22. [Libraries](#22-libraries)
23. [Difficult Parts of the Project](#23-difficult-parts-of-the-project)
24. [Challenges Solved](#24-challenges-solved)
25. [Trade-offs](#25-trade-offs)
26. [Possible Improvements](#26-possible-improvements)
27. [Resume Points Explained](#27-resume-points-explained)
28. [Placement Interview Explanation](#28-placement-interview-explanation)
29. [Frequently Asked Questions (100+)](#29-frequently-asked-questions)
30. [Code Walkthrough](#30-code-walkthrough)
31. [Key Learnings](#31-key-learnings)
32. [Production Readiness](#32-production-readiness)
33. [Project Strengths](#33-project-strengths)
34. [Weaknesses](#34-weaknesses)
35. [Final Placement Cheat Sheet](#35-final-placement-cheat-sheet)

---

## 1. Project Overview

| | |
|---|---|
| **Name** | Coal GCV Predictor (repo: `GCV_Prediction`) |
| **Type** | Final-year engineering capstone — applied ML + data product |
| **Purpose** | Predict the Gross Calorific Value (GCV) of a coal sample — its energy content in kcal/kg — from four cheap, fast laboratory measurements, instead of a slow/expensive bomb-calorimeter test |
| **Problem statement** | Coal-fired power plants and coal traders need GCV to price coal and plan combustion, but the standard test (bomb calorimetry) takes time and lab equipment. Proximate analysis (Moisture, Volatile Matter, Fixed Carbon, Ash) is fast and routinely measured anyway. This project builds a regression model that predicts GCV directly from those four numbers, with honest uncertainty and an explanation of *why* — not just a black-box number |
| **Why built** | Academic capstone demonstrating an end-to-end, production-style ML workflow: data validation → cleaning → multi-model comparison → hyperparameter tuning → uncertainty quantification → explainability (XAI) → a deployed interactive UI. Chosen specifically to go beyond "train a model in a notebook" and demonstrate software engineering discipline (TDD, modular pipeline, 44 automated tests) |
| **Real-world use case** | Coal quality estimation at mine mouth, in power-plant fuel-receipt inspection, or in a trading desk — anywhere a quick proximate-analysis reading is available but a calorimeter reading is not yet ready |
| **Target users** | Coal-quality lab technicians, power-plant fuel engineers, coal traders/analysts, and (in its current form) recruiters/interviewers evaluating the author's ML engineering skill |
| **Expected business impact** | Cuts the feedback loop for coal-quality decisions from a lab-test cycle to instant, while making the prediction trustworthy via a confidence interval (so a user knows when *not* to trust it) and explainable via SHAP (so a domain expert can sanity-check the reasoning against combustion chemistry) |

---

## 2. One-Minute Elevator Pitch

> "Coal power plants and traders need to know a coal sample's Gross Calorific Value — its energy content — but the standard lab test is slow. I built an ML system that predicts GCV instantly from four numbers that are already measured routinely: Moisture, Volatile Matter, Fixed Carbon, and Ash.
>
> On the backend, I built a modular Python pipeline that validates the data against the physical constraint that those four numbers must sum to 100%, cleans it, then trains and tunes ten different regression models — from linear regression to XGBoost, CatBoost and Extra Trees — using RandomizedSearchCV with 5-fold cross-validation, and automatically picks the best one by test RMSE. The winner, an Extra Trees Regressor, hits an R² of 0.99 and an RMSE of about ±153 kcal/kg.
>
> But I didn't stop at a number. Every prediction ships with a **90% statistical confidence interval** — combining the ensemble's internal disagreement with the model's measured test error, and automatically widening if someone enters values outside the training data's range — and a **SHAP explanation** that tells you, in plain English, which feature pushed the prediction up or down and by how much.
>
> The whole thing is wrapped in a custom-themed Streamlit dashboard with a glassmorphism UI, deployed live on Streamlit Community Cloud, and backed by 44 automated pytest tests covering the data cleaning, model training, confidence scoring, and SHAP math end to end."

---

## 3. Architecture

### Overall architecture

This is **not a client-server web app with a persistent backend**. It is a **two-stage, single-process ML application**:

1. **Offline/training stage** (`ml/` package) — run manually/periodically by the developer, not by end users. Produces versioned artifacts on disk (`models/*.joblib`, `*.json`).
2. **Online/inference stage** (`streamlit_app.py`) — a single Python process that both renders the UI and runs inference in-process. Streamlit's server-side rendering model means there is no separate REST API layer, no client-side JS bundle, and no network hop between "frontend" and "model" — they're the same process, same memory space.

```
┌─────────────────────────────┐         ┌──────────────────────────────────┐
│   OFFLINE (developer-run)   │         │   ONLINE (Streamlit Cloud)       │
│                             │         │                                  │
│  data/raw/coal_all.csv      │         │  Browser (any device)            │
│         │                   │         │       │  HTTP + WebSocket        │
│         ▼                   │         │       ▼                          │
│  ml/run_clean.py            │         │  streamlit_app.py (server proc)  │
│         │                   │         │   ├─ loads models/best_model.    │
│         ▼                   │  writes │   │   joblib (cached)            │
│  data/processed/            │◄────────┤   ├─ ui/ (theme/styles/          │
│    coal_clean.csv           │  reads  │   │   components/charts)         │
│         │                   │  (see  │   ├─ ml/src/models/confidence.py │
│         ▼                   │   note)│   │   (CI math, reused directly) │
│  ml/run_train.py            │         │   └─ ml/src/explain/             │
│   (10 models, RandomizedSCV)│         │       shap_explain.py (reused)   │
│         │                   │         │                                  │
│         ▼                   │         │  Renders HTML/CSS + Plotly       │
│  models/best_model.joblib ──┼────────►│  charts back over the same       │
│  models/model_metadata.json │  loads  │  WebSocket connection             │
│  models/model_comparison... │         │                                  │
│         │                   │         │                                  │
│         ▼                   │         │                                  │
│  ml/run_explain.py (SHAP)   │         │                                  │
│         │                   │         │                                  │
│         ▼                   │         │                                  │
│  models/shap_global.json    │         │                                  │
└─────────────────────────────┘         └──────────────────────────────────┘
```

**Key architectural decision:** the Streamlit app does not duplicate any ML logic. It imports the *same* `ml/src/models/confidence.py` and `ml/src/explain/shap_explain.py` modules used during training (via a `sys.path.insert` shim — see `streamlit_app.py:10-13`). This guarantees the confidence interval and SHAP math the user sees in production is byte-identical to what's unit-tested in `ml/tests/`.

### Client-server communication

There is no custom client-server protocol. Streamlit itself provides:
- An initial HTTP request that serves the page shell.
- A persistent WebSocket connection over which every widget interaction (slider drag, button click) sends the new value to the server, the server re-runs the *entire* Python script top-to-bottom (Streamlit's execution model), and the resulting HTML/Plotly-JSON diff is pushed back to re-render the DOM.

There is no JSON REST API, no `fetch`/`axios` calls, and no client-side state — all state lives server-side in `st.session_state` (see [Section 14](#14-state-management)).

### Data flow

```
User drags a slider (e.g. Moisture = 6.5)
   │
   ▼
Streamlit sends new widget value over WebSocket
   │
   ▼
Entire streamlit_app.py script re-executes top-to-bottom
   │
   ├─ st.session_state["Moisture"] is already updated by the framework
   ├─ render_feature_input() redraws the slider + value label
   ├─ render_sum_pill() recomputes Moisture+VM+FC+Ash and shows ✓/⚠
   │
   ▼
User clicks "Predict"
   │
   ├─ raw_input = {four features} from session_state
   ├─ artifact = load_artifact()            (cached — loads once per server process)
   ├─ feature_vector → scaler.transform (if needed) → estimator.predict → prediction_mj
   ├─ compute_confidence_interval(artifact, raw_input, prediction_mj) → (low, high) in MJ/kg
   ├─ unit conversion MJ/kg → kcal/kg (×1000/4.1868)
   ├─ explainer = load_explainer(artifact)   (cached)
   ├─ explain_single_prediction(...) → per-feature SHAP contributions
   ├─ generate_explanation_sentence(...) → plain-English sentence
   │
   ▼
Three KPI cards + confidence-interval chart + SHAP waterfall chart + sentence render
```

### Request lifecycle (per Streamlit "run")

1. Browser opens `/` → Streamlit server spins up (or reuses) a session, runs `streamlit_app.py` from top to bottom.
2. Module-level code executes: `st.set_page_config`, default session-state seeding (`st.session_state.setdefault`), theme resolution, CSS injection.
3. Widgets are declared in execution order; Streamlit binds each to its `key` in `session_state` and reconciles values sent from the client.
4. If `predict_clicked` is `True` (button was just pressed on this run), the prediction block executes and renders results; otherwise the script ends after rendering the input form — no prediction is shown until a user explicitly asks for one.
5. Every subsequent user interaction (a different slider move, a button press) triggers the *same* full-script re-run — there's no persistent server loop or long-running request; each run is stateless except for what's stored in `st.session_state` and the `@st.cache_resource`/`@st.cache_data` caches.

### Folder structure

```
gcv_coal_project/
├── streamlit_app.py            # Single-file Streamlit app entrypoint (the "frontend + backend")
├── requirements.txt             # Deployment-only, pinned deps (Streamlit Cloud)
├── .python-version              # "3.11" — pins Streamlit Cloud's Python runtime
├── streamlit_test2.log          # Local dev log artifact (not part of deployed app)
│
├── ui/                          # Presentation layer, imported by streamlit_app.py
│   ├── __init__.py
│   ├── theme.py                 # Theme dataclass + DARK/LIGHT palettes, fonts
│   ├── styles.py                # inject_global_css() — glassmorphism CSS-in-Python
│   ├── components.py             # render_header/kpi_card/feature_input/footer/sum_pill
│   └── charts.py                 # confidence_interval_chart(), shap_waterfall() (Plotly)
│
├── ml/                           # Self-contained ML pipeline (its own venv/tests/deps)
│   ├── run_pipeline.py           # Orchestrator: clean → train → explain
│   ├── run_clean.py              # Stage 1 entrypoint
│   ├── run_train.py              # Stage 2 entrypoint
│   ├── run_explain.py            # Stage 3 entrypoint
│   ├── requirements.txt          # Dev/training deps (xgboost, catboost, pytest, jupyter, …)
│   ├── pytest.ini                # pythonpath=. so `src` resolves as a package
│   ├── README.md                 # Sub-project README
│   ├── src/
│   │   ├── config.py             # All paths + constants (single source of truth)
│   │   ├── data/
│   │   │   ├── schema.py         # Pydantic CoalRecord + validate_dataframe()
│   │   │   └── clean.py          # dedup, column selection, imputation, IQR outliers
│   │   ├── models/
│   │   │   ├── registry.py       # ModelSpec dataclass + build_registry() — 7–10 models
│   │   │   ├── train.py          # train_and_tune() + select_best()
│   │   │   └── confidence.py     # compute_spread / compute_confidence / compute_confidence_interval
│   │   └── explain/
│   │       └── shap_explain.py   # SHAP explainer build + global/local explanation helpers
│   ├── notebooks/
│   │   ├── build_eda_notebook.py # Programmatically generates the notebook below (reproducible EDA)
│   │   └── 01_eda.ipynb          # Generated notebook: distributions, correlation, scatter plots
│   └── tests/                    # 12 modules, 44 tests (see Section 6)
│
├── data/
│   ├── raw/coal_all.csv          # Original 4,540-row dataset (10 columns)
│   └── processed/                # Git-ignored in principle but currently committed
│       ├── coal_clean.csv        # 4,513 rows × 5 columns (4 features + GCV)
│       └── data_quality_report.json
│
├── models/                       # Git-tracked trained artifacts (small enough to commit)
│   ├── best_model.joblib         # 11-key artifact bundle (estimator + everything confidence/SHAP need)
│   ├── model_metadata.json       # Winner name, timestamp, feature order, metrics
│   ├── model_comparison.json     # All 10 models' RMSE/MAE/R²
│   └── shap_global.json          # Global feature importance + 200-row SHAP sample (currently unused by the app UI — see Section 34)
│
└── docs/superpowers/             # Design/process docs (specs + implementation plans), not user docs
    ├── specs/2026-06-28-ml-pipeline-design.md
    ├── specs/2026-06-29-streamlit-app-design.md
    └── plans/2026-06-29-ml-pipeline.md
```

**Not present:** `frontend/` (React), `backend/` (FastAPI/Node), `routes/`, `controllers/`, `services/`, `public/`, `Dockerfile`, `.github/workflows/`. These appear in the *original* design spec as future sub-projects but were never built — the Streamlit app absorbed their role.

---

## 4. Tech Stack

| Technology | Purpose | Why chosen | Alternative(s) |
|---|---|---|---|
| **Python 3.11** | Language for the entire project | Pinned (not the system default 3.14) because `shap`, `xgboost`, `catboost`, `lightgbm` have reliable prebuilt Windows wheels for 3.11; newer Python often lacks them at release time | Python 3.10/3.12 |
| **pandas** (3.0.4 pinned in root, `>=2.2` in ml/) | Tabular data loading/cleaning/manipulation | De facto standard; needed for CSV I/O, `groupby`/`qcut`, dataframe ops throughout cleaning and training | Polars (faster, less mature ecosystem for this use case) |
| **NumPy** (2.4.6 / `>=1.26`) | Numerical arrays, vectorized math (confidence intervals, distances) | Required by every other library in the stack; used directly for `np.linalg.norm`, `np.exp`, `np.sqrt` in `confidence.py` | — (foundational, no real alternative) |
| **scikit-learn** (1.9.0 / `>=1.4`) | Regression models, `StandardScaler`, `train_test_split`, `RandomizedSearchCV`, metrics | Industry-standard, consistent `fit/predict` API lets all 7 core models share one training loop (`train_and_tune`) | Statsmodels (less ML-pipeline-oriented) |
| **XGBoost** (`>=2.0`) | Gradient-boosted trees candidate model | Strong baseline for tabular regression; optional (registry degrades gracefully if not installed) | LightGBM, CatBoost (both also included) |
| **CatBoost** (`>=1.2`) | Gradient-boosted trees candidate model | Handles categorical features natively (not exercised here, but future-proofs the registry) and is often competitive with less tuning | XGBoost, LightGBM |
| **LightGBM** (`>=4.3`) | Gradient-boosted trees candidate model | Very fast training via histogram-based splits; included for comparison breadth | XGBoost, CatBoost |
| **Pydantic** (v2, `>=2.6`) | Row-level schema validation (`CoalRecord`) | Declarative validators (`field_validator`, `model_validator`) express the domain rule ("proximate values sum to ~100%") clearly and raise structured `ValidationError`s the pipeline can collect | Manual `if`/`raise` checks, `cerberus`, `marshmallow` |
| **SHAP** (0.51.0 / `>=0.45`) | Model explainability (per-prediction + global feature importance) | `TreeExplainer`/`LinearExplainer` give *exact*, fast, additive explanations for tree/linear models — no approximate `KernelExplainer` needed | LIME (approximate, model-agnostic but slower/less precise for these model families) |
| **joblib** (1.5.3 / `>=1.3`) | Serializing the trained model bundle to disk | Standard for persisting scikit-learn estimators (handles NumPy arrays efficiently); used to bundle the estimator + scaler + training data + metadata into one artifact | Python `pickle` directly, ONNX export |
| **Streamlit** (`>=1.38`) | The entire "frontend" — UI framework and web server in one | Turns a Python script into a reactive web app with zero JS/HTML/CSS boilerplate required, ideal for a solo-developer capstone that needs a demo fast | Flask/FastAPI + a real frontend (React/Vue), Gradio, Dash |
| **Plotly** (`>=5.24`) | Interactive charts (confidence interval bar, SHAP waterfall) | Native Streamlit integration (`st.plotly_chart`), interactive hover/zoom "for free", supports the `Waterfall` chart type needed for SHAP contributions | Matplotlib (static only), Altair |
| **pytest** (`>=8.0`) | Test runner | Standard Python test framework; simple fixtures, clear assertion introspection | unittest (more boilerplate) |
| **nbformat / nbconvert / jupyter / ipykernel** | Programmatic notebook generation + execution-as-a-test | The EDA notebook is *built by a script* (`build_eda_notebook.py`) rather than hand-edited, so it can be regenerated deterministically and executed headlessly in CI-less "testing" (`test_eda_notebook.py` runs `nbconvert --execute`) | Hand-maintained `.ipynb` (drifts, hard to diff) |
| **matplotlib / seaborn** | Static plotting inside the EDA notebook | Standard for quick exploratory histograms/heatmaps in a notebook context | Plotly (used instead in the live app for interactivity) |
| **Streamlit Community Cloud** | Hosting | Free, zero-ops hosting purpose-built for Streamlit apps, deploys directly from a GitHub branch | Render, Railway, Heroku, a VM + Docker |

---

## 5. Folder-by-Folder Explanation

| Folder | Responsibility |
|---|---|
| `/` (root) | Deployment entrypoint (`streamlit_app.py`), deployment dependency pin (`requirements.txt`), Python version pin (`.python-version`) — everything Streamlit Community Cloud needs to boot the app |
| `/ui` | Pure presentation layer: theme tokens, CSS injection, reusable render functions, and Plotly chart builders. Contains **no ML logic** — it only ever receives already-computed numbers (predictions, intervals, SHAP contributions) and renders them |
| `/ml` | The entire ML pipeline as a self-contained sub-project with its own `requirements.txt`, `pytest.ini`, and virtual environment convention (`../.venv`). Designed to be run independently of the Streamlit app (`run_pipeline.py`) to retrain the model |
| `/ml/src/data` | Input validation (`schema.py`) and cleaning/deduplication/outlier-reporting (`clean.py`) — the only place raw data is touched before it becomes "modeling data" |
| `/ml/src/models` | Everything about turning clean data into a deployable model: the candidate model catalogue (`registry.py`), the training/tuning/selection loop (`train.py`), and post-hoc uncertainty quantification (`confidence.py`) |
| `/ml/src/explain` | Explainability only — wraps SHAP so both the offline `run_explain.py` script and the live Streamlit app can build explainers and format contributions identically |
| `/ml/run_*.py` | Thin orchestration scripts — each does I/O (read CSV/joblib, call into `src/`, write CSV/JSON/joblib) and nothing else. This keeps `src/` fully unit-testable without touching disk |
| `/ml/tests` | 12 pytest modules, 44 tests, mirroring the `src/` structure 1:1 plus one "smoke test" per `run_*.py` script and one for the EDA notebook |
| `/ml/notebooks` | Exploratory Data Analysis — regenerated by a script rather than hand-edited, so its output is reproducible and testable |
| `/data/raw` | The untouched original dataset — never modified by the pipeline |
| `/data/processed` | Pipeline *output*, not input — the cleaned CSV and the JSON data-quality report, both fully regenerable by re-running `run_clean.py` |
| `/models` | The pipeline's *final* output — everything the inference layer (Streamlit) needs, small enough to commit directly to git rather than needing an artifact store |
| `/docs/superpowers` | Design specifications and implementation plans written *before* coding (a spec-driven-development process), not end-user documentation |

**Folders explicitly absent** (present in a typical full-stack template, not here): `/frontend`, `/backend`, `/routes`, `/controllers`, `/services` (in the Node/Express sense), `/public`, `/config` (as a dedicated folder — config lives in `ml/src/config.py` instead), `/scripts` (the `run_*.py` files serve this role inside `ml/`).

---

## 6. File-by-File Summary

### Root

| File | Purpose | Key functions/classes | Connects to |
|---|---|---|---|
| `streamlit_app.py` | The entire live application: UI layout + inference orchestration | `load_artifact()`, `load_explainer()`, `load_clean_dataset()`, `load_metadata()`, `reset_inputs()`, `load_example()` | Imports `ml/src/config.py`, `ml/src/models/confidence.py`, `ml/src/explain/shap_explain.py`, and every `ui/*.py` module |
| `requirements.txt` | Pinned runtime deps for Streamlit Cloud | — | Read by Streamlit Cloud's build step |
| `.python-version` | Pins Python 3.11 for Streamlit Cloud | — | Read by Streamlit Cloud's build step |

### `ui/`

| File | Purpose | Key functions/classes | Connects to |
|---|---|---|---|
| `theme.py` | Design tokens for dark/light mode | `Theme` (frozen dataclass), `DARK`, `LIGHT`, `get_theme(name)` | Consumed by `styles.py`, `components.py`, `charts.py` |
| `styles.py` | Injects one large `<style>` block implementing the glassmorphism look | `inject_global_css(theme)`, `_build_css(theme)` | Called once per run from `streamlit_app.py` |
| `components.py` | All custom HTML/Streamlit-widget render functions | `render_header`, `render_sum_pill`, `render_kpi_card`, `render_feature_input`, `render_footer`, `_step_value` | Called from `streamlit_app.py`; reads/writes `st.session_state` |
| `charts.py` | Builds the two Plotly figures used in the app | `confidence_interval_chart()`, `shap_waterfall()`, `_transparent_layout()` | Called from `streamlit_app.py` with numbers computed via `ml/src/*` |

### `ml/` (orchestration scripts)

| File | Purpose | Main function | Connects to |
|---|---|---|---|
| `run_clean.py` | Stage 1: validate + clean raw CSV | `main()` | Reads `data/raw/coal_all.csv`; calls `src.data.clean.clean_pipeline`; writes `data/processed/coal_clean.csv` + `data_quality_report.json` |
| `run_train.py` | Stage 2: train/tune/select/persist | `main()` | Reads `coal_clean.csv`; calls `build_registry` + `train_and_tune` + `select_best`; writes `models/best_model.joblib`, `model_metadata.json`, `model_comparison.json` |
| `run_explain.py` | Stage 3: compute global SHAP artifacts | `main()` | Loads `best_model.joblib`; calls `build_explainer`, `global_feature_importance`, `summary_plot_data`; writes `models/shap_global.json` |
| `run_pipeline.py` | Orchestrates all three stages in order | `main()` | Imports and calls the three scripts above sequentially |

### `ml/src/`

| File | Purpose | Key functions/classes | Notes |
|---|---|---|---|
| `config.py` | Single source of truth for every path and constant | `RAW_DATA_PATH`, `CLEAN_DATA_PATH`, `MODELS_DIR`, `FEATURE_COLUMNS`, `TARGET_COLUMN`, `RANDOM_SEED=42`, `TEST_SIZE=0.2`, `PROXIMATE_SUM_TOLERANCE=0.5` | `PROJECT_ROOT` resolved via `Path(__file__).resolve().parents[2]` — makes the whole pipeline path-independent of cwd |
| `data/schema.py` | Row-level validation | `CoalRecord` (Pydantic model, 5 fields with aliases matching CSV column names), `validate_dataframe(df)` | Enforces non-negativity, ≤100 per feature, and the proximate-sum-≈100 domain rule; returns `(valid_df, errors)` — a *reporting*, not exception-raising, contract at the dataframe level |
| `data/clean.py` | Cleaning pipeline (pure functions + one composed pipeline) | `drop_duplicate_rows`, `select_modeling_columns`, `impute_missing_values`, `detect_outliers_iqr`, `clean_pipeline` | `clean_pipeline` composes all four steps and returns `(clean_df, report_dict)` — the report is what becomes `data_quality_report.json` |
| `models/registry.py` | Declarative catalogue of every candidate model | `ModelSpec` (dataclass), `build_registry(random_seed)`, `HAS_XGBOOST/HAS_CATBOOST/HAS_LIGHTGBM` | Each `ModelSpec` bundles a factory lambda, its hyperparameter search space, whether it needs scaling, whether it's tree-based, and which confidence-scoring branch applies to it |
| `models/train.py` | Generic train/tune/evaluate loop, model-agnostic | `TrainedModel` (dataclass), `train_and_tune(spec, X_train, y_train, X_test, y_test, seed)`, `select_best(list[TrainedModel])` | Applies `StandardScaler` only if `spec.needs_scaling`; runs `RandomizedSearchCV(n_iter=10, cv=5)` only if `spec.param_distributions` is non-empty (plain `.fit()` otherwise, e.g. for `linear_regression`) |
| `models/confidence.py` | Per-prediction uncertainty quantification | `compute_spread`, `_bagging_ensemble_spread`, `_single_tree_spread`, `_knn_residual_spread`, `_extrapolation_penalty`, `compute_confidence`, `compute_confidence_interval` | `compute_confidence` (0–100 score) exists and is tested but is **not called by `streamlit_app.py`**; the live app uses `compute_confidence_interval` only (see [Section 34](#34-weaknesses)) |
| `explain/shap_explain.py` | SHAP wrapper, shared by training and inference | `build_explainer`, `_expected_value`, `global_feature_importance`, `summary_plot_data`, `explain_single_prediction`, `generate_explanation_sentence` | `build_explainer` dispatches on `is_tree_based` to pick `TreeExplainer` vs `LinearExplainer` — no `KernelExplainer` anywhere |

### `ml/tests/` (44 tests / 12 modules)

| File | # tests | What it covers |
|---|---|---|
| `test_config.py` | 3 | Path existence, feature/target column values, output-path directory structure |
| `test_schema.py` | 5 | Valid record passes; negative value, >100 value, and bad proximate-sum are rejected; `validate_dataframe` splits valid/invalid rows correctly |
| `test_clean.py` | 5 | Dedup count, ultimate-analysis column drop, median imputation, IQR outlier flagging, full `clean_pipeline` integration |
| `test_run_clean.py` | 1 | End-to-end: real dataset cleaned to exactly 4,513 rows with the 5 expected columns |
| `test_registry.py` | 4 | Core models present, correct `confidence_method` assigned per model, all factories produce valid unfitted estimators, optional models (xgboost/catboost/lightgbm) included iff installed |
| `test_train.py` | 3 | Decision tree trains with no scaler + valid metrics; linear regression trains *with* a scaler; `select_best` picks the lowest-RMSE model |
| `test_run_train.py` | 3 | Persisted artifact has all 11 required keys with correct feature order; metadata/comparison JSON structurally correct with ≥7 models compared |
| `test_confidence.py` | 11 | All three confidence-method branches (bagging/single-tree/knn) return scores in `[0,100]`; extrapolation penalty is 1.0 in-range and <1.0 out-of-range; (per the plan, also covers the unknown-method `ValueError` branch) |
| `test_shap_explain.py` | 6 | SHAP contributions sum to `prediction − base_value` (additivity property) for both tree and linear explainers; global importance covers all features and is non-negative; summary-plot row count = rows × features; explanation sentence mentions both increasing and decreasing features |
| `test_run_explain.py` | 1 | `models/shap_global.json` is produced with the expected structure |
| `test_run_pipeline.py` | 1 | Full `clean → train → explain` pipeline runs end-to-end without error (smoke test) |
| `test_eda_notebook.py` | 1 | The EDA notebook is rebuilt (`build_eda_notebook.py`) and executes cleanly end-to-end via `nbconvert --execute` — this doubles as a test *and* a regeneration step |

### `ml/notebooks/`

| File | Purpose |
|---|---|
| `build_eda_notebook.py` | Programmatically constructs `01_eda.ipynb` cell-by-cell (`nbformat`) — describe stats, 2×2 histogram grid of feature distributions, correlation heatmap, GCV-vs-feature scatter grid, and a markdown "Observations" cell |
| `01_eda.ipynb` | The generated notebook (build artifact, regenerated by the script above, executed by `test_eda_notebook.py`) |

### `data/` and `models/` (data artifacts, not code)

| File | Purpose |
|---|---|
| `data/raw/coal_all.csv` | Original dataset: 4,540 rows × 10 columns (`Moisture, Volatile_matter, Fixed_Carbon, Std.Ash, Hydrogen, Carbon, Nitrogen, Oxygen, Sulfur, GCV`) |
| `data/processed/coal_clean.csv` | 4,513 rows × 5 columns — the exact cleaned modeling dataset |
| `data/processed/data_quality_report.json` | Machine-readable cleaning report: rows in/out, duplicates removed (27), validation failures (0), missing-value counts (all 0), IQR outlier bounds+counts per column |
| `models/best_model.joblib` | The 11-key artifact bundle (see Section 3) |
| `models/model_metadata.json` | `{"model_name": "extra_trees", "trained_at": "...", "feature_columns": [...], "target_column": "GCV", "metrics": {"rmse": 0.6425, "mae": 0.4578, "r2": 0.9904}}` |
| `models/model_comparison.json` | RMSE/MAE/R² for all 10 candidate models |
| `models/shap_global.json` | `global_feature_importance` (mean |SHAP| per feature) + a 200-row `summary_plot_sample` |

---

## 7. Complete Backend Flow (training-time)

There is no request-driven "backend" in the traditional sense (no HTTP routes/controllers/services). The closest analogue is the **offline training pipeline**, which has an equivalent layered structure:

```
Raw CSV (data/raw/coal_all.csv)
        │
        ▼
┌───────────────────────────────────────────────────────────┐
│ "Router"-equivalent: run_clean.py / run_train.py /          │
│ run_explain.py — the entrypoints a developer invokes         │
└───────────────────┬───────────────────────────────────────┘
                     ▼
┌───────────────────────────────────────────────────────────┐
│ "Controller"-equivalent: each run_*.py's main() function     │
│  — orchestrates I/O and calls into src/ business logic       │
└───────────────────┬───────────────────────────────────────┘
                     ▼
┌───────────────────────────────────────────────────────────┐
│ "Service"-equivalent: ml/src/{data,models,explain}/*.py       │
│  — validate_dataframe, clean_pipeline, build_registry,       │
│    train_and_tune, select_best, compute_confidence_interval, │
│    build_explainer, explain_single_prediction                │
│  — all pure functions/classes, no I/O, fully unit-tested     │
└───────────────────┬───────────────────────────────────────┘
                     ▼
┌───────────────────────────────────────────────────────────┐
│ "Database"-equivalent: the filesystem — joblib + JSON         │
│  artifacts under data/processed/ and models/                 │
└───────────────────┬───────────────────────────────────────┘
                     ▼
"Response": console print statements (row counts, winning model
 name + metrics) — the human developer reads stdout, there is
 no client waiting on an HTTP response
```

Step by step, running `python run_pipeline.py`:

1. **`run_clean.main()`** — reads `coal_all.csv` with pandas → `clean_pipeline(raw_df)`:
   - `validate_dataframe` runs every row through the Pydantic `CoalRecord` model, splitting valid rows from a list of structured error dicts (0 rows fail on the real dataset).
   - `drop_duplicate_rows` removes 27 exact duplicate rows via `pandas.DataFrame.drop_duplicates()`.
   - `select_modeling_columns` drops the five ultimate-analysis columns (Hydrogen, Carbon, Nitrogen, Oxygen, Sulfur) — they exist in the raw data but were excluded from modeling by design decision.
   - `impute_missing_values` median-imputes any nulls per column (0 in practice — defensive code for future/dirtier data).
   - `detect_outliers_iqr` computes IQR bounds and flags counts per column *for visibility only* — outliers are **not removed** (they represent real coal-type diversity, e.g. lignite vs. anthracite, not data errors).
   - Writes `coal_clean.csv` (4,513 × 5) and `data_quality_report.json`.
2. **`run_train.main()`** — reads the clean CSV, builds `X`/`y` NumPy arrays, stratifies by GCV quartile (`pd.qcut(y, q=4)`), splits 80/20 with `random_state=42`. Builds the model registry (`build_registry`), trains+tunes each of the (up to 10) `ModelSpec`s via `train_and_tune` (StandardScaler if needed, `RandomizedSearchCV` if a search space is defined), picks the lowest-test-RMSE model via `select_best`, recomputes training residuals for the winner, computes per-feature training ranges, and serializes the full artifact bundle + two JSON reports.
3. **`run_explain.main()`** — loads the persisted artifact, samples 200 rows from the clean dataset, builds a SHAP explainer matching the winning model's type, computes global importance and a per-row SHAP sample, writes `shap_global.json`.

---

## 8. Complete Frontend Flow (inference-time)

There is no React component tree, no hooks, no client-side API calls, and no client-side state manager — Streamlit replaces all of that with **server-side re-execution + `st.session_state`**. Mapping the requested flow onto what actually exists:

```
User interaction (drag slider / click +/− / click Predict/Reset/Load-example)
        │
        ▼
Streamlit widget callback (or plain rerun) updates st.session_state
        │  — this is the closest equivalent to a "hook" (useState) in this stack
        ▼
streamlit_app.py re-executes top-to-bottom (this is the "component render")
        │
        ├─ ui/components.py render_* functions redraw the DOM from session_state
        │  ("React component" equivalent — pure render functions parameterized by state)
        │
        ▼
If Predict was clicked this run:
        │
        ├─ "API call" equivalent: direct in-process Python calls —
        │    artifact = load_artifact()                       (st.cache_resource)
        │    prediction = estimator.predict(...)
        │    interval = compute_confidence_interval(...)       (ml/src/models/confidence.py)
        │    explanation = explain_single_prediction(...)      (ml/src/explain/shap_explain.py)
        │
        ▼
"State management" equivalent: results are local Python variables for
this run only (not persisted in session_state) — a rerun without a fresh
Predict click will not redisplay the last prediction
        │
        ▼
Rendering: ui/charts.py builds Plotly figures, st.plotly_chart renders them;
ui/components.py render_kpi_card renders the three number/label cards
```

Concretely, in `streamlit_app.py`:
- **Inputs**: `render_feature_input()` (in `ui/components.py`) draws a glass-card with an icon, a live value readout, a native `st.slider` (bound to `st.session_state[key]`), and custom +/− buttons whose `on_click=_step_value` callback clamps and rounds the value.
- **Validation feedback**: `render_sum_pill()` recomputes `Moisture+VM+FC+Ash` on every run and shows a green ✓ or amber ⚠ pill — this is advisory only; it does not block the Predict button (see [Section 12](#12-business-logic)).
- **Buttons**: `Reset` calls `reset_inputs()` (writes the four `DEFAULTS` back into session_state); `Load example` calls `load_example()` (samples one real row from the cached clean dataset via `@st.cache_data load_clean_dataset()` and writes its four feature values into session_state); `Predict` just sets a local boolean flag for this run via `st.button(...)`'s return value.
- **On Predict**: builds the raw feature vector in `FEATURE_COLUMNS` order, applies the persisted `scaler` if the winning model needs one, calls `estimator.predict`, converts MJ/kg → kcal/kg, computes the 90% CI, builds/reuses a cached SHAP explainer, computes per-feature contributions (also converted to kcal/kg), and generates the natural-language sentence.
- **Rendering**: three `st.columns` KPI cards (Predicted GCV, 90% CI Plotly chart, Model RMSE), then a "Why this prediction?" card with the sentence and a SHAP waterfall Plotly chart.

---

## 9. Database Design

**There is no database** (no PostgreSQL/MySQL/MongoDB/SQLite, no ORM, no migrations). Persistence is entirely **filesystem-based**, using the dataset/artifact files as the "schema":

### "Tables" (files) and their "columns" (fields)

**`data/raw/coal_all.csv`** (input "table", 4,540 rows)

| Column | Type | Constraint |
|---|---|---|
| Moisture | float | 0 ≤ x ≤ 100 |
| Volatile_matter | float | 0 ≤ x ≤ 100 |
| Fixed_Carbon | float | 0 ≤ x ≤ 100 |
| Std.Ash | float | 0 ≤ x ≤ 100 |
| Hydrogen, Carbon, Nitrogen, Oxygen, Sulfur | float | present in raw data, **dropped**, never used as model inputs |
| GCV | float | target; ≥ 0 |
| *(implicit constraint)* | — | Moisture + Volatile_matter + Fixed_Carbon + Std.Ash ≈ 100 (±0.5) — proximate-analysis convention |

**`data/processed/coal_clean.csv`** (modeling "table", 4,513 rows × 5 columns) — same schema as above minus the 5 ultimate-analysis columns, post dedup/validation.

**`models/best_model.joblib`** (the closest thing to a "row" the live app reads) — a single Python `dict` with these 11 keys (this exact key set is treated as a contract across the codebase):

| Key | Type | Role |
|---|---|---|
| `estimator` | fitted sklearn estimator | Makes predictions |
| `scaler` | `StandardScaler` or `None` | Applied before prediction if the model needs scaled inputs |
| `feature_columns` | `list[str]` | Canonical input order |
| `target_column` | `str` | `"GCV"` |
| `confidence_method` | `str` | One of `bagging_ensemble` / `single_tree` / `knn_residual` — dispatch key for `confidence.py` |
| `is_tree_based` | `bool` | Dispatch key for SHAP explainer choice |
| `X_train` | `np.ndarray` | Training features (post-scaling if applicable) — used for kNN-residual lookups and as SHAP background |
| `y_train` | `np.ndarray` | Training targets |
| `residual_std` | `float` | Test-set RMSE (used as the irreducible-error term in the CI) |
| `residuals_train` | `np.ndarray` | Per-sample training residuals — used for local-spread lookups |
| `feature_ranges` | `dict[str, tuple[float, float]]` | Per-feature (min, max) — extrapolation-penalty input |

### "Relationships"

There is no relational schema, but there is a clear **data lineage/dependency graph**:

```
coal_all.csv ──▶ coal_clean.csv ──▶ best_model.joblib ──▶ model_metadata.json
                        │                    │                       │
                        │                    └──▶ model_comparison.json
                        │                    │
                        └────────────────────┴──▶ shap_global.json
```

### ASCII "ER diagram" (artifact dependency graph, since there are no relational entities)

```
┌────────────────┐      cleans       ┌────────────────┐
│  coal_all.csv   │ ────────────────▶ │ coal_clean.csv  │
│ (4540 × 10)     │                   │ (4513 × 5)      │
└────────────────┘                   └───────┬────────┘
                                              │ trains on
                                              ▼
                                     ┌────────────────────┐
                                     │ best_model.joblib   │
                                     │ (11-key artifact)   │
                                     └───┬────────┬───────┘
                          describes/     │        │ explains
                          summarizes     │        │
                                         ▼        ▼
                          ┌────────────────┐  ┌────────────────┐
                          │model_metadata /│  │ shap_global.json│
                          │comparison.json │  │                │
                          └────────────────┘  └────────────────┘
```

### Indexes / constraints

None in the database sense. The closest analogues are:
- **Pydantic validators** in `CoalRecord` (non-negativity, ≤100, proximate-sum tolerance) — enforced at clean-time, not query-time.
- **`FEATURE_COLUMNS` order** in `config.py` — implicitly acts as a "foreign key"/contract that every downstream module (training, confidence, SHAP, the UI) must respect, since NumPy arrays are positional, not named.

### Why this design was chosen

A relational database would be pure overhead here: there are no concurrent writers, no transactional requirements, no user accounts, and the entire "database" (4,513 rows) fits comfortably in memory and in git. Flat files (CSV/JSON/joblib) are simpler to version-control, diff, and reproduce deterministically — appropriate for a research/capstone artifact pipeline, not appropriate if this were a multi-user production service ingesting live lab results (see [Section 26](#26-possible-improvements)).

---

## 10. APIs

**There is no REST/GraphQL API** — no Flask/FastAPI/Express routes, no HTTP endpoints, no request/response JSON contracts. The system's only "endpoints" are Python function signatures called in-process. The table below documents those as the functional equivalent of an API surface (this is what a backend API would have wrapped, had sub-project 2 of the original plan been built):

| "Endpoint" (function) | "Method" | Input | Output | Purpose |
|---|---|---|---|---|
| `src.data.clean.clean_pipeline(raw_df)` | pure function | raw `pd.DataFrame` | `(clean_df, report: dict)` | Validate + clean the raw dataset |
| `src.models.registry.build_registry(random_seed)` | pure function | `int` | `list[ModelSpec]` | Enumerate candidate models + their search spaces |
| `src.models.train.train_and_tune(spec, X_train, y_train, X_test, y_test, seed)` | pure function | model spec + arrays | `TrainedModel` | Train, tune, and score one model |
| `src.models.train.select_best(trained_models)` | pure function | `list[TrainedModel]` | `TrainedModel` | Pick lowest-test-RMSE model |
| `src.models.confidence.compute_confidence_interval(artifact, raw_input, prediction, z=1.645)` | pure function | artifact dict + `{feature: value}` + predicted value | `(low, high)` tuple | **The function a "predict" endpoint would call** to get a 90% CI |
| `src.models.confidence.compute_confidence(artifact, raw_input)` | pure function | artifact + raw input | `float` in `[0,100]` | Legacy/tested 0–100 confidence score (unused by the current UI) |
| `src.explain.shap_explain.build_explainer(is_tree_based, estimator, background)` | pure function | flags + fitted model + background array | SHAP explainer object | Construct the correct SHAP explainer |
| `src.explain.shap_explain.explain_single_prediction(explainer, X_row, feature_names, raw_values)` | pure function | explainer + one row | `{"base_value", "contributions", "prediction"}` | **The function a "predict" endpoint would call** for the SHAP breakdown |
| `src.explain.shap_explain.generate_explanation_sentence(contributions, top_n=3, unit="MJ/kg")` | pure function | contributions list | `str` | Human-readable explanation sentence |

If a real backend API existed (per the original design spec's sub-project 2), it would almost certainly look like:

```
POST /predict
  body: { "moisture": 5.2, "volatile_matter": 31.1, "fixed_carbon": 40.7, "std_ash": 23.0 }
  response: {
    "prediction_kcal": 6234,
    "confidence_interval": { "low": 6081, "high": 6387 },
    "explanation": {
      "base_value": 5890,
      "contributions": [ {"feature": "Fixed_Carbon", "shap_value": 210.4}, ... ],
      "sentence": "Fixed_Carbon (+210 kcal/kg) increased the predicted GCV, while Std.Ash (-95 kcal/kg) decreased it."
    }
  }
```

— but this endpoint **does not exist in the codebase**; `streamlit_app.py` calls the equivalent Python functions directly in-process instead of over HTTP.

---

## 11. Important Algorithms

### 11.1 Proximate-sum validation + IQR outlier detection (`ml/src/data/`)

- **Logic**: Pydantic validators run per-row (`CoalRecord.model_validate`); IQR bounds computed once per column (`Q1 - 1.5×IQR`, `Q3 + 1.5×IQR`), then a boolean mask flags out-of-bound rows.
- **Time complexity**: row validation is `O(n)` (n = 4,540 rrows, iterated via `df.iterrows()` — see [Section 18](#18-performance-optimizations) for why this is *not* vectorized); IQR bound computation is `O(n log n)` per column (quantile requires a sort); outlier masking is `O(n)`.
- **Space complexity**: `O(n)` for the boolean masks and error list.
- **Why chosen**: IQR is a simple, distribution-agnostic, industry-standard rule for flagging (not necessarily removing) anomalies; chosen over z-score because the GCV/feature distributions aren't guaranteed normal (coal type varies widely).

### 11.2 Train/test split with quartile stratification

- **Logic**: `pd.qcut(y, q=4)` buckets GCV into 4 quantile bins, then `train_test_split(..., stratify=strata)` ensures the 80/20 split has a proportional mix of low/medium/high-GCV coal in both sets.
- **Complexity**: `qcut` is `O(n log n)` (needs sorted quantile boundaries); the split itself is `O(n)`.
- **Why chosen**: prevents a "lucky"/"unlucky" split where, e.g., all the high-GCV anthracite samples land in the test set, which would make the reported RMSE misleading.

### 11.3 RandomizedSearchCV hyperparameter tuning (`ml/src/models/train.py`)

- **Logic**: for each of the (up to 10) candidate models with a non-empty `param_distributions`, samples `n_iter=10` hyperparameter combinations, evaluates each with 5-fold CV (`cv=5`) scored by negative RMSE, keeps the best.
- **Time complexity**: `O(n_iter × cv_folds × single_fit_cost)` = 50 fits per tuned model, versus exhaustive `GridSearchCV`'s combinatorial explosion (e.g., Random Forest's 3×4×3 = 36 combinations × 5 folds = 180 fits, and worse for finer grids).
- **Why chosen over GridSearchCV**: explicitly a *practical compromise* across 10 candidate models — random sampling covers the space "well enough" in a fixed time budget rather than exhaustively.

### 11.4 Extra Trees Regressor (the winning model)

- **Logic**: an ensemble of extremely randomized decision trees — like Random Forest, but split thresholds are chosen *randomly* among candidate features rather than searched for the locally-optimal split, and (in scikit-learn's implementation) the whole training set is used per tree rather than a bootstrap sample.
- **Time complexity (training)**: `O(n_estimators × n log n × n_features)` roughly, similar order to Random Forest but with cheaper per-split decisions.
- **Time complexity (inference)**: `O(n_estimators × depth)` per prediction — trivial at this dataset's scale (4 features).
- **Why it won**: the extra randomization reduces variance further than Random Forest without a real accuracy cost on this dataset (test RMSE 0.6425 vs Random Forest's 0.6613), because the underlying feature→GCV relationship is smooth enough that exhaustive per-node split search offers little benefit over random split candidates while overfitting less.

### 11.5 Confidence-interval computation (`compute_confidence_interval`)

- **Logic**: `total_uncertainty = √(spread² + residual_std²)`, where `spread` is a **model-type-dispatched epistemic uncertainty estimate**:
  - `bagging_ensemble` (Random Forest/Extra Trees): standard deviation of each individual tree's prediction for this input — `O(n_estimators)` predict calls.
  - `single_tree` (Decision Tree): standard deviation of training residuals *among the training rows that land in the same leaf* as this input — `O(n_train)` to compute `estimator.apply()` over the whole training set (this is recomputed per-request; see [Section 18](#18-performance-optimizations)).
  - `knn_residual` (linear models, boosting models): standard deviation of residuals among the 15 nearest training neighbors by Euclidean distance in feature space — `O(n_train × n_features)` for the distance computation plus `O(n_train log n_train)` for the full `np.argsort` (only the top 15 are used — a partial `argpartition` would be `O(n_train)`, see [Section 26](#26-possible-improvements)).
  - `residual_std` is the winning model's **test-set RMSE** (a fixed, precomputed number — the model's irreducible error).
  - `half_width = z × total_uncertainty / extrapolation_penalty`, `z = 1.645` for a two-sided 90% interval.
- **Why chosen**: combines *epistemic* uncertainty (how much the model itself disagrees with itself near this input) with *aleatoric*/measured error (test RMSE), and widens gracefully rather than failing when a user enters an out-of-distribution input.

### 11.6 Extrapolation penalty (`_extrapolation_penalty`)

- **Logic**: for each feature, if the input falls within `[min, max]` observed in training, no penalty; if outside, `penalty *= max(0.5, 1 - distance/span)` (multiplicative across features, floored per-feature at 0.5 so one wildly out-of-range feature can't zero out the interval width entirely, but *can* halve confidence per offending feature).
- **Complexity**: `O(features)` = `O(4)` — trivial.
- **Why chosen**: a cheap, interpretable proxy for "is this a reliable prediction," without needing a full outlier-detection model.

### 11.7 SHAP TreeExplainer / LinearExplainer

- **Logic**: `TreeExplainer` implements Lundberg's **Tree SHAP** algorithm — computes *exact* Shapley values for tree ensembles in polynomial time by exploiting the tree structure (tracking feature-subset paths through internal nodes), rather than the exponential-time brute-force Shapley definition. `LinearExplainer` computes exact Shapley values for linear models directly from the coefficients and background feature covariance/means.
- **Time complexity**: Tree SHAP is `O(T·L·D²)` where T = number of trees, L = max leaves per tree, D = max tree depth — dramatically cheaper than the `O(2^features)` exact brute-force Shapley computation, and exact (unlike sampling-based `KernelExplainer`, which is model-agnostic but approximate and slower).
- **Why chosen**: both candidate model families (tree ensembles and linear models) have exact, fast explainers available, so there was no need for the generic (slower, approximate) `KernelExplainer`.
- **Additivity check** (verified by `test_shap_explain.py`): `base_value + Σ(shap_values) == prediction` within floating-point tolerance — this is the mathematical guarantee that makes the "why this prediction" sentence trustworthy rather than a plausible-sounding fabrication.

---

## 12. Business Logic

The entire "business rule" of this project is a single **domain law of coal chemistry**, encoded in three places with three different strictness levels — worth understanding deeply, since it's the single most interview-relevant design decision in the codebase:

1. **Physical law**: In proximate analysis, Moisture % + Volatile Matter % + Fixed Carbon % + Ash % **must** sum to 100% by definition (they partition the whole sample by mass).
2. **Training-time enforcement (strict/hard)**: `CoalRecord.proximate_components_sum_to_100` (`ml/src/data/schema.py:30-37`) rejects any row where the sum deviates from 100 by more than `PROXIMATE_SUM_TOLERANCE = 0.5`. This is a **hard gate** — such rows never enter the training set.
3. **Inference-time enforcement (soft/advisory)**: `render_sum_pill()` (`ui/components.py:41-53`) recomputes the same sum for the user's four slider values and shows a colored pill (✓ green if within ±0.5, ⚠ amber otherwise) — but **does not disable the Predict button**. A user *can* submit an input that violates the physical law.

**Why the asymmetry is actually a defensible design, not a bug**: blocking prediction entirely on a soft UI constraint would prevent users from exploring "what if" scenarios (e.g., stress-testing the model, or entering approximate remembered values); instead, the system leans on `compute_confidence_interval`'s extrapolation penalty to automatically **widen the confidence interval** for physically implausible inputs rather than refusing to answer — a "degrade gracefully, don't hard-fail" philosophy that's arguably more appropriate for a decision-support tool than a hard validation error would be. (This is a good trade-off to be able to defend in an interview — see [Section 25](#25-trade-offs).)

**Other business rules:**
- **Ultimate-analysis columns are permanently excluded** from modeling (`select_modeling_columns`), even though they exist in the raw data — a deliberate scope decision matching the stated UI input contract (only proximate analysis, which is what's cheaply available).
- **Exact duplicate rows are removed**, but **statistical outliers are not** — duplicates are noise (measurement/data-entry artifacts), while outliers may be legitimate rare coal types (e.g., very high-rank anthracite or very wet lignite) whose signal the model should learn from, not discard.
- **Model selection is purely RMSE-driven** (`select_best` = `min(..., key=rmse)`) — MAE and R² are recorded for context but never used as tie-breakers or selection criteria.

---

## 13. Important Components

| Component | Responsibilities | Inputs | Outputs | Interacts with |
|---|---|---|---|---|
| **Data Cleaning Pipeline** (`clean_pipeline`) | Validate, dedup, select columns, impute, report outliers | Raw `DataFrame` | Clean `DataFrame` + report `dict` | `schema.validate_dataframe`; called by `run_clean.py` |
| **Model Registry** (`build_registry`) | Declare every candidate model + its tuning space + metadata needed later (scaling flag, tree flag, confidence method) | `random_seed` | `list[ModelSpec]` | Consumed by `train.py` and `run_train.py` |
| **Training Loop** (`train_and_tune` + `select_best`) | Fit/tune every candidate, pick the winner by test RMSE | Registry + train/test arrays | `TrainedModel` (winner) | Feeds `run_train.py`'s artifact assembly |
| **Confidence Engine** (`confidence.py`) | Turn a raw input into a spread estimate and a 90% CI, penalized for extrapolation | Artifact + raw feature dict + prediction | `(low, high)` tuple / `float` score | Shared by `streamlit_app.py` and its own test suite |
| **Explainability Engine** (`shap_explain.py`) | Build a SHAP explainer, compute global + per-prediction attributions, generate a sentence | Artifact/estimator + training background + one input row | Contributions list + sentence | Shared by `run_explain.py` and `streamlit_app.py` |
| **Streamlit App** (`streamlit_app.py`) | Orchestrate the entire live user experience: load cached artifacts, gather input, call the engines above, render results | User widget state | Rendered UI | `ui/*`, `ml/src/*` |
| **UI Component Library** (`ui/components.py`, `ui/charts.py`) | Pure rendering — sliders, KPI cards, header/footer, Plotly figures | Already-computed numbers + `Theme` | Streamlit-rendered HTML/Plotly | `theme.py` for tokens; called by `streamlit_app.py` |
| **Theme System** (`ui/theme.py`, `ui/styles.py`) | Centralize every color/font/shadow token; inject as one CSS block | Theme name (`"dark"`/`"light"`) | `Theme` dataclass / CSS string | Read by `components.py` and `charts.py` |

---

## 14. State Management

There is no Redux/Zustand/Context API — **state lives in `st.session_state`**, Streamlit's built-in per-browser-session key-value store, which persists across script reruns within the same session but resets on a hard page reload/new session.

| State key(s) | Where set | Where read | Lifetime |
|---|---|---|---|
| `Moisture`, `Volatile_matter`, `Fixed_Carbon`, `Std.Ash` | Seeded from `DEFAULTS` via `setdefault` (`streamlit_app.py:81-82`); mutated by the slider widget itself, by `_step_value` (+/− buttons), by `reset_inputs()`, and by `load_example()` | `render_feature_input` (display), the `raw_input` dict built each run (`streamlit_app.py:99`) | Whole browser session |
| `dark_mode_toggle` | The `st.toggle` widget in `render_header` | `theme = get_theme(...)` at module level | Whole browser session |
| *(prediction results)* | **Not stored in session_state at all** — computed as local variables only inside the `if predict_clicked:` block | Only within the same run that clicked Predict | **One run only** — a slider tweak after predicting clears the displayed result until Predict is clicked again |

**Why this approach was used**: Streamlit's programming model is "rerun the whole script on every interaction," so anything that must survive a rerun *must* go through `session_state` — there is no closures-over-time or component-instance state like React. Widget values are automatically synced to `session_state[key]` by the framework, which is why every input widget in `ui/components.py` is given an explicit `key=` matching the feature name — this is the load-bearing mechanism that makes `Reset`/`Load example` work (they simply overwrite `st.session_state[key]` directly, and Streamlit's next render picks it up).

**Caching (a form of state that spans sessions, not just reruns)**:
- `@st.cache_resource` — used for `load_artifact()` and `load_explainer()`: these hold non-serializable objects (a fitted estimator, a SHAP explainer) and are cached **per server process**, shared across all users' sessions, computed once until the server restarts or cache is cleared.
- `@st.cache_data` — used for `load_clean_dataset()` and `load_metadata()`: these return serializable data (a DataFrame, a dict) and are cached by content hash of inputs.

---

## 15. Third-Party Services

**None of Google Maps, Firebase, Supabase, MongoDB, Redis, Cloudinary, OpenAI, or Stripe are used.** The only external service in this project is:

| Service | Why used | Auth | API limits | Error handling |
|---|---|---|---|---|
| **Streamlit Community Cloud** | Free hosting purpose-built for Streamlit apps; deploys directly from the GitHub `master` branch with zero infrastructure setup | GitHub OAuth (developer's account only, to authorize the deploy — no runtime API keys) | Free tier resource limits (CPU/RAM caps, may sleep after inactivity) — not explicitly handled in code | None needed — no network calls are made *from* the app to any external API at runtime; it's fully self-contained (reads local files bundled in the git repo) |
| **Google Fonts** (`fonts.googleapis.com`) | Loads "Space Grotesk" and "Plus Jakarta Sans" via a `@import url(...)` in the injected CSS (`ui/theme.py:68-72`, `ui/styles.py:13`) | None (public CDN) | None enforced/handled | If the font CDN is unreachable, the browser silently falls back to the CSS `sans-serif` fallback — no error handling needed since this is a pure CSS `@import`, not a JS fetch |

No API keys, secrets, or `.env` variables are required to run this project at all — a notable simplicity relative to most portfolio projects, and worth mentioning proactively in interviews as a deliberate scope choice, not an oversight.

---

## 16. Authentication Flow

**Not present.** There is no login, registration, session/JWT/cookie-based auth, or RBAC anywhere in the codebase. The Streamlit app is fully public — anyone with the URL can use it, and there are no user accounts, no per-user data, and nothing to authorize access to. This is an intentional simplicity for a single-purpose public demo tool, not an omission that matters for its current scope — but it's the first thing that would need to be added if this became a multi-tenant internal enterprise tool (see [Section 26](#26-possible-improvements)).

---

## 17. Error Handling

| Layer | Mechanism |
|---|---|
| **Row-level validation** | Pydantic raises `ValidationError` per invalid row inside `validate_dataframe`'s `try/except`; the error is *caught and recorded* (not propagated), so one bad row never crashes the whole cleaning run — the row is simply excluded and logged into `data_quality_report.json["validation_errors"]` |
| **Optional heavy ML dependencies** | `registry.py` wraps `xgboost`/`catboost`/`lightgbm` imports in `try/except ImportError`, setting `HAS_XGBOOST` etc. to `False` — the registry (and thus the whole pipeline) degrades gracefully to 7 models instead of 10 rather than crashing if a native-wheel-only package fails to install on a given machine |
| **Unknown confidence method** | `compute_spread` (`confidence.py:47`) raises `ValueError(f"Unknown confidence_method: {method}")` for any method string outside the three known branches — a deliberate **fail-fast** here (unlike the graceful-degrade philosophy elsewhere) because an unknown method would silently produce a meaningless confidence number, which is worse than crashing |
| **Streamlit runtime errors** | None custom-written — relies on Streamlit's default behavior of rendering a traceback in the browser if any exception escapes the script (acceptable for a low-traffic public demo; not acceptable for a production service — see [Section 26](#26-possible-improvements)) |
| **Retries / fallbacks / circuit breakers** | **None** — there are no network calls to retry (no external API dependency at runtime) |
| **Logging** | No structured application logging framework; `streamlit_test2.log` at the repo root appears to be a local development artifact (Streamlit's own console output redirected to a file), not a deliberate logging feature |

---

## 18. Performance Optimizations

| Optimization | Where | What it does |
|---|---|---|
| **`@st.cache_resource` for the model artifact & SHAP explainer** | `streamlit_app.py:49-56` | Loads `best_model.joblib` from disk and builds the SHAP explainer **once per server process**, not once per user interaction — critical because SHAP explainer construction and joblib deserialization are non-trivial costs that would otherwise repeat on every slider tweak |
| **`@st.cache_data` for the clean dataset & metadata** | `streamlit_app.py:59-66` | Avoids re-reading `coal_clean.csv` (4,513 rows) and `model_metadata.json` from disk on every rerun |
| **Precomputed artifact bundle** | `models/best_model.joblib` | All expensive computation (training, hyperparameter search, residual computation, feature ranges) happens **offline once**; the live app only ever does cheap inference-time math (`estimator.predict`, a handful of NumPy vector ops) — this is the single biggest performance decision in the system |
| **`RandomizedSearchCV` instead of exhaustive `GridSearchCV`** | `ml/src/models/train.py` | Caps tuning cost at `n_iter × cv` fits per model regardless of how large a model's hyperparameter grid is, at the cost of not guaranteeing the global-best combination is found |
| **`n_jobs=-1` in `RandomizedSearchCV`** | `train.py:44` | Parallelizes cross-validation folds across all available CPU cores during (offline) training |
| **Exact SHAP explainers (Tree/Linear) instead of `KernelExplainer`** | `shap_explain.py` | Polynomial-time exact Shapley values instead of exponential-time brute force or slow sampling-based approximation |
| **200-row SHAP sample instead of the full dataset** | `run_explain.py:9,16` | Bounds the cost of computing `summary_plot_data` for the global-importance artifact rather than running SHAP over all 4,513 rows |

**Explicitly *not* present / not applicable** (and worth being honest about this in an interview, since the checklist below is a generic web-app list that doesn't map cleanly onto a single-user-at-a-time ML demo):
- **Pagination / batching / parallel request processing** — not applicable; there is no list of records being paged through, and each Streamlit session handles one user's one prediction at a time, not a batch of concurrent requests.
- **Debouncing/throttling** on the sliders — not implemented; every slider drag tick triggers a full script rerun (Streamlit's default), which is acceptable at this UI's scale but is a real, identifiable optimization opportunity (see [Section 26](#26-possible-improvements)).
- **Database indexing** — not applicable (no database).
- **Memoization beyond Streamlit's own decorators** — none custom-written.
- **Vectorized row validation** — `validate_dataframe` uses `df.iterrows()` (`schema.py:43`), which is `O(n)` but with high per-row Python/Pydantic overhead compared to a vectorized pandas/NumPy boolean-mask approach; acceptable at 4,540 rows (sub-second), but would not scale to millions of rows without rewriting as vectorized checks.
- **kNN neighbor search uses full `np.argsort`** (`confidence.py:31-32`) rather than `np.argpartition`, doing `O(n log n)` work to find the top-15 nearest neighbors when `O(n)` would suffice — negligible at 4,513 training rows, but a legitimate "how would you optimize this at scale" interview answer (see [Section 26](#26-possible-improvements)).

---

## 19. Security

| Concern | Status in this project |
|---|---|
| **Input validation** | Present at training time (Pydantic `CoalRecord`) but this validates *training data*, not live user input — the Streamlit sliders enforce bounds (`min_value=0.0, max_value=100.0`) at the **widget level**, which is a legitimate but framework-provided (not custom-coded) form of input constraint |
| **Sanitization / XSS prevention** | `st.markdown(..., unsafe_allow_html=True)` is used extensively (`ui/components.py`, `ui/styles.py`) to render custom HTML/CSS — but every string passed to it is either a hardcoded template or a **server-computed, formatted number** (e.g., `f"{prediction_kcal:,.0f}"`), never raw free-text user input. There is no user-controlled text field anywhere in the app, so there is currently no reflected-XSS attack surface — but this pattern (`unsafe_allow_html=True` + f-strings) would need careful escaping if any free-text input were ever added |
| **Authentication / Authorization** | None — not applicable (see [Section 16](#16-authentication-flow)) |
| **Secrets / environment variables** | None required or present — no API keys, no `.env` file, nothing in `.gitignore` suggests secrets were ever handled (only `.env` is gitignored defensively, unused) |
| **Rate limiting** | None — not implemented; relies entirely on Streamlit Community Cloud's platform-level resource limits |
| **CORS** | Not applicable — Streamlit serves its own single-origin app; there is no separate API origin to configure CORS for |
| **Helmet / CSRF** | Not applicable — no custom HTTP server/Express app to attach middleware to; Streamlit handles its own transport |
| **SQL Injection** | Not applicable — no SQL database, no raw queries anywhere |
| **Dependency-pin discipline** | Root `requirements.txt` pins *exact* versions (`pandas==3.0.4`, `numpy==2.4.6`, `scikit-learn==1.9.0`, `joblib==1.5.3`, `shap==0.51.0`) — a deliberate security-*and*-correctness measure: version drift in these libraries can silently break `joblib.load()` deserialization of a pickled model bundle (pickle files are bound to the class definitions of the library version that created them) |

**Overall assessment**: the OWASP Top 10 largely doesn't apply to this system's actual attack surface, because it has no persistent user data, no authentication, no database, and no free-text input fields — the honest answer to "how did you secure this" in an interview is "the attack surface is intentionally minimal because there's nothing here to attack (no accounts, no stored user data, no SQL, no server-rendered user-controlled strings)," not "I added security features."

---

## 20. Deployment

| Aspect | Detail |
|---|---|
| **Hosting** | Streamlit Community Cloud (free tier), live at `gcvprediction-z2gkxweznrnkrqp6fgvwfd.streamlit.app` |
| **Environment variables** | None used |
| **Build process** | Streamlit Cloud reads `.python-version` (pins Python 3.11) and `requirements.txt` (exact-pinned deps) at the repo root, builds a container image, and runs `streamlit run streamlit_app.py` |
| **Deployment pipeline** | Push to `master` on GitHub → Streamlit Cloud auto-redeploys (per `docs/superpowers/specs/2026-06-29-streamlit-app-design.md:28-30`) — no manual step, no separate CI trigger |
| **Docker** | **Not present** — no `Dockerfile`/`docker-compose.yml` anywhere in the repo (verified) |
| **CI/CD** | **Not present** — no `.github/workflows/` directory, no CI config of any kind (verified). Tests (`ml/tests/`) are run manually by the developer, not automatically on push |
| **Production architecture** | Single Streamlit process serving all users; the trained model (`models/best_model.joblib`) is committed directly to git and loaded from the local filesystem at boot — there is no model registry, no separate model-serving infrastructure, and no artifact store |
| **Retraining in production** | Manual only: developer runs `ml/run_pipeline.py` locally, commits the updated `models/*.joblib`/`*.json`, pushes to `master`; the app picks up the new model on next process restart (the README notes `st.cache_resource` requires a server restart/cache-clear to pick up a changed file, since the cache key doesn't include file mtime) |

---

## 21. Configuration Files

| File | Purpose |
|---|---|
| `requirements.txt` (root) | Deployment-only deps for Streamlit Cloud, **exact-pinned** for joblib-unpickle compatibility with the locally-trained model: `streamlit>=1.38`, `plotly>=5.24`, `pandas==3.0.4`, `numpy==2.4.6`, `scikit-learn==1.9.0`, `joblib==1.5.3`, `shap==0.51.0` |
| `.python-version` | `3.11` — read by Streamlit Community Cloud (and `pyenv`-style tools) to select the Python runtime |
| `ml/requirements.txt` | Broader **development/training** deps, mostly `>=` (loose) pins: `pandas`, `numpy`, `scikit-learn`, `pydantic`, `shap`, `joblib`, `xgboost`, `catboost`, `lightgbm`, `pytest`, `nbformat`, `nbconvert`, `jupyter`, `ipykernel`, `matplotlib`, `seaborn` — intentionally looser than the root file since this environment never needs to match a pickled artifact's exact library versions bit-for-bit, only train a *new* one |
| `ml/pytest.ini` | `[pytest] pythonpath = . / testpaths = tests` — makes `src.*` imports resolve without a package install (`pip install -e .`) and scopes test discovery to `tests/` |
| `.gitignore` | Excludes `__pycache__/`, `*.pyc`, `.venv/`/`venv/`, `node_modules/` (defensive, unused), `dist/`/`build/`, `.env` (defensive, unused), `*.log`, `.DS_Store`, `.ipynb_checkpoints/`, `.superpowers/`, `.claude/`, `catboost_info/` (CatBoost's training-log side-effect directory) |

**Not present**: `package.json`, `tsconfig.json`, `vite.config`, `webpack.config`, `.eslintrc`, `.prettierrc`, `docker-compose.yml`, `Dockerfile` — there is no JavaScript/TypeScript toolchain anywhere in this repository.

---

## 22. Libraries

(Consolidating [Section 4](#4-tech-stack)'s "why chosen" column into a scannable list — see that section for the full table with alternatives.)

**Core (both root and `ml/`)**: pandas, NumPy, scikit-learn, joblib, SHAP — the non-negotiable foundation for tabular ML with explainability.

**Training-only (`ml/requirements.txt`)**: Pydantic (schema validation), XGBoost/CatBoost/LightGBM (extra candidate models, all optional at import-time), pytest (testing), nbformat/nbconvert/jupyter/ipykernel (reproducible EDA notebook), matplotlib/seaborn (static EDA plots).

**Deployment-only (root `requirements.txt`)**: Streamlit (UI framework/server), Plotly (interactive charts) — added on top of the core ML libs, but *without* Pydantic/pytest/xgboost/catboost/lightgbm/jupyter, since the deployed app never trains a model, only loads and infers from an already-persisted one. This split keeps the production deploy's dependency footprint (and thus Streamlit Cloud's build time and disk usage) meaningfully smaller than the full dev environment.

---

## 23. Difficult Parts of the Project

1. **Most complex module: `ml/src/models/confidence.py`.** It has to produce a *statistically defensible* 90% interval from three structurally different model families (bagging ensembles, a single tree, and everything else via kNN-on-residuals) using one unified public function (`compute_confidence_interval`), while also handling the extrapolation case gracefully. Getting the three `_*_spread` branches to each return a comparable "spread" quantity (so they can be combined the same way with `residual_std` downstream) required real statistical reasoning, not just API plumbing.
2. **Most difficult logic: the model-type → confidence-method mapping in `registry.py`.** Deciding *which* uncertainty-quantification technique is valid for *which* model family (bagging ensembles get tree-disagreement spread; a lone decision tree gets leaf-local residual spread; anything else falls back to kNN-in-feature-space residual spread) required understanding the internal structure of each estimator type (e.g., that `RandomForestRegressor.estimators_` exposes individual trees, but `GradientBoostingRegressor`'s trees are *additive stages*, not independent votes, so bagging-style disagreement doesn't apply to it — hence it's mapped to `knn_residual` instead).
3. **Most innovative implementation: reusing the exact same `ml/src/*` modules in both the offline pipeline and the live Streamlit app** via a `sys.path.insert` shim. This guarantees zero logic drift between what's unit-tested and what's actually shown to users — a meaningfully more rigorous choice than the common anti-pattern of "copy the model logic into the app and hope it matches."
4. **Most optimized section: the artifact bundle contract in `run_train.py`/`best_model.joblib`.** By precomputing and persisting `residuals_train`, `residual_std`, and `feature_ranges` at training time, every live prediction avoids recomputing anything expensive — inference-time cost is just a handful of vector ops, regardless of how expensive the offline training/tuning was.

---

## 24. Challenges Solved

1. **"How do I give a regression prediction without lying about its precision?"** Solved by combining ensemble/local spread with measured test RMSE into a single interval, rather than reporting a bare point estimate (which most student ML projects stop at).
2. **"How do I explain a black-box ensemble model's prediction to a non-ML domain expert (a coal engineer)?"** Solved with exact SHAP attributions converted into both a waterfall chart *and* a templated natural-language sentence — two redundant, mutually-reinforcing explanations for different audiences.
3. **"How do I support 10 wildly different model types (linear to gradient boosting) with one training loop, without an if/elif chain per model?"** Solved via the `ModelSpec` dataclass + registry pattern — `train_and_tune` is entirely generic, driven only by declarative flags (`needs_scaling`, `param_distributions`) on each spec.
4. **"How do I make the ML pipeline resilient to a developer's machine not having every optional heavy dependency (XGBoost/CatBoost/LightGBM, which can be finicky to install, especially on Windows) installed?"** Solved with `try/except ImportError` + module-level boolean flags, tested explicitly (`test_optional_models_included_only_if_installed`).
5. **"How do I avoid drift between the model logic that's tested and the model logic that's deployed?"** Solved by having `streamlit_app.py` import `ml/src/*` directly rather than re-implementing prediction/confidence/SHAP logic inline.
6. **"How do I keep a joblib-pickled model loadable in a different (cloud) environment than the one that trained it?"** Solved by exact-pinning the deployment `requirements.txt` to match the locally-installed versions used at training time.

---

## 25. Trade-offs

| Decision | Chosen approach | Alternative considered | Why the trade-off was accepted |
|---|---|---|---|
| Streamlit vs. FastAPI + React | Single-file Streamlit app | The original 5-sub-project plan (FastAPI backend + React frontend) | Faster to ship a working, deployable demo as a solo developer on a capstone timeline; costs some UI flexibility/customizability and rules out a "real" client-server split, but the project explicitly reused the ML pipeline modules directly instead of duplicating logic behind an API, so little was lost architecturally |
| `RandomizedSearchCV` vs. `GridSearchCV` | Randomized, 10 iterations, 5-fold | Exhaustive grid search | Bounded, predictable tuning time across 10 models at the cost of not guaranteeing the globally-optimal hyperparameters are found |
| Soft (UI pill) vs. hard (blocking) proximate-sum validation at inference time | Soft — warns but allows Predict | Disable the Predict button until the sum ≈ 100 | Preserves exploratory "what-if" usability; leans on the extrapolation penalty to keep the reported confidence honest instead |
| Outliers reported but not removed | Keep them, flag in the quality report | Auto-drop IQR outliers | Outliers likely represent real, rare coal types (lignite/anthracite extremes) whose signal the model should learn from — removing them would bias the model toward "average" coal only |
| Docker/CI/CD explicitly descoped | None | Add Docker + GitHub Actions | Project owner's own words: *"this is a college project, kept simple and explainable, not a DevOps showcase"* — deliberately kept the scope focused on the ML/data-science depth rather than spreading effort into infra |
| Exact vs. loose dependency pinning | Root `requirements.txt` exact-pinned; `ml/requirements.txt` loose | Pin everything exactly everywhere | Training environment benefits from staying current (loose pins); the deployed environment must match the *exact* versions that pickled the model, so it's pinned tightly instead |
| `compute_confidence` (0–100 score) kept but unused by the UI | Kept in code + tests, replaced in the UI by `compute_confidence_interval` | Delete the unused function | Represents an earlier design iteration (confirmed by commit history: `ea8099a "replace confidence-score gauge with a real confidence interval"`) that was superseded but not cleaned up — a real, honestly-acknowledged bit of technical debt rather than a trade-off with upside (see [Section 34](#34-weaknesses)) |

---

## 26. Possible Improvements

**Scalability**
- Move from flat-file artifacts to a proper model registry (e.g., MLflow) if multiple model versions ever need to be tracked/rolled back.
- Vectorize `validate_dataframe`'s row-by-row Pydantic loop (or use `pandas`-native boolean masks) if the raw dataset grows past hundreds of thousands of rows.
- Replace the `knn_residual` branch's full `np.argsort` with `np.argpartition` for O(n) top-k neighbor selection instead of O(n log n).

**Monitoring**
- No metrics/observability exist at all today (no request counts, no prediction-distribution drift detection, no uptime dashboard). Adding basic logging of predictions (with SHAP attributions) would enable retroactive model-drift analysis.

**Testing**
- All 44 tests are unit/integration-level; there are no UI/browser tests of the Streamlit app itself (e.g., via Playwright) verifying the rendered charts/sliders actually behave correctly end-to-end in a browser.
- No CI pipeline runs these tests automatically on push — currently 100% manual (`pytest` run locally before committing).

**Security**
- If this ever gained user accounts or free-text fields, `unsafe_allow_html=True` usage would need an explicit escaping audit.
- Add basic abuse/rate limiting if traffic ever became a concern (not needed today).

**Performance**
- Debounce slider drags so a full script rerun doesn't fire on every pixel of drag movement (Streamlit reruns on every widget state change by default).
- `shap_global.json` is computed but never read by `streamlit_app.py` — either wire it into a "Model Performance"/"Analytics" page (as the original design spec envisioned) or stop computing it to save a pipeline stage.

**Architecture**
- Reinstate the originally-planned backend API layer if this needs to serve non-Streamlit clients (e.g., a mobile app or a lab's LIMS integration) — the `ml/src/*` modules are already structured cleanly enough to drop behind a FastAPI route with minimal change.
- Delete or actively use the unused `compute_confidence` (0–100 score) function to remove dead-but-tested code.

---

## 27. Resume Points Explained

> Suggested resume bullets, each explained: what it means, where it lives in the code, and how to defend it in an interview.

**"Built an end-to-end ML pipeline predicting Coal GCV with automated model selection across 10 regression algorithms, achieving R²=0.99 (RMSE ≈153 kcal/kg) via RandomizedSearchCV tuning."**
- *Means*: a fully automated `clean → train/tune/compare → select` pipeline, not a single hand-picked model.
- *Where*: `ml/src/models/registry.py` (10 candidates), `train.py` (`train_and_tune`/`select_best`), `run_train.py` (orchestration), `models/model_comparison.json` (the receipts).
- *Interview justification*: be ready to explain *why* Extra Trees won (lower variance than Random Forest via extra split-randomization, on a dataset where the feature→target relationship is smooth) and why RMSE (not R²/MAE) was the selection criterion.

**"Designed a per-prediction 90% confidence interval combining ensemble uncertainty and measured test error, with automatic widening for out-of-distribution inputs."**
- *Means*: uncertainty quantification beyond a bare point estimate.
- *Where*: `ml/src/models/confidence.py` (`compute_spread`, `compute_confidence_interval`, `_extrapolation_penalty`), tested in `test_confidence.py`.
- *Interview justification*: explain the `√(spread² + residual_std²)` formula (combining two independent-ish error sources in quadrature) and walk through the three model-type-specific spread computations.

**"Implemented explainable AI using SHAP TreeExplainer/LinearExplainer, generating both visual waterfall charts and natural-language explanations for individual predictions."**
- *Means*: model interpretability for a non-ML audience.
- *Where*: `ml/src/explain/shap_explain.py`, `ui/charts.py:53-76` (`shap_waterfall`), tested for additivity in `test_shap_explain.py`.
- *Interview justification*: know the additivity guarantee (`base_value + Σshap == prediction`) and why exact explainers were possible here (tree/linear model families both have them) versus needing an approximate method for a truly black-box model.

**"Wrote 44 automated pytest tests across 12 modules covering data validation, model training, confidence scoring, and SHAP correctness, achieving 100% pass rate."**
- *Means*: engineering discipline beyond "the model works on my machine."
- *Where*: `ml/tests/*`.
- *Interview justification*: be ready to explain the TDD-ish workflow evident in the plan doc (write failing test → implement → verify pass → commit) and the deliberate exception for the three "slow" tasks (full training run, full SHAP run) where red/green-per-line TDD wasn't practical.

**"Built and deployed a custom-themed interactive Streamlit dashboard with dark/light mode, live on Streamlit Community Cloud."**
- *Means*: shipped a real, usable product, not just a notebook.
- *Where*: `streamlit_app.py`, `ui/*`.
- *Interview justification*: be ready to explain Streamlit's rerun-on-interaction execution model and how `st.session_state`/`st.cache_resource`/`st.cache_data` were used deliberately, not just "it's what Streamlit gives you."

---

## 28. Placement Interview Explanation

**How to frame this project per company archetype** (what each is likely to probe, based on their typical interview emphasis):

| Company archetype | Likely focus | How this project answers it |
|---|---|---|
| **Google** | Algorithmic rigor, complexity analysis, "why this data structure/algorithm" | Walk through Tree SHAP's polynomial-time exact algorithm vs. exponential brute-force Shapley; the RandomizedSearchCV vs GridSearchCV complexity trade-off; the kNN-residual `O(n log n)` vs achievable `O(n)` improvement |
| **Amazon** | Leadership Principles / ownership / "customer obsession", scale thinking | Frame the confidence-interval + explanation features as "not stopping at a number — building for the end user's actual decision-making need"; be honest about what wouldn't scale (flat-file storage, no monitoring) and how you'd evolve it |
| **Microsoft** | Engineering fundamentals, testing discipline, clean design | Lead with the 44-test suite and the `ModelSpec` registry pattern that avoids an if/elif explosion across 10 models |
| **Atlassian** | Collaboration, incremental delivery, documentation | Point to the spec-driven-development docs (`docs/superpowers/specs/*`, `plans/*`) showing design-before-code discipline and incremental, testable task breakdown |
| **Uber** | Real-world operational systems, data pipelines at scale | Be candid: this is dataset-scale (4.5K rows), not Uber-scale — pivot to how you'd redesign it for millions of rows (vectorized validation, a real model registry, monitoring for feature/prediction drift) |
| **Flipkart** | Practical product sense, trade-off articulation | The proximate-sum soft-validation trade-off ([Section 25](#25-trade-offs)) is a great example of "product usability vs. strict correctness" reasoning |
| **Oracle** | Data modeling, correctness, schema discipline | Explain the `FEATURE_COLUMNS` ordering-as-contract discipline, and the Pydantic schema as a lightweight but rigorous data-quality gate |
| **Goldman Sachs** | Quantitative rigor, risk/uncertainty communication | The confidence-interval design (combining epistemic + measured error, in quadrature, widened by extrapolation) is directly analogous to risk-interval reasoning in finance — emphasize this framing heavily |
| **High-growth startups** | Speed, pragmatism, shipping a full loop | Emphasize going from raw CSV to a live, deployed, explainable product solo, in a scoped and shipped MVP rather than an open-ended research exercise |

---

## 29. Frequently Asked Questions

### Basic

1. **What does this project do?** Predicts coal's Gross Calorific Value (kcal/kg) from four proximate-analysis inputs, with a confidence interval and SHAP explanation.
2. **What are the four inputs?** Moisture, Volatile Matter, Fixed Carbon, Ash (`Std.Ash`) — all percentages that sum to ~100%.
3. **What's the output?** Predicted GCV in kcal/kg, a 90% confidence interval, the model's test RMSE, and a SHAP-based explanation sentence + chart.
4. **What language/framework is this built in?** Python 3.11; scikit-learn for ML, Streamlit for the UI.
5. **Is there a backend server?** No — Streamlit runs the UI and the inference logic in the same process; there's no separate REST API.
6. **Where's the trained model stored?** `models/best_model.joblib`, committed directly to git.
7. **How big is the dataset?** 4,540 raw rows, cleaned down to 4,513.
8. **Which model won?** `ExtraTreesRegressor`, chosen from 10 candidates by lowest test RMSE.
9. **What's the model's accuracy?** R² = 0.9904, RMSE = 0.6425 MJ/kg (~±153 kcal/kg).
10. **Is the app deployed anywhere?** Yes, Streamlit Community Cloud.

### Intermediate

11. **Why is GCV predicted rather than measured directly?** Because bomb calorimetry is slow/needs lab equipment, while proximate analysis is fast and already routinely measured.
12. **Why do the four features sum to 100%?** Proximate analysis partitions the whole coal sample by mass into those four components by definition.
13. **What happens if a user's slider inputs don't sum to 100%?** A warning pill appears, but the app still predicts — the confidence interval widens via the extrapolation penalty instead of blocking.
14. **How is the dataset cleaned?** Row validation (Pydantic), duplicate removal, ultimate-analysis column drop, median imputation (defensive), IQR outlier reporting (not removal).
15. **How many duplicate rows were found and removed?** 27 exact duplicates.
16. **Were there any missing values in the raw data?** No — verified by the data quality report, though the imputation code path exists defensively.
17. **Why drop the ultimate-analysis columns (Hydrogen, Carbon, etc.)?** They're not part of the intended four-input UI/product scope — a deliberate scope decision, not a data-quality issue.
18. **What's a `ModelSpec`?** A dataclass declaring one candidate model's factory function, hyperparameter search space, scaling requirement, tree-based flag, and confidence-scoring method.
19. **How are hyperparameters tuned?** `RandomizedSearchCV`, 10 iterations, 5-fold CV, scored by negative RMSE.
20. **Why RandomizedSearchCV instead of GridSearchCV?** Bounds tuning time consistently across 10 very differently-sized hyperparameter grids.
21. **How is the best model selected?** Lowest test-set RMSE among all tuned candidates.
22. **What is the artifact bundle?** An 11-key dict (`estimator, scaler, feature_columns, target_column, confidence_method, is_tree_based, X_train, y_train, residual_std, residuals_train, feature_ranges`) serialized via joblib.
23. **Why bundle training data (`X_train`, `residuals_train`) into the deployed artifact?** The confidence engine needs them at inference time (kNN-residual lookups, leaf-local residuals) — they must travel with the model, not just the estimator.
24. **What is `residual_std` set to?** The winning model's test-set RMSE (used as the irreducible-error term of the confidence interval).
25. **How does the app know which confidence method to use?** Each `ModelSpec` in the registry declares its `confidence_method`; that string travels into the artifact and dispatches `compute_spread`.

### Advanced

26. **Explain the 90% confidence interval formula.** `half_width = Z₉₀ × √(spread² + residual_std²) / extrapolation_penalty`, where `Z₉₀ = 1.645`, `spread` is model-type-specific epistemic uncertainty, `residual_std` is test RMSE, and the penalty widens the interval for out-of-range inputs.
27. **Why combine spread and residual_std in quadrature (√(a²+b²)) rather than adding them?** Treats the two uncertainty sources as (approximately) independent variances, which is the standard way to combine independent error terms.
28. **How is "spread" computed for a bagging ensemble (Random Forest/Extra Trees)?** Standard deviation of each individual tree's prediction for the given input.
29. **How is "spread" computed for a single Decision Tree?** Standard deviation of training residuals among rows that land in the same leaf as the input (via `estimator.apply()`).
30. **How is "spread" computed for linear/boosting models?** Standard deviation of residuals among the 15 nearest training neighbors (Euclidean distance in feature space) — a kNN-residual approach.
31. **Why does Gradient Boosting use kNN-residual rather than bagging-ensemble spread, even though it's an ensemble of trees?** Its trees are additive boosting *stages* correcting each other's errors, not independent bootstrap votes — there's no meaningful "disagreement between trees" signal to use the way there is in Random Forest/Extra Trees.
32. **What is the extrapolation penalty and how is it computed?** Per-feature: if the input value is within the training range, penalty factor = 1; if outside, `max(0.5, 1 - distance_beyond_range/range_span)`, multiplied across all four features.
33. **Why is the extrapolation penalty floored at 0.5 per feature rather than allowed to go to 0?** So one wildly out-of-range feature can't make the confidence interval infinitely wide/degenerate — it's a bounded, interpretable penalty rather than an unbounded one.
34. **How does SHAP TreeExplainer differ from KernelExplainer?** TreeExplainer computes *exact* Shapley values in polynomial time by exploiting tree structure; KernelExplainer is model-agnostic but approximate and much slower (sampling-based).
35. **What does SHAP additivity mean, and how is it tested?** `base_value + sum(shap_values) == prediction` (within floating-point tolerance) — verified explicitly in `test_shap_explain.py`.
36. **How is the natural-language explanation sentence generated?** `generate_explanation_sentence` takes the sorted SHAP contributions, picks the top-N positive and top-N negative, and formats them into a templated sentence — not an LLM call, purely deterministic string templating.
37. **Why does the Streamlit app convert predictions from MJ/kg to kcal/kg?** The model is trained on GCV in MJ/kg (the dataset's native unit), but the coal industry conventionally reports GCV in kcal/kg — conversion factor `1000/4.1868`.
38. **Where is that conversion applied, and to what?** In `streamlit_app.py`, applied to the prediction, the confidence interval bounds, the RMSE display, and the SHAP contribution values — everything the user sees is in kcal/kg, while everything on disk (metrics, artifact) stays in MJ/kg.
39. **Why use `st.cache_resource` for the model but `st.cache_data` for the dataset?** `cache_resource` is for non-serializable/singleton objects (a fitted estimator, a SHAP explainer object); `cache_data` is for serializable data (a DataFrame, a dict) that Streamlit can hash and potentially copy safely.
40. **What does the leading underscore in `load_explainer(_artifact)` do?** Tells Streamlit's caching decorator not to attempt to hash that argument (some objects, like fitted estimators, aren't reliably hashable) — a documented Streamlit convention.

### Architecture

41. **Is this a monolith or microservices?** Neither in the traditional sense — it's a single-process application with an internally modular (not network-separated) architecture; the "ML pipeline" and "UI" are separate Python packages within one deployable unit.
42. **How do the offline pipeline and the live app share logic without duplicating it?** `streamlit_app.py` inserts `ml/` onto `sys.path` and imports the same `ml/src/*` modules the pipeline scripts and test suite use.
43. **Why wasn't a FastAPI backend built, given the original design spec planned one?** It was descoped in favor of shipping a working Streamlit MVP directly reusing the pipeline modules — trading a "real" client-server split for faster, simpler delivery.
44. **What would need to change to add a real backend API later?** Wrap the existing `ml/src/*` functions (`compute_confidence_interval`, `explain_single_prediction`) behind FastAPI routes — minimal change since they're already pure, dependency-injected functions, not tangled into the UI.
45. **How is configuration centralized?** `ml/src/config.py` — every path and constant (feature columns, target column, seed, tolerances) lives there, imported everywhere else.
46. **Why is `PROJECT_ROOT` computed via `Path(__file__).resolve().parents[2]` instead of a hardcoded path?** Makes the pipeline runnable regardless of the current working directory it's invoked from.
47. **What's the data lineage from raw CSV to a live prediction?** `coal_all.csv → (clean) → coal_clean.csv → (train) → best_model.joblib → (loaded by Streamlit) → live predictions`.

### Backend (ML pipeline, the closest analogue)

48. **What's the "controller" layer in this codebase?** The `main()` functions in `run_clean.py`/`run_train.py`/`run_explain.py` — they do I/O and call into `src/`.
49. **What's the "service" layer?** `ml/src/data/*`, `ml/src/models/*`, `ml/src/explain/*` — pure, unit-tested business logic with no direct file I/O.
50. **Why keep I/O out of the `src/` modules?** So they're testable with in-memory fixtures (synthetic arrays, small DataFrames) without touching disk — enables fast, deterministic unit tests.
51. **How would you test `run_train.py` itself, given it takes minutes to run against real data?** It isn't red/green TDD'd line-by-line — it's run once against real data, then a fast verification test (`test_run_train.py`) checks the *persisted artifact's* structure and key values, an explicitly acknowledged deviation from strict TDD ordering for slow, one-shot pipeline stages.

### Frontend (Streamlit, the closest analogue)

52. **What's the Streamlit execution model?** Every user interaction triggers the entire script to re-run top-to-bottom; there's no persistent component tree like React's.
53. **How do widgets keep their value across reruns?** Via an explicit `key=` argument binding them to `st.session_state`.
54. **How were the custom +/− step buttons implemented, since Streamlit sliders don't have them natively?** As separate `st.button`s with an `on_click=_step_value` callback that reads/clamps/writes `st.session_state[key]` directly.
55. **How is the glassmorphism look achieved without a CSS framework?** A single large CSS string is generated in Python (`ui/styles.py:_build_css`) using theme tokens, then injected via `st.markdown(..., unsafe_allow_html=True)` — targeting Streamlit's internal `data-testid` attributes and `st-key-*` classes.
56. **How does dark/light mode work?** A `Theme` frozen dataclass holds every color token; `get_theme("dark"/"light")` picks one; `st.toggle` in the header writes to `st.session_state["dark_mode_toggle"]`, which the top of the script reads to choose the theme before rendering anything.
57. **Are the Plotly charts interactive?** Yes (hover tooltips work), but the mode bar is disabled (`config={"displayModeBar": False}`) for a cleaner look.

### Database (filesystem, the closest analogue)

58. **Why no database?** No concurrent writers, no transactions needed, no user accounts, dataset fits comfortably in memory/git — a database would be pure overhead.
59. **What plays the role of a "primary key" across modules?** The `FEATURE_COLUMNS` list order in `config.py` — every NumPy array in the pipeline is positional and must respect this exact order.
60. **Where would you add a database if this had to support many concurrent users submitting predictions to be reviewed later?** A predictions-log table (timestamp, inputs, prediction, interval, model version) — currently nothing is persisted per-prediction at all.

### API (function-level, the closest analogue)

61. **What would a `/predict` endpoint's request/response look like if built?** See the example JSON contract in [Section 10](#10-apis) — request has the four features, response has the prediction, interval, and SHAP breakdown.
62. **Is there input validation on this hypothetical endpoint today?** Only at the Streamlit widget level (min/max on sliders) — there's no server-side request-body validation layer since there's no server-side request body at all.

### React / Node

63. **Is React used anywhere?** No — confirmed no `frontend/`, no `package.json`, no JS framework anywhere in the repo.
64. **Is Node.js used anywhere?** No — the entire stack is Python.
65. **If you had to port the UI to React, what would change?** You'd need a real backend API (see Q43/Q61) since React can't import Python modules directly the way Streamlit's single-process model allows.

### Performance

66. **What's the single biggest performance win in this system?** Precomputing everything expensive (training, tuning) offline, so the live app only ever does cheap inference-time math.
67. **What's inefficient but acceptable at current scale?** `validate_dataframe`'s row-by-row Python loop and the kNN-residual branch's full `np.argsort` instead of `np.argpartition` — both `O(n log n)`-ish where a more scalable approach exists, but negligible at ~4,500 rows.
68. **How would you speed up validation for a much larger dataset?** Vectorize the Pydantic checks into pandas boolean-mask operations instead of iterating rows.
69. **How would you speed up the kNN-residual confidence branch at scale?** Use `np.argpartition` for O(n) top-k selection instead of a full O(n log n) sort, or precompute a spatial index (e.g., a KD-tree) if this were called at high request volume.
70. **Does the app do any caching?** Yes — `st.cache_resource` (model/explainer) and `st.cache_data` (dataset/metadata), both Streamlit-native decorators.

### Security

71. **What's the biggest security consideration in this app?** There largely isn't one — no auth, no database, no user-controlled free text, no external API calls at runtime.
72. **Is `unsafe_allow_html=True` a security risk here?** Not currently, because every string passed through it is either hardcoded or a server-formatted number, never raw user input — but it would need auditing if free-text input were ever added.
73. **Are there any secrets in the repo?** No — no API keys, no `.env` file is used (only defensively gitignored).
74. **How are exact dependency versions a security consideration?** Pinning `pandas==3.0.4`, `numpy==2.4.6`, etc. exactly in the deployment `requirements.txt` prevents a supply-chain-style "surprise" from a transitive version bump silently breaking `joblib.load()` on the cloud, since pickled objects are bound to the library versions that created them.

### Deployment

75. **How is this deployed?** Push to `master` → Streamlit Community Cloud auto-builds from `.python-version` + `requirements.txt` and runs `streamlit run streamlit_app.py`.
76. **Is there a Dockerfile?** No — confirmed absent from the repo.
77. **Is there a CI/CD pipeline?** No — confirmed no `.github/workflows/`; tests are run manually.
78. **How do you retrain and redeploy the model?** Run `ml/run_pipeline.py` locally, commit the new `models/*` files, push — the live app picks them up on its next process restart/cache clear.
79. **Why are dependencies pinned differently in root vs. `ml/`?** Root must exactly match the training environment's versions (for joblib compatibility); `ml/` can stay loosely pinned since it always trains a fresh artifact.

### Behavioral

80. **What was the hardest bug you fixed on this project?** (Use the real commit history: `c9d0864 "fix(ml): calibrate confidence score against test RMSE instead of train residuals"` — explain that using *training* residuals as the irreducible-error baseline understated real-world uncertainty, since training residuals are optimistic; fixing it to use test-set RMSE gave an honest interval.)
81. **What would you do differently if you started over?** Reasonable honest answers: wire up `shap_global.json` into an actual dashboard page, or vectorize the row-validation loop from the start.
82. **How did you decide what NOT to build?** Point to the explicit descoping of Docker/CI/CD and the backend API/React frontend, each with a stated reason (keep scope focused, ship an MVP) rather than "ran out of time."
83. **How did you validate the model was good enough to ship?** Test-set RMSE/R² comparison across all 10 candidates (`model_comparison.json`), not just training accuracy.
84. **Tell me about a design decision you're proud of.** The soft (non-blocking) proximate-sum validation at inference paired with the extrapolation penalty — degrade gracefully rather than hard-block.
85. **Tell me about a limitation you'd flag to a stakeholder.** `shap_global.json` is computed but unused by the live UI — a small, honest example of following through on cleanup.

### Design

86. **Why glassmorphism / dark mode by default?** A deliberate visual-design choice for a portfolio-facing demo — blur + translucency + gradient accents read as "modern data product," and dark-by-default suits a technical/analytical tool aesthetic.
87. **Why Space Grotesk + Plus Jakarta Sans specifically?** Space Grotesk (display font) has a technical/geometric character fitting the "predictor" branding; Plus Jakarta Sans (body font) is a clean, highly legible sans-serif — loaded once via Google Fonts `@import`.
88. **Why three separate KPI cards (prediction, CI, RMSE) instead of one combined view?** Separates the point estimate, its uncertainty range, and the model's overall known error into distinct, scannable units rather than one cluttered number.

### Optimization

89. **What's the most expensive operation in this codebase, and when does it run?** Full 10-model `RandomizedSearchCV` tuning in `run_train.py` — takes roughly 1–3 minutes, runs *offline only*, never in the live request path.
90. **What's the most expensive *live* operation?** Building the SHAP explainer (cached per process) and per-request `estimator.predict` + confidence math — all sub-second.
91. **Could this run on serverless/lambda-style infrastructure?** Not ideally as-is, since `st.cache_resource` assumes a long-lived process to amortize the model-load cost across requests; a cold-start-per-request model would repay that cost every time.

### Edge Cases

92. **What happens if a user enters Moisture=0, Volatile Matter=0, Fixed Carbon=0, Ash=100?** The app still predicts (sum = 100, passes the soft validation), but this is likely far outside any real coal sample's feature distribution for at least some features — the extrapolation penalty would widen the confidence interval if any single feature falls outside its observed training range.
93. **What happens if the sum is wildly off (e.g., 40%)?** The warning pill turns amber, but Predict still works — expect a wider confidence interval from the extrapolation penalty for whichever features are out of their training range.
94. **What if `residual_std` is 0?** `compute_confidence`/`compute_spread`-based scoring returns 100.0 (perfect confidence) as a defensive branch — though in practice the winning model's test RMSE is never exactly 0.
95. **What happens if XGBoost/CatBoost/LightGBM aren't installed?** The registry silently includes only the 7 core scikit-learn models instead of erroring — explicitly tested (`test_optional_models_included_only_if_installed`).
96. **What if a raw data row has a proximate sum of exactly 100.5 (right at the tolerance boundary)?** It fails validation, since the check is `abs(total - 100) > PROXIMATE_SUM_TOLERANCE (0.5)` — exactly 100.5 is right at the boundary and is *rejected* (`>` not `>=`... actually equal to tolerance is NOT `>` so it would pass; a value must *exceed* 100.5 to fail).
97. **Can the app show a stale prediction after inputs change?** Yes — since prediction results are local variables (not in `session_state`), a later slider tweak without clicking Predict again simply hides the old result on the next rerun rather than showing an inconsistent one, because the whole prediction block only renders `if predict_clicked` on *that specific rerun*.
98. **What if the user clicks "Load example" then immediately "Predict"?** Works as expected — `load_example()` overwrites the four `session_state` feature values with a real sampled row before the rerun, and the subsequent Predict click reads those fresh values.
99. **What happens on the very first page load, before any Predict click?** Only the input form and sum-validation pill render; no KPI cards or SHAP chart are shown until the user explicitly clicks Predict.
100. **What if the trained artifact file is missing or corrupted on the server?** Not explicitly handled — `joblib.load()` would raise an exception that Streamlit would surface as an unhandled traceback in the browser (see [Section 17](#17-error-handling) and [Section 26](#26-possible-improvements)).
101. **How does the system behave for a coal type completely unlike anything in the training data (e.g., anthracite with very low volatile matter)?** The kNN-residual/leaf-local/tree-disagreement spread would likely be small if similar rows exist in the training set at all (the dataset does span from lignite to anthracite), but the extrapolation penalty is the actual safeguard for genuinely novel feature combinations outside the observed ranges.
102. **Is there any protection against a user submitting negative numbers?** Yes, but only at the Streamlit widget level (`min_value=0.0` on every slider) — there's no separate server-side re-validation, since the widget itself constrains the value that can ever reach `session_state`.

---

## 30. Code Walkthrough

**From `streamlit run streamlit_app.py` to a rendered prediction:**

1. **Process boot**: Streamlit's server starts, imports `streamlit_app.py`. Module-level code runs once: `sys.path` is extended with `ml/` and the project root so `src.*` and `ui.*` are importable without a package install; all the `from src...`/`from ui...` imports execute; `KCAL_PER_MJ` and `DEFAULTS` constants are defined; `st.set_page_config(...)` runs.
2. **First script run (new session)**: `st.session_state.setdefault(column, value)` seeds the four feature keys with `DEFAULTS` (only takes effect the very first time, since `setdefault` is a no-op on subsequent reruns where the key already exists). `dark_mode` reads `session_state.get("dark_mode_toggle", True)` (defaults to dark). `theme = get_theme(...)`, `inject_global_css(theme)` writes the CSS `<style>` block. `render_header(theme)` draws the title, status pill, and the dark/light `st.toggle`.
3. **Input form renders**: two `st.columns`, each calling `render_feature_input` twice — each call draws a glass card with an icon/label, the current value, a `−` button, the native slider (bound by `key=`), and a `+` button. `raw_input` is assembled from `session_state` and `render_sum_pill(total, theme)` shows the validation pill. Three action buttons (`Predict`/`Reset`/`Load example`) render in a bordered container; `Reset`/`Load example` are wired via `on_click=` callbacks, while `Predict`'s return value (`predict_clicked`) is just read directly this run.
4. **User clicks Predict**: Streamlit sends the click over the WebSocket, the whole script reruns; this time `predict_clicked` is `True` at the point the script reaches the `if predict_clicked:` block.
5. **Inference**: `load_artifact()` returns the cached (or freshly-loaded-once) `best_model.joblib` dict. The four `session_state` values become a NumPy vector in `FEATURE_COLUMNS` order; if the winning model needed scaling, `artifact["scaler"].transform(...)` is applied. `artifact["estimator"].predict(...)` returns the raw MJ/kg prediction; it's converted to kcal/kg.
6. **Confidence interval**: `compute_confidence_interval(artifact, raw_input, prediction_mj)` (from `ml/src/models/confidence.py`) dispatches to the correct `_*_spread` function based on `artifact["confidence_method"]`, combines it with `artifact["residual_std"]` in quadrature, applies the extrapolation penalty, and returns `(low, high)` in MJ/kg — converted to kcal/kg for display.
7. **Explanation**: `load_explainer(artifact)` builds (once, cached) a SHAP `TreeExplainer`/`LinearExplainer` matching the model type. `explain_single_prediction(...)` computes per-feature SHAP values for this exact input, sorted by absolute magnitude, alongside the base value; contributions are converted to kcal/kg. `generate_explanation_sentence(...)` formats the top increasing/decreasing features into one readable sentence.
8. **Rendering the result**: `load_metadata()` (cached) supplies the model name + RMSE for display. Three `st.columns` render the KPI cards (`render_kpi_card` for prediction and RMSE; a Plotly `confidence_interval_chart` for the CI). A fourth container renders the explanation sentence and the `shap_waterfall` Plotly chart. `render_footer(...)` renders the closing credit/link row.
9. **Any subsequent interaction** (another slider drag, another button click) restarts this entire sequence from step 4 onward — there is no persistent "server loop" beyond Streamlit's own session/rerun machinery.

**First successful user interaction, end to end**: opening the URL → seeing default slider values (Moisture 5.2, VM 31.1, FC 40.7, Ash 23.0 — a real, "sum = 100" example baked in as the default) → clicking **Predict** with no changes → seeing a predicted GCV, a 90% CI chart, the model's RMSE, and a SHAP waterfall with an explanation sentence, all within roughly one Streamlit rerun (sub-second, since the model is already cached in the server process after the very first load).

---

## 31. Key Learnings

Building this project would teach a developer:

- **How to quantify and communicate model uncertainty**, not just accuracy — the difference between "the model says 6200 kcal/kg" and "the model says 6200 ± 150 kcal/kg, and here's why it's less sure this time" is a genuinely advanced ML-engineering skill most student projects skip entirely.
- **How model interpretability (SHAP) actually works mathematically** — the additivity property, why tree/linear models get exact explainers while arbitrary black-box models need approximations, and how to translate SHAP values into a decision-maker-friendly sentence.
- **How to design a registry/strategy pattern** so that adding an 11th candidate model requires *zero* changes to the training loop, only a new `ModelSpec` entry — a transferable software-design lesson well beyond ML.
- **The practical realities of shipping ML to production (even a small one)**: pickled-model version pinning, cache invalidation on redeploy, and the gap between "works in a notebook" and "reproducible via a scripted, tested pipeline."
- **How to scope a project honestly** — recognizing when a fuller architecture (separate backend/frontend, Docker, CI/CD) is *not* worth building for the actual requirement, and saying so explicitly rather than over-engineering for imagined future needs.
- **Streamlit's rerun-based programming model** as a genuinely different paradigm from React's component/hook model — useful contrast to be able to articulate in interviews even if you never touch Streamlit again.

---

## 32. Production Readiness

| Module | Score (1–10) | Why |
|---|---|---|
| Data cleaning (`ml/src/data/`) | 8 | Well-tested, clear domain rules, but the row-by-row Pydantic loop wouldn't scale past hundreds of thousands of rows without vectorization |
| Model training (`ml/src/models/registry.py`, `train.py`) | 8 | Clean registry pattern, graceful optional-dependency handling, well-tested; lacks experiment tracking (no MLflow-style run history beyond the single latest `model_comparison.json`) |
| Confidence engine (`confidence.py`) | 7 | Statistically reasonable and well-tested across all three branches, but has an unused/duplicate public function (`compute_confidence`) that's dead code relative to the live app, and the kNN branch isn't optimized for scale |
| SHAP explainability (`shap_explain.py`) | 8 | Correctly verified (additivity tests), reused cleanly between training and serving; the `shap_global.json` output is produced but currently orphaned (unused by the UI) |
| Streamlit app (`streamlit_app.py`, `ui/*`) | 6 | Functionally solid and well-cached, but has zero automated UI tests, no error boundary for a missing/corrupt model file, and no monitoring/logging of live usage |
| Testing (`ml/tests/`) | 8 | 44 tests, meaningful coverage of correctness properties (additivity, structural contracts), but no CI automation running them on every push |
| Deployment/Infra | 3 | Works for a single-developer demo on a free tier; no CI/CD, no Docker, no rollback strategy, no environment separation (dev/staging/prod), no monitoring — appropriate *for its stated scope*, not for a multi-user production system |
| Security | 5 | Not because it's insecure, but because it's untested against any adversarial input (no fuzzing, no explicit threat model document) — its low attack surface is more "by absence" than "by design," even though the absence itself is a reasonable outcome here |
| Documentation | 8 | A genuinely good root README + `ml/README.md` + design-spec docs; this very document fills the remaining "why," not just "what," gap |

**Overall**: excellent as a capstone/portfolio artifact demonstrating ML engineering depth; explicitly not intended as, and not ready as, a multi-tenant production service without the improvements in [Section 26](#26-possible-improvements) — and that gap is honestly documented rather than hidden, which is itself a strength to highlight in interviews.

---

## 33. Project Strengths

- **Goes meaningfully beyond "train a model and report accuracy"** — ships uncertainty quantification and explainability as first-class, tested features, not afterthoughts.
- **Zero logic duplication between the tested pipeline and the deployed app** — a discipline many production ML systems fail at.
- **Clean, extensible model registry** — adding a new candidate model is a one-entry change, not a refactor.
- **Real automated test coverage (44 tests) of genuinely non-trivial correctness properties** (SHAP additivity, confidence-interval bounds, artifact structural contracts) rather than trivial smoke tests only.
- **Honest, deliberate scope discipline** — Docker/CI/CD/backend-API were consciously descoped with a stated reason, not silently skipped.
- **A polished, custom-designed UI** that doesn't look like a default Streamlit demo, showing product/design sensibility alongside ML skill.
- **Reproducible-by-construction**: fixed random seeds everywhere, a script-generated (not hand-edited) EDA notebook, and a fully scripted pipeline from raw CSV to deployed artifact.

---

## 34. Weaknesses

Stated honestly, as the instructions require:

- **`shap_global.json` is computed by the pipeline but never consumed by `streamlit_app.py`** — dead output, or an unfinished feature (the original design envisioned an Analytics/Model Performance dashboard page that was never built in the Streamlit MVP).
- **`compute_confidence` (the 0–100 score) is implemented and tested but unused by the live app**, which uses `compute_confidence_interval` instead — leftover from an earlier design iteration, not cleaned up.
- **No automated tests exercise the Streamlit UI itself** — all 44 tests cover the `ml/` pipeline; the presentation layer (`ui/*`, `streamlit_app.py`) has no test coverage at all.
- **No CI/CD** — tests are run manually; nothing prevents a broken commit from being pushed and auto-deployed by Streamlit Cloud.
- **No monitoring or logging of live predictions** — there is no way to detect model drift, unusual input patterns, or app errors in production without manually checking the Streamlit Cloud dashboard.
- **Row-level validation isn't vectorized** — fine at ~4,500 rows, a real bottleneck at much larger scale.
- **No error handling for a missing/corrupt `best_model.joblib`** on the deployed server — would surface as a raw traceback to end users.
- **No experiment-tracking history** — `model_comparison.json` only ever holds the *latest* training run's results; there's no record of how metrics changed across retraining iterations over time.

---

## 35. Final Placement Cheat Sheet

**30-second explanation**
> "I built a Coal GCV Predictor — an ML app that predicts coal's energy content from four quick lab measurements instead of a slow calorimeter test, with a 90% confidence interval and a SHAP explanation for every prediction, deployed live as a Streamlit app."

**1-minute explanation**
> See [Section 2](#2-one-minute-elevator-pitch) verbatim.

**3-minute explanation**
> Start with the 1-minute pitch, then add: "The pipeline is modular — a `ModelSpec` registry lets me train and tune 10 different regression models (Linear through Extra Trees and XGBoost/CatBoost/LightGBM) through one generic loop using RandomizedSearchCV, and automatically picks the winner by test RMSE — Extra Trees won with R²=0.99. Rather than stopping at a bare prediction, I quantify uncertainty per-model-family — ensemble disagreement for Random Forest/Extra Trees, leaf-local residuals for a single tree, k-nearest-neighbor residuals for everything else — combine that with the model's measured test error, and automatically widen the interval if someone enters an out-of-distribution input. I also compute exact SHAP attributions using TreeExplainer/LinearExplainer and turn them into both a waterfall chart and a plain-English sentence. Everything is backed by 44 pytest tests, including a test that verifies the SHAP values mathematically sum to the prediction. And critically, the Streamlit app imports the exact same tested modules the pipeline uses — there's no duplicated, untested copy of the model logic in the UI layer."

**Architecture**: single-process Streamlit app; offline `ml/` pipeline produces a joblib artifact; no backend API, no database, no Docker/CI-CD (explicitly descoped).

**APIs**: none (no REST layer) — direct in-process Python function calls (`compute_confidence_interval`, `explain_single_prediction`).

**Database**: none — flat files (`CSV`/`JSON`/`joblib`); artifact bundle's 11-key dict is the closest thing to a schema.

**Most difficult feature**: the confidence-interval engine's three model-family-specific spread computations, unified behind one public function.

**Biggest optimization**: precomputing everything expensive (training/tuning/residuals) offline so live inference is just cheap vector math + Streamlit's own resource/data caching.

**Biggest challenge**: giving an honest, non-overconfident uncertainty estimate across structurally different model types without an if/elif mess, and without duplicating logic between the tested pipeline and the deployed app.

**Security**: minimal attack surface by design — no auth, no DB, no user-controlled free text, no runtime external API calls; the honest answer is "there's very little here to attack," not "I added security hardening."

**Performance**: `st.cache_resource`/`st.cache_data`, RandomizedSearchCV instead of exhaustive grid search, exact (not approximate) SHAP explainers.

**Deployment**: Streamlit Community Cloud, auto-deploy from GitHub `master`, exact-pinned deployment dependencies for joblib compatibility; no Docker/CI-CD (deliberate scope decision).

**Tech stack**: Python 3.11 · pandas · NumPy · scikit-learn · XGBoost/CatBoost/LightGBM · Pydantic · SHAP · joblib · Streamlit · Plotly · pytest.

**Important algorithms**: RandomizedSearchCV tuning, Extra Trees ensemble, IQR outlier detection, Tree/Linear SHAP, quadrature-combined confidence interval, extrapolation penalty.

**Numbers/metrics to remember**: 4,540 raw rows → 4,513 clean rows (27 duplicates removed, 0 validation failures); 10 candidate models; winner = ExtraTreesRegressor; test RMSE = 0.6425 MJ/kg (≈ ±153 kcal/kg); R² = 0.9904; 44 tests across 12 modules, 100% passing; Z₉₀ = 1.645.

**Resume talking points**: automated 10-model comparison + selection; statistically-grounded per-prediction confidence intervals; exact SHAP explainability with a plain-language summary; 44-test suite including mathematical correctness checks (SHAP additivity); zero logic duplication between tested pipeline and deployed app.

**Possible cross-questions**: *"Why not build a real backend API?"* / *"How would this scale to millions of rows?"* / *"Why is `shap_global.json` unused?"* / *"What happens if the model file is missing?"* — all answered candidly in [Sections 25](#25-trade-offs), [26](#26-possible-improvements), and [34](#34-weaknesses).

**Best answers**: always pair a limitation with the reason it was an acceptable trade-off *for this project's actual scope* (a capstone demo, not a multi-tenant production system) — interviewers respond far better to "I chose not to build X because Y, and here's how I'd add it if the scope required it" than to either overclaiming completeness or apologizing for a student project not being enterprise infrastructure.

**Things NOT to say**:
- ❌ "It has a full backend and frontend" — it doesn't; be precise that Streamlit *is* both.
- ❌ "It's production-ready" — it's a well-engineered capstone, not a hardened multi-tenant service; say so directly if asked.
- ❌ "I used [React/Node/MongoDB/Docker/CI-CD]" — none of these are in the codebase; don't claim technologies that aren't there.
- ❌ "The confidence score and confidence interval are the same thing" — they're two different (one now-unused) implementations; know the distinction if probed.
- ❌ Overstating security work — the honest, better answer is "the attack surface is minimal by design," not a fabricated list of security features.
