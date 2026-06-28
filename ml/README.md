# ML Pipeline — Coal GCV Prediction

Trains, evaluates, and explains a regression model that predicts Coal Gross
Calorific Value (GCV) from four proximate-analysis inputs (Moisture,
Volatile Matter, Fixed Carbon, Ash).

## Setup

```bash
py -3.11 -m venv ../.venv
../.venv/Scripts/python.exe -m pip install -r requirements.txt
```

## Run the full pipeline

```bash
../.venv/Scripts/python.exe run_pipeline.py
```

This runs, in order:
1. `run_clean.py` — validates and cleans `data/raw/coal_all.csv`, writes
   `data/processed/coal_clean.csv` and `data/processed/data_quality_report.json`.
2. `run_train.py` — trains and tunes up to 10 regression models, selects the
   best by test RMSE, writes `models/best_model.joblib`,
   `models/model_metadata.json`, `models/model_comparison.json`.
3. `run_explain.py` — computes SHAP feature importance and a summary-plot
   sample, writes `models/shap_global.json`.

## Run tests

```bash
../.venv/Scripts/python.exe -m pytest
```

## Project layout

- `src/config.py` — paths and constants shared across the pipeline.
- `src/data/` — schema validation (`schema.py`) and cleaning (`clean.py`).
- `src/models/` — model registry (`registry.py`), training/tuning
  (`train.py`), confidence scoring (`confidence.py`).
- `src/explain/` — SHAP explainability (`shap_explain.py`).
- `notebooks/` — exploratory data analysis.
- `tests/` — pytest suite.
