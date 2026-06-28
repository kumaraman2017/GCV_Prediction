# ML Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a tested, end-to-end ML pipeline that cleans the coal dataset, trains/tunes/compares regression models, selects the best one, computes a per-prediction confidence score, and produces SHAP explainability artifacts — all persisted to disk for the backend API (a later sub-project) to load.

**Architecture:** A `ml/` package with `src/data` (validation + cleaning), `src/models` (registry, training, confidence scoring), and `src/explain` (SHAP) modules, each with pure, independently-testable functions. Three thin orchestration scripts (`run_clean.py`, `run_train.py`, `run_explain.py`) run those modules against the real dataset and persist artifacts; `run_pipeline.py` chains all three.

**Tech Stack:** Python 3.11, pandas, NumPy, scikit-learn, Pydantic v2, SHAP, joblib, XGBoost, CatBoost, LightGBM, pytest, nbformat/nbconvert.

## Global Constraints

- Python version: **3.11**, via `py -3.11 -m venv .venv` at the project root — chosen over the system default 3.14 because shap/xgboost/catboost/lightgbm have reliable prebuilt Windows wheels for 3.11 and may not for 3.14.
- Random seed: `RANDOM_SEED = 42`, used everywhere a seed is accepted (splits, model `random_state`, search `random_state`), for reproducibility.
- Feature columns, in this exact order everywhere they appear: `["Moisture", "Volatile_matter", "Fixed_Carbon", "Std.Ash"]`. Target column: `"GCV"`.
- Proximate-sum validation tolerance: rows are valid only if `Moisture+Volatile_matter+Fixed_Carbon+Std.Ash` is within ±0.5 of 100.
- Train/test split: 80/20 (`TEST_SIZE = 0.2`), stratified by GCV quartile.
- All commands below assume cwd = project root (`gcv_coal_project/`) with the venv created in Task 1, and are written as `cd ml && ../.venv/Scripts/python.exe ...`.
- Tasks 3, 6, and 9 run a script against the full real dataset (data load, full hyperparameter search across up to 10 models, or SHAP over a 200-row sample) and are slow (seconds to a few minutes). For these: write the script, run it once against real data, then write a fast verification test against the artifacts it leaves on disk. Full red/green TDD against the real dataset isn't practical when a step takes minutes — this is an intentional, scoped deviation from strict TDD ordering, not a license to skip testing.
- Out of scope for this plan: FastAPI serving, frontend rendering, Docker/CI/CD — separate sub-projects (DevOps is explicitly descoped for this project per the owner's decision: keep it simple and explainable, not a DevOps showcase).

---

### Task 1: ML Environment Setup & Config Module

**Files:**
- Create: `ml/requirements.txt`
- Create: `ml/pytest.ini`
- Create: `ml/src/__init__.py`
- Create: `ml/src/config.py`
- Test: `ml/tests/test_config.py`

**Interfaces:**
- Produces: module-level constants `RAW_DATA_PATH`, `PROCESSED_DATA_DIR`, `CLEAN_DATA_PATH`, `DATA_QUALITY_REPORT_PATH`, `MODELS_DIR`, `BEST_MODEL_PATH`, `MODEL_METADATA_PATH`, `MODEL_COMPARISON_PATH`, `SHAP_GLOBAL_PATH` (all `pathlib.Path`), `FEATURE_COLUMNS: list[str]`, `TARGET_COLUMN: str`, `RANDOM_SEED: int`, `TEST_SIZE: float`, `PROXIMATE_SUM_TOLERANCE: float` — consumed by every later task.

- [ ] **Step 1: Create `ml/requirements.txt`**

```text
pandas>=2.2
numpy>=1.26
scikit-learn>=1.4
pydantic>=2.6
shap>=0.45
joblib>=1.3
xgboost>=2.0
catboost>=1.2
lightgbm>=4.3
pytest>=8.0
nbformat>=5.10
nbconvert>=7.16
jupyter>=1.0
ipykernel>=6.29
matplotlib>=3.8
seaborn>=0.13
```

- [ ] **Step 2: Create the virtual environment and install dependencies**

Run (from project root):
```bash
py -3.11 -m venv .venv
./.venv/Scripts/python.exe -m pip install --upgrade pip
./.venv/Scripts/python.exe -m pip install -r ml/requirements.txt
```
Expected: completes with no errors (this can take a few minutes — catboost/xgboost/lightgbm are large packages). If `xgboost`, `catboost`, or `lightgbm` fail to build, that's tolerable (Task 4 handles their absence gracefully) — but `pandas`, `numpy`, `scikit-learn`, `pydantic`, `shap`, `joblib`, and `pytest` installing successfully is required to proceed.

- [ ] **Step 3: Create `ml/pytest.ini`**

```ini
[pytest]
pythonpath = .
testpaths = tests
```

- [ ] **Step 4: Write the failing test**

`ml/tests/test_config.py`:
```python
from src.config import (
    CLEAN_DATA_PATH,
    DATA_QUALITY_REPORT_PATH,
    FEATURE_COLUMNS,
    MODELS_DIR,
    RAW_DATA_PATH,
    TARGET_COLUMN,
)


def test_raw_data_path_points_to_existing_file():
    assert RAW_DATA_PATH.exists()
    assert RAW_DATA_PATH.name == "coal_all.csv"


def test_feature_and_target_columns_are_defined():
    assert FEATURE_COLUMNS == ["Moisture", "Volatile_matter", "Fixed_Carbon", "Std.Ash"]
    assert TARGET_COLUMN == "GCV"


def test_output_paths_are_under_expected_directories():
    assert CLEAN_DATA_PATH.parent.name == "processed"
    assert DATA_QUALITY_REPORT_PATH.parent.name == "processed"
    assert MODELS_DIR.name == "models"
```

- [ ] **Step 5: Run test, verify it fails**

Run: `cd ml && ../.venv/Scripts/python.exe -m pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src'`

- [ ] **Step 6: Implement `ml/src/__init__.py` and `ml/src/config.py`**

`ml/src/__init__.py`: empty file.

`ml/src/config.py`:
```python
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "coal_all.csv"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
CLEAN_DATA_PATH = PROCESSED_DATA_DIR / "coal_clean.csv"
DATA_QUALITY_REPORT_PATH = PROCESSED_DATA_DIR / "data_quality_report.json"

MODELS_DIR = PROJECT_ROOT / "models"
BEST_MODEL_PATH = MODELS_DIR / "best_model.joblib"
MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.json"
MODEL_COMPARISON_PATH = MODELS_DIR / "model_comparison.json"
SHAP_GLOBAL_PATH = MODELS_DIR / "shap_global.json"

FEATURE_COLUMNS = ["Moisture", "Volatile_matter", "Fixed_Carbon", "Std.Ash"]
TARGET_COLUMN = "GCV"
RANDOM_SEED = 42
TEST_SIZE = 0.2
PROXIMATE_SUM_TOLERANCE = 0.5
```

- [ ] **Step 7: Run test, verify it passes**

Run: `cd ml && ../.venv/Scripts/python.exe -m pytest tests/test_config.py -v`
Expected: 3 passed

- [ ] **Step 8: Commit**

```bash
git add ml/requirements.txt ml/pytest.ini ml/src/__init__.py ml/src/config.py ml/tests/test_config.py
git commit -m "feat(ml): add project config and environment setup"
```

---

### Task 2: Data Schema Validation

**Files:**
- Create: `ml/src/data/__init__.py`
- Create: `ml/src/data/schema.py`
- Test: `ml/tests/test_schema.py`

**Interfaces:**
- Consumes: `src.config.PROXIMATE_SUM_TOLERANCE` (Task 1)
- Produces: `CoalRecord` (Pydantic model), `validate_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]` — consumed by Task 3's `clean_pipeline`.

- [ ] **Step 1: Write the failing tests**

`ml/tests/test_schema.py`:
```python
import pandas as pd
import pytest
from pydantic import ValidationError

from src.data.schema import CoalRecord, validate_dataframe


def test_valid_record_passes():
    record = CoalRecord.model_validate({
        "Moisture": 5.20, "Volatile_matter": 31.10, "Fixed_Carbon": 40.70,
        "Std.Ash": 23.00, "GCV": 22.84477109,
    })
    assert record.moisture == 5.20


def test_negative_value_rejected():
    with pytest.raises(ValidationError):
        CoalRecord.model_validate({
            "Moisture": -1.0, "Volatile_matter": 31.10, "Fixed_Carbon": 40.70,
            "Std.Ash": 30.90, "GCV": 22.84477109,
        })


def test_value_over_100_rejected():
    with pytest.raises(ValidationError):
        CoalRecord.model_validate({
            "Moisture": 5.20, "Volatile_matter": 31.10, "Fixed_Carbon": 40.70,
            "Std.Ash": 123.00, "GCV": 22.84477109,
        })


def test_proximate_sum_must_be_close_to_100():
    with pytest.raises(ValidationError):
        CoalRecord.model_validate({
            "Moisture": 5.20, "Volatile_matter": 31.10, "Fixed_Carbon": 40.70,
            "Std.Ash": 10.00, "GCV": 22.84477109,
        })


def test_validate_dataframe_splits_valid_and_invalid_rows():
    df = pd.DataFrame([
        {"Moisture": 5.20, "Volatile_matter": 31.10, "Fixed_Carbon": 40.70, "Std.Ash": 23.00, "GCV": 22.84477109},
        {"Moisture": -1.0, "Volatile_matter": 31.10, "Fixed_Carbon": 40.70, "Std.Ash": 30.90, "GCV": 22.84477109},
        {"Moisture": 10.70, "Volatile_matter": 28.00, "Fixed_Carbon": 30.00, "Std.Ash": 31.30, "GCV": 15.49101361},
    ])
    valid_df, errors = validate_dataframe(df)
    assert len(valid_df) == 2
    assert len(errors) == 1
    assert errors[0]["index"] == 1
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `cd ml && ../.venv/Scripts/python.exe -m pytest tests/test_schema.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.data'`

- [ ] **Step 3: Implement the schema**

`ml/src/data/__init__.py`: empty file.

`ml/src/data/schema.py`:
```python
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from src.config import PROXIMATE_SUM_TOLERANCE


class CoalRecord(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    moisture: float = Field(alias="Moisture")
    volatile_matter: float = Field(alias="Volatile_matter")
    fixed_carbon: float = Field(alias="Fixed_Carbon")
    std_ash: float = Field(alias="Std.Ash")
    gcv: float = Field(alias="GCV")

    @field_validator("moisture", "volatile_matter", "fixed_carbon", "std_ash", "gcv")
    @classmethod
    def must_be_non_negative(cls, value: float, info) -> float:
        if value < 0:
            raise ValueError(f"{info.field_name} must be non-negative, got {value}")
        return value

    @field_validator("moisture", "volatile_matter", "fixed_carbon", "std_ash")
    @classmethod
    def must_be_at_most_100(cls, value: float, info) -> float:
        if value > 100:
            raise ValueError(f"{info.field_name} must be <= 100, got {value}")
        return value

    @model_validator(mode="after")
    def proximate_components_sum_to_100(self) -> "CoalRecord":
        total = self.moisture + self.volatile_matter + self.fixed_carbon + self.std_ash
        if abs(total - 100) > PROXIMATE_SUM_TOLERANCE:
            raise ValueError(
                f"Moisture+Volatile_matter+Fixed_Carbon+Std.Ash = {total:.2f}, expected ~100"
            )
        return self


def validate_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    valid_indices = []
    errors = []
    for idx, row in df.iterrows():
        try:
            CoalRecord.model_validate(row.to_dict())
            valid_indices.append(idx)
        except ValidationError as exc:
            errors.append({"index": int(idx), "errors": exc.errors()})
    return df.loc[valid_indices].reset_index(drop=True), errors
```

- [ ] **Step 4: Run tests, verify they pass**

Run: `cd ml && ../.venv/Scripts/python.exe -m pytest tests/test_schema.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add ml/src/data/__init__.py ml/src/data/schema.py ml/tests/test_schema.py
git commit -m "feat(ml): add Pydantic schema validation for coal records"
```

---

### Task 3: Data Cleaning Pipeline

**Files:**
- Create: `ml/src/data/clean.py`
- Create: `ml/run_clean.py`
- Test: `ml/tests/test_clean.py`
- Test: `ml/tests/test_run_clean.py`

**Interfaces:**
- Consumes: `src.config.FEATURE_COLUMNS`, `src.config.TARGET_COLUMN`, `src.config.RAW_DATA_PATH`, `src.config.CLEAN_DATA_PATH`, `src.config.DATA_QUALITY_REPORT_PATH`, `src.config.PROCESSED_DATA_DIR` (Task 1); `validate_dataframe` (Task 2)
- Produces: `clean_pipeline(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]`; on-disk `data/processed/coal_clean.csv` (5 columns: `Moisture, Volatile_matter, Fixed_Carbon, Std.Ash, GCV`) and `data/processed/data_quality_report.json` — consumed by Task 6, 9, 10.

- [ ] **Step 1: Write the failing unit tests**

`ml/tests/test_clean.py`:
```python
import pandas as pd

from src.data.clean import (
    clean_pipeline,
    detect_outliers_iqr,
    drop_duplicate_rows,
    impute_missing_values,
    select_modeling_columns,
)


def _valid_row(overrides=None):
    row = {"Moisture": 5.20, "Volatile_matter": 31.10, "Fixed_Carbon": 40.70, "Std.Ash": 23.00, "GCV": 22.84477109}
    if overrides:
        row.update(overrides)
    return row


def test_drop_duplicate_rows_removes_exact_duplicates():
    df = pd.DataFrame([_valid_row(), _valid_row(), _valid_row({"GCV": 15.0})])
    deduped, n_removed = drop_duplicate_rows(df)
    assert len(deduped) == 2
    assert n_removed == 1


def test_select_modeling_columns_drops_ultimate_analysis_columns():
    df = pd.DataFrame([{**_valid_row(), "Hydrogen": 4.9, "Carbon": 57.9}])
    modeling_df = select_modeling_columns(df)
    assert list(modeling_df.columns) == ["Moisture", "Volatile_matter", "Fixed_Carbon", "Std.Ash", "GCV"]


def test_impute_missing_values_fills_with_median():
    df = pd.DataFrame([_valid_row(), _valid_row({"Moisture": None}), _valid_row({"Moisture": 7.0})])
    imputed_df, counts = impute_missing_values(df)
    assert counts["Moisture"] == 1
    assert imputed_df["Moisture"].isnull().sum() == 0


def test_detect_outliers_iqr_flags_extreme_value():
    rows = [_valid_row({"GCV": 22.0 + i * 0.1}) for i in range(20)]
    rows.append(_valid_row({"GCV": 100.0}))
    df = pd.DataFrame(rows)
    report = detect_outliers_iqr(df)
    assert report["GCV"]["count"] >= 1


def test_clean_pipeline_removes_duplicates_and_invalid_rows():
    rows = [_valid_row(), _valid_row(), _valid_row({"Moisture": -1.0, "Std.Ash": 30.90})]
    df = pd.DataFrame(rows)
    clean_df, report = clean_pipeline(df)
    assert report["duplicates_removed"] == 1
    assert report["rows_failed_validation"] == 1
    assert len(clean_df) == 1
    assert list(clean_df.columns) == ["Moisture", "Volatile_matter", "Fixed_Carbon", "Std.Ash", "GCV"]
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `cd ml && ../.venv/Scripts/python.exe -m pytest tests/test_clean.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.data.clean'`

- [ ] **Step 3: Implement `ml/src/data/clean.py`**

```python
import pandas as pd

from src.config import FEATURE_COLUMNS, TARGET_COLUMN
from src.data.schema import validate_dataframe

MODELING_COLUMNS = FEATURE_COLUMNS + [TARGET_COLUMN]


def drop_duplicate_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    deduped = df.drop_duplicates().reset_index(drop=True)
    return deduped, len(df) - len(deduped)


def select_modeling_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df[MODELING_COLUMNS].copy()


def impute_missing_values(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = df.copy()
    imputed_counts = {}
    for column in MODELING_COLUMNS:
        missing = int(df[column].isnull().sum())
        imputed_counts[column] = missing
        if missing > 0:
            df[column] = df[column].fillna(df[column].median())
    return df, imputed_counts


def detect_outliers_iqr(df: pd.DataFrame) -> dict:
    report = {}
    for column in MODELING_COLUMNS:
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        count = int(((df[column] < lower_bound) | (df[column] > upper_bound)).sum())
        report[column] = {
            "lower_bound": float(lower_bound),
            "upper_bound": float(upper_bound),
            "count": count,
        }
    return report


def clean_pipeline(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    valid_df, validation_errors = validate_dataframe(raw_df)
    deduped_df, n_duplicates = drop_duplicate_rows(valid_df)
    modeling_df = select_modeling_columns(deduped_df)
    imputed_df, imputed_counts = impute_missing_values(modeling_df)
    outliers = detect_outliers_iqr(imputed_df)

    report = {
        "raw_row_count": int(len(raw_df)),
        "rows_failed_validation": len(validation_errors),
        "validation_errors": validation_errors,
        "duplicates_removed": n_duplicates,
        "rows_after_cleaning": int(len(imputed_df)),
        "missing_values_imputed": imputed_counts,
        "outliers": outliers,
    }
    return imputed_df, report
```

- [ ] **Step 4: Run tests, verify they pass**

Run: `cd ml && ../.venv/Scripts/python.exe -m pytest tests/test_clean.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit the cleaning logic**

```bash
git add ml/src/data/clean.py ml/tests/test_clean.py
git commit -m "feat(ml): add data cleaning pipeline (dedup, impute, outlier report)"
```

- [ ] **Step 6: Implement `ml/run_clean.py` and run it against the real dataset**

```python
import json

import pandas as pd

from src.config import CLEAN_DATA_PATH, DATA_QUALITY_REPORT_PATH, PROCESSED_DATA_DIR, RAW_DATA_PATH
from src.data.clean import clean_pipeline


def main() -> None:
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    raw_df = pd.read_csv(RAW_DATA_PATH)
    clean_df, report = clean_pipeline(raw_df)
    clean_df.to_csv(CLEAN_DATA_PATH, index=False)
    DATA_QUALITY_REPORT_PATH.write_text(json.dumps(report, indent=2))
    print(f"Cleaned {report['raw_row_count']} rows -> {report['rows_after_cleaning']} rows")
    print(f"Wrote {CLEAN_DATA_PATH} and {DATA_QUALITY_REPORT_PATH}")


if __name__ == "__main__":
    main()
```

Run: `cd ml && ../.venv/Scripts/python.exe run_clean.py`
Expected output: `Cleaned 4540 rows -> 4513 rows`

- [ ] **Step 7: Write and run the verification test**

`ml/tests/test_run_clean.py`:
```python
import pandas as pd

from src.config import CLEAN_DATA_PATH, DATA_QUALITY_REPORT_PATH


def test_run_clean_produces_expected_outputs():
    assert CLEAN_DATA_PATH.exists()
    assert DATA_QUALITY_REPORT_PATH.exists()

    clean_df = pd.read_csv(CLEAN_DATA_PATH)
    assert len(clean_df) == 4513
    assert list(clean_df.columns) == ["Moisture", "Volatile_matter", "Fixed_Carbon", "Std.Ash", "GCV"]
```

Run: `cd ml && ../.venv/Scripts/python.exe -m pytest tests/test_run_clean.py -v`
Expected: 1 passed

- [ ] **Step 8: Commit**

```bash
git add ml/run_clean.py ml/tests/test_run_clean.py data/processed/coal_clean.csv data/processed/data_quality_report.json
git commit -m "feat(ml): run cleaning pipeline on real dataset, persist outputs"
```

---

### Task 4: Model Registry

**Files:**
- Create: `ml/src/models/__init__.py`
- Create: `ml/src/models/registry.py`
- Test: `ml/tests/test_registry.py`

**Interfaces:**
- Consumes: `src.config.RANDOM_SEED` (Task 1)
- Produces: `ModelSpec` dataclass (`name: str`, `estimator_factory: Callable[[], Any]`, `param_distributions: dict`, `needs_scaling: bool`, `is_tree_based: bool`, `confidence_method: str` — one of `"bagging_ensemble" | "single_tree" | "knn_residual"`), `build_registry(random_seed: int) -> list[ModelSpec]`, module flags `HAS_XGBOOST`, `HAS_CATBOOST`, `HAS_LIGHTGBM` — consumed by Task 5, 6, 9.

- [ ] **Step 1: Write the failing tests**

`ml/tests/test_registry.py`:
```python
from src.models.registry import build_registry


def test_registry_contains_core_models():
    registry = build_registry(random_seed=42)
    names = {spec.name for spec in registry}
    core_models = {
        "linear_regression", "ridge", "lasso", "decision_tree",
        "random_forest", "extra_trees", "gradient_boosting",
    }
    assert core_models.issubset(names)


def test_registry_assigns_correct_confidence_method():
    registry = build_registry(random_seed=42)
    by_name = {spec.name: spec for spec in registry}
    assert by_name["random_forest"].confidence_method == "bagging_ensemble"
    assert by_name["extra_trees"].confidence_method == "bagging_ensemble"
    assert by_name["decision_tree"].confidence_method == "single_tree"
    assert by_name["linear_regression"].confidence_method == "knn_residual"
    assert by_name["gradient_boosting"].confidence_method == "knn_residual"


def test_registry_estimator_factories_produce_unfitted_estimators():
    registry = build_registry(random_seed=42)
    for spec in registry:
        estimator = spec.estimator_factory()
        assert hasattr(estimator, "fit")
        assert hasattr(estimator, "predict")


def test_optional_models_included_only_if_installed():
    import src.models.registry as registry_module
    registry = build_registry(random_seed=42)
    names = {spec.name for spec in registry}
    assert ("xgboost" in names) == registry_module.HAS_XGBOOST
    assert ("catboost" in names) == registry_module.HAS_CATBOOST
    assert ("lightgbm" in names) == registry_module.HAS_LIGHTGBM
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `cd ml && ../.venv/Scripts/python.exe -m pytest tests/test_registry.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.models'`

- [ ] **Step 3: Implement `ml/src/models/registry.py`**

`ml/src/models/__init__.py`: empty file.

`ml/src/models/registry.py`:
```python
from dataclasses import dataclass
from typing import Any, Callable

from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor

try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    from catboost import CatBoostRegressor
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

try:
    from lightgbm import LGBMRegressor
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False


@dataclass
class ModelSpec:
    name: str
    estimator_factory: Callable[[], Any]
    param_distributions: dict
    needs_scaling: bool
    is_tree_based: bool
    confidence_method: str


def build_registry(random_seed: int) -> list[ModelSpec]:
    registry = [
        ModelSpec(
            name="linear_regression",
            estimator_factory=lambda: LinearRegression(),
            param_distributions={},
            needs_scaling=True,
            is_tree_based=False,
            confidence_method="knn_residual",
        ),
        ModelSpec(
            name="ridge",
            estimator_factory=lambda: Ridge(random_state=random_seed),
            param_distributions={"alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
            needs_scaling=True,
            is_tree_based=False,
            confidence_method="knn_residual",
        ),
        ModelSpec(
            name="lasso",
            estimator_factory=lambda: Lasso(random_state=random_seed),
            param_distributions={"alpha": [0.001, 0.01, 0.1, 1.0, 10.0]},
            needs_scaling=True,
            is_tree_based=False,
            confidence_method="knn_residual",
        ),
        ModelSpec(
            name="decision_tree",
            estimator_factory=lambda: DecisionTreeRegressor(random_state=random_seed),
            param_distributions={"max_depth": [3, 5, 8, 12, None], "min_samples_leaf": [1, 2, 5, 10]},
            needs_scaling=False,
            is_tree_based=True,
            confidence_method="single_tree",
        ),
        ModelSpec(
            name="random_forest",
            estimator_factory=lambda: RandomForestRegressor(random_state=random_seed),
            param_distributions={
                "n_estimators": [100, 200, 300],
                "max_depth": [5, 10, 15, None],
                "min_samples_leaf": [1, 2, 5],
            },
            needs_scaling=False,
            is_tree_based=True,
            confidence_method="bagging_ensemble",
        ),
        ModelSpec(
            name="extra_trees",
            estimator_factory=lambda: ExtraTreesRegressor(random_state=random_seed),
            param_distributions={
                "n_estimators": [100, 200, 300],
                "max_depth": [5, 10, 15, None],
                "min_samples_leaf": [1, 2, 5],
            },
            needs_scaling=False,
            is_tree_based=True,
            confidence_method="bagging_ensemble",
        ),
        ModelSpec(
            name="gradient_boosting",
            estimator_factory=lambda: GradientBoostingRegressor(random_state=random_seed),
            param_distributions={
                "n_estimators": [100, 200],
                "max_depth": [2, 3, 4],
                "learning_rate": [0.01, 0.05, 0.1],
            },
            needs_scaling=False,
            is_tree_based=True,
            confidence_method="knn_residual",
        ),
    ]

    if HAS_XGBOOST:
        registry.append(
            ModelSpec(
                name="xgboost",
                estimator_factory=lambda: XGBRegressor(random_state=random_seed, verbosity=0),
                param_distributions={
                    "n_estimators": [100, 200],
                    "max_depth": [3, 4, 6],
                    "learning_rate": [0.01, 0.05, 0.1],
                },
                needs_scaling=False,
                is_tree_based=True,
                confidence_method="knn_residual",
            )
        )

    if HAS_CATBOOST:
        registry.append(
            ModelSpec(
                name="catboost",
                estimator_factory=lambda: CatBoostRegressor(random_state=random_seed, verbose=False),
                param_distributions={
                    "iterations": [200, 400],
                    "depth": [4, 6, 8],
                    "learning_rate": [0.01, 0.05, 0.1],
                },
                needs_scaling=False,
                is_tree_based=True,
                confidence_method="knn_residual",
            )
        )

    if HAS_LIGHTGBM:
        registry.append(
            ModelSpec(
                name="lightgbm",
                estimator_factory=lambda: LGBMRegressor(random_state=random_seed, verbosity=-1),
                param_distributions={
                    "n_estimators": [100, 200],
                    "max_depth": [3, 5, -1],
                    "learning_rate": [0.01, 0.05, 0.1],
                },
                needs_scaling=False,
                is_tree_based=True,
                confidence_method="knn_residual",
            )
        )

    return registry
```

- [ ] **Step 4: Run tests, verify they pass**

Run: `cd ml && ../.venv/Scripts/python.exe -m pytest tests/test_registry.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add ml/src/models/__init__.py ml/src/models/registry.py ml/tests/test_registry.py
git commit -m "feat(ml): add model registry with confidence-method classification"
```

---

### Task 5: Model Training & Tuning Logic

**Files:**
- Create: `ml/src/models/train.py`
- Test: `ml/tests/test_train.py`

**Interfaces:**
- Consumes: `ModelSpec` (Task 4)
- Produces: `TrainedModel` dataclass (`spec: ModelSpec`, `estimator: Any`, `scaler: StandardScaler | None`, `metrics: dict` with keys `rmse, mae, r2`), `train_and_tune(spec, X_train, y_train, X_test, y_test, random_seed) -> TrainedModel`, `select_best(trained_models: list[TrainedModel]) -> TrainedModel` — consumed by Task 6.

- [ ] **Step 1: Write the failing tests**

`ml/tests/test_train.py`:
```python
import numpy as np

from src.models.registry import ModelSpec, build_registry
from src.models.train import TrainedModel, select_best, train_and_tune


def _synthetic_data(n=200, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.uniform(0, 50, size=(n, 4))
    y = X[:, 0] * 0.3 + X[:, 1] * 0.5 - X[:, 2] * 0.2 + rng.normal(0, 1, size=n)
    return X[:150], y[:150], X[150:], y[150:]


def test_train_and_tune_decision_tree_produces_metrics():
    X_train, y_train, X_test, y_test = _synthetic_data()
    spec = next(s for s in build_registry(random_seed=42) if s.name == "decision_tree")
    trained = train_and_tune(spec, X_train, y_train, X_test, y_test, random_seed=42)
    assert isinstance(trained, TrainedModel)
    assert trained.metrics["rmse"] >= 0
    assert trained.metrics["mae"] >= 0
    assert trained.metrics["r2"] <= 1.0
    assert trained.scaler is None


def test_train_and_tune_applies_scaler_for_linear_models():
    X_train, y_train, X_test, y_test = _synthetic_data()
    spec = next(s for s in build_registry(random_seed=42) if s.name == "linear_regression")
    trained = train_and_tune(spec, X_train, y_train, X_test, y_test, random_seed=42)
    assert trained.scaler is not None
    assert trained.metrics["rmse"] >= 0


def test_select_best_picks_lowest_rmse():
    fake_spec = ModelSpec("fake", lambda: None, {}, False, False, "knn_residual")
    good = TrainedModel(spec=fake_spec, estimator=None, scaler=None, metrics={"rmse": 1.0, "mae": 1.0, "r2": 0.9})
    bad = TrainedModel(spec=fake_spec, estimator=None, scaler=None, metrics={"rmse": 5.0, "mae": 3.0, "r2": 0.5})
    assert select_best([good, bad]) is good
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `cd ml && ../.venv/Scripts/python.exe -m pytest tests/test_train.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.models.train'`

- [ ] **Step 3: Implement `ml/src/models/train.py`**

```python
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler

from src.models.registry import ModelSpec


@dataclass
class TrainedModel:
    spec: ModelSpec
    estimator: Any
    scaler: Optional[StandardScaler]
    metrics: dict


def train_and_tune(
    spec: ModelSpec,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    random_seed: int,
) -> TrainedModel:
    scaler = None
    X_train_used, X_test_used = X_train, X_test

    if spec.needs_scaling:
        scaler = StandardScaler().fit(X_train)
        X_train_used = scaler.transform(X_train)
        X_test_used = scaler.transform(X_test)

    if spec.param_distributions:
        search = RandomizedSearchCV(
            spec.estimator_factory(),
            param_distributions=spec.param_distributions,
            n_iter=10,
            cv=5,
            scoring="neg_root_mean_squared_error",
            random_state=random_seed,
            n_jobs=-1,
        )
        search.fit(X_train_used, y_train)
        estimator = search.best_estimator_
    else:
        estimator = spec.estimator_factory()
        estimator.fit(X_train_used, y_train)

    predictions = estimator.predict(X_test_used)
    metrics = {
        "rmse": float(root_mean_squared_error(y_test, predictions)),
        "mae": float(mean_absolute_error(y_test, predictions)),
        "r2": float(r2_score(y_test, predictions)),
    }
    return TrainedModel(spec=spec, estimator=estimator, scaler=scaler, metrics=metrics)


def select_best(trained_models: list[TrainedModel]) -> TrainedModel:
    return min(trained_models, key=lambda trained: trained.metrics["rmse"])
```

- [ ] **Step 4: Run tests, verify they pass**

Run: `cd ml && ../.venv/Scripts/python.exe -m pytest tests/test_train.py -v`
Expected: 3 passed (may take ~10-20s due to RandomizedSearchCV)

- [ ] **Step 5: Commit**

```bash
git add ml/src/models/train.py ml/tests/test_train.py
git commit -m "feat(ml): add model training, tuning, and selection logic"
```

---

### Task 6: Full Model Training on Real Data

**Files:**
- Create: `ml/run_train.py`
- Test: `ml/tests/test_run_train.py`

**Interfaces:**
- Consumes: `CLEAN_DATA_PATH`, `FEATURE_COLUMNS`, `TARGET_COLUMN`, `RANDOM_SEED`, `TEST_SIZE`, `BEST_MODEL_PATH`, `MODEL_METADATA_PATH`, `MODEL_COMPARISON_PATH`, `MODELS_DIR` (Task 1); `build_registry` (Task 4); `train_and_tune`, `select_best` (Task 5)
- Produces: on-disk `models/best_model.joblib` — a dict with keys `estimator, scaler, feature_columns, target_column, confidence_method, is_tree_based, X_train, y_train, residual_std, residuals_train, feature_ranges` (this exact key set is the contract Task 7 and Task 9 rely on); `models/model_metadata.json`; `models/model_comparison.json`.

- [ ] **Step 1: Implement `ml/run_train.py`**

```python
import json
from datetime import datetime, timezone

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    BEST_MODEL_PATH,
    CLEAN_DATA_PATH,
    FEATURE_COLUMNS,
    MODEL_COMPARISON_PATH,
    MODEL_METADATA_PATH,
    MODELS_DIR,
    RANDOM_SEED,
    TARGET_COLUMN,
    TEST_SIZE,
)
from src.models.registry import build_registry
from src.models.train import select_best, train_and_tune


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(CLEAN_DATA_PATH)

    X = df[FEATURE_COLUMNS].to_numpy()
    y = df[TARGET_COLUMN].to_numpy()
    strata = pd.qcut(df[TARGET_COLUMN], q=4, labels=False)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=strata
    )

    registry = build_registry(random_seed=RANDOM_SEED)
    trained_models = [
        train_and_tune(spec, X_train, y_train, X_test, y_test, random_seed=RANDOM_SEED)
        for spec in registry
    ]
    best = select_best(trained_models)

    X_train_for_best = X_train
    if best.scaler is not None:
        X_train_for_best = best.scaler.transform(X_train)
    residuals_train = y_train - best.estimator.predict(X_train_for_best)

    feature_ranges = {
        column: (float(X_train[:, i].min()), float(X_train[:, i].max()))
        for i, column in enumerate(FEATURE_COLUMNS)
    }

    artifact = {
        "estimator": best.estimator,
        "scaler": best.scaler,
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "confidence_method": best.spec.confidence_method,
        "is_tree_based": best.spec.is_tree_based,
        "X_train": X_train_for_best,
        "y_train": y_train,
        "residual_std": float(residuals_train.std()),
        "residuals_train": residuals_train,
        "feature_ranges": feature_ranges,
    }
    joblib.dump(artifact, BEST_MODEL_PATH)

    metadata = {
        "model_name": best.spec.name,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "metrics": best.metrics,
    }
    MODEL_METADATA_PATH.write_text(json.dumps(metadata, indent=2))

    comparison = {trained.spec.name: trained.metrics for trained in trained_models}
    MODEL_COMPARISON_PATH.write_text(json.dumps(comparison, indent=2))

    print(f"Best model: {best.spec.name} (RMSE={best.metrics['rmse']:.3f}, R2={best.metrics['r2']:.3f})")
    print(f"Saved {BEST_MODEL_PATH}, {MODEL_METADATA_PATH}, {MODEL_COMPARISON_PATH}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it against the real cleaned dataset**

Run: `cd ml && ../.venv/Scripts/python.exe run_train.py`
Expected: prints the winning model name and metrics; takes roughly 1-3 minutes (RandomizedSearchCV across up to 10 models). No errors.

- [ ] **Step 3: Write and run the verification test**

`ml/tests/test_run_train.py`:
```python
import json

import joblib

from src.config import BEST_MODEL_PATH, MODEL_COMPARISON_PATH, MODEL_METADATA_PATH


def test_run_train_persists_best_model_bundle():
    artifact = joblib.load(BEST_MODEL_PATH)
    required_keys = {
        "estimator", "scaler", "feature_columns", "target_column",
        "confidence_method", "is_tree_based", "X_train", "y_train",
        "residual_std", "residuals_train", "feature_ranges",
    }
    assert required_keys.issubset(artifact.keys())
    assert artifact["feature_columns"] == ["Moisture", "Volatile_matter", "Fixed_Carbon", "Std.Ash"]
    assert artifact["confidence_method"] in {"bagging_ensemble", "single_tree", "knn_residual"}


def test_run_train_persists_metadata_and_comparison():
    metadata = json.loads(MODEL_METADATA_PATH.read_text())
    assert "model_name" in metadata
    assert set(metadata["metrics"].keys()) == {"rmse", "mae", "r2"}

    comparison = json.loads(MODEL_COMPARISON_PATH.read_text())
    assert metadata["model_name"] in comparison
    assert len(comparison) >= 7
```

Run: `cd ml && ../.venv/Scripts/python.exe -m pytest tests/test_run_train.py -v`
Expected: 2 passed

- [ ] **Step 4: Commit**

```bash
git add ml/run_train.py ml/tests/test_run_train.py models/best_model.joblib models/model_metadata.json models/model_comparison.json
git commit -m "feat(ml): train, tune, and select best model on real dataset"
```

---

### Task 7: Confidence Score Module

**Files:**
- Create: `ml/src/models/confidence.py`
- Test: `ml/tests/test_confidence.py`

**Interfaces:**
- Consumes: the `best_model.joblib` artifact contract from Task 6 (keys: `feature_columns, scaler, confidence_method, estimator, X_train, residuals_train, residual_std, feature_ranges`)
- Produces: `compute_confidence(artifact: dict, raw_input: dict[str, float]) -> float` (returns a value in `[0, 100]`) — to be consumed by the backend sub-project later; exercised directly by this task's tests.

- [ ] **Step 1: Write the failing tests**

`ml/tests/test_confidence.py`:
```python
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from src.models.confidence import _extrapolation_penalty, compute_confidence

FEATURE_COLUMNS = ["Moisture", "Volatile_matter", "Fixed_Carbon", "Std.Ash"]


def _synthetic_training_data(n=100, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.uniform(5, 40, size=(n, 4))
    y = X[:, 0] * 0.3 + X[:, 1] * 0.5 - X[:, 2] * 0.2 + rng.normal(0, 0.5, size=n)
    return X, y


def _feature_ranges(X: np.ndarray) -> dict:
    return {
        column: (float(X[:, i].min()), float(X[:, i].max()))
        for i, column in enumerate(FEATURE_COLUMNS)
    }


def _sample_input(X: np.ndarray) -> dict:
    row = X[0]
    return {column: float(row[i]) for i, column in enumerate(FEATURE_COLUMNS)}


def _base_artifact(estimator, X, y, method):
    residuals = y - estimator.predict(X)
    return {
        "feature_columns": FEATURE_COLUMNS,
        "scaler": None,
        "confidence_method": method,
        "estimator": estimator,
        "X_train": X,
        "y_train": y,
        "residuals_train": residuals,
        "residual_std": float(residuals.std()),
        "feature_ranges": _feature_ranges(X),
    }


def test_bagging_ensemble_confidence_in_valid_range():
    X, y = _synthetic_training_data()
    estimator = RandomForestRegressor(n_estimators=20, random_state=42).fit(X, y)
    artifact = _base_artifact(estimator, X, y, "bagging_ensemble")
    score = compute_confidence(artifact, _sample_input(X))
    assert 0.0 <= score <= 100.0


def test_single_tree_confidence_in_valid_range():
    X, y = _synthetic_training_data()
    estimator = DecisionTreeRegressor(max_depth=4, random_state=42).fit(X, y)
    artifact = _base_artifact(estimator, X, y, "single_tree")
    score = compute_confidence(artifact, _sample_input(X))
    assert 0.0 <= score <= 100.0


def test_knn_residual_confidence_in_valid_range():
    X, y = _synthetic_training_data()
    estimator = LinearRegression().fit(X, y)
    artifact = _base_artifact(estimator, X, y, "knn_residual")
    score = compute_confidence(artifact, _sample_input(X))
    assert 0.0 <= score <= 100.0


def test_extrapolation_penalty_is_one_when_in_range():
    feature_ranges = {col: (0.0, 50.0) for col in FEATURE_COLUMNS}
    raw_input = {"Moisture": 10.0, "Volatile_matter": 20.0, "Fixed_Carbon": 30.0, "Std.Ash": 25.0}
    assert _extrapolation_penalty(raw_input, feature_ranges) == 1.0


def test_extrapolation_penalty_below_one_when_out_of_range():
    feature_ranges = {col: (0.0, 50.0) for col in FEATURE_COLUMNS}
    raw_input = {"Moisture": 100.0, "Volatile_matter": 20.0, "Fixed_Carbon": 30.0, "Std.Ash": 25.0}
    penalty = _extrapolation_penalty(raw_input, feature_ranges)
    assert 0.5 <= penalty < 1.0
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `cd ml && ../.venv/Scripts/python.exe -m pytest tests/test_confidence.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.models.confidence'`

- [ ] **Step 3: Implement `ml/src/models/confidence.py`**

```python
import numpy as np


def _to_feature_vector(raw_input: dict, feature_columns: list[str]) -> np.ndarray:
    return np.array([raw_input[column] for column in feature_columns], dtype=float)


def _transform_for_model(raw_vector: np.ndarray, scaler) -> np.ndarray:
    if scaler is None:
        return raw_vector
    return scaler.transform(raw_vector.reshape(1, -1))[0]


def _bagging_ensemble_confidence(estimator, X_row: np.ndarray, residual_std: float) -> float:
    tree_predictions = np.array([tree.predict(X_row.reshape(1, -1))[0] for tree in estimator.estimators_])
    spread = float(tree_predictions.std())
    if residual_std <= 0:
        return 100.0
    return 100.0 * float(np.exp(-spread / residual_std))


def _single_tree_confidence(estimator, X_train: np.ndarray, residuals_train: np.ndarray, X_row: np.ndarray, residual_std: float) -> float:
    leaf_id = estimator.apply(X_row.reshape(1, -1))[0]
    train_leaf_ids = estimator.apply(X_train)
    mask = train_leaf_ids == leaf_id
    local_residuals = residuals_train[mask] if mask.sum() > 1 else np.array([residual_std])
    local_spread = float(np.abs(local_residuals).std()) if len(local_residuals) > 1 else float(np.abs(local_residuals).mean())
    if residual_std <= 0:
        return 100.0
    return 100.0 * float(np.exp(-local_spread / residual_std))


def _knn_residual_confidence(X_train: np.ndarray, residuals_train: np.ndarray, X_row: np.ndarray, residual_std: float, k: int = 15) -> float:
    distances = np.linalg.norm(X_train - X_row, axis=1)
    neighbor_count = min(k, len(distances))
    nearest_idx = np.argsort(distances)[:neighbor_count]
    local_spread = float(np.abs(residuals_train[nearest_idx]).std()) if neighbor_count > 1 else float(np.abs(residuals_train[nearest_idx]).mean())
    if residual_std <= 0:
        return 100.0
    return 100.0 * float(np.exp(-local_spread / residual_std))


def _extrapolation_penalty(raw_input: dict, feature_ranges: dict) -> float:
    penalty = 1.0
    for feature, value in raw_input.items():
        low, high = feature_ranges[feature]
        span = high - low
        if value < low:
            distance = low - value
        elif value > high:
            distance = value - high
        else:
            continue
        penalty *= max(0.5, 1.0 - (distance / span))
    return penalty


def compute_confidence(artifact: dict, raw_input: dict) -> float:
    feature_columns = artifact["feature_columns"]
    raw_vector = _to_feature_vector(raw_input, feature_columns)
    X_row = _transform_for_model(raw_vector, artifact["scaler"])

    method = artifact["confidence_method"]
    residual_std = artifact["residual_std"]

    if method == "bagging_ensemble":
        base_score = _bagging_ensemble_confidence(artifact["estimator"], X_row, residual_std)
    elif method == "single_tree":
        base_score = _single_tree_confidence(
            artifact["estimator"], artifact["X_train"], artifact["residuals_train"], X_row, residual_std
        )
    elif method == "knn_residual":
        base_score = _knn_residual_confidence(
            artifact["X_train"], artifact["residuals_train"], X_row, residual_std
        )
    else:
        raise ValueError(f"Unknown confidence_method: {method}")

    penalty = _extrapolation_penalty(raw_input, artifact["feature_ranges"])
    return max(0.0, min(100.0, base_score * penalty))
```

- [ ] **Step 4: Run tests, verify they pass**

Run: `cd ml && ../.venv/Scripts/python.exe -m pytest tests/test_confidence.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add ml/src/models/confidence.py ml/tests/test_confidence.py
git commit -m "feat(ml): add per-prediction confidence scoring with extrapolation penalty"
```

---

### Task 8: SHAP Explainability Module

**Files:**
- Create: `ml/src/explain/__init__.py`
- Create: `ml/src/explain/shap_explain.py`
- Test: `ml/tests/test_shap_explain.py`

**Interfaces:**
- Produces: `build_explainer(is_tree_based: bool, estimator, background: np.ndarray)`, `global_feature_importance(explainer, X_sample, feature_names) -> dict[str, float]`, `summary_plot_data(explainer, X_sample, raw_X_sample, feature_names) -> list[dict]`, `explain_single_prediction(explainer, X_row, feature_names, raw_values) -> dict` (keys: `base_value, contributions, prediction`), `generate_explanation_sentence(contributions, top_n=3) -> str` — consumed by Task 9.

- [ ] **Step 1: Write the failing tests**

`ml/tests/test_shap_explain.py`:
```python
import numpy as np
from sklearn.tree import DecisionTreeRegressor

from src.explain.shap_explain import (
    build_explainer,
    explain_single_prediction,
    generate_explanation_sentence,
    global_feature_importance,
    summary_plot_data,
)

FEATURE_COLUMNS = ["Moisture", "Volatile_matter", "Fixed_Carbon", "Std.Ash"]


def _fit_model():
    rng = np.random.RandomState(0)
    X = rng.uniform(5, 40, size=(100, 4))
    y = X[:, 0] * 0.3 + X[:, 1] * 0.5 - X[:, 2] * 0.2 + X[:, 3] * 0.1
    model = DecisionTreeRegressor(max_depth=4, random_state=42).fit(X, y)
    return model, X, y


def test_shap_values_sum_to_prediction_minus_base_value():
    model, X, y = _fit_model()
    explainer = build_explainer(is_tree_based=True, estimator=model, background=X)
    raw_values = {name: float(X[0, idx]) for idx, name in enumerate(FEATURE_COLUMNS)}

    result = explain_single_prediction(explainer, X[0], FEATURE_COLUMNS, raw_values)
    actual_prediction = float(model.predict(X[0].reshape(1, -1))[0])

    shap_sum = sum(c["shap_value"] for c in result["contributions"])
    assert abs(result["base_value"] + shap_sum - actual_prediction) < 1e-4
    assert abs(result["prediction"] - actual_prediction) < 1e-4


def test_global_feature_importance_has_all_features():
    model, X, y = _fit_model()
    explainer = build_explainer(is_tree_based=True, estimator=model, background=X)
    importance = global_feature_importance(explainer, X, FEATURE_COLUMNS)
    assert set(importance.keys()) == set(FEATURE_COLUMNS)
    assert all(value >= 0 for value in importance.values())


def test_summary_plot_data_has_expected_row_count():
    model, X, y = _fit_model()
    explainer = build_explainer(is_tree_based=True, estimator=model, background=X)
    records = summary_plot_data(explainer, X, X, FEATURE_COLUMNS)
    assert len(records) == X.shape[0] * len(FEATURE_COLUMNS)


def test_generate_explanation_sentence_mentions_increasing_and_decreasing_features():
    contributions = [
        {"feature": "Fixed_Carbon", "value": 40.0, "shap_value": 2.3},
        {"feature": "Std.Ash", "value": 23.0, "shap_value": -3.0},
        {"feature": "Moisture", "value": 5.0, "shap_value": 0.01},
    ]
    sentence = generate_explanation_sentence(contributions)
    assert "Fixed_Carbon" in sentence
    assert "Std.Ash" in sentence
    assert "increased" in sentence
    assert "decreased" in sentence
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `cd ml && ../.venv/Scripts/python.exe -m pytest tests/test_shap_explain.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.explain'`

- [ ] **Step 3: Implement `ml/src/explain/shap_explain.py`**

`ml/src/explain/__init__.py`: empty file.

`ml/src/explain/shap_explain.py`:
```python
from typing import Any

import numpy as np
import shap


def build_explainer(is_tree_based: bool, estimator: Any, background: np.ndarray):
    if is_tree_based:
        return shap.TreeExplainer(estimator)
    return shap.LinearExplainer(estimator, background)


def _expected_value(explainer) -> float:
    base_value = explainer.expected_value
    if isinstance(base_value, (list, np.ndarray)):
        return float(np.asarray(base_value).flatten()[0])
    return float(base_value)


def global_feature_importance(explainer, X_sample: np.ndarray, feature_names: list[str]) -> dict:
    shap_values = np.asarray(explainer.shap_values(X_sample))
    mean_abs = np.abs(shap_values).mean(axis=0)
    return {name: float(value) for name, value in zip(feature_names, mean_abs)}


def summary_plot_data(explainer, X_sample: np.ndarray, raw_X_sample: np.ndarray, feature_names: list[str]) -> list[dict]:
    shap_values = np.asarray(explainer.shap_values(X_sample))
    records = []
    for row_idx in range(X_sample.shape[0]):
        for col_idx, name in enumerate(feature_names):
            records.append({
                "feature": name,
                "feature_value": float(raw_X_sample[row_idx, col_idx]),
                "shap_value": float(shap_values[row_idx, col_idx]),
            })
    return records


def explain_single_prediction(explainer, X_row: np.ndarray, feature_names: list[str], raw_values: dict) -> dict:
    shap_values = np.asarray(explainer.shap_values(X_row.reshape(1, -1)))[0]
    base_value = _expected_value(explainer)
    contributions = [
        {"feature": name, "value": raw_values[name], "shap_value": float(shap_values[idx])}
        for idx, name in enumerate(feature_names)
    ]
    contributions.sort(key=lambda c: abs(c["shap_value"]), reverse=True)
    return {
        "base_value": base_value,
        "contributions": contributions,
        "prediction": base_value + float(shap_values.sum()),
    }


def generate_explanation_sentence(contributions: list[dict], top_n: int = 3) -> str:
    increasing = [c for c in contributions if c["shap_value"] > 0][:top_n]
    decreasing = [c for c in contributions if c["shap_value"] < 0][:top_n]

    parts = []
    if increasing:
        described = ", ".join(f"{c['feature']} ({c['shap_value']:+.2f} MJ/kg)" for c in increasing)
        parts.append(f"{described} increased the predicted GCV")
    if decreasing:
        described = ", ".join(f"{c['feature']} ({c['shap_value']:+.2f} MJ/kg)" for c in decreasing)
        parts.append(f"{described} decreased it")

    if not parts:
        return "No feature had a significant effect on this prediction."
    return ", while ".join(parts) + "."
```

- [ ] **Step 4: Run tests, verify they pass**

Run: `cd ml && ../.venv/Scripts/python.exe -m pytest tests/test_shap_explain.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add ml/src/explain/__init__.py ml/src/explain/shap_explain.py ml/tests/test_shap_explain.py
git commit -m "feat(ml): add SHAP explainability module with additivity-verified tests"
```

---

### Task 9: SHAP Explanations on Real Trained Model

**Files:**
- Create: `ml/run_explain.py`
- Test: `ml/tests/test_run_explain.py`

**Interfaces:**
- Consumes: `BEST_MODEL_PATH`, `CLEAN_DATA_PATH`, `FEATURE_COLUMNS`, `SHAP_GLOBAL_PATH` (Task 1); the `best_model.joblib` artifact (Task 6); `build_explainer`, `global_feature_importance`, `summary_plot_data` (Task 8)
- Produces: on-disk `models/shap_global.json` with keys `global_feature_importance` (dict of 4 features) and `summary_plot_sample` (list of `{feature, feature_value, shap_value}` records).

- [ ] **Step 1: Implement `ml/run_explain.py`**

```python
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
```

- [ ] **Step 2: Run it against the real trained model**

Run: `cd ml && ../.venv/Scripts/python.exe run_explain.py`
Expected: `Wrote .../shap_global.json (800 summary-plot records)` (200 sampled rows × 4 features), no errors.

- [ ] **Step 3: Write and run the verification test**

`ml/tests/test_run_explain.py`:
```python
import json

from src.config import FEATURE_COLUMNS, SHAP_GLOBAL_PATH


def test_run_explain_persists_global_feature_importance():
    output = json.loads(SHAP_GLOBAL_PATH.read_text())
    assert set(output["global_feature_importance"].keys()) == set(FEATURE_COLUMNS)
    assert len(output["summary_plot_sample"]) > 0
    assert set(output["summary_plot_sample"][0].keys()) == {"feature", "feature_value", "shap_value"}
```

Run: `cd ml && ../.venv/Scripts/python.exe -m pytest tests/test_run_explain.py -v`
Expected: 1 passed

- [ ] **Step 4: Commit**

```bash
git add ml/run_explain.py ml/tests/test_run_explain.py models/shap_global.json
git commit -m "feat(ml): generate SHAP global importance and summary-plot data"
```

---

### Task 10: EDA Notebook

**Files:**
- Create: `ml/notebooks/build_eda_notebook.py`
- Create (generated by the script above, not hand-written): `ml/notebooks/01_eda.ipynb`
- Test: `ml/tests/test_eda_notebook.py`

**Interfaces:**
- Consumes: `data/processed/coal_clean.csv` (Task 3)
- Produces: `ml/notebooks/01_eda.ipynb`, executable end-to-end.

- [ ] **Step 1: Implement `ml/notebooks/build_eda_notebook.py`**

```python
from pathlib import Path

import nbformat as nbf

NOTEBOOK_PATH = Path(__file__).resolve().parent / "01_eda.ipynb"


def _markdown(source: str):
    return nbf.v4.new_markdown_cell(source)


def _code(source: str):
    return nbf.v4.new_code_cell(source)


def build_notebook() -> "nbf.NotebookNode":
    notebook = nbf.v4.new_notebook()
    notebook.metadata = {
        "kernelspec": {
            "name": "python3",
            "display_name": "Python 3",
            "language": "python",
        }
    }
    notebook.cells = [
        _markdown(
            "# Coal GCV — Exploratory Data Analysis\n\n"
            "Distribution, correlation, and relationship analysis of the four "
            "proximate-analysis features against GCV."
        ),
        _code(
            "import sys\n"
            "from pathlib import Path\n\n"
            "import matplotlib.pyplot as plt\n"
            "import pandas as pd\n"
            "import seaborn as sns\n\n"
            "sys.path.insert(0, str(Path.cwd().parent))\n"
            "from src.config import CLEAN_DATA_PATH, FEATURE_COLUMNS, TARGET_COLUMN\n\n"
            "df = pd.read_csv(CLEAN_DATA_PATH)\n"
            "df.describe()"
        ),
        _markdown("## Feature distributions"),
        _code(
            "fig, axes = plt.subplots(2, 2, figsize=(10, 8))\n"
            "for ax, column in zip(axes.flatten(), FEATURE_COLUMNS):\n"
            "    sns.histplot(df[column], kde=True, ax=ax)\n"
            "    ax.set_title(column)\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
        _markdown("## Correlation heatmap"),
        _code(
            "correlation = df[FEATURE_COLUMNS + [TARGET_COLUMN]].corr()\n"
            "plt.figure(figsize=(6, 5))\n"
            "sns.heatmap(correlation, annot=True, cmap=\"coolwarm\", vmin=-1, vmax=1)\n"
            "plt.title(\"Correlation between proximate features and GCV\")\n"
            "plt.show()"
        ),
        _markdown("## GCV vs. each feature"),
        _code(
            "fig, axes = plt.subplots(2, 2, figsize=(10, 8))\n"
            "for ax, column in zip(axes.flatten(), FEATURE_COLUMNS):\n"
            "    ax.scatter(df[column], df[TARGET_COLUMN], alpha=0.3, s=10)\n"
            "    ax.set_xlabel(column)\n"
            "    ax.set_ylabel(TARGET_COLUMN)\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
        _markdown(
            "## Observations\n\n"
            "- Fixed Carbon typically shows the strongest positive correlation with "
            "GCV; Moisture and Ash typically show negative correlation, consistent "
            "with combustion chemistry.\n"
            "- The four features sum to 100% by construction (proximate analysis "
            "convention).\n"
            "- Findings here inform the Analytics dashboard page."
        ),
    ]
    return notebook


def main() -> None:
    notebook = build_notebook()
    nbf.write(notebook, str(NOTEBOOK_PATH))
    print(f"Wrote {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Register a Jupyter kernel for this venv**

`nbconvert --execute` runs notebooks against a named, registered kernel — it does not simply use whichever Python ran the command. The notebook's `kernelspec.name` is `"python3"` (set in Step 1), so a kernel with that exact name must be registered for this venv before execution will work.

Run:
```bash
cd ml && ../.venv/Scripts/python.exe -m ipykernel install --user --name python3 --display-name "Python 3 (gcv_coal_project)"
```
Expected: prints `Installed kernelspec python3 in ...`

- [ ] **Step 3: Run the build script to generate the notebook**

Run: `cd ml && ../.venv/Scripts/python.exe notebooks/build_eda_notebook.py`
Expected: `Wrote .../ml/notebooks/01_eda.ipynb`

- [ ] **Step 4: Write and run the verification test**

`ml/tests/test_eda_notebook.py`:
```python
import subprocess
import sys
from pathlib import Path

NOTEBOOK_PATH = Path(__file__).resolve().parents[1] / "notebooks" / "01_eda.ipynb"
BUILD_SCRIPT = Path(__file__).resolve().parents[1] / "notebooks" / "build_eda_notebook.py"


def test_eda_notebook_builds_and_executes_without_error():
    subprocess.run([sys.executable, str(BUILD_SCRIPT)], check=True)
    assert NOTEBOOK_PATH.exists()

    result = subprocess.run(
        [
            sys.executable, "-m", "nbconvert",
            "--to", "notebook", "--execute", "--output", "01_eda.ipynb",
            str(NOTEBOOK_PATH),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
```

Run: `cd ml && ../.venv/Scripts/python.exe -m pytest tests/test_eda_notebook.py -v`
Expected: 1 passed (takes ~10-20s — it executes the full notebook)

- [ ] **Step 5: Commit**

```bash
git add ml/notebooks/build_eda_notebook.py ml/notebooks/01_eda.ipynb ml/tests/test_eda_notebook.py
git commit -m "feat(ml): add programmatically-generated EDA notebook"
```

---

### Task 11: Pipeline Orchestration Script & ml/README.md

**Files:**
- Create: `ml/run_pipeline.py`
- Create: `ml/README.md`
- Test: `ml/tests/test_run_pipeline.py`

**Interfaces:**
- Consumes: `run_clean.main`, `run_train.main`, `run_explain.main` (Tasks 3, 6, 9)
- Produces: a single entry point that runs the full pipeline in order.

- [ ] **Step 1: Write the failing test**

`ml/tests/test_run_pipeline.py`:
```python
from unittest.mock import Mock, call, patch

import run_pipeline


def test_run_pipeline_calls_steps_in_order():
    with patch("run_pipeline.run_clean") as mock_clean, \
         patch("run_pipeline.run_train") as mock_train, \
         patch("run_pipeline.run_explain") as mock_explain:
        manager = Mock()
        manager.attach_mock(mock_clean.main, "clean")
        manager.attach_mock(mock_train.main, "train")
        manager.attach_mock(mock_explain.main, "explain")

        run_pipeline.main()

        manager.assert_has_calls([call.clean(), call.train(), call.explain()])
```

- [ ] **Step 2: Run test, verify it fails**

Run: `cd ml && ../.venv/Scripts/python.exe -m pytest tests/test_run_pipeline.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'run_pipeline'`

- [ ] **Step 3: Implement `ml/run_pipeline.py`**

```python
import run_clean
import run_explain
import run_train


def main() -> None:
    print("Step 1/3: cleaning data...")
    run_clean.main()
    print("Step 2/3: training and selecting best model...")
    run_train.main()
    print("Step 3/3: generating SHAP explanations...")
    run_explain.main()
    print("Pipeline complete.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test, verify it passes**

Run: `cd ml && ../.venv/Scripts/python.exe -m pytest tests/test_run_pipeline.py -v`
Expected: 1 passed

- [ ] **Step 5: Write `ml/README.md`**

```markdown
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
```

- [ ] **Step 6: Commit**

```bash
git add ml/run_pipeline.py ml/README.md ml/tests/test_run_pipeline.py
git commit -m "feat(ml): add pipeline orchestration script and ml subproject README"
```

---

## Final Verification

After Task 11, run the full suite once more from a clean perspective to confirm the sub-project is complete:

```bash
cd ml && ../.venv/Scripts/python.exe -m pytest -v
```
Expected: all tests across all 11 tasks pass.

```bash
cd ml && ../.venv/Scripts/python.exe run_pipeline.py
```
Expected: completes end-to-end, reprinting the same artifacts (idempotent — safe to re-run).

Confirm these files exist before considering this sub-project done: `data/processed/coal_clean.csv`, `data/processed/data_quality_report.json`, `models/best_model.joblib`, `models/model_metadata.json`, `models/model_comparison.json`, `models/shap_global.json`, `ml/notebooks/01_eda.ipynb`.
