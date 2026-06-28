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
