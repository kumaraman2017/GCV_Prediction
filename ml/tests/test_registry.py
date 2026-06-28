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
