"""Central registry for all Model Arena competitive estimators."""

from typing import Any, Callable, Dict, List
from nexus9.exceptions import ModelNotFoundError
from nexus9.ml_lab.models.bagging import create_generic_bagging_model, create_random_forest_model
from nexus9.ml_lab.models.baselines import create_dummy_baseline, create_logistic_baseline
from nexus9.ml_lab.models.boosting import create_gradient_boosting_model, create_xgboost_model
from nexus9.ml_lab.models.svm import create_svm_model
from nexus9.ml_lab.models.voting import create_voting_ensemble_model

MODEL_REGISTRY: Dict[str, Callable[..., Any]] = {
    "baseline_dummy": create_dummy_baseline,
    "baseline_logistic": create_logistic_baseline,
    "bagging_random_forest": create_random_forest_model,
    "bagging_canonical": create_generic_bagging_model,
    "boosting_xgboost": create_xgboost_model,
    "boosting_sklearn": create_gradient_boosting_model,
    "voting_ensemble": create_voting_ensemble_model,
    "svm_classifier": create_svm_model,
}

def get_model(name: str, **kwargs: Any) -> Any:
    """Factory to instantiate registered model families."""
    if name not in MODEL_REGISTRY:
        raise ModelNotFoundError(
            f"Model '{name}' not found. Available models: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[name](**kwargs)

def list_registered_models() -> List[str]:
    """Return names of all registered model families."""
    return list(MODEL_REGISTRY.keys())
