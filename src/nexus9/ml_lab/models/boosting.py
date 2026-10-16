"""Boosting Ensemble Family: XGBoost & Gradient Boosting Classifiers."""

from typing import Any, Dict, Optional
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from nexus9.data.preprocessing import Nexus9Preprocessor

def create_xgboost_model(
    params: Optional[Dict[str, Any]] = None,
    random_state: int = 20261016,
) -> Pipeline:
    """Build enterprise XGBoost Classifier pipeline."""
    default_params = {
        "n_estimators": 120,
        "max_depth": 5,
        "learning_rate": 0.08,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "random_state": random_state,
        "eval_metric": "logloss",
        "n_jobs": -1,
    }
    if params:
        default_params.update(params)

    return Pipeline([
        ("preprocessor", Nexus9Preprocessor()),
        ("model", XGBClassifier(**default_params)),
    ])

def create_gradient_boosting_model(
    params: Optional[Dict[str, Any]] = None,
    random_state: int = 20261016,
) -> Pipeline:
    """Build Scikit-Learn Gradient Boosting pipeline."""
    default_params = {
        "n_estimators": 100,
        "learning_rate": 0.1,
        "max_depth": 4,
        "random_state": random_state,
    }
    if params:
        default_params.update(params)

    return Pipeline([
        ("preprocessor", Nexus9Preprocessor()),
        ("model", GradientBoostingClassifier(**default_params)),
    ])
