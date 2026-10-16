"""Bagging Ensemble Family: Random Forest & Bagging Classifiers."""

from typing import Any, Dict, Optional
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from nexus9.data.preprocessing import Nexus9Preprocessor

def create_random_forest_model(
    params: Optional[Dict[str, Any]] = None,
    random_state: int = 20261016,
) -> Pipeline:
    """Build tuned Random Forest Bagging pipeline."""
    default_params = {
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 5,
        "class_weight": "balanced",
        "random_state": random_state,
        "n_jobs": -1,
    }
    if params:
        default_params.update(params)

    return Pipeline([
        ("preprocessor", Nexus9Preprocessor()),
        ("model", RandomForestClassifier(**default_params)),
    ])

def create_generic_bagging_model(
    n_estimators: int = 50,
    random_state: int = 20261016,
) -> Pipeline:
    """Build canonical Bagging Classifier with Decision Trees."""
    base = DecisionTreeClassifier(max_depth=8, class_weight="balanced", random_state=random_state)
    return Pipeline([
        ("preprocessor", Nexus9Preprocessor()),
        ("model", BaggingClassifier(
            estimator=base,
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
        )),
    ])
