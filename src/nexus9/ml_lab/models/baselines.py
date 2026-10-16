"""Baseline classifiers for the Model Arena benchmark."""

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from nexus9.data.preprocessing import Nexus9Preprocessor

def create_dummy_baseline(strategy: str = "stratified", random_state: int = 20261016) -> Pipeline:
    """Create naive baseline classifier."""
    return Pipeline([
        ("preprocessor", Nexus9Preprocessor()),
        ("model", DummyClassifier(strategy=strategy, random_state=random_state)),
    ])

def create_logistic_baseline(random_state: int = 20261016) -> Pipeline:
    """Create regularized Logistic Regression baseline."""
    return Pipeline([
        ("preprocessor", Nexus9Preprocessor()),
        ("model", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=random_state)),
    ])
