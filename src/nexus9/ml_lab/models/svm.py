"""Support Vector Machine Family: SVC with calibrated probability estimates."""

from typing import Any, Dict, Optional
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from nexus9.data.preprocessing import Nexus9Preprocessor

def create_svm_model(
    params: Optional[Dict[str, Any]] = None,
    random_state: int = 20261016,
) -> Pipeline:
    """Build calibrated Support Vector Classifier pipeline."""
    default_params = {
        "kernel": "rbf",
        "C": 1.5,
        "gamma": "scale",
        "probability": True,
        "class_weight": "balanced",
        "random_state": random_state,
    }
    if params:
        default_params.update(params)

    # Use standard scaler for SVM optimal hyperplane margins
    return Pipeline([
        ("preprocessor", Nexus9Preprocessor(scaler_type="standard")),
        ("model", SVC(**default_params)),
    ])
