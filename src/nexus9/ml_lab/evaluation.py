"""Comprehensive model evaluation suite with cross-validation and test metrics."""

from typing import Any, Dict, List
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate

def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
) -> Dict[str, float]:
    """Compute full statistical evaluation metrics."""
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }

    if y_prob is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
            metrics["brier_score"] = float(brier_score_loss(y_true, y_prob))
            metrics["log_loss"] = float(log_loss(y_true, y_prob))
        except Exception:
            pass

    return metrics

def cross_validate_pipeline(
    pipeline: Any,
    X: pd.DataFrame,
    y: pd.Series,
    cv: int = 5,
    random_state: int = 20261016,
) -> Dict[str, Any]:
    """Perform stratified k-fold cross validation."""
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]

    cv_results = cross_validate(
        pipeline,
        X,
        y,
        cv=skf,
        scoring=scoring,
        return_train_score=False,
        n_jobs=-1,
    )

    summary = {}
    for metric in scoring:
        key = f"test_{metric}"
        if key in cv_results:
            vals = cv_results[key]
            summary[f"{metric}_mean"] = float(np.mean(vals))
            summary[f"{metric}_std"] = float(np.std(vals))

    return summary
