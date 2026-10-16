"""Reliable binary-classification metrics and cross-validation helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate


def evaluate_predictions(
    y_true: Any,
    y_pred: Any,
    y_prob: Any | None = None,
) -> dict[str, float]:
    """Compute explicit binary-classification metrics.

    Probability inputs must be positive-class probabilities in [0, 1].
    Invalid shapes and non-finite values fail loudly rather than silently
    suppressing metric errors. The target is expected to use labels 0 and 1.
    """
    truth = np.asarray(y_true).reshape(-1)
    pred = np.asarray(y_pred).reshape(-1)

    if truth.size == 0:
        raise ValueError("y_true must contain at least one observation.")
    if truth.shape != pred.shape:
        raise ValueError("y_true and y_pred must have the same number of observations.")
    if not set(np.unique(truth)).issubset({0, 1}):
        raise ValueError("Binary target labels must be encoded as 0 and 1.")
    if not set(np.unique(pred)).issubset({0, 1}):
        raise ValueError("Binary predictions must be encoded as 0 and 1.")

    tn, fp, fn, tp = confusion_matrix(truth, pred, labels=[0, 1]).ravel()
    specificity = float(tn / (tn + fp)) if (tn + fp) else 0.0

    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(truth, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, pred)),
        "precision": float(precision_score(truth, pred, zero_division=0)),
        "recall": float(recall_score(truth, pred, zero_division=0)),
        "specificity": specificity,
        "f1": float(f1_score(truth, pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(truth, pred)),
        "true_negative": float(tn),
        "false_positive": float(fp),
        "false_negative": float(fn),
        "true_positive": float(tp),
    }

    if y_prob is not None:
        prob = np.asarray(y_prob, dtype=float).reshape(-1)
        if prob.shape != truth.shape:
            raise ValueError("y_prob and y_true must have the same number of observations.")
        if not np.isfinite(prob).all():
            raise ValueError("y_prob must contain only finite values.")
        if ((prob < 0.0) | (prob > 1.0)).any():
            raise ValueError("y_prob values must lie in [0, 1].")

        # Ranking metrics are undefined when the test labels contain one class.
        if np.unique(truth).size == 2:
            metrics["roc_auc"] = float(roc_auc_score(truth, prob))
            metrics["average_precision"] = float(average_precision_score(truth, prob))
        else:
            metrics["roc_auc"] = float("nan")
            metrics["average_precision"] = float("nan")

        metrics["brier_score"] = float(brier_score_loss(truth, prob))
        metrics["log_loss"] = float(log_loss(truth, np.column_stack([1.0 - prob, prob]), labels=[0, 1]))

    return metrics


def cross_validate_pipeline(
    pipeline: Any,
    X: pd.DataFrame,
    y: pd.Series,
    cv: int = 5,
    random_state: int = 20261016,
) -> dict[str, Any]:
    """Run reproducible stratified CV and report mean and sample standard deviation."""
    if cv < 2:
        raise ValueError("cv must be at least 2.")
    if len(X) != len(y):
        raise ValueError("X and y must contain the same number of observations.")
    if y.value_counts().min() < cv:
        raise ValueError("Each class must contain at least cv observations.")

    splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
        "average_precision": "average_precision",
        "balanced_accuracy": "balanced_accuracy",
    }
    cv_results = cross_validate(
        pipeline,
        X,
        y,
        cv=splitter,
        scoring=scoring,
        return_train_score=False,
        n_jobs=-1,
        error_score="raise",
    )

    summary: dict[str, Any] = {
        "cv_folds": cv,
        "random_state": random_state,
        "n_samples": int(len(y)),
    }
    for metric in scoring:
        values = cv_results[f"test_{metric}"]
        summary[f"{metric}_mean"] = float(np.mean(values))
        summary[f"{metric}_std"] = float(np.std(values, ddof=1))

    return summary
