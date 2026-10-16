"""Regression tests for classification metric correctness and validation."""

import numpy as np
import pytest

from nexus9.ml_lab.evaluation import evaluate_predictions


def test_binary_metrics_include_specificity_mcc_and_confusion_counts():
    metrics = evaluate_predictions(
        y_true=np.array([0, 0, 0, 1, 1, 1]),
        y_pred=np.array([0, 0, 1, 1, 0, 1]),
        y_prob=np.array([0.05, 0.20, 0.70, 0.80, 0.35, 0.95]),
    )

    assert metrics["accuracy"] == pytest.approx(4 / 6)
    assert metrics["specificity"] == pytest.approx(2 / 3)
    assert metrics["recall"] == pytest.approx(2 / 3)
    assert metrics["true_negative"] == 2
    assert metrics["false_positive"] == 1
    assert metrics["false_negative"] == 1
    assert metrics["true_positive"] == 2
    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert 0.0 <= metrics["average_precision"] <= 1.0
    assert metrics["brier_score"] >= 0.0


@pytest.mark.parametrize(
    ("y_true", "y_pred", "y_prob"),
    [
        ([0, 1], [0], [0.2, 0.8]),
        ([0, 1], [0, 1], [0.2]),
        ([0, 1], [0, 1], [0.2, np.nan]),
        ([0, 1], [0, 1], [0.2, 1.2]),
        ([0, 2], [0, 1], [0.2, 0.8]),
    ],
)
def test_invalid_metric_inputs_fail_loudly(y_true, y_pred, y_prob):
    with pytest.raises(ValueError):
        evaluate_predictions(y_true, y_pred, y_prob)


def test_one_class_test_set_marks_ranking_metrics_undefined():
    metrics = evaluate_predictions([1, 1, 1], [1, 0, 1], [0.9, 0.2, 0.8])

    assert np.isnan(metrics["roc_auc"])
    assert np.isnan(metrics["average_precision"])
    assert metrics["recall"] == pytest.approx(2 / 3)
