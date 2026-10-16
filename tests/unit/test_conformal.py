"""Unit test for Split Conformal Classifier."""

from nexus9.ml_lab.conformal import SplitConformalClassifier
from nexus9.ml_lab.models.boosting import create_xgboost_model

def test_split_conformal_coverage(split_data):
    X_train, X_val, X_test, y_train, y_val, y_test = split_data
    pipe = create_xgboost_model({"n_estimators": 15, "max_depth": 3})
    pipe.fit(X_train.head(400), y_train.head(400))

    conformal = SplitConformalClassifier(pipe, alpha=0.10)
    conformal.calibrate(X_val.head(200), y_val.head(200))

    metrics = conformal.evaluate_coverage(X_test.head(100), y_test.head(100))
    assert metrics["empirical_coverage"] >= 0.80 # Target is ~0.90
    assert 1.0 <= metrics["average_set_size"] <= 2.0
