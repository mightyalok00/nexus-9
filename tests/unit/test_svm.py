"""Unit tests for Calibrated Support Vector Classifier."""

from nexus9.ml_lab.models.svm import create_svm_model

def test_svm_training(split_data):
    X_train, X_val, X_test, y_train, y_val, y_test = split_data
    pipe = create_svm_model({"C": 1.0})
    pipe.fit(X_train.head(300), y_train.head(300))

    preds = pipe.predict(X_test.head(30))
    probs = pipe.predict_proba(X_test.head(30))

    assert len(preds) == 30
    assert probs.shape == (30, 2)
