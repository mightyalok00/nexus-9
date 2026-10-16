"""Unit tests for Boosting / XGBoost models."""

from nexus9.ml_lab.models.boosting import create_xgboost_model

def test_xgboost_training_and_inference(split_data):
    X_train, X_val, X_test, y_train, y_val, y_test = split_data
    pipe = create_xgboost_model({"n_estimators": 15, "max_depth": 3})
    pipe.fit(X_train.head(500), y_train.head(500))

    preds = pipe.predict(X_test.head(50))
    probs = pipe.predict_proba(X_test.head(50))

    assert len(preds) == 50
    assert probs.shape == (50, 2)
    assert set(preds).issubset({0, 1})
