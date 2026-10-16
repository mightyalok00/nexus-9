"""Integration test for complete ML Pipeline: Loader -> Preprocessing -> Training -> Evaluation."""

from nexus9.data.loader import load_futurebench_data
from nexus9.data.splitting import create_train_val_test_splits
from nexus9.ml_lab.registry import get_model
from nexus9.ml_lab.evaluation import evaluate_predictions

def test_complete_ml_pipeline_run():
    df = load_futurebench_data()
    X_train, X_val, X_test, y_train, y_val, y_test = create_train_val_test_splits(df)

    pipe = get_model("boosting_xgboost", params={"n_estimators": 20, "max_depth": 3})
    pipe.fit(X_train.head(1000), y_train.head(1000))

    preds = pipe.predict(X_test.head(100))
    probs = pipe.predict_proba(X_test.head(100))[:, 1]
    metrics = evaluate_predictions(y_test.head(100), preds, probs)

    assert "accuracy" in metrics
    assert "f1" in metrics
    assert "roc_auc" in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0
