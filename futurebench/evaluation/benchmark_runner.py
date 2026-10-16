"""Reproducible benchmark runner for predictive models and causal discovery."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from nexus9.causal_engine.discovery import discover_causal_skeleton_correlation
from nexus9.causal_engine.graph import CausalGraph
from nexus9.data.loader import load_futurebench_data
from nexus9.data.splitting import create_train_val_test_splits
from nexus9.ml_lab.evaluation import evaluate_predictions
from nexus9.ml_lab.registry import get_model
from futurebench.evaluation.causal_metrics import evaluate_causal_discovery

RANDOM_STATE = 20261016
MODEL_NAMES = (
    "baseline_dummy",
    "baseline_logistic",
    "bagging_random_forest",
    "boosting_xgboost",
    "voting_ensemble",
    "svm_classifier",
)


def run_complete_benchmark() -> dict[str, Any]:
    """Run the held-out model benchmark and causal-DAG benchmark.

    The returned metadata records the split sizes, random seed, and target
    prevalence so benchmark artifacts can be interpreted and reproduced.
    """
    df = load_futurebench_data()
    X_train, X_val, X_test, y_train, y_val, y_test = create_train_val_test_splits(
        df, random_state=RANDOM_STATE
    )

    model_results: list[dict[str, Any]] = []
    for name in MODEL_NAMES:
        model = get_model(name)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
        metrics = evaluate_predictions(y_test, y_pred, y_prob)
        metrics["model"] = name
        model_results.append(metrics)

    model_results.sort(key=lambda item: item["f1"], reverse=True)

    # This is a separate synthetic-data structure-recovery experiment, not a
    # predictive test-set score. Its input and interpretation are documented.
    discovered_cg = discover_causal_skeleton_correlation(df, threshold=0.25)
    ground_truth_cg = CausalGraph.load_ground_truth()
    causal_metrics = evaluate_causal_discovery(
        predicted_edges=discovered_cg.get_edges(),
        ground_truth_edges=ground_truth_cg.get_edges(),
    )

    return {
        "metadata": {
            "benchmark": "FutureBench synthetic benchmark",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "random_state": RANDOM_STATE,
            "n_rows": int(len(df)),
            "n_features": int(X_train.shape[1]),
            "split_sizes": {
                "train": int(len(X_train)),
                "validation": int(len(X_val)),
                "test": int(len(X_test)),
            },
            "test_positive_rate": float(y_test.mean()),
            "target": str(y_train.name),
            "dataset_type": "synthetic; not evidence of real-world performance",
        },
        "model_arena": model_results,
        "causal_discovery": causal_metrics,
    }
