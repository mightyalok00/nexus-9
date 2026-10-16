"""Automated benchmark runner orchestrating model and causal evaluations."""

from typing import Any, Dict
import pandas as pd
from nexus9.causal_engine.discovery import discover_causal_skeleton_correlation
from nexus9.causal_engine.graph import CausalGraph
from nexus9.data.loader import load_futurebench_data
from nexus9.data.splitting import create_train_val_test_splits
from nexus9.ml_lab.evaluation import evaluate_predictions
from nexus9.ml_lab.registry import get_model, list_registered_models
from futurebench.evaluation.causal_metrics import evaluate_causal_discovery

def run_complete_benchmark() -> Dict[str, Any]:
    """Run Model Arena benchmark and Causal Discovery benchmark against ground truth."""
    df = load_futurebench_data()
    X_train, X_val, X_test, y_train, y_val, y_test = create_train_val_test_splits(df)

    # 1. Model Arena
    model_names = [
        "baseline_dummy",
        "bagging_random_forest",
        "boosting_xgboost",
        "voting_ensemble",
        "svm_classifier",
    ]
    model_results = []

    for name in model_names:
        pipe = get_model(name)
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1] if hasattr(pipe, "predict_proba") else None
        metrics = evaluate_predictions(y_test, y_pred, y_prob)
        metrics["model"] = name
        model_results.append(metrics)

    # 2. Causal Discovery Benchmark
    discovered_cg = discover_causal_skeleton_correlation(df, threshold=0.25)
    gt_cg = CausalGraph.load_ground_truth()

    causal_metrics = evaluate_causal_discovery(
        predicted_edges=discovered_cg.get_edges(),
        ground_truth_edges=gt_cg.get_edges(),
    )

    return {
        "model_arena": model_results,
        "causal_discovery": causal_metrics,
    }
