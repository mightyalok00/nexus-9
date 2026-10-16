"""CLI entry point for training and evaluating the Model Arena."""

import json
from pathlib import Path

import pandas as pd

from nexus9.data.loader import load_futurebench_data
from nexus9.data.splitting import create_train_val_test_splits
from nexus9.evidence.ledger import EvidenceLedger
from nexus9.ml_lab.evaluation import evaluate_predictions
from nexus9.ml_lab.registry import get_model

RANDOM_STATE = 20261016
MODEL_NAMES = (
    "baseline_dummy",
    "baseline_logistic",
    "bagging_random_forest",
    "boosting_xgboost",
    "voting_ensemble",
    "svm_classifier",
)


def main() -> None:
    print("=== NEXUS-9 Model Arena: Training & Evaluation Pipeline ===")
    df = load_futurebench_data()
    X_train, X_val, X_test, y_train, y_val, y_test = create_train_val_test_splits(
        df, random_state=RANDOM_STATE
    )
    print(
        f"Dataset split: Train={len(X_train)}, Validation={len(X_val)}, "
        f"Test={len(X_test)} | seed={RANDOM_STATE}"
    )

    results = []
    ledger = EvidenceLedger()
    for name in MODEL_NAMES:
        print(f"Training [{name}]...")
        model = get_model(name)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
        metrics = evaluate_predictions(y_test, y_pred, y_prob)
        metrics["model"] = name
        results.append(metrics)
        print(
            f" -> Accuracy: {metrics['accuracy']:.4f} | F1: {metrics['f1']:.4f} | "
            f"MCC: {metrics['mcc']:.4f} | ROC-AUC: {metrics.get('roc_auc', float('nan')):.4f}"
        )

    results_df = pd.DataFrame(results).sort_values(by="f1", ascending=False)
    print()
    print("=== FINAL LEADERBOARD (sorted by F1) ===")
    print(results_df.to_string(index=False))

    out_dir = Path("reports/benchmark")
    out_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out_dir / "arena_leaderboard.csv", index=False)
    payload = {
        "random_state": RANDOM_STATE,
        "n_train": len(X_train),
        "n_validation": len(X_val),
        "n_test": len(X_test),
        "target_prevalence_test": float(y_test.mean()),
        "ranking_metric": "f1",
        "results": results_df.to_dict(orient="records"),
    }
    (out_dir / "arena_leaderboard.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8"
    )

    # Record the actual top-ranked model, not the first model in loop order.
    best_model = results_df.iloc[0]
    ledger.record_entry(
        experiment_type="Model Arena Evaluation",
        parameters={
            "evaluated_models": list(MODEL_NAMES),
            "random_state": RANDOM_STATE,
            "ranking_metric": "f1",
            "test_size": len(X_test),
        },
        metrics={
            "best_model": str(best_model["model"]),
            "f1": float(best_model["f1"]),
            "mcc": float(best_model["mcc"]),
        },
        agent_notes=(
            "Held-out FutureBench synthetic evaluation. Results are not a claim of "
            "real-world supply-chain performance."
        ),
    )
    print()
    print(
        "Results written to reports/benchmark/arena_leaderboard.csv and "
        "reports/benchmark/arena_leaderboard.json."
    )


if __name__ == "__main__":
    main()
