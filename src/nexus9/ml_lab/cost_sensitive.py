"""Asymmetric cost-sensitive decision threshold optimization."""

from typing import Any, Dict, Tuple
import numpy as np

def optimize_decision_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    cost_false_negative: float = 10000.0,
    cost_false_positive: float = 500.0,
    cost_true_positive: float = 100.0,
    cost_true_negative: float = 0.0,
) -> Dict[str, Any]:
    """Find classification threshold tau in [0, 1] that minimizes expected operational cost."""
    thresholds = np.linspace(0.01, 0.99, 99)
    total_costs = []

    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        tp = np.sum((preds == 1) & (y_true == 1))
        fp = np.sum((preds == 1) & (y_true == 0))
        fn = np.sum((preds == 0) & (y_true == 1))
        tn = np.sum((preds == 0) & (y_true == 0))

        cost = (
            fn * cost_false_negative +
            fp * cost_false_positive +
            tp * cost_true_positive +
            tn * cost_true_negative
        )
        total_costs.append(cost)

    best_idx = int(np.argmin(total_costs))
    best_threshold = float(thresholds[best_idx])
    default_cost = float(total_costs[49]) # Threshold = 0.50
    savings = float(default_cost - total_costs[best_idx])

    return {
        "optimal_threshold": best_threshold,
        "min_cost": float(total_costs[best_idx]),
        "cost_at_0_5": default_cost,
        "cost_savings": savings,
        "cost_reduction_percent": float((savings / max(1.0, default_cost)) * 100),
    }
