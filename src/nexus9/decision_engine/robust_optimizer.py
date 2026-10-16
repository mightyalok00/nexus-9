"""Distributionally robust optimization (DRO) and minimax decision policies."""

from typing import Any, Dict, List
import numpy as np

def minimax_decision(loss_matrix: np.ndarray) -> Dict[str, Any]:
    """Select policy that minimizes the worst-case loss across all regimes/states."""
    worst_case_per_policy = np.max(loss_matrix, axis=1)
    best_policy_idx = int(np.argmin(worst_case_per_policy))

    return {
        "worst_case_losses": worst_case_per_policy.tolist(),
        "minimax_optimal_policy_index": best_policy_idx,
        "minimax_loss": float(worst_case_per_policy[best_policy_idx]),
    }
