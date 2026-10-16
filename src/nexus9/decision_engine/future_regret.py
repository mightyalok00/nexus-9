"""Counterfactual regret minimization and Expected Opportunity Loss (EOL)."""

from typing import Any, Dict, List
import numpy as np

def calculate_expected_regret(
    payoff_matrix: np.ndarray,
    state_probabilities: np.ndarray,
) -> Dict[str, Any]:
    """Compute expected regret for each candidate policy given state distribution.

    Args:
        payoff_matrix: 2D array of shape (n_policies, n_states), where values are costs/losses.
        state_probabilities: 1D array of shape (n_states,) summing to 1.

    Returns:
        Dict containing regrets per policy and optimal minimum-regret policy.
    """
    # Min cost in each state (best possible action)
    best_cost_per_state = np.min(payoff_matrix, axis=0)

    # Regret = cost(policy, state) - best_cost(state)
    regret_matrix = payoff_matrix - best_cost_per_state

    # Expected regret = E_{states}[regret]
    expected_regret = np.dot(regret_matrix, state_probabilities)
    best_policy_idx = int(np.argmin(expected_regret))

    return {
        "expected_regrets": expected_regret.tolist(),
        "min_regret_policy_index": best_policy_idx,
        "min_regret_value": float(expected_regret[best_policy_idx]),
    }
