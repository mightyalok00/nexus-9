"""Quantified epistemic and aleatoric uncertainty estimators."""

from typing import List, Tuple
import numpy as np

def compute_confidence_bounds(
    trajectories: List[List[float]],
    confidence: float = 0.95,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return mean, lower bound, and upper bound across Monte Carlo rollouts."""
    arr = np.array(trajectories) # shape: (n_simulations, horizon)
    mean = np.mean(arr, axis=0)
    alpha = (1.0 - confidence) / 2.0
    lower = np.percentile(arr, alpha * 100, axis=0)
    upper = np.percentile(arr, (1.0 - alpha) * 100, axis=0)
    return mean, lower, upper
