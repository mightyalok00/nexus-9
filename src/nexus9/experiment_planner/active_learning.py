"""Uncertainty sampling and pool-based active learning selection."""

from typing import List
import numpy as np
from sklearn.pipeline import Pipeline

def select_uncertain_samples(
    pipeline: Pipeline,
    pool_X: np.ndarray,
    n_samples: int = 10,
) -> List[int]:
    """Select the most uncertain samples (entropy sampling) to prioritize experimental acquisition."""
    probs = pipeline.predict_proba(pool_X)
    # Entropy: -sum(p * log(p))
    entropy = -np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0)), axis=1)
    top_indices = np.argsort(entropy)[-n_samples:].tolist()
    return top_indices
