"""Sensitivity analysis and refutation tests for causal unobserved confounding."""

from typing import Dict
import numpy as np
import pandas as pd
from nexus9.causal_engine.effects import estimate_average_treatment_effect

def placebo_treatment_refuter(
    df: pd.DataFrame,
    treatment: str,
    outcome: str,
    confounders: list | None = None,
    n_permutations: int = 20,
) -> Dict[str, float]:
    """Placebo refutation test: randomizes treatment column to ensure estimated effect collapses to ~0."""
    effects = []
    shuffled_df = df.copy()

    for _ in range(n_permutations):
        shuffled_df[treatment] = np.random.permutation(shuffled_df[treatment].values)
        res = estimate_average_treatment_effect(shuffled_df, treatment, outcome, confounders)
        effects.append(res["ate"])

    return {
        "placebo_ate_mean": float(np.mean(effects)),
        "placebo_ate_std": float(np.std(effects)),
        "passed_refutation": bool(abs(np.mean(effects)) < 0.1),
    }
