"""Continuous causal dose-response curve estimation."""

from typing import Dict, List
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

def estimate_dose_response_curve(
    df: pd.DataFrame,
    treatment: str,
    outcome: str,
    confounders: List[str],
    n_points: int = 20,
) -> Dict[str, Any]:
    """Estimate E[Y | do(T = t)] across continuous sweep of treatment levels."""
    clean_df = df[[treatment, outcome] + confounders].dropna()
    X = clean_df[[treatment] + confounders]
    y = clean_df[outcome]

    model = GradientBoostingRegressor(n_estimators=60, max_depth=3, random_state=20261016)
    model.fit(X, y)

    t_min = float(clean_df[treatment].quantile(0.05))
    t_max = float(clean_df[treatment].quantile(0.95))
    treatment_grid = np.linspace(t_min, t_max, n_points)

    expected_outcomes = []
    # G-computation / marginal standardization
    for t_val in treatment_grid:
        synth_X = X.copy()
        synth_X[treatment] = t_val
        preds = model.predict(synth_X)
        expected_outcomes.append(float(np.mean(preds)))

    return {
        "treatment": treatment,
        "outcome": outcome,
        "treatment_levels": treatment_grid.tolist(),
        "expected_outcomes": expected_outcomes,
    }
