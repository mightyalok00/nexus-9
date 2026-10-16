import math
"""Partially Linear Double Machine Learning (DML) with Cross-Fitting."""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold

def estimate_double_machine_learning_ate(
    df: pd.DataFrame,
    treatment: str,
    outcome: str,
    confounders: List[str],
    n_splits: int = 3,
    random_state: int = 20261016,
) -> Dict[str, float]:
    """Robinson's Orthogonalized DML Estimator:
    Y = theta * T + g(X) + epsilon
    T = m(X) + eta
    Uses gradient boosted trees for nuisance functions g(X) and m(X).
    """
    clean_df = df[[treatment, outcome] + confounders].dropna()
    Y = clean_df[outcome].values
    T = clean_df[treatment].values
    X = clean_df[confounders].values

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    Y_res = np.zeros(len(Y))
    T_res = np.zeros(len(T))

    for train_idx, test_idx in kf.split(X):
        # 1. Nuisance model for outcome E[Y|X]
        model_y = GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=random_state)
        model_y.fit(X[train_idx], Y[train_idx])
        Y_res[test_idx] = Y[test_idx] - model_y.predict(X[test_idx])

        # 2. Nuisance model for treatment E[T|X]
        model_t = GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=random_state)
        model_t.fit(X[train_idx], T[train_idx])
        T_res[test_idx] = T[test_idx] - model_t.predict(X[test_idx])

    # Orthogonalized OLS: theta = Cov(Y_res, T_res) / Var(T_res)
    denom = np.mean(T_res ** 2)
    theta = float(np.mean(Y_res * T_res) / max(denom, 1e-8))

    # Asymptotic variance & 95% Confidence Interval
    psi = (Y_res - theta * T_res) * T_res
    variance = np.mean(psi ** 2) / (denom ** 2 * len(Y))
    std_error = float(np.sqrt(max(variance, 1e-12)))
    ci_lower = theta - 1.96 * std_error
    ci_upper = theta + 1.96 * std_error

    return {
        "treatment": treatment,
        "outcome": outcome,
        "dml_ate": theta,
        "std_error": std_error,
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "p_value": float(2 * (1 - 0.5 * (1 + math.erf(abs(theta) / (std_error * np.sqrt(2)))))),
    }
