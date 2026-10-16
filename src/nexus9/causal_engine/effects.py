"""Average Treatment Effect (ATE) and Conditional Average Treatment Effect (CATE) estimators."""

from typing import Dict, Optional
import numpy as np
import pandas as pd
import statsmodels.api as sm

def estimate_average_treatment_effect(
    df: pd.DataFrame,
    treatment: str,
    outcome: str,
    confounders: Optional[list] = None,
) -> Dict[str, float]:
    """Estimate ATE using multivariate OLS adjustment with robust standard errors."""
    clean_df = df.dropna().copy()
    confounders = confounders or []

    # Valid confounders present in dataframe
    valid_confounders = [c for c in confounders if c in clean_df.columns and c not in (treatment, outcome)]

    X = clean_df[[treatment] + valid_confounders]
    X = sm.add_constant(X)
    y = clean_df[outcome]

    model = sm.OLS(y, X).fit(cov_type="HC3")

    ate = float(model.params[treatment])
    p_val = float(model.pvalues[treatment])
    ci_lower, ci_upper = model.conf_int().loc[treatment]

    return {
        "treatment": treatment,
        "outcome": outcome,
        "ate": ate,
        "p_value": p_val,
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
    }
