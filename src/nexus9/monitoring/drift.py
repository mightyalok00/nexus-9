"""Data and concept drift detection (Kolmogorov-Smirnov & Population Stability Index)."""

from typing import Any, Dict
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

def detect_feature_drift_ks(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    alpha: float = 0.01,
) -> Dict[str, Dict[str, Any]]:
    """Run two-sample Kolmogorov-Smirnov test on all numerical features."""
    results = {}
    cols = reference_df.select_dtypes(include=[np.number]).columns

    for col in cols:
        if col in current_df.columns:
            s1 = reference_df[col].dropna()
            s2 = current_df[col].dropna()
            stat, p_val = ks_2samp(s1, s2)
            results[col] = {
                "statistic": float(stat),
                "p_value": float(p_val),
                "drift_detected": bool(p_val < alpha),
            }

    return results
