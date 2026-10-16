"""Mutual information and Bayesian information gain estimators."""

from typing import List
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

def rank_features_by_information_gain(
    X: pd.DataFrame,
    y: pd.Series,
) -> pd.DataFrame:
    """Calculate mutual information gain for each predictor feature relative to target."""
    numeric_X = X.select_dtypes(include=[np.number]).fillna(0)
    mi = mutual_info_classif(numeric_X, y, random_state=20261016)

    df_mi = pd.DataFrame({
        "feature": numeric_X.columns,
        "information_gain": mi,
    }).sort_values(by="information_gain", ascending=False).reset_index(drop=True)

    return df_mi
