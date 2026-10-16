"""Leakage-safe stratified and temporal-consistent data splitting."""

from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
from nexus9.config import settings
from nexus9.data.validation import assert_no_data_leakage

def get_feature_and_target_matrices(
    df: pd.DataFrame,
    include_leakage: bool = False,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Extract early-warning feature matrix X and target y.

    Args:
        df: Raw input DataFrame.
        include_leakage: If True, retains outcome variables (useful ONLY for post-hoc diagnostic testing).

    Returns:
        (X, y): Clean feature dataframe and target series.
    """
    target_col = settings.features.target_column
    id_col = settings.features.id_column
    leakage_cols = settings.features.leakage_columns

    drop_cols = [id_col, target_col]
    if not include_leakage:
        drop_cols.extend(leakage_cols)

    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    y = df[target_col]

    if not include_leakage:
        assert_no_data_leakage(list(X.columns))

    return X, y

def create_train_val_test_splits(
    df: pd.DataFrame,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 20261016,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Produce stratified Train, Validation, and Test splits.

    Returns:
        (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    X, y = get_feature_and_target_matrices(df, include_leakage=False)

    # First split off test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # Split remaining into train and validation
    adjusted_val_size = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=adjusted_val_size,
        random_state=random_state,
        stratify=y_temp,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test
