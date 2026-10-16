"""Preprocessing pipeline: Imputation, categorical encoding, and outlier clipping."""

from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler
from nexus9.config import settings
from nexus9.data.validation import assert_no_data_leakage

class Nexus9Preprocessor(BaseEstimator, TransformerMixin):
    """Leakage-safe preprocessor for NEXUS-9 tabular early-warning features."""

    def __init__(
        self,
        numerical_features: Optional[List[str]] = None,
        categorical_features: Optional[List[str]] = None,
        scaler_type: str = "robust",
    ):
        self.numerical_features = numerical_features or settings.features.numerical_features
        self.categorical_features = categorical_features or settings.features.categorical_features
        self.scaler_type = scaler_type
        self.pipeline: Optional[ColumnTransformer] = None

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "Nexus9Preprocessor":
        # Validate no leakage in input
        assert_no_data_leakage(list(X.columns))

        # Numerical pipeline with median imputation (handles supplier_reliability NaNs)
        scaler = RobustScaler() if self.scaler_type == "robust" else StandardScaler()
        num_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", scaler),
        ])

        # Categorical pipeline
        cat_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)),
        ])

        self.pipeline = ColumnTransformer(
            transformers=[
                ("num", num_pipe, self.numerical_features),
                ("cat", cat_pipe, self.categorical_features),
            ],
            remainder="drop",
        )
        self.pipeline.fit(X, y)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if self.pipeline is None:
            raise RuntimeError("Preprocessor has not been fitted.")
        assert_no_data_leakage(list(X.columns))
        return self.pipeline.transform(X)

    def get_feature_names_out(self) -> List[str]:
        if self.pipeline is None:
            return []
        names = []
        for name, trans, cols in self.pipeline.transformers_:
            if name == "num":
                names.extend(cols)
            elif name == "cat":
                encoder = trans.named_steps["encoder"]
                cat_names = encoder.get_feature_names_out(cols)
                names.extend(list(cat_names))
        return names
