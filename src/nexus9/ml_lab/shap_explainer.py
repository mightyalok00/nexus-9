"""SHAP (SHapley Additive exPlanations) for model interpretability."""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

class ModelExplainer:
    """Extracts feature importance and Shapley values for early-warning models."""

    def __init__(self, pipeline: Any):
        self.pipeline = pipeline
        self.model = pipeline.named_steps.get("model", pipeline)
        self.preprocessor = pipeline.named_steps.get("preprocessor", None)

    def explain(self, X_sample: pd.DataFrame, n_background: int = 50) -> Dict[str, Any]:
        """Compute SHAP values or feature importance ranking."""
        if self.preprocessor is not None:
            X_trans = self.preprocessor.transform(X_sample)
            feature_names = self.preprocessor.get_feature_names_out()
        else:
            X_trans = X_sample.values
            feature_names = list(X_sample.columns)

        # Check if shap is installed
        try:
            import shap
            explainer = shap.Explainer(self.model, X_trans[:n_background])
            shap_vals = explainer(X_trans)
            mean_abs_shap = np.mean(np.abs(shap_vals.values), axis=0)
            if mean_abs_shap.ndim > 1:
                mean_abs_shap = mean_abs_shap[:, 1] # Class 1
        except Exception:
            # Fallback to model feature importances
            if hasattr(self.model, "feature_importances_"):
                mean_abs_shap = self.model.feature_importances_
            else:
                mean_abs_shap = np.ones(len(feature_names)) / len(feature_names)

        ranking = pd.DataFrame({
            "feature": feature_names,
            "importance": mean_abs_shap,
        }).sort_values(by="importance", ascending=False).reset_index(drop=True)

        return {
            "feature_importance": ranking.to_dict(orient="records"),
            "top_features": ranking.head(5)["feature"].tolist(),
        }
