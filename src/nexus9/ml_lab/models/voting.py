"""Voting Ensemble Family: Heterogeneous Ensembles across Trees, Boosting, and SVM."""

from typing import Any, Dict, List, Optional
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from xgboost import XGBClassifier
from nexus9.data.preprocessing import Nexus9Preprocessor

def create_voting_ensemble_model(
    voting: str = "soft",
    weights: Optional[List[int]] = None,
    random_state: int = 20261016,
) -> Pipeline:
    """Build heterogeneous Voting Classifier combining Bagging, Boosting, and SVM."""
    clf1 = RandomForestClassifier(n_estimators=80, max_depth=8, class_weight="balanced", random_state=random_state)
    clf2 = XGBClassifier(n_estimators=80, max_depth=4, learning_rate=0.08, eval_metric="logloss", random_state=random_state)
    clf3 = SVC(kernel="rbf", C=1.0, probability=True, class_weight="balanced", random_state=random_state)

    ensemble = VotingClassifier(
        estimators=[
            ("rf", clf1),
            ("xgb", clf2),
            ("svm", clf3),
        ],
        voting=voting,
        weights=weights or [1, 2, 1],
        n_jobs=-1,
    )

    return Pipeline([
        ("preprocessor", Nexus9Preprocessor()),
        ("model", ensemble),
    ])
