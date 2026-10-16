"""Optuna hyperparameter tuning integration for Model Arena."""

from typing import Any, Callable, Dict
import optuna
from sklearn.model_selection import StratifiedKFold, cross_val_score
from nexus9.ml_lab.models.boosting import create_xgboost_model
from nexus9.ml_lab.models.bagging import create_random_forest_model

optuna.logging.set_verbosity(optuna.logging.WARNING)

def tune_xgboost(
    X_train: Any,
    y_train: Any,
    n_trials: int = 15,
    random_state: int = 20261016,
) -> Dict[str, Any]:
    """Tune XGBoost hyperparameters via Optuna."""
    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 180),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        }
        pipe = create_xgboost_model(params=params, random_state=random_state)
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
        scores = cross_val_score(pipe, X_train, y_train, cv=skf, scoring="f1", n_jobs=-1)
        return float(scores.mean())

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    return {
        "best_params": study.best_params,
        "best_score": study.best_value,
    }
