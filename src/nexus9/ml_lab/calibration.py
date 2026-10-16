"""Model probability calibration and diagnostics (Brier Score, ECE)."""

from typing import Any, Dict
import numpy as np
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss

def calibrate_model(
    model: Any,
    X_train: Any,
    y_train: Any,
    method: str = "isotonic",
    cv: int = 3,
) -> CalibratedClassifierCV:
    """Wrap an estimator in CalibratedClassifierCV."""
    calibrated = CalibratedClassifierCV(estimator=model, method=method, cv=cv)
    calibrated.fit(X_train, y_train)
    return calibrated

def compute_calibration_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, Any]:
    """Calculate reliability curve coordinates and Expected Calibration Error (ECE)."""
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")
    brier = brier_score_loss(y_true, y_prob)

    # Expected Calibration Error (ECE)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    binids = np.digitize(y_prob, bins) - 1
    ece = 0.0
    for i in range(n_bins):
        mask = binids == i
        if np.sum(mask) > 0:
            prop = np.mean(y_true[mask])
            conf = np.mean(y_prob[mask])
            ece += (np.sum(mask) / len(y_prob)) * np.abs(prop - conf)

    return {
        "prob_true": prob_true.tolist(),
        "prob_pred": prob_pred.tolist(),
        "brier_score": float(brier),
        "ece": float(ece),
    }
