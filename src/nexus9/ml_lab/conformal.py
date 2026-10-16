"""Split Conformal Prediction for distribution-free finite-sample coverage guarantees."""

from typing import Any, Dict, List, Tuple
import numpy as np
from sklearn.pipeline import Pipeline

class SplitConformalClassifier:
    """Guarantees 1 - alpha marginal coverage: P(Y in C(X)) >= 1 - alpha."""

    def __init__(self, pipeline: Pipeline, alpha: float = 0.05):
        self.pipeline = pipeline
        self.alpha = alpha
        self.q_hat: float = 1.0
        self.is_calibrated: bool = False

    def calibrate(self, X_cal: Any, y_cal: Any) -> "SplitConformalClassifier":
        """Compute non-conformity scores on dedicated calibration set."""
        probs = self.pipeline.predict_proba(X_cal)
        n = len(y_cal)
        y_cal_arr = np.array(y_cal)

        # Non-conformity score: 1 - P(Y = true_label | X)
        prob_true = probs[np.arange(n), y_cal_arr]
        scores = 1.0 - prob_true

        # Conformal quantile level: ceil((n + 1) * (1 - alpha)) / n
        level = np.ceil((n + 1) * (1.0 - self.alpha)) / n
        level = min(1.0, max(0.0, level))
        self.q_hat = float(np.quantile(scores, level, method="higher"))
        self.is_calibrated = True
        return self

    def predict_sets(self, X_test: Any) -> List[List[int]]:
        """Generate guaranteed prediction sets {0}, {1}, or {0, 1}."""
        if not self.is_calibrated:
            raise RuntimeError("Conformal predictor must be calibrated first.")

        probs = self.pipeline.predict_proba(X_test)
        pred_sets = []

        for p in probs:
            s0 = 1.0 - p[0]
            s1 = 1.0 - p[1]
            pset = []
            if s0 <= self.q_hat:
                pset.append(0)
            if s1 <= self.q_hat:
                pset.append(1)
            # Empty set safeguard
            if not pset:
                pset.append(int(np.argmax(p)))
            pred_sets.append(pset)

        return pred_sets

    def evaluate_coverage(self, X_test: Any, y_test: Any) -> Dict[str, float]:
        """Measure empirical coverage and average set efficiency size."""
        sets = self.predict_sets(X_test)
        y_arr = np.array(y_test)
        covered = [y_arr[i] in sets[i] for i in range(len(y_arr))]
        coverage = float(np.mean(covered))
        avg_size = float(np.mean([len(s) for s in sets]))
        singleton_pct = float(np.mean([1 for s in sets if len(s) == 1]))

        return {
            "target_coverage": 1.0 - self.alpha,
            "empirical_coverage": coverage,
            "average_set_size": avg_size,
            "singleton_rate": singleton_pct,
            "conformal_quantile_qhat": self.q_hat,
        }
