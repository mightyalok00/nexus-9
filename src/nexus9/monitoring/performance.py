"""Real-time model performance degradation and telemetry tracking."""

from typing import Dict, List
import numpy as np

class PerformanceMonitor:
    """Monitors operational metrics against pre-established baselines."""

    def __init__(self, baseline_f1: float = 0.85, alert_threshold: float = 0.10):
        self.baseline_f1 = baseline_f1
        self.alert_threshold = alert_threshold

    def evaluate_batch(self, current_f1: float) -> Dict[str, Any]:
        drop = self.baseline_f1 - current_f1
        is_alert = drop >= self.alert_threshold
        return {
            "current_f1": current_f1,
            "baseline_f1": self.baseline_f1,
            "performance_drop": float(drop),
            "alert_triggered": is_alert,
        }
