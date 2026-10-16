"""Decision Evaluation: Regret, EOL, and Value of Information metrics."""

from typing import Dict
import numpy as np

def compute_decision_regret_metrics(
    chosen_loss: float,
    optimal_loss: float,
    worst_case_loss: float,
) -> Dict[str, float]:
    """Evaluate the quality of a chosen policy compared to oracle and adversary."""
    regret = max(0.0, chosen_loss - optimal_loss)
    suboptimality_ratio = (chosen_loss / max(1e-6, optimal_loss)) - 1.0
    adversarial_gap = worst_case_loss - chosen_loss

    return {
        "absolute_regret": float(regret),
        "suboptimality_ratio": float(suboptimality_ratio),
        "adversarial_gap": float(adversarial_gap),
    }
