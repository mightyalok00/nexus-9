"""Causal Discovery Evaluation: Structural Hamming Distance (SHD), Precision, Recall, and F1."""

from typing import Dict, Set, Tuple

def evaluate_causal_discovery(
    predicted_edges: Set[Tuple[str, str]],
    ground_truth_edges: Set[Tuple[str, str]],
) -> Dict[str, float]:
    """Compute causal discovery evaluation metrics against known synthetic ground truth.

    Args:
        predicted_edges: Set of (source, target) tuples inferred by discovery algorithm.
        ground_truth_edges: Set of true (source, target) tuples.

    Returns:
        Dict with SHD, Precision, Recall, and F1 score.
    """
    tp = len(predicted_edges.intersection(ground_truth_edges))
    fp = len(predicted_edges - ground_truth_edges)
    fn = len(ground_truth_edges - predicted_edges)

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    # Structural Hamming Distance = False Positives + False Negatives
    shd = fp + fn

    return {
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "shd": int(shd),
    }
