"""Red-Team Reviewer Agent: Audits evidence for data leakage, causal fallacies, and overconfidence."""

from typing import Any, Dict, List
from nexus9.config import settings

class RedTeamReviewerAgent:
    """Adversarial auditor: challenges claims, detects data leakage and spurious correlations."""

    def audit_model_features(self, feature_list: List[str]) -> Dict[str, Any]:
        """Check for leakage columns: delivery_delay_days, inventory_gap, service_level, scenario_id."""
        leaked = [col for col in feature_list if col in settings.features.leakage_columns or col == settings.features.id_column]
        passed = len(leaked) == 0
        return {
            "check": "data_leakage_audit",
            "passed": passed,
            "leaked_columns": leaked,
            "verdict": "APPROVED" if passed else "REJECTED - Leaks target definition",
        }

    def audit_causal_claim(self, claim: Dict[str, Any], ground_truth_edges: set) -> Dict[str, Any]:
        """Audit whether a discovered causal edge actually aligns with ground truth."""
        edge = (claim.get("treatment"), claim.get("outcome"))
        is_true_edge = edge in ground_truth_edges
        return {
            "check": "ground_truth_alignment",
            "proposed_edge": edge,
            "ground_truth_supported": is_true_edge,
            "verdict": "VERIFIED" if is_true_edge else "CHALLENGED - Potential spurious correlation or reverse causality",
        }
