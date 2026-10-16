"""Hypothesis Scientist Agent: Formulates testable causal and statistical hypotheses."""

from typing import Any, Dict, List

class HypothesisScientistAgent:
    """Generates falsifiable hypotheses based strictly on observed evidence."""

    def formulate_hypotheses(self, investigation_evidence: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate candidate hypotheses."""
        hypotheses = []
        drivers = investigation_evidence.get("key_drivers", {})

        for var, stats in drivers.items():
            if abs(stats["delta"]) > 0.15:
                direction = "increases" if stats["delta"] > 0 else "decreases"
                hypotheses.append({
                    "id": f"HYP_{var.upper()}",
                    "statement": f"Elevated {var} statistically {direction} the probability of supply chain disruption.",
                    "treatment_variable": var,
                    "outcome_variable": "disruption_risk",
                    "status": "PROPOSED",
                })

        if not hypotheses:
            hypotheses.append({
                "id": "HYP_BASELINE",
                "statement": "System disruption risk is invariant to exogenous shocks under current policy strength.",
                "treatment_variable": "policy_strength",
                "outcome_variable": "disruption_risk",
                "status": "PROPOSED",
            })

        return hypotheses
