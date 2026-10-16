"""Research Agent Orchestrator: Drives multi-agent consensus workflows."""

from typing import Any, Dict, List
import pandas as pd
from nexus9.agents.investigator import InvestigatorAgent
from nexus9.agents.hypothesis_scientist import HypothesisScientistAgent
from nexus9.agents.experiment_scientist import ExperimentScientistAgent
from nexus9.agents.red_team import RedTeamReviewerAgent
from nexus9.causal_engine.graph import CausalGraph

class ResearchOrchestrator:
    """Orchestrates autonomous research cycles: investigate -> hypothesize -> plan -> red-team audit."""

    def __init__(self):
        self.investigator = InvestigatorAgent()
        self.hypothesis_scientist = HypothesisScientistAgent()
        self.experiment_scientist = ExperimentScientistAgent()
        self.red_team = RedTeamReviewerAgent()

    def run_investigation_cycle(
        self,
        simulation_df: pd.DataFrame,
        feature_names: List[str],
    ) -> Dict[str, Any]:
        """Execute one full multi-agent collaborative inquiry cycle."""
        # 1. Red team feature audit
        feature_audit = self.red_team.audit_model_features(feature_names)

        # 2. Investigator analyzes data
        evidence = self.investigator.analyze_simulation_run(simulation_df)

        # 3. Hypothesis scientist formulates hypotheses
        hypotheses = self.hypothesis_scientist.formulate_hypotheses(evidence)

        # 4. Experiment scientist creates experiment plan
        plans = [self.experiment_scientist.design_experiment(h) for h in hypotheses]

        # 5. Red team audits claims against causal ground truth
        gt_graph = CausalGraph.load_ground_truth()
        gt_edges = gt_graph.get_edges()

        audited_hypotheses = []
        for h in hypotheses:
            audit = self.red_team.audit_causal_claim(
                {"treatment": h["treatment_variable"], "outcome": h["outcome_variable"]},
                gt_edges,
            )
            audited_hypotheses.append({
                "hypothesis": h,
                "red_team_audit": audit,
            })

        return {
            "status": "COMPLETED",
            "feature_audit": feature_audit,
            "empirical_evidence": evidence,
            "audited_hypotheses": audited_hypotheses,
            "experiment_plans": plans,
        }
