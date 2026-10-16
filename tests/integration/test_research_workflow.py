"""Integration test for Autonomous Research Cycle and Evidence Ledger."""

from nexus9.agents.orchestrator import ResearchOrchestrator
from nexus9.evidence.ledger import EvidenceLedger
from nexus9.simulator.runner import run_scenario
from nexus9.simulator.scenario import Scenario
from nexus9.config import settings

def test_research_workflow_ledger_recording(tmp_path):
    scen = Scenario(
        name="Integration Test Shock",
        description="Shock scenario",
        regime="demand_surge",
        interventions={"economic_pressure": 1.2},
        duration_steps=6,
    )
    sim_df = run_scenario(scen, n_runs=5)

    orchestrator = ResearchOrchestrator()
    feature_names = settings.features.numerical_features + settings.features.categorical_features
    result = orchestrator.run_investigation_cycle(sim_df, feature_names)

    assert result["status"] == "COMPLETED"
    assert result["feature_audit"]["passed"] is True

    # Record into ledger
    ledger_file = tmp_path / "test_ledger.json"
    ledger = EvidenceLedger(ledger_file=ledger_file)
    entry = ledger.record_entry(
        experiment_type="Autonomous Research Cycle",
        parameters={"scenario": scen.name},
        metrics={"disruption_rate": result["empirical_evidence"]["disruption_rate"]},
        agent_notes="Integration test passed",
    )

    assert entry["hash"] is not None
    assert len(ledger.get_all_entries()) == 1
