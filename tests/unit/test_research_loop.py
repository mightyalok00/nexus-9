"""Unit test for AutonomousResearchLoop self-correction."""

from nexus9.agents.research_loop import AutonomousResearchLoop
from nexus9.simulator.runner import run_scenario
from nexus9.simulator.scenario import Scenario
from nexus9.config import settings

def test_self_correcting_research_loop():
    scen = Scenario(name="Test Loop", description="Shock", regime="normal", duration_steps=4)
    sim_df = run_scenario(scen, n_runs=2)

    loop = AutonomousResearchLoop(max_iterations=2)
    feature_names = settings.features.numerical_features + settings.features.categorical_features
    res = loop.run_self_correcting_cycle(sim_df, feature_names)

    assert res["status"] in ("APPROVED", "CHALLENGED")
    assert len(res["cycle_history"]) >= 1
    assert "ledger_entry" in res
