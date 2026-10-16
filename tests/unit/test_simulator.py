"""Unit tests for World Model & Simulator."""

from nexus9.simulator.runner import run_scenario
from nexus9.simulator.scenario import Scenario

def test_simulator_reproducibility():
    scen = Scenario(
        name="Test Scenario",
        description="Unit test scenario",
        regime="normal",
        interventions={"demand_shock": 1.0},
        duration_steps=5,
    )
    df1 = run_scenario(scen, n_runs=2, seed=123)
    df2 = run_scenario(scen, n_runs=2, seed=123)

    assert len(df1) == 10
    assert df1["disruption_risk"].tolist() == df2["disruption_risk"].tolist()
