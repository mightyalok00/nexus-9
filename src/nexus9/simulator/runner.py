"""Scenario execution engine and Monte Carlo rollouts."""

from typing import Dict, List
import pandas as pd
from nexus9.simulator.environment import SimulationEnvironment
from nexus9.simulator.scenario import Scenario
from nexus9.world_model.state import WorldState

def run_scenario(
    scenario: Scenario,
    n_runs: int = 25,
    seed: int = 20261016,
) -> pd.DataFrame:
    """Execute Monte Carlo replications for a given scenario."""
    records: List[Dict] = []

    for run_idx in range(n_runs):
        env = SimulationEnvironment(seed=seed + run_idx)
        for t in range(scenario.duration_steps):
            state = env.step(interventions=scenario.interventions)
            rec = state.to_dict()
            rec["run_id"] = run_idx
            rec["scenario_name"] = scenario.name
            records.append(rec)

    return pd.DataFrame(records)
