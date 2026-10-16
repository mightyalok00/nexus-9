"""Automated extreme stress-testing suite for systemic fragility identification."""

import pandas as pd
from nexus9.simulator.runner import run_scenario
from nexus9.simulator.scenario import Scenario

def run_stress_test_suite() -> pd.DataFrame:
    """Run simultaneous multi-factor breakdown tests."""
    severe_scenario = Scenario(
        name="Compound Black Swan Disruption",
        description="Simultaneous supplier failure, demand shock, and weather freeze",
        regime="extreme_weather",
        interventions={
            "weather_severity": 0.98,
            "supplier_reliability": 0.1,
            "demand_shock": 3.0,
            "economic_pressure": 2.0,
        },
        duration_steps=10,
    )
    df = run_scenario(severe_scenario, n_runs=10)
    return df
