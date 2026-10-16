"""Investigator Agent: Mines metrics, tracks anomalies, and uncovers empirical anomalies."""

from typing import Any, Dict, List
import pandas as pd
import numpy as np

class InvestigatorAgent:
    """Investigates dataset shifts, model degradation, and simulation anomalies."""

    def __init__(self, anomaly_threshold_std: float = 2.5):
        self.anomaly_threshold_std = anomaly_threshold_std

    def analyze_simulation_run(self, sim_df: pd.DataFrame) -> Dict[str, Any]:
        """Extract empirical evidence of disruption triggers from simulation data."""
        disrupted = sim_df[sim_df["disruption_risk"] == 1]
        stable = sim_df[sim_df["disruption_risk"] == 0]

        evidence = {
            "total_steps": len(sim_df),
            "disruption_rate": float(len(disrupted) / max(1, len(sim_df))),
            "key_drivers": {},
        }

        numeric_cols = ["weather_severity", "supplier_reliability", "demand_shock", "inventory_gap", "delivery_delay_days"]
        for col in numeric_cols:
            if col in sim_df.columns:
                mean_dis = float(disrupted[col].mean()) if len(disrupted) > 0 else 0.0
                mean_stb = float(stable[col].mean()) if len(stable) > 0 else 0.0
                evidence["key_drivers"][col] = {
                    "disrupted_mean": mean_dis,
                    "stable_mean": mean_stb,
                    "delta": mean_dis - mean_stb,
                }

        return evidence
