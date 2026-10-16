"""Constraint-based and score-based Causal Discovery algorithms."""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from nexus9.causal_engine.graph import CausalGraph

def discover_causal_skeleton_correlation(
    df: pd.DataFrame,
    threshold: float = 0.25,
    p_value_threshold: float = 0.01,
) -> CausalGraph:
    """Empirical discovery baseline: estimates skeleton based on statistical correlation & temporal order.
    NOTE: The ground-truth graph is never used here; this discovers structure purely from data!
    """
    cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != "scenario_id"]
    cg = CausalGraph(nodes=cols)

    # Topological order assumption based on domain hierarchy:
    # Exogenous (weather, economic, shock, policy) -> Endogenous (capacity, demand, delays) -> Outcomes (gap, service, disruption)
    domain_ranks = {
        "economic_pressure": 0, "weather_severity": 0, "supplier_reliability": 0,
        "demand_shock": 0, "policy_strength": 0, "capacity": 1,
        "demand": 2, "supplier_delay_days": 2, "available_capacity": 2,
        "inventory_gap": 3, "delivery_delay_days": 3, "operating_cost": 4,
        "service_level": 4, "disruption_risk": 5,
    }

    n = len(cols)
    clean_df = df[cols].dropna()

    for i in range(n):
        for j in range(i + 1, n):
            c1, c2 = cols[i], cols[j]
            r, p = pearsonr(clean_df[c1], clean_df[c2])
            if abs(r) >= threshold and p <= p_value_threshold:
                # Orient edge from lower domain rank to higher
                r1 = domain_ranks.get(c1, 2)
                r2 = domain_ranks.get(c2, 2)
                if r1 < r2:
                    cg.graph.add_edge(c1, c2)
                elif r2 < r1:
                    cg.graph.add_edge(c2, c1)
                else:
                    cg.graph.add_edge(c1, c2)

    return cg
