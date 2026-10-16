"""Synthetic supply chain scenario generator for FutureBench."""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd

def generate_futurebench_dataset(
    n_samples: int = 12000,
    seed: int = 20261016,
) -> pd.DataFrame:
    """Synthesizes a controlled FutureBench dataset adhering to structural causal laws."""
    rng = np.random.default_rng(seed)

    # 1. Exogenous variables
    economic_pressure = rng.normal(0.0, 1.0, size=n_samples)
    weather_severity = rng.uniform(0.0, 1.0, size=n_samples)

    # Missingness introduced deliberately into supplier_reliability (e.g. 5%)
    supplier_reliability = rng.beta(5.0, 1.5, size=n_samples)
    missing_mask = rng.uniform(0.0, 1.0, size=n_samples) < 0.05
    supplier_reliability_col = np.where(missing_mask, np.nan, supplier_reliability)

    demand_shock = rng.normal(0.0, 1.0, size=n_samples)
    policy_strength = rng.uniform(0.0, 1.0, size=n_samples)
    capacity = rng.normal(100.0, 15.0, size=n_samples).clip(40.0, 180.0)

    # Environment regime derived from weather & economic pressure
    regime = []
    for w, e in zip(weather_severity, economic_pressure):
        if w > 0.75:
            regime.append("extreme_weather")
        elif e > 1.2:
            regime.append("demand_surge")
        else:
            regime.append("normal")

    # 2. Endogenous variables
    demand = (80.0 + 12.0 * economic_pressure + 8.0 * demand_shock - 6.0 * policy_strength + rng.normal(0, 3, n_samples)).clip(10.0, 220.0)

    # Supplier delay days
    sup_rel_effective = np.nan_to_num(supplier_reliability_col, nan=0.75)
    supplier_delay_days = (14.0 * weather_severity * (1.0 - sup_rel_effective) + rng.exponential(1.5, n_samples)).clip(0.0, 45.0)

    # Available capacity
    avail_factor = (0.25 + 0.75 * sup_rel_effective) * (1.0 - 0.45 * weather_severity)
    available_capacity = (capacity * avail_factor + rng.normal(0, 2, n_samples)).clip(5.0, 200.0)

    # Inventory gap: demand minus available capacity, floored at 0
    inventory_gap = np.maximum(0.0, demand - available_capacity)

    # Delivery delay days
    delivery_delay_days = (0.22 * inventory_gap + 0.35 * supplier_delay_days + rng.normal(0, 0.5, n_samples)).clip(0.0, 50.0)

    # Operating cost with small extreme outlier tail
    base_cost = 1000.0 + 18.0 * inventory_gap + 40.0 * supplier_delay_days + 75.0 * weather_severity + rng.normal(0, 50, n_samples)
    outlier_mask = rng.uniform(0.0, 1.0, size=n_samples) < 0.01
    operating_cost = np.where(outlier_mask, base_cost * rng.uniform(3.0, 6.0, size=n_samples), base_cost)

    # Service level (0 - 100)
    service_level = (100.0 - 0.75 * inventory_gap - 2.8 * delivery_delay_days + rng.normal(0, 1.0, n_samples)).clip(0.0, 100.0)

    # Target: disruption_risk derived from outcome variables
    disruption_risk = (
        (delivery_delay_days > 5.0) |
        (inventory_gap > 25.0) |
        (service_level < 85.0)
    ).astype(int)

    df = pd.DataFrame({
        "scenario_id": np.arange(1, n_samples + 1),
        "environment_regime": regime,
        "economic_pressure": economic_pressure,
        "weather_severity": weather_severity,
        "supplier_reliability": supplier_reliability_col,
        "demand_shock": demand_shock,
        "policy_strength": policy_strength,
        "capacity": capacity,
        "demand": demand,
        "supplier_delay_days": supplier_delay_days,
        "available_capacity": available_capacity,
        "inventory_gap": inventory_gap,
        "delivery_delay_days": delivery_delay_days,
        "operating_cost": operating_cost,
        "service_level": service_level,
        "disruption_risk": disruption_risk,
    })

    return df
