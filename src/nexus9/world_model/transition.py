"""Deterministic and stochastic state transition functions."""

import numpy as np
from nexus9.world_model.state import WorldState

def transition_state(
    current: WorldState,
    rng: np.random.Generator,
    interventions: dict | None = None,
) -> WorldState:
    """Compute next world state using structural causal dynamics."""
    interventions = interventions or {}
    nxt = WorldState(step=current.step + 1)

    # 1. Exogenous variables with optional policy/intervention overrides
    nxt.environment_regime = interventions.get("environment_regime", current.environment_regime)
    nxt.economic_pressure = interventions.get("economic_pressure", current.economic_pressure + rng.normal(0, 0.1))
    nxt.weather_severity = float(np.clip(
        interventions.get("weather_severity", current.weather_severity + rng.normal(0, 0.05)),
        0.0, 1.0
    ))
    nxt.supplier_reliability = float(np.clip(
        interventions.get("supplier_reliability", current.supplier_reliability + rng.normal(0, 0.03)),
        0.0, 1.0
    ))
    nxt.demand_shock = interventions.get("demand_shock", float(rng.normal(0, 0.5)))
    nxt.policy_strength = float(np.clip(
        interventions.get("policy_strength", current.policy_strength),
        0.0, 1.0
    ))

    # 2. Endogenous variables
    base_cap = interventions.get("capacity", current.capacity)
    nxt.capacity = max(10.0, base_cap)

    # Demand: influenced by economic pressure, demand shock, policy strength
    raw_demand = 80.0 + 12.0 * nxt.economic_pressure + 8.0 * nxt.demand_shock - 6.0 * nxt.policy_strength
    nxt.demand = max(5.0, float(raw_demand))

    # Supplier delay days: driven by weather and supplier unreliability
    raw_sup_delay = 15.0 * nxt.weather_severity * (1.0 - nxt.supplier_reliability) + rng.exponential(1.0)
    nxt.supplier_delay_days = max(0.0, float(raw_sup_delay))

    # Available capacity: base capacity penalized by weather and supplier reliability
    avail_factor = (0.3 + 0.7 * nxt.supplier_reliability) * (1.0 - 0.4 * nxt.weather_severity)
    nxt.available_capacity = max(5.0, float(nxt.capacity * avail_factor))

    # Inventory gap: demand minus available capacity, floored at 0
    nxt.inventory_gap = max(0.0, float(nxt.demand - nxt.available_capacity))

    # Delivery delay days: driven by inventory gap and supplier delays
    raw_del_delay = 0.18 * nxt.inventory_gap + 0.4 * nxt.supplier_delay_days + rng.normal(0, 0.2)
    nxt.delivery_delay_days = max(0.0, float(raw_del_delay))

    # Operating cost: inventory holding/shortage + supplier delays + weather mitigation
    nxt.operating_cost = float(1000.0 + 15.0 * nxt.inventory_gap + 45.0 * nxt.supplier_delay_days + 80.0 * nxt.weather_severity)

    # Service level: clipped to 0 - 100
    nxt.service_level = float(np.clip(100.0 - 0.7 * nxt.inventory_gap - 2.5 * nxt.delivery_delay_days, 0.0, 100.0))

    # Disruption risk: true if high delivery delay, severe inventory gap, or service level collapse
    nxt.disruption_risk = int(
        (nxt.delivery_delay_days > 5.0) or
        (nxt.inventory_gap > 25.0) or
        (nxt.service_level < 85.0)
    )

    return nxt
