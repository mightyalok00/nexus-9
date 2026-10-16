"""State definitions and invariants for the supply chain world model."""

from dataclasses import dataclass, field
from typing import Dict, Any

@dataclass
class WorldState:
    """Encapsulates snapshot state of the industrial supply network."""
    step: int = 0
    environment_regime: str = "normal"
    economic_pressure: float = 0.0
    weather_severity: float = 0.0
    supplier_reliability: float = 1.0
    demand_shock: float = 0.0
    policy_strength: float = 0.5
    capacity: float = 100.0
    demand: float = 80.0
    supplier_delay_days: float = 0.0
    available_capacity: float = 100.0
    inventory_gap: float = 0.0
    delivery_delay_days: float = 0.0
    operating_cost: float = 1000.0
    service_level: float = 100.0
    disruption_risk: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "environment_regime": self.environment_regime,
            "economic_pressure": self.economic_pressure,
            "weather_severity": self.weather_severity,
            "supplier_reliability": self.supplier_reliability,
            "demand_shock": self.demand_shock,
            "policy_strength": self.policy_strength,
            "capacity": self.capacity,
            "demand": self.demand,
            "supplier_delay_days": self.supplier_delay_days,
            "available_capacity": self.available_capacity,
            "inventory_gap": self.inventory_gap,
            "delivery_delay_days": self.delivery_delay_days,
            "operating_cost": self.operating_cost,
            "service_level": self.service_level,
            "disruption_risk": self.disruption_risk,
        }
