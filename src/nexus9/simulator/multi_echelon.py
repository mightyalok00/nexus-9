"""Multi-echelon supply network simulation: Tier-2 -> Tier-1 -> Central DC -> Regional Outlets.
Models Bullwhip effect and classic (s, S) & (r, Q) replenishment policies.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple
import numpy as np

@dataclass
class EchelonState:
    name: str
    inventory_on_hand: float = 100.0
    inventory_in_transit: float = 0.0
    backorders: float = 0.0
    lead_time_steps: int = 2
    safety_stock: float = 30.0
    reorder_point_s: float = 50.0
    order_up_to_S: float = 120.0

class MultiEchelonNetwork:
    """4-Echelon industrial supply network."""

    def __init__(self, seed: int = 20261016):
        self.rng = np.random.default_rng(seed)
        self.retail = EchelonState(name="Retail Outlets", lead_time_steps=1, reorder_point_s=40, order_up_to_S=100)
        self.dc = EchelonState(name="Central DC", lead_time_steps=2, reorder_point_s=80, order_up_to_S=200)
        self.tier1 = EchelonState(name="Tier-1 Assembly", lead_time_steps=3, reorder_point_s=150, order_up_to_S=300)
        self.tier2 = EchelonState(name="Tier-2 Raw Materials", lead_time_steps=2, reorder_point_s=200, order_up_to_S=400)
        self.history: List[Dict] = []

    def step(self, customer_demand: float) -> Dict[str, Any]:
        """Advance network one period with (s, S) replenishment and pipeline delay."""
        # 1. Retail echelon
        retail_demand = customer_demand
        net_retail = self.retail.inventory_on_hand - retail_demand - self.retail.backorders
        if net_retail >= 0:
            self.retail.inventory_on_hand = net_retail
            self.retail.backorders = 0.0
            retail_service = 100.0
        else:
            self.retail.backorders = abs(net_retail)
            self.retail.inventory_on_hand = 0.0
            retail_service = max(0.0, 100.0 - (self.retail.backorders / max(1.0, retail_demand)) * 100)

        # Retail (s, S) order to DC
        inventory_position_retail = self.retail.inventory_on_hand + self.retail.inventory_in_transit - self.retail.backorders
        order_to_dc = max(0.0, self.retail.order_up_to_S - inventory_position_retail) if inventory_position_retail <= self.retail.reorder_point_s else 0.0

        # 2. Central DC
        dc_demand = order_to_dc
        net_dc = self.dc.inventory_on_hand - dc_demand
        self.dc.inventory_on_hand = max(0.0, net_dc)
        order_to_tier1 = max(0.0, self.dc.order_up_to_S - (self.dc.inventory_on_hand + self.dc.inventory_in_transit)) if self.dc.inventory_on_hand <= self.dc.reorder_point_s else 0.0

        # 3. Tier-1
        t1_demand = order_to_tier1
        net_t1 = self.tier1.inventory_on_hand - t1_demand
        self.tier1.inventory_on_hand = max(0.0, net_t1)
        order_to_tier2 = max(0.0, self.tier1.order_up_to_S - self.tier1.inventory_on_hand) if self.tier1.inventory_on_hand <= self.tier1.reorder_point_s else 0.0

        record = {
            "customer_demand": customer_demand,
            "retail_order": order_to_dc,
            "dc_order": order_to_tier1,
            "tier1_order": order_to_tier2,
            "retail_inventory": self.retail.inventory_on_hand,
            "retail_service_level": retail_service,
            "bullwhip_ratio": float((order_to_tier2 + 1e-4) / (customer_demand + 1e-4)),
        }
        self.history.append(record)
        return record
