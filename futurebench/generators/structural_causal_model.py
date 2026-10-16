"""Structural Causal Model (SCM) specifications for ground-truth data generation."""

from typing import Dict, List, Tuple
import networkx as nx

class StructuralCausalModel:
    """Represents the structural equations and causal DAG."""

    def __init__(self):
        self.dag = nx.DiGraph()
        self._build_standard_dag()

    def _build_standard_dag(self):
        edges = [
            ("economic_pressure", "demand"),
            ("demand_shock", "demand"),
            ("policy_strength", "demand"),
            ("weather_severity", "supplier_delay_days"),
            ("supplier_reliability", "supplier_delay_days"),
            ("capacity", "available_capacity"),
            ("supplier_reliability", "available_capacity"),
            ("weather_severity", "available_capacity"),
            ("demand", "inventory_gap"),
            ("available_capacity", "inventory_gap"),
            ("inventory_gap", "delivery_delay_days"),
            ("supplier_delay_days", "delivery_delay_days"),
            ("demand", "operating_cost"),
            ("supplier_delay_days", "operating_cost"),
            ("inventory_gap", "operating_cost"),
            ("weather_severity", "operating_cost"),
            ("inventory_gap", "service_level"),
            ("delivery_delay_days", "service_level"),
            ("supplier_reliability", "service_level"),
            ("delivery_delay_days", "disruption_risk"),
            ("inventory_gap", "disruption_risk"),
            ("service_level", "disruption_risk"),
        ]
        self.dag.add_edges_from(edges)

    def get_ground_truth_edges(self) -> set:
        return set(self.dag.edges())
