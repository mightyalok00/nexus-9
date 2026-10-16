"""Intervention operators for causal do-calculus and scenario shock injection."""

from typing import Any, Dict
from nexus9.world_model.state import WorldState

class Intervention:
    """Represents a do(X = x) policy intervention."""
    def __init__(self, target_variable: str, value: Any):
        self.target_variable = target_variable
        self.value = value

    def apply(self, state: WorldState) -> WorldState:
        if hasattr(state, self.target_variable):
            setattr(state, self.target_variable, self.value)
        return state
