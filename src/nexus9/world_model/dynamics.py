"""System dynamics, feedback loops, and differential response functions."""

from typing import List
from nexus9.world_model.state import WorldState

class SystemDynamics:
    """Analyzes momentum, stability, and trajectory oscillations."""

    @staticmethod
    def calculate_resilience_index(trajectory: List[WorldState]) -> float:
        """Compute normalized recovery resilience score based on service level restoration."""
        if not trajectory:
            return 1.0
        service_levels = [s.service_level for s in trajectory]
        min_sl = min(service_levels)
        final_sl = service_levels[-1]
        if min_sl >= 95.0:
            return 1.0
        recovery = (final_sl - min_sl) / (100.0 - min_sl + 1e-6)
        return float(min(1.0, max(0.0, recovery)))
