"""Stateful simulation environment for supply network trajectories."""

import numpy as np
from typing import Dict, List, Optional
from nexus9.world_model.state import WorldState
from nexus9.world_model.transition import transition_state

class SimulationEnvironment:
    """Interactive supply chain environment."""

    def __init__(self, initial_state: Optional[WorldState] = None, seed: int = 20261016):
        self.rng = np.random.default_rng(seed)
        self.initial_state = initial_state or WorldState()
        self.current_state = self.initial_state
        self.history: List[WorldState] = [self.initial_state]

    def reset(self) -> WorldState:
        self.current_state = self.initial_state
        self.history = [self.initial_state]
        return self.current_state

    def step(self, interventions: Optional[Dict] = None) -> WorldState:
        nxt = transition_state(self.current_state, self.rng, interventions=interventions)
        self.current_state = nxt
        self.history.append(nxt)
        return nxt
