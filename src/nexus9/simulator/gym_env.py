"""Gymnasium Reinforcement Learning Environment for Autonomous Policy Training."""

from typing import Any, Dict, Optional, Tuple
import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
    HAS_GYM = True
except ImportError:
    HAS_GYM = False
    # Fallback dummy class
    class gym:
        class Env:
            pass

class SupplyChainGymEnv(gym.Env if HAS_GYM else object):
    """Custom Gym Environment:
    Action: Continuous safety buffer adjustments [-1.0, 1.0]
    Observation: [economic_pressure, weather_severity, inventory_on_hand, demand, supplier_delay]
    Reward: -operating_cost - 50 * disruption_penalty
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, seed: int = 20261016):
        super().__init__()
        self.rng = np.random.default_rng(seed)
        self.max_steps = 20
        self.current_step = 0

        if HAS_GYM:
            # Continuous action: buffer capacity scaling
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
            # Observations: 5 continuous metrics
            self.observation_space = spaces.Box(
                low=np.array([-5.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
                high=np.array([5.0, 1.0, 500.0, 300.0, 60.0], dtype=np.float32),
                dtype=np.float32,
            )

        self.state = np.zeros(5, dtype=np.float32)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.current_step = 0
        self.state = np.array([
            self.rng.normal(0.0, 1.0),   # Economic pressure
            self.rng.uniform(0.1, 0.4),  # Weather severity
            100.0,                        # Inventory
            80.0,                         # Demand
            2.0,                          # Delay days
        ], dtype=np.float32)
        return self.state, {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        self.current_step += 1
        buffer_adj = float(action[0]) * 20.0

        # State transition
        econ, weather, inv, dem, delay = self.state
        new_econ = econ + self.rng.normal(0, 0.1)
        new_weather = float(np.clip(weather + self.rng.normal(0, 0.05), 0.0, 1.0))
        new_dem = float(max(10.0, 80.0 + 10.0 * new_econ + self.rng.normal(0, 5)))
        new_delay = float(max(0.0, 12.0 * new_weather + self.rng.exponential(1.0)))

        effective_capacity = max(10.0, 100.0 + buffer_adj - 30.0 * new_weather)
        gap = max(0.0, new_dem - effective_capacity)
        new_inv = max(0.0, inv + effective_capacity - new_dem)

        is_disrupted = bool((gap > 20.0) or (new_delay > 6.0))
        operating_cost = 1000.0 + 15.0 * gap + 40.0 * new_delay + abs(buffer_adj) * 10.0
        reward = -operating_cost - (5000.0 if is_disrupted else 0.0)

        self.state = np.array([new_econ, new_weather, new_inv, new_dem, new_delay], dtype=np.float32)
        terminated = bool(self.current_step >= self.max_steps)
        truncated = False

        info = {
            "step": self.current_step,
            "gap": gap,
            "is_disrupted": is_disrupted,
            "operating_cost": operating_cost,
        }
        return self.state, float(reward), terminated, truncated, info
