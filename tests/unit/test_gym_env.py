"""Unit test for SupplyChainGymEnv."""

import numpy as np
from nexus9.simulator.gym_env import SupplyChainGymEnv

def test_gym_env_step_and_reset():
    env = SupplyChainGymEnv(seed=42)
    obs, info = env.reset()
    assert len(obs) == 5

    action = np.array([0.5], dtype=np.float32)
    next_obs, reward, terminated, truncated, step_info = env.step(action)
    assert len(next_obs) == 5
    assert isinstance(reward, float)
    assert "operating_cost" in step_info
