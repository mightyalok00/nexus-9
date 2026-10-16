"""Decision policies under computational budget constraints."""

from typing import Any, Dict, List

class ExperimentBudgetPolicy:
    """Allocates experimental steps between exploration (high uncertainty) and exploitation."""

    def __init__(self, total_budget: int = 50, exploration_ratio: float = 0.3):
        self.total_budget = total_budget
        self.exploration_ratio = exploration_ratio
        self.spent_budget = 0

    def allocate_next_batch(self, batch_size: int = 5) -> Dict[str, int]:
        available = min(batch_size, self.total_budget - self.spent_budget)
        if available <= 0:
            return {"explore": 0, "exploit": 0}
        n_explore = int(available * self.exploration_ratio)
        n_exploit = available - n_explore
        self.spent_budget += available
        return {"explore": n_explore, "exploit": n_exploit}
