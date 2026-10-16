"""Scenario configuration schemas and loaders."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict
import yaml

@dataclass
class Scenario:
    name: str
    description: str = ""
    regime: str = "normal"
    interventions: Dict[str, Any] = field(default_factory=dict)
    duration_steps: int = 12

    @classmethod
    def from_yaml(cls, path: Path | str) -> "Scenario":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(
            name=data.get("name", "Unnamed"),
            description=data.get("description", ""),
            regime=data.get("regime", "normal"),
            interventions=data.get("interventions", {}),
            duration_steps=data.get("duration_steps", 12),
        )
