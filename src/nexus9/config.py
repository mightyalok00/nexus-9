"""Global configuration management for NEXUS-9."""

import os
from pathlib import Path
from typing import Any, Dict, List
import yaml
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

class PathSettings(BaseModel):
    raw_data: str = "datasets/raw/nexus9_futurebench.csv"
    metadata_dir: str = "datasets/metadata"
    causal_graph: str = "datasets/metadata/ground_truth_causal_graph.json"
    data_dictionary: str = "datasets/metadata/data_dictionary.csv"
    evidence_ledger: str = "reports/experiments/ledger.json"
    model_artifacts: str = "reports/model_cards"

class FeatureSettings(BaseModel):
    target_column: str = "disruption_risk"
    id_column: str = "scenario_id"
    leakage_columns: List[str] = [
        "delivery_delay_days",
        "inventory_gap",
        "service_level",
    ]
    categorical_features: List[str] = ["environment_regime"]
    numerical_features: List[str] = [
        "economic_pressure",
        "weather_severity",
        "supplier_reliability",
        "demand_shock",
        "policy_strength",
        "capacity",
        "demand",
        "supplier_delay_days",
        "available_capacity",
        "operating_cost",
    ]

class Settings(BaseModel):
    project_name: str = "nexus-9"
    version: str = "0.1.0"
    random_seed: int = 20261016
    log_level: str = "INFO"
    paths: PathSettings = Field(default_factory=PathSettings)
    features: FeatureSettings = Field(default_factory=FeatureSettings)

    @classmethod
    def load_from_yaml(cls, config_path: str | Path | None = None) -> "Settings":
        if config_path is None:
            config_path = PROJECT_ROOT / "configs" / "default.yaml"
        config_path = Path(config_path)
        if not config_path.exists():
            return cls()

        with open(config_path, "r", encoding="utf-8") as f:
            raw_data = yaml.safe_load(f) or {}

        system_cfg = raw_data.get("system", {})
        return cls(
            project_name=system_cfg.get("project_name", "nexus-9"),
            version=system_cfg.get("version", "0.1.0"),
            random_seed=system_cfg.get("random_seed", 20261016),
            log_level=system_cfg.get("log_level", "INFO"),
            paths=PathSettings(**raw_data.get("paths", {})),
            features=FeatureSettings(**raw_data.get("features", {})),
        )

# Global singleton
settings = Settings.load_from_yaml()
