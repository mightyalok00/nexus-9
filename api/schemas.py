"""Pydantic schemas for NEXUS-9 REST API."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

class PredictionRequest(BaseModel):
    model_name: str = "boosting_xgboost"
    instances: List[Dict[str, Any]]

class PredictionResponse(BaseModel):
    model_name: str
    predictions: List[int]
    probabilities: Optional[List[float]] = None

class SimulationRequest(BaseModel):
    scenario_name: str = "Custom Intervention"
    duration_steps: int = 12
    interventions: Dict[str, Any] = Field(default_factory=dict)
    n_runs: int = 10

class SimulationResponse(BaseModel):
    scenario_name: str
    n_runs: int
    duration_steps: int
    mean_disruption_rate: float
    mean_service_level: float
    mean_operating_cost: float

class CausalEffectRequest(BaseModel):
    treatment: str = "weather_severity"
    outcome: str = "disruption_risk"
    confounders: List[str] = ["economic_pressure", "policy_strength"]

class CausalEffectResponse(BaseModel):
    treatment: str
    outcome: str
    ate: float
    p_value: float
    ci_lower: float
    ci_upper: float

class ResearchCycleRequest(BaseModel):
    scenario_name: str = "demand_surge"
    n_steps: int = 15

class ResearchCycleResponse(BaseModel):
    status: str
    feature_audit: Dict[str, Any]
    hypotheses_count: int
    summary: str
