"""Causal inference and autonomous agent research cycle endpoints."""

from fastapi import APIRouter, HTTPException
from api.schemas import (
    CausalEffectRequest,
    CausalEffectResponse,
    ResearchCycleRequest,
    ResearchCycleResponse,
)
from nexus9.agents.orchestrator import ResearchOrchestrator
from nexus9.causal_engine.effects import estimate_average_treatment_effect
from nexus9.config import settings
from nexus9.data.loader import load_futurebench_data
from nexus9.simulator.runner import run_scenario
from nexus9.simulator.scenario import Scenario

router = APIRouter(prefix="/experiments", tags=["Experiments"])

@router.post("/causal-effect", response_model=CausalEffectResponse)
def compute_causal_effect(request: CausalEffectRequest):
    try:
        df = load_futurebench_data()
        res = estimate_average_treatment_effect(
            df=df,
            treatment=request.treatment,
            outcome=request.outcome,
            confounders=request.confounders,
        )
        return CausalEffectResponse(**res)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/run-research-cycle", response_model=ResearchCycleResponse)
def run_research_cycle(request: ResearchCycleRequest):
    try:
        # 1. Run simulation
        scen = Scenario(
            name=request.scenario_name,
            description="Trigger for autonomous research inquiry",
            regime="demand_surge",
            interventions={"economic_pressure": 1.5, "demand_shock": 2.0},
            duration_steps=request.n_steps,
        )
        sim_df = run_scenario(scen, n_runs=15)

        # 2. Orchestrator coordinates agents
        orchestrator = ResearchOrchestrator()
        feature_names = settings.features.numerical_features + settings.features.categorical_features
        result = orchestrator.run_investigation_cycle(sim_df, feature_names)

        return ResearchCycleResponse(
            status=result["status"],
            feature_audit=result["feature_audit"],
            hypotheses_count=len(result["audited_hypotheses"]),
            summary=f"Investigated {len(sim_df)} scenario steps. Disruption rate: {result['empirical_evidence']['disruption_rate']:.2%}.",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
