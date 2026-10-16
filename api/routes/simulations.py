"""Supply chain world simulator endpoints."""

from fastapi import APIRouter, HTTPException
from api.schemas import SimulationRequest, SimulationResponse
from nexus9.simulator.runner import run_scenario
from nexus9.simulator.scenario import Scenario

router = APIRouter(prefix="/simulations", tags=["Simulations"])

@router.post("/run", response_model=SimulationResponse)
def run_simulation(request: SimulationRequest):
    try:
        scen = Scenario(
            name=request.scenario_name,
            description="API invoked simulation scenario",
            regime="normal",
            interventions=request.interventions,
            duration_steps=request.duration_steps,
        )
        sim_df = run_scenario(scen, n_runs=request.n_runs)

        return SimulationResponse(
            scenario_name=request.scenario_name,
            n_runs=request.n_runs,
            duration_steps=request.duration_steps,
            mean_disruption_rate=float(sim_df["disruption_risk"].mean()),
            mean_service_level=float(sim_df["service_level"].mean()),
            mean_operating_cost=float(sim_df["operating_cost"].mean()),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
