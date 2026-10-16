"""CLI script to execute Future World simulator scenarios."""

from pathlib import Path
from nexus9.simulator.runner import run_scenario
from nexus9.simulator.scenario import Scenario

def main():
    print("=== NEXUS-9 Future Simulator: Scenario Execution ===")
    scen_path = Path("futurebench/scenarios/extreme_weather.yaml")
    scen = Scenario.from_yaml(scen_path)
    print(f"Running scenario: '{scen.name}' ({scen.description})")

    df = run_scenario(scen, n_runs=20)
    print(f"Completed {len(df)} total simulation steps across 20 runs.")
    print(f"Mean Disruption Risk: {df['disruption_risk'].mean():.2%}")
    print(f"Mean Service Level:   {df['service_level'].mean():.1f}%")
    print(f"Mean Operating Cost:  ${df['operating_cost'].mean():,.2f}")

    out_dir = Path("reports/experiments")
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "simulation_extreme_weather_results.csv", index=False)
    print(f"Simulation trajectory saved to: {out_dir / 'simulation_extreme_weather_results.csv'}")

if __name__ == "__main__":
    main()
