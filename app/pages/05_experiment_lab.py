import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
"""Page 05: Autonomous Research Agents Collaboration & Multi-Turn Self-Correction Loop."""

import streamlit as st
import pandas as pd
from nexus9.agents.research_loop import AutonomousResearchLoop
from nexus9.simulator.runner import run_scenario
from nexus9.simulator.scenario import Scenario
from nexus9.config import settings

st.header(" Autonomous Research Agents: Self-Correcting Inquiry Loop")

st.markdown("""
NEXUS-9 deploys a cooperative 4-agent research society:
1. **Investigator**: Mines empirical patterns and statistical anomalies.
2. **Hypothesis Scientist**: Formulates falsifiable structural hypotheses.
3. **Experiment Scientist**: Designs controlled intervention protocols.
4. **Red-Team Reviewer**: Adversarial auditor challenging data leakage and spurious correlation.
""")

max_iter = st.slider("Maximum Iterations for Red-Team Consensus", 1, 5, 3)

if st.button(" Execute Autonomous Research Inquiry Loop", key="run_loop_btn"):
    with st.spinner("Agents conducting collaborative inquiry loop..."):
        scen = Scenario(
            name="Research Shock Scenario",
            description="Autonomous inquiry trigger",
            regime="extreme_weather",
            interventions={"weather_severity": 0.85, "demand_shock": 2.0},
            duration_steps=12,
        )
        sim_df = run_scenario(scen, n_runs=15)

        loop = AutonomousResearchLoop(max_iterations=max_iter)
        feature_names = settings.features.numerical_features + settings.features.categorical_features
        results = loop.run_self_correcting_cycle(sim_df, feature_names)

        st.success(f"Final Loop Status: **{results['status']}** (Completed {len(results['cycle_history'])} round(s))")

        for round_info in results["cycle_history"]:
            with st.expander(f"Inquiry Round #{round_info['iteration']} - Audit Status: {'APPROVED' if round_info['approved'] else 'CHALLENGED'}"):
                st.write(f"**Hypotheses Generated:** {round_info['hypotheses_count']}")
                st.info(f"**Red-Team Statement:** {round_info['red_team_statement']}")
                st.json(round_info["feature_audit"])

        st.subheader(" Tamper-Evident Ledger Block Recorded")
        st.json(results["ledger_entry"])
