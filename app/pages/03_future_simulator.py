import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
"""Page 03: Future World Simulator, Multi-Echelon Bullwhip, and Gymnasium RL."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from nexus9.simulator.runner import run_scenario
from nexus9.simulator.scenario import Scenario
from nexus9.simulator.multi_echelon import MultiEchelonNetwork
from nexus9.simulator.gym_env import SupplyChainGymEnv
from app.components.charts import create_simulation_trajectory_plot
from app.components.scenario_controls import render_scenario_controls

st.header(" Future World Simulator: Digital Twin, Bullwhip & Gym RL")

tab1, tab2, tab3 = st.tabs([" Macro Scenario Simulation", " Multi-Echelon Bullwhip Network", " Gymnasium RL Agent"])

with tab1:
    controls = render_scenario_controls()
    horizon = st.sidebar.slider("Simulation Horizon (steps)", 5, 30, 12)
    n_runs = st.sidebar.slider("Monte Carlo Runs", 5, 50, 15)

    if st.button(" Run Scenario Simulation"):
        with st.spinner("Running Monte Carlo rollouts..."):
            scen = Scenario(
                name=f"Interactive Regime: {controls['environment_regime']}",
                description="User-configured shock simulation",
                regime=controls["environment_regime"],
                interventions=controls,
                duration_steps=horizon,
            )
            sim_df = run_scenario(scen, n_runs=n_runs)

            col1, col2, col3 = st.columns(3)
            col1.metric("Mean Disruption Rate", f"{sim_df['disruption_risk'].mean():.2%}")
            col2.metric("Mean Service Level", f"{sim_df['service_level'].mean():.1f}%")
            col3.metric("Avg Operating Cost", f"${sim_df['operating_cost'].mean():,.2f}")

            st.plotly_chart(create_simulation_trajectory_plot(sim_df), use_container_width=True)

with tab2:
    st.subheader(" 4-Echelon Network: Tier-2 -> Tier-1 -> Central DC -> Retail Outlets")
    st.markdown("Simulates order amplification (the **Bullwhip Effect**) across upstream supply tiers using classical $(s, S)$ policies.")

    steps = st.slider("Simulation Steps", 10, 50, 25)
    demand_volatility = st.slider("Customer Demand Volatility (Std Dev)", 5.0, 30.0, 15.0)

    if st.button("Run Multi-Echelon Simulation", key="echelon_btn"):
        net = MultiEchelonNetwork()
        for t in range(steps):
            d = max(10.0, float(np.random.normal(80.0, demand_volatility)))
            net.step(customer_demand=d)

        df_net = pd.DataFrame(net.history)
        df_net["step"] = range(1, len(df_net) + 1)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_net["step"], y=df_net["customer_demand"], name="1. Customer Demand", line=dict(color="#636EFA", width=2)))
        fig.add_trace(go.Scatter(x=df_net["step"], y=df_net["retail_order"], name="2. Retail -> DC Order", line=dict(color="#00CC96", width=2)))
        fig.add_trace(go.Scatter(x=df_net["step"], y=df_net["dc_order"], name="3. DC -> Tier-1 Order", line=dict(color="#FFA15A", width=2)))
        fig.add_trace(go.Scatter(x=df_net["step"], y=df_net["tier1_order"], name="4. Tier-1 -> Tier-2 Order", line=dict(color="#EF553B", width=3)))

        fig.update_layout(title="Bullwhip Effect Order Amplification Across Echelons", xaxis_title="Time Period", yaxis_title="Units Ordered")
        st.plotly_chart(fig, use_container_width=True)

        avg_bullwhip = df_net["bullwhip_ratio"].mean()
        st.metric("Upstream Bullwhip Amplification Ratio", f"{avg_bullwhip:.2f}x")

with tab3:
    st.subheader(" Gymnasium Reinforcement Learning Environment")
    st.markdown("`SupplyChainGymEnv` standard interface for PPO/SAC autonomous policy learning.")

    if st.button("Run Sample Gym RL Agent Rollout", key="gym_btn"):
        env = SupplyChainGymEnv(seed=20261016)
        obs, _ = env.reset()

        rollout_records = []
        total_reward = 0.0

        for step in range(20):
            # Dynamic policy action
            action = np.array([np.sin(step / 3.0) * 0.8], dtype=np.float32)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            rollout_records.append({
                "step": step + 1,
                "action_buffer_adj": float(action[0]),
                "reward": reward,
                "operating_cost": info["operating_cost"],
                "disrupted": int(info["is_disrupted"]),
            })

        df_rl = pd.DataFrame(rollout_records)
        c1, c2 = st.columns(2)
        c1.metric("Cumulative Episode Reward", f"{total_reward:,.1f}")
        c2.metric("Episode Disruption Rate", f"{df_rl['disrupted'].mean():.2%}")
        st.dataframe(df_rl, use_container_width=True)
