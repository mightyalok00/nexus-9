"""Interactive sidebar controls for shock interventions."""

import streamlit as st

def render_scenario_controls() -> dict:
    st.sidebar.subheader("Shock & Intervention Controls")
    regime = st.sidebar.selectbox("Environment Regime", ["normal", "demand_surge", "extreme_weather", "supplier_failure"])
    weather = st.sidebar.slider("Weather Severity", 0.0, 1.0, 0.25, 0.05)
    supplier_rel = st.sidebar.slider("Supplier Reliability", 0.0, 1.0, 0.85, 0.05)
    demand_shock = st.sidebar.slider("Demand Shock", -3.0, 3.0, 0.0, 0.2)
    policy_str = st.sidebar.slider("Policy Strength", 0.0, 1.0, 0.5, 0.05)

    return {
        "environment_regime": regime,
        "weather_severity": weather,
        "supplier_reliability": supplier_rel,
        "demand_shock": demand_shock,
        "policy_strength": policy_str,
    }
