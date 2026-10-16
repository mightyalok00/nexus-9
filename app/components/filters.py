"""Interactive Multi-Dimensional Filter Controls for FutureBench Explorer."""

import streamlit as st
import pandas as pd
import numpy as np

def render_data_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Render rich interactive sidebar filters without encoding corruption."""
    st.sidebar.markdown("### **Scenario Filter Controls**")

    # 1. Regime Filter
    available_regimes = sorted(list(df["environment_regime"].unique()))
    regime_display = [f"Regime: {r}" for r in available_regimes]

    selected_display = st.sidebar.multiselect(
        "Environment Regime",
        options=regime_display,
        default=regime_display,
        help="Filter by macro-environmental regime",
    )
    selected_regimes = [d.replace("Regime: ", "") for d in selected_display]

    # 2. Target Disruption Filter
    risk_choice = st.sidebar.radio(
        "Disruption Status",
        options=["All Scenarios", "Disrupted Only (Risk = 1)", "Normal Only (Risk = 0)"],
        index=0,
    )

    # 3. Sliders for drivers
    st.sidebar.markdown("---")
    st.sidebar.markdown("##### **Driver Ranges**")

    w_min, w_max = float(df["weather_severity"].min()), float(df["weather_severity"].max())
    weather_range = st.sidebar.slider(
        "Weather Severity Range",
        min_value=round(w_min, 2),
        max_value=round(w_max, 2),
        value=(round(w_min, 2), round(w_max, 2)),
        step=0.05,
    )

    e_min, e_max = float(df["economic_pressure"].min()), float(df["economic_pressure"].max())
    econ_range = st.sidebar.slider(
        "Economic Pressure Range",
        min_value=round(e_min, 2),
        max_value=round(e_max, 2),
        value=(round(e_min, 2), round(e_max, 2)),
        step=0.1,
    )

    # 4. Supplier Reliability filter with missingness toggle
    include_missing_rel = st.sidebar.checkbox("Include Scenarios with Missing Supplier Data (NaN)", value=True)

    # Apply filters
    filtered_df = df.copy()

    # Apply regime
    if selected_regimes:
        filtered_df = filtered_df[filtered_df["environment_regime"].isin(selected_regimes)]
    else:
        filtered_df = filtered_df.iloc[0:0]

    # Apply target
    if "Disrupted Only" in risk_choice:
        filtered_df = filtered_df[filtered_df["disruption_risk"] == 1]
    elif "Normal Only" in risk_choice:
        filtered_df = filtered_df[filtered_df["disruption_risk"] == 0]

    # Apply weather
    filtered_df = filtered_df[
        (filtered_df["weather_severity"] >= weather_range[0]) &
        (filtered_df["weather_severity"] <= weather_range[1])
    ]

    # Apply economic
    filtered_df = filtered_df[
        (filtered_df["economic_pressure"] >= econ_range[0]) &
        (filtered_df["economic_pressure"] <= econ_range[1])
    ]

    # Apply missingness
    if not include_missing_rel:
        filtered_df = filtered_df.dropna(subset=["supplier_reliability"])

    # Sidebar Filter Summary Box
    st.sidebar.markdown("---")
    pct_kept = (len(filtered_df) / max(1, len(df))) * 100
    st.sidebar.caption(f"Active Filtered View: {len(filtered_df):,} of {len(df):,} scenarios ({pct_kept:.1f}%)")

    return filtered_df
