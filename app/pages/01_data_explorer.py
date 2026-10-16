import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
"""Page 01: Data Explorer, Interactive Filter Suite & Anti-Leakage Audit."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from nexus9.data.loader import load_futurebench_data
from nexus9.data.validation import validate_schema, inspect_missingness
from app.components.theme import render_hero_banner, apply_custom_theme
from app.components.filters import render_data_filters
from app.components.charts import create_correlation_heatmap

apply_custom_theme()

render_hero_banner(
    title="?? FutureBench Data Explorer & Filter Suite",
    subtitle="Interactive Multi-Dimensional Filtering ? Regime Distributions ? Anti-Leakage Verification",
    badge_text="12,000 SCENARIOS",
)

try:
    raw_df = load_futurebench_data()

    # Apply Sidebar Filters
    filtered_df = render_data_filters(raw_df)

    # Top KPI Metrics for Active Filter
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Active Scenarios</div>
            <div class="metric-value">{len(filtered_df):,}</div>
            <span class="metric-badge badge-info">{len(filtered_df)/len(raw_df):.1%} of Total</span>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        risk_rate = filtered_df["disruption_risk"].mean() if len(filtered_df) > 0 else 0.0
        delta_risk = risk_rate - raw_df["disruption_risk"].mean()
        badge_cls = "badge-danger" if risk_rate > 0.5 else "badge-success"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Filtered Disruption Rate</div>
            <div class="metric-value">{risk_rate:.1%}</div>
            <span class="metric-badge {badge_cls}">? {delta_risk:+.1%} vs Baseline</span>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        avg_cost = filtered_df["operating_cost"].mean() if len(filtered_df) > 0 else 0.0
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Avg Operating Cost</div>
            <div class="metric-value">${avg_cost:,.0f}</div>
            <span class="metric-badge badge-warning">Tail Outliers Included</span>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        avg_service = filtered_df["service_level"].mean() if len(filtered_df) > 0 else 0.0
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Avg Service Level</div>
            <div class="metric-value">{avg_service:.1f}%</div>
            <span class="metric-badge badge-success">Target: > 85%</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Tabs: Data View, Distribution Charts, Schema & Missingness
    tab1, tab2, tab3 = st.tabs(["?? Filtered Data Table", "?? Distribution & Correlations", "?? Schema & Missingness Audit"])

    with tab1:
        st.subheader("Filtered Scenario Records")
        search_kw = st.text_input("?? Quick Search by Regime or Metric Range", placeholder="Type keywords or scenario IDs...")
        display_df = filtered_df
        if search_kw:
            display_df = filtered_df[filtered_df.astype(str).apply(lambda row: row.str.contains(search_kw, case=False).any(), axis=1)]

        st.dataframe(display_df.head(25), width="stretch")

        csv_data = filtered_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="?? Download Active Filtered Dataset (CSV)",
            data=csv_data,
            file_name="nexus9_filtered_futurebench.csv",
            mime="text/csv",
        )

    with tab2:
        st.subheader("Feature Distributions Across Regimes")
        c1, c2 = st.columns(2)
        with c1:
            fig_hist = px.histogram(
                filtered_df,
                x="operating_cost",
                color="environment_regime",
                nbins=40,
                title="Operating Cost Distribution by Regime",
                color_discrete_sequence=["#60a5fa", "#f59e0b", "#ef4444"],
            )
            st.plotly_chart(fig_hist, width="stretch")

        with c2:
            fig_scatter = px.scatter(
                filtered_df.sample(min(1000, len(filtered_df)), random_state=42) if len(filtered_df) > 0 else filtered_df,
                x="weather_severity",
                y="supplier_delay_days",
                color="disruption_risk",
                title="Weather Severity vs. Supplier Delay Days",
                color_continuous_scale="Viridis",
            )
            st.plotly_chart(fig_scatter, width="stretch")

        st.subheader("Predictor & Outcome Correlation Matrix")
        st.plotly_chart(create_correlation_heatmap(filtered_df), width="stretch")

    with tab3:
        st.subheader("Missingness & Schema Integrity")
        missing_s = inspect_missingness(raw_df)
        missing_df = pd.DataFrame({"Missing Records": missing_s, "Percentage": (missing_s / len(raw_df)) * 100})
        st.dataframe(missing_df, width="stretch")

        st.info("""
        **?? Critical Data Design Rule Verification:**
        `disruption_risk` is directly derived from `delivery_delay_days`, `inventory_gap`, and `service_level`.
        All early-warning models strictly exclude those three outcome columns to prevent label leakage.
        """)

except Exception as e:
    st.error(f"Error loading dataset: {e}")
