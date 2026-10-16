"""Reusable KPI metric cards for NEXUS-9."""

import streamlit as st

def display_kpi_row(metrics: dict):
    cols = st.columns(len(metrics))
    for col, (label, val) in zip(cols, metrics.items()):
        with col:
            st.metric(label=label, value=val)
