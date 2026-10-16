"""Reusable Plotly charts for NEXUS-9 Dashboard."""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

def create_correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    corr = df.select_dtypes(include=[np.number]).corr()
    fig = px.imshow(
        corr,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="Viridis",
        title="Predictor & Outcome Correlation Matrix",
    )
    fig.update_layout(margin=dict(l=20, r=20, t=40, b=20))
    return fig

def create_model_comparison_bar(results_df: pd.DataFrame, metric: str = "f1") -> go.Figure:
    fig = px.bar(
        results_df,
        x="model",
        y=metric,
        color=metric,
        color_continuous_scale="Plasma",
        title=f"Model Comparison: {metric.upper()}",
        text_auto=".3f",
    )
    fig.update_layout(yaxis_range=[0, 1.05], margin=dict(l=20, r=20, t=40, b=20))
    return fig

def create_simulation_trajectory_plot(sim_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    # Group by step across runs
    step_summary = sim_df.groupby("step").agg({
        "service_level": ["mean", "std"],
        "delivery_delay_days": "mean",
        "inventory_gap": "mean",
    }).reset_index()

    step_summary.columns = ["step", "sl_mean", "sl_std", "delay_mean", "gap_mean"]

    fig.add_trace(go.Scatter(
        x=step_summary["step"],
        y=step_summary["sl_mean"],
        mode="lines+markers",
        name="Service Level (%)",
        line=dict(color="#00CC96", width=3),
    ))

    fig.add_trace(go.Scatter(
        x=step_summary["step"],
        y=step_summary["delay_mean"],
        mode="lines+markers",
        name="Delivery Delay (days)",
        line=dict(color="#EF553B", width=2, dash="dash"),
        yaxis="y2",
    ))

    fig.update_layout(
        title="Simulation Dynamic Trajectory (Averaged Over Replications)",
        xaxis_title="Simulation Step",
        yaxis=dict(title="Service Level (%)", range=[0, 105]),
        yaxis2=dict(title="Delivery Delay (days)", overlaying="y", side="right"),
        hovermode="x unified",
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig
