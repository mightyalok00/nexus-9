import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
"""Page 04: Causal Discovery, Double Machine Learning (DML), and Dose-Response Curves."""

import streamlit as st
import pandas as pd
import plotly.express as px
from nexus9.data.loader import load_futurebench_data
from nexus9.causal_engine.graph import CausalGraph
from nexus9.causal_engine.discovery import discover_causal_skeleton_correlation
from nexus9.causal_engine.effects import estimate_average_treatment_effect
from nexus9.causal_engine.double_ml import estimate_double_machine_learning_ate
from nexus9.causal_engine.dose_response import estimate_dose_response_curve
from nexus9.causal_engine.sensitivity import placebo_treatment_refuter
from futurebench.evaluation.causal_metrics import evaluate_causal_discovery

st.header("??? Causal Engine: Discovery, DML & Continuous Interventions")

df = load_futurebench_data()
gt_cg = CausalGraph.load_ground_truth()

tab1, tab2, tab3 = st.tabs(["?? DAG Discovery", "?? Double Machine Learning (DML)", "?? Continuous Dose-Response"])

with tab1:
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Causal Discovery vs. Synthetic Ground Truth")
        threshold = st.slider("Correlation Threshold for Edge Discovery", 0.1, 0.6, 0.25, 0.05)

        if st.button("Run Causal Discovery"):
            disc_cg = discover_causal_skeleton_correlation(df, threshold=threshold)
            metrics = evaluate_causal_discovery(disc_cg.get_edges(), gt_cg.get_edges())
            st.json(metrics)
            st.metric("Structural Hamming Distance (SHD)", metrics["shd"])
            st.metric("Discovery F1-Score", f"{metrics['f1']:.3f}")

    with col2:
        st.subheader("Estimate OLS ATE & Refutation")
        treatment = st.selectbox("Treatment", ["weather_severity", "economic_pressure", "policy_strength", "supplier_delay_days"])
        outcome = st.selectbox("Outcome", ["disruption_risk", "operating_cost", "service_level"])

        if st.button("Estimate OLS ATE & Refute"):
            ate_res = estimate_average_treatment_effect(df, treatment=treatment, outcome=outcome)
            st.json(ate_res)
            ref_res = placebo_treatment_refuter(df, treatment=treatment, outcome=outcome, n_permutations=10)
            st.json(ref_res)

with tab2:
    st.subheader("?? Robinson's Double Machine Learning (DML)")
    st.markdown("Uses **orthogonalized residual cross-fitting** with Gradient Boosted Trees to eliminate non-linear confounding bias.")

    t_dml = st.selectbox("DML Treatment", ["weather_severity", "economic_pressure", "supplier_delay_days"], key="t_dml")
    o_dml = st.selectbox("DML Outcome", ["disruption_risk", "operating_cost"], key="o_dml")

    if st.button("Run Double Machine Learning Estimation", key="dml_btn"):
        with st.spinner("Fitting nuisance gradient boosting models with cross-fitting..."):
            dml_res = estimate_double_machine_learning_ate(
                df=df,
                treatment=t_dml,
                outcome=o_dml,
                confounders=["economic_pressure", "policy_strength", "capacity"],
                n_splits=3,
            )
            col1, col2, col3 = st.columns(3)
            col1.metric("Debiased DML ATE (?)", f"{dml_res['dml_ate']:.4f}")
            col2.metric("Standard Error", f"{dml_res['std_error']:.4f}")
            col3.metric("p-value", f"{dml_res['p_value']:.4e}")
            st.info(f"95% Confidence Interval: [{dml_res['ci_lower']:.4f}, {dml_res['ci_upper']:.4f}]")
            st.json(dml_res)

with tab3:
    st.subheader("?? Continuous Causal Dose-Response Curve E[Y | do(T = t)]")
    st.markdown("Simulates causal response across continuous policy or shock levels.")

    t_dose = st.selectbox("Treatment Intervention Variable", ["policy_strength", "weather_severity"], key="t_dose")
    o_dose = st.selectbox("Response Variable", ["service_level", "operating_cost"], key="o_dose")

    if st.button("Generate Dose-Response Curve", key="dose_btn"):
        with st.spinner("Computing G-computation dose response..."):
            dose_res = estimate_dose_response_curve(
                df=df,
                treatment=t_dose,
                outcome=o_dose,
                confounders=["economic_pressure", "capacity", "demand"],
                n_points=25,
            )
            df_plot = pd.DataFrame({
                t_dose: dose_res["treatment_levels"],
                f"Expected {o_dose}": dose_res["expected_outcomes"]
            })
            fig = px.line(
                df_plot,
                x=t_dose,
                y=f"Expected {o_dose}",
                title=f"Causal Dose-Response: do({t_dose}) -> E[{o_dose}]",
                markers=True,
            )
            st.plotly_chart(fig, use_container_width=True)
