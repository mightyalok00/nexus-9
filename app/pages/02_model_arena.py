import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
"""Page 02: Model Arena Benchmark, Conformal Prediction, and SHAP Explainability."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from nexus9.data.loader import load_futurebench_data
from nexus9.data.splitting import create_train_val_test_splits
from nexus9.ml_lab.registry import get_model, list_registered_models
from nexus9.ml_lab.evaluation import evaluate_predictions
from nexus9.ml_lab.conformal import SplitConformalClassifier
from nexus9.ml_lab.shap_explainer import ModelExplainer
from nexus9.ml_lab.cost_sensitive import optimize_decision_threshold
from app.components.charts import create_model_comparison_bar

st.header("?? Model Arena: Competitive Benchmarks, Conformal Sets & SHAP")

df = load_futurebench_data()
X_train, X_val, X_test, y_train, y_val, y_test = create_train_val_test_splits(df)

tab1, tab2, tab3, tab4 = st.tabs(["?? Leaderboard", "??? Conformal Prediction", "?? SHAP Explainability", "?? Cost-Sensitive Thresholds"])

with tab1:
    st.write(f"Train split: **{len(X_train):,}** | Val split: **{len(X_val):,}** | Test split: **{len(X_test):,}**")
    if st.button("?? Run Model Arena Benchmark", key="arena_bench"):
        with st.spinner("Training competitive models..."):
            models = ["baseline_dummy", "baseline_logistic", "bagging_random_forest", "boosting_xgboost", "voting_ensemble", "svm_classifier"]
            results = []
            for name in models:
                pipe = get_model(name)
                pipe.fit(X_train, y_train)
                y_pred = pipe.predict(X_test)
                y_prob = pipe.predict_proba(X_test)[:, 1] if hasattr(pipe, "predict_proba") else None
                m = evaluate_predictions(y_test, y_pred, y_prob)
                m["model"] = name
                results.append(m)

            res_df = pd.DataFrame(results)[["model", "accuracy", "precision", "recall", "f1", "roc_auc", "brier_score"]]
            st.session_state["leaderboard_df"] = res_df
            st.dataframe(res_df.style.highlight_max(axis=0, subset=["accuracy", "f1", "roc_auc"]), use_container_width=True)

            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(create_model_comparison_bar(res_df, metric="f1"), use_container_width=True)
            with col2:
                st.plotly_chart(create_model_comparison_bar(res_df, metric="roc_auc"), use_container_width=True)

with tab2:
    st.subheader("??? Split Conformal Prediction (Guaranteed Finite-Sample Coverage)")
    st.markdown("Replaces uncalibrated point estimates with **statistically valid prediction sets** $C(X) \subseteq \{0, 1\}$ such that $P(Y \in C(X)) \ge 1 - \alpha$.")

    confidence_level = st.slider("Target Confidence Level (1 - ?)", 0.80, 0.99, 0.90, 0.01)
    if st.button("Calculate Conformal Prediction Sets", key="conformal_btn"):
        with st.spinner("Calibrating conformal predictor on held-out validation set..."):
            pipe = get_model("boosting_xgboost", params={"n_estimators": 60, "max_depth": 4})
            pipe.fit(X_train, y_train)

            alpha = 1.0 - confidence_level
            conformal = SplitConformalClassifier(pipe, alpha=alpha)
            conformal.calibrate(X_val, y_val)
            metrics = conformal.evaluate_coverage(X_test, y_test)

            col1, col2, col3 = st.columns(3)
            col1.metric("Empirical Coverage", f"{metrics['empirical_coverage']:.2%}", delta=f"{metrics['empirical_coverage'] - (1.0 - alpha):.2%}")
            col2.metric("Average Set Size", f"{metrics['average_set_size']:.2f}")
            col3.metric("Singleton Certainty Rate", f"{metrics['singleton_rate']:.2%}")

            st.json(metrics)

with tab3:
    st.subheader("?? SHAP Feature Importance & Interpretability")
    st.markdown("TreeExplainer Shapley values for early-warning predictor attribution without leaked outcomes.")
    if st.button("Compute SHAP Values", key="shap_btn"):
        with st.spinner("Calculating SHAP explanations..."):
            pipe = get_model("boosting_xgboost", params={"n_estimators": 50, "max_depth": 4})
            pipe.fit(X_train, y_train)

            explainer = ModelExplainer(pipe)
            shap_res = explainer.explain(X_test.head(100))
            df_shap = pd.DataFrame(shap_res["feature_importance"])

            fig = px.bar(
                df_shap,
                x="importance",
                y="feature",
                orientation="h",
                title="Mean Absolute SHAP Value (Predictor Impact on Disruption Risk)",
                color="importance",
                color_continuous_scale="Viridis",
            )
            fig.update_layout(yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.subheader("?? Asymmetric Cost-Sensitive Threshold Tuning")
    cost_fn = st.number_input("Cost of False Negative (Unanticipated Disruption Shutdown)", value=10000.0, step=1000.0)
    cost_fp = st.number_input("Cost of False Positive (Unnecessary Buffer Inventory)", value=500.0, step=50.0)

    if st.button("Optimize Threshold", key="cost_btn"):
        pipe = get_model("boosting_xgboost", params={"n_estimators": 60})
        pipe.fit(X_train, y_train)
        y_prob = pipe.predict_proba(X_test)[:, 1]

        cost_res = optimize_decision_threshold(
            y_true=y_test.values,
            y_prob=y_prob,
            cost_false_negative=cost_fn,
            cost_false_positive=cost_fp,
        )

        st.success(f"Optimal Decision Threshold: **{cost_res['optimal_threshold']:.2f}** (Default: 0.50)")
        c1, c2, c3 = st.columns(3)
        c1.metric("Optimized Expected Cost", f"${cost_res['min_cost']:,.0f}")
        c2.metric("Default Cost (at 0.50)", f"${cost_res['cost_at_0_5']:,.0f}")
        c3.metric("Financial Savings", f"${cost_res['cost_savings']:,.0f}", delta=f"{cost_res['cost_reduction_percent']:.1f}%")
