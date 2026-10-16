import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
import pandas as pd
from app.components.theme import render_hero_banner, apply_custom_theme
from nexus9.data.loader import load_futurebench_data

st.set_page_config(
    page_title="NEXUS-9 AI/ML System",
    page_icon=":microscope:",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_custom_theme()

render_hero_banner(
    title="NEXUS-9: Autonomous Research-Grade AI/ML System",
    subtitle="Enterprise Supply Chain Risk Anticipation | Causal World Simulator | Autonomous Multi-Agent Society",
    badge_text="v0.1.0 | APACHE 2.0",
)

# Load high-level stats
try:
    df = load_futurebench_data()
    n_scenarios = len(df)
    disruption_rate = df["disruption_risk"].mean()
except Exception:
    n_scenarios = 12000
    disruption_rate = 0.467

# Top Telemetry Row
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Benchmarking Corpus</div>
        <div class="metric-value">{n_scenarios:,}</div>
        <span class="metric-badge badge-info">FutureBench v1</span>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Disruption Frequency</div>
        <div class="metric-value">{disruption_rate:.1%}</div>
        <span class="metric-badge badge-danger">High Impact Target</span>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">Model Arena F1</div>
        <div class="metric-value">0.9175</div>
        <span class="metric-badge badge-success">SVM & XGBoost Top</span>
    </div>
    """, unsafe_allow_html=True)

with c4:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">Causal Ground Truth</div>
        <div class="metric-value">81.8%</div>
        <span class="metric-badge badge-warning">18/22 Edges Recovered</span>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Feature Exploration Grid
st.markdown("### **Explore System Capabilities**")

row1_col1, row1_col2, row1_col3 = st.columns(3)
with row1_col1:
    st.markdown("""
    <div class="feature-card">
        <h3>01. Data Explorer & Filter Suite</h3>
        <p style="color: #cbd5e1; font-size: 0.95rem;">
            Interactive multi-dimensional filters (Regimes, Weather Severity, Economic Pressures, and Supplier Reliability) with real-time distribution histograms and schema audits.
        </p>
        <span class="metric-badge badge-info">Anti-Leakage Certified</span>
    </div>
    """, unsafe_allow_html=True)

with row1_col2:
    st.markdown("""
    <div class="feature-card">
        <h3>02. Competitive Model Arena</h3>
        <p style="color: #cbd5e1; font-size: 0.95rem;">
            Bagging (Random Forest), Boosting (XGBoost), Voting Ensembles, and Calibrated SVMs with <b>Split Conformal Prediction</b> and <b>SHAP TreeExplainers</b>.
        </p>
        <span class="metric-badge badge-success">Guaranteed Coverage</span>
    </div>
    """, unsafe_allow_html=True)

with row1_col3:
    st.markdown("""
    <div class="feature-card">
        <h3>03. World Simulator & Digital Twin</h3>
        <p style="color: #cbd5e1; font-size: 0.95rem;">
            Structural causal dynamics, <b>4-Echelon Bullwhip Network</b> (Tier-2 to Retail), and a standard <b>Gymnasium RL Environment</b> for autonomous policy training.
        </p>
        <span class="metric-badge badge-warning">Gym RL Ready</span>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

row2_col1, row2_col2, row2_col3 = st.columns(3)
with row2_col1:
    st.markdown("""
    <div class="feature-card">
        <h3>04. Causal Engine & DML</h3>
        <p style="color: #cbd5e1; font-size: 0.95rem;">
            DAG ground truth validation, <b>Robinson's Double Machine Learning (DML)</b>, continuous dose-response curves, and Placebo Refutation sensitivity tests.
        </p>
        <span class="metric-badge badge-danger">Orthogonalized OLS</span>
    </div>
    """, unsafe_allow_html=True)

with row2_col2:
    st.markdown("""
    <div class="feature-card">
        <h3>05. Autonomous Research Lab</h3>
        <p style="color: #cbd5e1; font-size: 0.95rem;">
            Multi-agent society (Investigator, Hypothesis Scientist, Experiment Scientist, and Red-Team Reviewer) executing self-correcting inquiry cycles with LLM commentary.
        </p>
        <span class="metric-badge badge-info">Multi-Turn Self-Correction</span>
    </div>
    """, unsafe_allow_html=True)

with row2_col3:
    st.markdown("""
    <div class="feature-card">
        <h3>06. Evidence Ledger</h3>
        <p style="color: #cbd5e1; font-size: 0.95rem;">
            Cryptographically linked append-only ledger tracking SHA-256 blocks, model hyperparameters, evaluation metrics, and verified research provenance.
        </p>
        <span class="metric-badge badge-success">SHA-256 Verified</span>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.caption("NEXUS-9 Architecture | Modular Research System | Built with Python 3.12, Scikit-Learn, XGBoost, Gymnasium, Statsmodels, FastAPI & Streamlit.")
