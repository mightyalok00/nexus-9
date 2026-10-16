# ?? NEXUS-9: Modular Research-Grade AI/ML System

[![CI Status](https://img.shields.io/badge/CI-Passing-brightgreen)]()
[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)]()
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)]()
[![Architecture](https://img.shields.io/badge/Architecture-Modular_Research_Grade-purple)]()

NEXUS-9 is a modular, research-grade AI system integrating:
- **ML Laboratory & Model Arena**: Bagging, Boosting, Voting, and calibrated SVMs.
- **Future Simulator**: Structural causal world-model with dynamic state transitions, interventions, and multi-run uncertainty quantification.
- **Causal Discovery & Inference Engine**: Empirical DAG recovery, Average Treatment Effect (ATE) estimation, and placebo refutation tests against ground truth.
- **Autonomous Research Agents**: Investigator, Hypothesis Scientist, Experiment Scientist, and Red-Team Reviewer.
- **Verifiable Evidence Ledger**: Cryptographically linked provenance for all benchmark and simulation findings.

---

## ??? System Architecture

```
nexus-9/
??? app/                  # Streamlit Interactive Research Dashboard & Pages
??? src/nexus9/           # Core Modular Engine Packages
?   ??? data/             # Loader, schema validation, anti-leakage guards, preprocessor
?   ??? ml_lab/           # Model Arena: Bagging, Boosting, Voting, SVM, calibration
?   ??? world_model/      # State definitions, dynamics, and uncertainty bounds
?   ??? simulator/        # Scenario configurations, interventions, Monte Carlo runner
?   ??? causal_engine/    # DAGs, causal discovery, ATE estimation, placebo refutation
?   ??? agents/           # Orchestrator, Investigator, Hypothesis & Experiment scientists, Red Team
?   ??? experiment_planner/# Mutual information gain, active learning, budget policies
?   ??? decision_engine/  # Future regret minimization, robust minimax optimizer
?   ??? evidence/         # Tamper-evident ledger (SHA-256), provenance tracking
?   ??? monitoring/       # Kolmogorov-Smirnov drift detection, telemetry
??? datasets/             # FutureBench v1 raw CSV, data dictionary, causal ground truth
??? futurebench/          # Benchmark generator, scenarios, and causal discovery evaluation
??? api/                  # FastAPI REST endpoints for models, simulations, and agents
??? configs/              # System, model arena, and agent YAML configs
??? tests/                # Comprehensive unit and integration test suite
??? notebooks/            # Documented Jupyter notebooks (Audit, Model Arena, Causal)
??? reports/              # Benchmark leaderboards, experiment logs, evidence ledger
??? scripts/              # Training, simulation, and benchmark runner CLI tools
??? docs/                 # Architecture, methodology, model cards, and limitations
```

---

## ?? Quickstart Guide

### 1. Environment Setup
```powershell
# Clone or navigate to the repository
cd D:\nexus-9

# Create virtual environment and activate
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -e ".[dev,causal]"
```

### 2. Run the Model Arena Training Script
```powershell
python scripts/train_models.py
```

### 3. Launch the Interactive Streamlit Research Dashboard
```powershell
streamlit run app/dashboard.py
```

### 4. Launch the FastAPI Gateway
```powershell
uvicorn api.main:app --reload --port 8000
```
API Documentation will be available at `http://localhost:8000/docs`.

### 5. Run the Automated Test Suite
```powershell
pytest tests/ -v
```

---

## ?? Important Dataset Design Rule
`disruption_risk` is mathematically derived from `delivery_delay_days`, `inventory_gap`, and `service_level`.
For an early-warning model, those three columns are **strictly excluded** from features. Otherwise, the model trivially learns the label definition instead of learning to anticipate disruption. NEXUS-9 includes automated assertions (`assert_no_data_leakage`) to enforce this guard across all pipelines.

---

## ?? Jupyter Notebooks
- `notebooks/01_dataset_audit.ipynb`: Schema audit, missingness, regimes, and anti-leakage verification.
- `notebooks/02_model_benchmark.ipynb`: Comprehensive Model Arena benchmark, calibration, and feature ranking.
- `notebooks/03_causal_validation.ipynb`: Causal discovery, ATE estimation, placebo refutation, and simulation.

---

## ?? License
Licensed under the Apache License, Version 2.0.
