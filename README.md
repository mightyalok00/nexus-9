# NEXUS-9: Modular Research-Grade AI/ML System

[![CI](https://github.com/mightyalok00/nexus-9/actions/workflows/ci.yml/badge.svg)](https://github.com/mightyalok00/nexus-9/actions/workflows/ci.yml)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)]()
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)]()
[![Architecture](https://img.shields.io/badge/Architecture-Modular_Research_Grade-purple)]()

NEXUS-9 is a modular, research-grade AI system for industrial supply chain risk anticipation, causal inference, and dynamic simulation.

The core engineering principle: **The ML and simulation engines produce verifiable evidence; autonomous research agents interpret, challenge, and explain it.**

---

## System Architecture

```
nexus-9/
|-- app/                            # Interactive Streamlit Research Dashboard & Pages
|   |-- dashboard.py                # Main Executive Telemetry Hub
|   |-- pages/
|   |   |-- 01_data_explorer.py     # Multi-dimensional filter suite & schema audit
|   |   |-- 02_model_arena.py       # Model Arena, Conformal Prediction, SHAP, Cost tuning
|   |   |-- 03_future_simulator.py  # Monte Carlo shocks, 4-Echelon Bullwhip, Gym RL agent
|   |   |-- 04_causal_explorer.py   # Ground-truth DAG discovery, DML, Dose-Response curves
|   |   |-- 05_experiment_lab.py    # Self-correcting autonomous multi-agent inquiry loop
|   |   \-- 06_evidence_ledger.py   # Tamper-evident SHA-256 cryptographic audit ledger
|   \-- components/
|       |-- theme.py                # Glassmorphic UI theme & typography styling
|       |-- filters.py              # Interactive multi-dimensional sidebar filters
|       |-- charts.py               # Plotly interactive trajectory & comparison charts
|       |-- metrics.py              # KPI summary cards
|       \-- scenario_controls.py    # Shock intervention parameter sliders
|
|-- src/nexus9/                     # Core Modular Engine Packages
|   |-- data/                       # Loader, schema validation, anti-leakage guards, preprocessor
|   |-- ml_lab/                     # Model Arena: Bagging, Boosting, Voting, SVM, calibration, SHAP
|   |-- world_model/                # State definitions, causal transitions, uncertainty bounds
|   |-- simulator/                  # Monte Carlo runner, 4-echelon Bullwhip network, Gymnasium RL
|   |-- causal_engine/              # DAG validation, DML ATE estimation, placebo refuters
|   |-- agents/                     # Investigator, Hypothesis Scientist, Experiment Scientist, Red Team
|   |-- experiment_planner/         # Information gain ranking, active learning, budget policies
|   |-- decision_engine/            # Future regret minimization, robust minimax optimizer
|   |-- evidence/                   # Append-only SHA-256 ledger, provenance, MLflow tracking
|   \-- monitoring/                 # Kolmogorov-Smirnov drift detection, telemetry
|
|-- datasets/                       # FutureBench v1 raw CSV, data dictionary, causal ground truth
|   |-- raw/nexus9_futurebench.csv  # 12,000 synthetic industrial scenarios
|   \-- metadata/                   # Schema dictionary & 22-edge causal DAG
|
|-- futurebench/                    # FutureBench Benchmark & Generators
|   |-- generators/                 # Structural causal model & dataset synthesis
|   |-- scenarios/                  # YAML shock definitions (demand surge, supplier failure, weather)
|   \-- evaluation/                 # Causal SHD metrics, regret metrics, benchmark runner
|
|-- api/                            # FastAPI REST Gateway
|   |-- main.py                     # API gateway with CORS middleware
|   |-- schemas.py                  # Pydantic request & response schemas
|   \-- routes/                     # Endpoints for datasets, models, simulations, and experiments
|
|-- configs/                        # System, model arena, and agent YAML configurations
|-- tests/                          # 18 passing unit and integration tests (Pytest)
|-- notebooks/                      # Pre-rendered documented Jupyter notebooks (Audit, Arena, Causal)
|-- reports/                        # Benchmark leaderboards, experiment logs, evidence ledger
|-- scripts/                        # CLI runners (train_models.py, run_simulation.py, run_benchmarks.py)
\-- docs/                           # Architecture, methodology, model cards, and limitations
```

---

## Previously Reported Benchmark Snapshot (Synthetic FutureBench)\n\n> These figures are a repository-reported snapshot, not results executed or verified by this change. Regenerate the benchmark with `python scripts/run_benchmarks.py` before citing the numbers. FutureBench is synthetic; strong scores on it do not establish performance on real supply-chain data. The runner now emits split/seed metadata and expanded classification metrics.

All models are trained and evaluated strictly on early-warning predictors (zero label leakage):

| Model Family | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Brier Score |
|---|---|---|---|---|---|---|
| **Support Vector Classifier (SVM)** | **91.33%** | **93.03%** | 90.50% | **0.9175** | 0.9738 | 0.0632 |
| **Baseline Logistic Regression** | 90.94% | 91.36% | **91.65%** | 0.9151 | 0.9674 | 0.0677 |
| **Boosting (XGBoost)** | 90.78% | 92.49% | 89.98% | 0.9122 | 0.9739 | 0.0635 |
| **Voting Ensemble (RF + XGB + SVM)** | 90.56% | 92.09% | 89.98% | 0.9102 | **0.9746** | **0.0624** |
| **Bagging (Random Forest)** | 90.22% | 92.50% | 88.83% | 0.9063 | 0.9716 | 0.0658 |
| **Baseline (Dummy Stratified)** | 48.39% | 51.53% | 50.94% | 0.5123 | 0.4821 | 0.5161 |

---

## Quickstart Guide

### 1. Environment Setup
```powershell
cd D:\nexus-9
.\.venv\Scripts\Activate.ps1
pip install -e .
```

### 2. Launch the Streamlit Dashboard
```powershell
.\run_dashboard.ps1
# Or directly:
streamlit run app/dashboard.py
```
Open your browser at `http://localhost:8501`.

### 3. Run the Automated Test Suite
```powershell
pytest tests/ -v
```

### 4. Run CLI Scripts
```powershell
python scripts/train_models.py
python scripts/run_simulation.py
python scripts/run_benchmarks.py
```

---

## Evaluation and Reproducibility\n\nThe model benchmark uses a deterministic seed and reports train/validation/test sizes, test-set target prevalence, and the selected ranking metric. Classification metrics include F1, balanced accuracy, specificity, Matthews correlation coefficient (MCC), ROC-AUC, average precision, Brier score, log loss, and confusion-matrix counts when probabilities are available. Invalid probability arrays now fail explicitly rather than being silently ignored.\n\nThe model leaderboard is ranked by held-out F1 and records the actual top-ranked model in the evidence ledger. Results are still based on the synthetic FutureBench dataset; external validation remains required before making real-world claims.\n\n## Real-World Proxy Validation (Olist)

NEXUS-9 now includes an optional external validation harness for the real [Olist Brazilian E-Commerce dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce). It evaluates delivery-delay classification using chronological train/validation/test splits, a dummy baseline, logistic regression, random forest, and leakage-aware purchase-time features.

The Olist CSVs are **not included** in this repository. Download them from Kaggle and follow [the validation protocol](docs/real_world_validation.md). The dataset is licensed CC BY-NC-SA 4.0; review its terms before reuse.

```bash
python scripts/validate_olist.py --data-dir data/real_world/
```

This is a real-world proxy task—not direct validation of industrial supply-chain disruption. No real-data performance score should be claimed until the script is run on the dataset and the generated report is reviewed.

---

## Critical Anti-Leakage Design Rule

`disruption_risk` is mathematically defined from `delivery_delay_days`, `inventory_gap`, and `service_level`:

$$\text{disruption\_risk} = \mathbb{I}(\text{delivery\_delay\_days} > 5 \;\lor\; \text{inventory\_gap} > 25 \;\lor\; \text{service\_level} < 85)$$

In NEXUS-9, all early-warning models strictly exclude those three outcome columns. The `assert_no_data_leakage` validation module and Red-Team auditor verify this rule across all training and evaluation pipelines.

---

## Interactive Jupyter Notebooks

* `notebooks/01_dataset_audit.ipynb`: Schema validation, missingness profiling, outlier checks, and anti-leakage audit.
* `notebooks/02_model_benchmark.ipynb`: Model Arena evaluation, probability calibration curves, and feature information gain.
* `notebooks/03_causal_validation.ipynb`: Causal discovery vs. ground-truth DAG, Robinson DML ATE estimation, placebo refutations, and simulation rollouts.

---

## License
Licensed under the Apache License, Version 2.0.
