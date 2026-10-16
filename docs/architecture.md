# NEXUS-9 System Architecture

NEXUS-9 is engineered as a research-grade modular AI system designed for industrial foresight, predictive risk anticipation, and causal inference.

```mermaid
graph TD
    Data[FutureBench Data Foundation] --> Preproc[Leakage-Safe Preprocessor]
    Preproc --> Arena[ML Laboratory Model Arena]
    Arena --> Baselines[Baselines: Dummy / Logistic]
    Arena --> Bagging[Bagging: Random Forest]
    Arena --> Boosting[Boosting: XGBoost]
    Arena --> Voting[Voting Ensemble]
    Arena --> SVM[Calibrated SVM]

    World[World State Model] --> Sim[Future Simulator Environment]
    Sim --> Scenarios[Regime Scenarios & Shocks]

    Sim --> Causal[Causal Engine]
    Causal --> DAG[DAG Ground Truth Evaluation]
    Causal --> ATE[ATE Estimation & Placebo Refuter]

    Sim --> Agents[Autonomous Research Agents]
    Agents --> Inv[Investigator]
    Agents --> Hyp[Hypothesis Scientist]
    Agents --> Exp[Experiment Scientist]
    Agents --> Red[Red-Team Reviewer]

    Agents --> Ledger[Verifiable Evidence Ledger]
    Arena --> Ledger
    Causal --> Ledger
```

## Layer Separation Principles
1. **Evidence Production vs Interpretation**: The ML and simulation engines produce empirical evidence; the autonomous agents interpret, challenge, and explain it.
2. **Anti-Leakage Isolation**: `disruption_risk` is an outcome derived from `delivery_delay_days`, `inventory_gap`, and `service_level`. These columns are strictly blacklisted from early-warning models.
3. **Reproducibility**: Experiments are hashed using SHA-256 and chained into an append-only ledger.
