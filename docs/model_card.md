# Model Card: NEXUS-9 Model Arena Early-Warning Classifier

## Model Details
- **Developer:** NEXUS-9 Research Team
- **Model Date:** October 2026
- **Model Type:** Heterogeneous Voting Ensemble & XGBoost Classifier
- **License:** Apache 2.0

## Intended Use
- **Primary Use:** Early-warning identification of industrial supply network disruptions prior to service level collapse.
- **Out-of-Scope:** Post-incident diagnosis using direct delivery delay or inventory shortfall measurements.

## Training & Evaluation Data
- **Dataset:** NEXUS-9 FutureBench v1 (12,000 synthetic scenarios).
- **Splits:** 70% Train, 15% Validation, 15% Held-out Test (Stratified).
- **Leakage Guards:** Strict exclusion of outcome variables (`delivery_delay_days`, `inventory_gap`, `service_level`, `scenario_id`).

## Performance Summary
- **Accuracy:** > 94%
- **F1-Score:** > 0.93
- **ROC-AUC:** > 0.98
- **Brier Score:** < 0.05
