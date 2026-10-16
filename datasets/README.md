# NEXUS-9 FutureBench v1

## Files
- `nexus9_futurebench.csv`: 12,000 synthetic scenarios.
- `ground_truth_causal_graph.json`: intended data-generating graph.
- `data_dictionary.csv`: column descriptions and types.

## Intended use
Prototype the NEXUS-9 Model Arena, test preprocessing and missing-data handling,
compare Bagging / Boosting / Voting / SVM, and validate causal-discovery code
against a known synthetic graph.

## Important caveats
- This dataset is synthetic and is not evidence about real supply chains.
- `scenario_id` is an identifier and must not be used as a model feature.
- `disruption_risk` is derived from `delivery_delay_days`, `inventory_gap`, and
  `service_level`. Exclude those outcome variables when predicting risk if the
  goal is an early-warning model; otherwise the task leaks the label definition.
- `environment_regime` is derived from `weather_severity` and `economic_pressure`.
- Missingness and outliers were intentionally introduced in selected columns.
- The ground-truth graph records intended direct dependencies. It is provided
  for evaluation and should not be used as input features.

## Reproducibility
Generated with NumPy RNG seed `20261016`, 12,000 rows.
This is an original synthetic starter dataset generated for this project.
