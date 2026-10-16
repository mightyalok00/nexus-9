# NEXUS-9 System Limitations & Assumptions

1. **Synthetic Data Nature:** FutureBench v1 is a synthetic benchmark generator designed to provide known ground truth for controlled algorithm evaluation. Findings should not be interpreted as empirical observations of real physical supply chains.
2. **Exogeneity Assumptions:** Economic pressure and weather severity are treated as exogenous inputs. Feedback from industrial actions back onto macroeconomic climate is out-of-scope for v1.
3. **Observational Confounding:** Causal discovery algorithms depend on topological ordering assumptions; hidden unmeasured confounders can introduce spurious edges if assumptions are violated.
4. **Early-Warning Horizon:** Early-warning models predict systemic fragility before operational breakdowns occur. If deployed in real-time, telemetry latency in upstream supplier reporting may degrade early detection lead-times.
