# NEXUS-9 Research Methodology

## 1. Machine Learning Laboratory
We evaluate 4 canonical model paradigms:
- **Baseline**: Dummy stratified and regularized Logistic Regression.
- **Bagging**: Ensemble of low-bias high-variance decision trees with bootstrap aggregation.
- **Boosting**: Sequential gradient tree boosting via XGBoost optimizing log-loss.
- **Voting**: Heterogeneous soft-voting ensemble combining Random Forest, XGBoost, and SVM.
- **Support Vector Machine (SVM)**: Maximum-margin hyperplane classification with RBF kernel and Platt probability scaling.

## 2. Structural Causal Model & Discovery
- Causal assumptions are represented as Directed Acyclic Graphs (DAGs).
- Discovered skeletons are evaluated against synthetic ground truth using Structural Hamming Distance (SHD), Precision, Recall, and Causal F1.
- Treatment effects are estimated with robust covariate adjustment and tested against placebo refutation permutations.
