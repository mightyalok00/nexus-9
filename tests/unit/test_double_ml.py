"""Unit test for Double Machine Learning ATE estimator."""

from nexus9.causal_engine.double_ml import estimate_double_machine_learning_ate

def test_double_ml_estimation(raw_dataset):
    sample = raw_dataset.head(500)
    res = estimate_double_machine_learning_ate(
        df=sample,
        treatment="weather_severity",
        outcome="disruption_risk",
        confounders=["economic_pressure", "policy_strength"],
        n_splits=2,
    )
    assert "dml_ate" in res
    assert "ci_lower" in res
    assert "ci_upper" in res
    assert res["ci_lower"] <= res["dml_ate"] <= res["ci_upper"]
