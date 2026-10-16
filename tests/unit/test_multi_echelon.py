"""Unit test for MultiEchelonNetwork & Bullwhip dynamics."""

from nexus9.simulator.multi_echelon import MultiEchelonNetwork

def test_multi_echelon_bullwhip():
    net = MultiEchelonNetwork(seed=42)
    record = net.step(customer_demand=75.0)
    assert "customer_demand" in record
    assert "bullwhip_ratio" in record
    assert 0.0 <= record["retail_service_level"] <= 100.0
