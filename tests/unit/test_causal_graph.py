"""Unit tests for DAG validation and causal edge evaluation."""

from nexus9.causal_engine.graph import CausalGraph
from futurebench.evaluation.causal_metrics import evaluate_causal_discovery

def test_ground_truth_dag_is_valid():
    cg = CausalGraph.load_ground_truth()
    assert cg.is_dag() is True
    assert len(cg.graph.nodes) == 14
    assert len(cg.graph.edges) == 22

def test_causal_metrics():
    gt = {("A", "B"), ("B", "C"), ("A", "C")}
    pred = {("A", "B"), ("B", "C"), ("C", "D")}
    res = evaluate_causal_discovery(pred, gt)
    assert res["true_positives"] == 2
    assert res["false_positives"] == 1
    assert res["false_negatives"] == 1
    assert res["shd"] == 2
