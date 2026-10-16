"""Causal graph representations, DAG validation, and ground truth loading."""

import json
from pathlib import Path
from typing import Dict, List, Set, Tuple
import networkx as nx
from nexus9.config import PROJECT_ROOT, settings

class CausalGraph:
    """Directed Acyclic Graph (DAG) for structural causal inference."""

    def __init__(self, nodes: List[str] | None = None, edges: List[Tuple[str, str]] | None = None):
        self.graph = nx.DiGraph()
        if nodes:
            self.graph.add_nodes_from(nodes)
        if edges:
            self.graph.add_edges_from(edges)

    @classmethod
    def load_ground_truth(cls, path: Path | str | None = None) -> "CausalGraph":
        if path is None:
            path = PROJECT_ROOT / settings.paths.causal_graph
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        nodes = data.get("nodes", [])
        edges = [tuple(e) for e in data.get("edges", [])]
        return cls(nodes=nodes, edges=edges)

    def is_dag(self) -> bool:
        return nx.is_directed_acyclic_graph(self.graph)

    def get_parents(self, node: str) -> List[str]:
        if node in self.graph:
            return list(self.graph.predecessors(node))
        return []

    def get_children(self, node: str) -> List[str]:
        if node in self.graph:
            return list(self.graph.successors(node))
        return []

    def get_edges(self) -> Set[Tuple[str, str]]:
        return set(self.graph.edges())
