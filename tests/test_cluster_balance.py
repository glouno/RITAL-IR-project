import tempfile
import unittest

import networkx as nx

from eval.analyze_cluster_balance import analyze_communities, normalized_entropy
from eval.retrieval_context_benchmark import coverage_metrics


class ClusterBalanceTests(unittest.TestCase):
    def test_bridge_coverage_diagnostic_is_stable(self) -> None:
        metrics = coverage_metrics(
            "soil health",
            {
                "bridge_edges": [{"description": "soil organic matter supports health"}],
                "local_entities": ["SOIL HEALTH"],
                "communities": [(0, "Soil and compost")],
            },
        )
        self.assertGreater(metrics["bridge_vs_query_overlap"], 0.0)

    def test_normalized_entropy_bounds(self) -> None:
        self.assertEqual(normalized_entropy(["A", "A"]), 0.0)
        self.assertGreater(normalized_entropy(["A", "B"]), 0.9)

    def test_cluster_balance_flags_oversized_community(self) -> None:
        graph = nx.Graph()
        for index in range(4):
            graph.add_node(
                f"N{index}",
                entity_type="TYPE_A" if index < 2 else "TYPE_B",
                source_id=f"c{index}",
            )
        graph.add_edges_from([("N0", "N1"), ("N2", "N3")])
        reports = {
            "0": {
                "level": 0,
                "title": "demo",
                "nodes": ["N0", "N1", "N2", "N3"],
            }
        }
        rows = analyze_communities(
            graph,
            reports,
            max_nodes=3,
            entropy_threshold=0.99,
        )
        self.assertTrue(rows[0]["should_split"])


if __name__ == "__main__":
    unittest.main()
