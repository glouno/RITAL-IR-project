import asyncio
import tempfile
import unittest
from unittest.mock import patch

import networkx as nx
import numpy as np

from hirag import HiRAG, QueryParam
from hirag._op import (
    _find_path_with_required_nodes,
    _load_edge_embedding_disk_cache,
    _minmax_path,
    _minmax_budgeted_path,
    _ordered_unique,
    _rerank_entity_node_datas,
    _write_edge_embedding_disk_cache,
    _weighted_dijkstra_path,
)
from hirag._storage import NetworkXStorage


async def _dummy_llm(*_args, **_kwargs):
    return ""


async def _dummy_embedding(texts):
    return np.ones((len(texts), 2), dtype=float)


_dummy_embedding.embedding_dim = 2


class RetrievalExperimentTests(unittest.TestCase):
    def test_mode_presets_are_applied(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            graph = HiRAG(
                working_dir=tmpdir,
                best_model_func=_dummy_llm,
                cheap_model_func=_dummy_llm,
                embedding_func=_dummy_embedding,
                graph_storage_cls=NetworkXStorage,
                enable_llm_cache=False,
            )
            weighted = graph._resolve_query_param(QueryParam(mode="hi_weighted"))
            rerank = graph._resolve_query_param(QueryParam(mode="hi_rerank"))
            both = graph._resolve_query_param(QueryParam(mode="hi_rerank_weighted"))
            budgeted = graph._resolve_query_param(QueryParam(mode="hi_minmax_budgeted"))

        self.assertEqual(weighted.bridge_strategy, "query_weighted")
        self.assertEqual(rerank.local_rerank_strategy, "fastembed_late_interaction")
        self.assertEqual(both.bridge_strategy, "query_weighted")
        self.assertEqual(both.local_rerank_strategy, "fastembed_late_interaction")
        self.assertEqual(budgeted.bridge_strategy, "minmax_budgeted")

    def test_ordered_unique_preserves_retrieval_order(self) -> None:
        self.assertEqual(_ordered_unique(["B", "A", "B", "C", "A"]), ["B", "A", "C"])

    def test_unweighted_bridge_follows_shortest_path(self) -> None:
        graph = nx.Graph()
        graph.add_edges_from([("A", "B"), ("B", "C")])
        storage = NetworkXStorage(namespace="demo", global_config={"working_dir": tempfile.mkdtemp()})
        storage._graph = graph
        path, decisions = asyncio.run(
            _find_path_with_required_nodes(
                storage,
                ["A", "C"],
                QueryParam(bridge_strategy="unweighted"),
                {},
            )
        )
        self.assertEqual(path, ["A", "B", "C"])
        self.assertEqual(decisions[0]["decision"], "unweighted")

    def test_weighted_bridge_prefers_relevant_longer_path(self) -> None:
        graph = nx.Graph()
        graph.add_edges_from([("A", "B"), ("B", "D"), ("A", "C"), ("C", "E"), ("E", "D")])
        costs = {
            tuple(sorted(("A", "B"))): {"cost": 10.0},
            tuple(sorted(("B", "D"))): {"cost": 10.0},
            tuple(sorted(("A", "C"))): {"cost": 1.0},
            tuple(sorted(("C", "E"))): {"cost": 1.0},
            tuple(sorted(("E", "D"))): {"cost": 1.0},
        }
        self.assertEqual(_weighted_dijkstra_path(graph, "A", "D", costs), ["A", "C", "E", "D"])

    def test_minmax_avoids_bad_edge(self) -> None:
        graph = nx.Graph()
        graph.add_edges_from([("A", "B"), ("B", "D"), ("A", "C"), ("C", "E"), ("E", "D")])
        costs = {
            tuple(sorted(("A", "B"))): {"cost": 1.0},
            tuple(sorted(("B", "D"))): {"cost": 9.0},
            tuple(sorted(("A", "C"))): {"cost": 3.0},
            tuple(sorted(("C", "E"))): {"cost": 3.0},
            tuple(sorted(("E", "D"))): {"cost": 3.0},
        }
        self.assertEqual(_minmax_path(graph, "A", "D", costs), ["A", "C", "E", "D"])

    def test_budgeted_minmax_respects_path_limit(self) -> None:
        graph = nx.Graph()
        graph.add_edges_from([("A", "B"), ("B", "D"), ("A", "C"), ("C", "E"), ("E", "D")])
        costs = {
            tuple(sorted(("A", "B"))): {"cost": 1.0},
            tuple(sorted(("B", "D"))): {"cost": 9.0},
            tuple(sorted(("A", "C"))): {"cost": 3.0},
            tuple(sorted(("C", "E"))): {"cost": 3.0},
            tuple(sorted(("E", "D"))): {"cost": 3.0},
        }
        param = QueryParam(bridge_strategy="minmax_budgeted", bridge_max_path_edges=2)
        path, decision = _minmax_budgeted_path(graph, "A", "D", costs, param)
        self.assertEqual(path, ["A", "B", "D"])
        self.assertEqual(decision, "minmax_budgeted")

    def test_edge_embedding_cache_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_file = tempfile.NamedTemporaryFile(dir=tmpdir, delete=False)
            cache_file.close()
            data = {
                "edge_embeddings": {
                    "A|B": {"text_hash": "abc", "embedding": [1.0, 0.0]}
                }
            }
            _write_edge_embedding_disk_cache(cache_file.name, data)
            loaded = _load_edge_embedding_disk_cache(cache_file.name)
        self.assertEqual(loaded["edge_embeddings"]["A|B"]["embedding"], [1.0, 0.0])

    def test_reranker_fallback_preserves_dense_order(self) -> None:
        nodes = [
            {"entity_name": "A", "description": "alpha", "dense_score": 0.9},
            {"entity_name": "B", "description": "beta", "dense_score": 0.8},
        ]
        with patch("hirag._op._get_late_interaction_reranker", side_effect=RuntimeError("no model")):
            reranked = _rerank_entity_node_datas(
                "alpha",
                nodes,
                QueryParam(local_rerank_strategy="fastembed_late_interaction"),
                {},
            )
        self.assertEqual([node["entity_name"] for node in reranked], ["A", "B"])


if __name__ == "__main__":
    unittest.main()
