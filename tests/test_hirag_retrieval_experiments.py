import asyncio
import tempfile
import unittest
from unittest.mock import patch

import networkx as nx
import numpy as np

from hirag import HiRAG, QueryParam
from hirag._op import (
    _edge_embedding_batches,
    _find_path_with_required_nodes,
    _load_edge_embedding_disk_cache,
    _mcts_config_from_query,
    _minmax_path,
    _minmax_budgeted_path,
    _ordered_unique,
    _rerank_entity_node_datas,
    _text_chunk_source_ids,
    _text_unit_context_content,
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
            mcts = graph._resolve_query_param(QueryParam(mode="hi_mcts"))

        self.assertEqual(weighted.bridge_strategy, "query_weighted")
        self.assertEqual(rerank.local_rerank_strategy, "fastembed_late_interaction")
        self.assertEqual(both.bridge_strategy, "query_weighted")
        self.assertEqual(both.local_rerank_strategy, "fastembed_late_interaction")
        self.assertEqual(budgeted.bridge_strategy, "minmax_budgeted")
        self.assertEqual(mcts.bridge_strategy, "mcts")

    def test_ordered_unique_preserves_retrieval_order(self) -> None:
        self.assertEqual(_ordered_unique(["B", "A", "B", "C", "A"]), ["B", "A", "C"])

    def test_text_chunk_source_ids_ignore_cluster_provenance(self) -> None:
        self.assertEqual(
            _text_chunk_source_ids("chunk-a<SEP>cluster-0-12<SEP>chunk-b"),
            ["chunk-a", "chunk-b"],
        )

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

    def test_mcts_prefers_weighted_candidate_region(self) -> None:
        graph = nx.Graph()
        graph.add_edges_from(
            [
                ("A", "B"),
                ("B", "D"),
                ("A", "C"),
                ("C", "E"),
                ("E", "D"),
            ]
        )
        storage = NetworkXStorage(namespace="demo", global_config={"working_dir": tempfile.mkdtemp()})
        storage._graph = graph
        costs = {
            tuple(sorted(("A", "B"))): {"cost": 9.0},
            tuple(sorted(("B", "D"))): {"cost": 9.0},
            tuple(sorted(("A", "C"))): {"cost": 0.1},
            tuple(sorted(("C", "E"))): {"cost": 0.1},
            tuple(sorted(("E", "D"))): {"cost": 0.1},
        }
        param = QueryParam(
            bridge_strategy="mcts",
            bridge_max_path_edges=4,
            mcts_max_iterations=64,
        )
        path, decisions = asyncio.run(
            _find_path_with_required_nodes(
                storage,
                ["A", "D"],
                param,
                costs,
            )
        )
        self.assertEqual(path, ["A", "C", "E", "D"])
        self.assertIn(decisions[0]["decision"], {"mcts", "mcts_weighted_fallback"})
        self.assertEqual(decisions[0]["candidate_nodes"], 5)
        self.assertEqual(decisions[0]["candidate_hops"], 2)
        self.assertFalse(decisions[0]["candidate_radius_expanded"])

    def test_mcts_candidate_radius_expands_when_initial_hops_disconnect(self) -> None:
        graph = nx.path_graph(["A", "B", "C", "D", "E", "F", "G"])
        storage = NetworkXStorage(namespace="demo", global_config={"working_dir": tempfile.mkdtemp()})
        storage._graph = graph
        costs = {
            tuple(sorted((left, right))): {"cost": 0.1}
            for left, right in graph.edges()
        }
        param = QueryParam(
            bridge_strategy="mcts",
            mcts_candidate_hops=2,
            mcts_candidate_max_hops=3,
            bridge_max_path_edges=6,
            mcts_max_iterations=32,
        )
        path, decisions = asyncio.run(
            _find_path_with_required_nodes(
                storage,
                ["A", "G"],
                param,
                costs,
            )
        )
        self.assertEqual(path, ["A", "B", "C", "D", "E", "F", "G"])
        self.assertEqual(decisions[0]["candidate_hops"], 3)
        self.assertTrue(decisions[0]["candidate_radius_expanded"])
        self.assertEqual(decisions[0]["candidate_max_hops"], 3)

    def test_mcts_candidate_radius_stops_at_max_hops(self) -> None:
        graph = nx.path_graph(["A", "B", "C", "D", "E", "F", "G", "H", "I"])
        storage = NetworkXStorage(namespace="demo", global_config={"working_dir": tempfile.mkdtemp()})
        storage._graph = graph
        costs = {
            tuple(sorted((left, right))): {"cost": 0.1}
            for left, right in graph.edges()
        }
        param = QueryParam(
            bridge_strategy="mcts",
            mcts_candidate_hops=2,
            mcts_candidate_max_hops=3,
            bridge_max_path_edges=8,
            mcts_max_iterations=32,
        )
        path, decisions = asyncio.run(
            _find_path_with_required_nodes(
                storage,
                ["A", "I"],
                param,
                costs,
            )
        )
        self.assertEqual(path, ["I"])
        self.assertEqual(decisions[0]["decision"], "mcts_no_path")
        self.assertEqual(decisions[0]["reason"], "candidate_subgraph_disconnected")
        self.assertEqual(decisions[0]["candidate_hops"], 3)
        self.assertTrue(decisions[0]["candidate_radius_expanded"])
        self.assertEqual(decisions[0]["candidate_max_hops"], 3)

    def test_mcts_config_uses_bridge_budget_fields(self) -> None:
        config = _mcts_config_from_query(
            QueryParam(
                bridge_max_path_edges=9,
                max_token_for_bridge_knowledge=321,
                bridge_length_penalty=0.07,
                mcts_token_penalty=0.2,
                mcts_target_reward=1.5,
                mcts_candidate_max_hops=5,
            )
        )
        self.assertEqual(config.max_path_edges, 9)
        self.assertEqual(config.max_token_budget, 321)
        self.assertEqual(config.length_penalty, 0.07)
        self.assertEqual(config.token_penalty, 0.2)
        self.assertEqual(config.target_reward, 1.5)
        self.assertEqual(config.candidate_max_hops, 5)

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

    def test_edge_embedding_batches_bound_items_and_tokens(self) -> None:
        batches = _edge_embedding_batches(
            [("A", "B"), ("B", "C"), ("C", "D")],
            ["alpha beta", "gamma delta epsilon", "zeta eta theta"],
            max_item_tokens=2,
            max_batch_tokens=4,
            max_batch_items=2,
        )

        self.assertEqual(len(batches), 2)
        self.assertEqual([len(texts) for _keys, texts in batches], [2, 1])
        self.assertTrue(
            all(
                len(text.split()) <= 2
                for _keys, texts in batches
                for text in texts
            )
        )

    def test_text_unit_snippet_keeps_context_bounded(self) -> None:
        text = _text_unit_context_content(
            {"content": "abcdef"},
            QueryParam(text_unit_snippet_chars=3),
        )
        self.assertEqual(text, "abc ...")

    def test_text_unit_query_overlap_snippet_prefers_relevant_window(self) -> None:
        text = _text_unit_context_content(
            {
                "content": (
                    "Generic introduction about farms. "
                    "Irrigation scheduling and soil moisture sensors reduce water stress. "
                    "Unrelated closing note."
                )
            },
            QueryParam(
                text_unit_snippet_chars=70,
                text_unit_snippet_strategy="query_overlap",
            ),
            "How do soil moisture sensors help irrigation scheduling?",
        )
        self.assertIn("soil moisture sensors", text)
        self.assertIn("irrigation", text.lower())

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
