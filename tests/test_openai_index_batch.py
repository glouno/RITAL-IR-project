import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import networkx as nx

from eval.assemble_openai_index_base_graph import (
    build_graph,
    merge_entity_nodes,
    merge_relation_edges,
    write_outputs,
)
from eval.export_openai_index_entity_batch import (
    build_entity_prompt,
    request_body,
    stage_plan,
)
from eval.import_openai_index_entity_batch import (
    convert_rows,
    load_metadata,
    parse_entities,
)
from eval.export_openai_index_relation_batch import (
    build_relation_prompt,
    entity_names_for_chunk,
    request_body as relation_request_body,
)
from eval.import_openai_index_relation_batch import (
    convert_rows as convert_relation_rows,
    parse_relations,
)
from eval.openai_batch_retry_utils import body_with_completion_cap, is_retryable_batch_row
from eval.run_openai_batch_file import validate_batch_file
from eval.export_openai_index_cluster_summary_batch import build_cluster_prompt, graph_clusters
from eval.import_openai_index_cluster_summary_batch import convert_rows as convert_cluster_rows
from eval.import_openai_index_community_report_batch import convert_rows as convert_report_rows
from eval.apply_openai_index_cluster_summaries import apply_summaries
from eval.export_openai_embedding_batch import entity_items, request_body as embedding_request_body
from eval.import_openai_embedding_batch import convert_rows as convert_embedding_rows
from eval.materialize_openai_index_hirag_workdir import write_precomputed_vector_stores
from hirag.regimes import resolve_prompts


class OpenAIIndexBatchTests(unittest.TestCase):
    def test_entity_batch_request_uses_completion_tokens(self) -> None:
        body = request_body(
            Namespace(
                model="gpt-5.4-mini",
                completion_token_param="max_completion_tokens",
                temperature=None,
            ),
            "Extract entities",
            256,
        )
        self.assertEqual(body["model"], "gpt-5.4-mini")
        self.assertEqual(body["messages"][0]["role"], "user")
        self.assertEqual(body["max_completion_tokens"], 256)
        self.assertNotIn("max_tokens", body)

    def test_entity_prompt_contains_real_text(self) -> None:
        prompts = resolve_prompts("ultra_lean")
        prompt = build_entity_prompt("Compost improves soil health.", prompts)
        self.assertIn("Compost improves soil health.", prompt)
        self.assertIn("Entity_types", prompt)

    def test_stage_plan_explains_sequential_boundaries(self) -> None:
        plan = stage_plan({"chunk-1": {}}, entity_extract_max_gleaning=0)
        self.assertEqual(plan[0]["stage"], "entity_extract")
        self.assertTrue(plan[0]["batchable_now"])
        self.assertIn("relation_extract", [item["stage"] for item in plan])

    def test_parse_entity_batch_output(self) -> None:
        prompts = resolve_prompts("ultra_lean")
        text = (
            '("entity"<|>"Compost"<|>"concept"<|>"Compost is organic matter.")'
            "##"
            '("entity"<|>"Soil Health"<|>"concept"<|>"Soil health is soil quality.")'
            "<|COMPLETE|>"
        )
        entities = parse_entities(text, prompts, "chunk-1")
        self.assertEqual([entity["entity_name"] for entity in entities], ['"COMPOST"', '"SOIL HEALTH"'])
        self.assertEqual(entities[0]["source_id"], "chunk-1")

    def test_convert_batch_rows_with_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            metadata = tmp / "metadata.jsonl"
            output = tmp / "output.jsonl"
            metadata.write_text(
                json.dumps(
                    {
                        "custom_id": "index_entity|chunk-1",
                        "chunk_id": "chunk-1",
                        "full_doc_id": "doc-1",
                        "chunk_order_index": 0,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            output.write_text(
                json.dumps(
                    {
                        "custom_id": "index_entity|chunk-1",
                        "response": {
                            "status_code": 200,
                            "body": {
                                "choices": [
                                    {
                                        "message": {
                                            "content": (
                                                '("entity"<|>"Compost"<|>"concept"<|>"Compost is organic matter.")'
                                                "<|COMPLETE|>"
                                            )
                                        }
                                    }
                                ],
                                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                            },
                        },
                        "error": None,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            rows = convert_rows(output, load_metadata(metadata), resolve_prompts("ultra_lean"))
        self.assertEqual(rows[0]["chunk_id"], "chunk-1")
        self.assertEqual(rows[0]["entity_count"], 1)
        self.assertIsNone(rows[0]["error"])

    def test_relation_batch_request_uses_completion_tokens(self) -> None:
        body = relation_request_body(
            Namespace(
                model="gpt-5.4-mini",
                completion_token_param="max_completion_tokens",
                temperature=None,
            ),
            "Extract relations",
            384,
        )
        self.assertEqual(body["max_completion_tokens"], 384)
        self.assertNotIn("max_tokens", body)

    def test_relation_prompt_uses_imported_entities(self) -> None:
        prompts = resolve_prompts("ultra_lean")
        prompt = build_relation_prompt(
            "Compost improves soil health.",
            ['"COMPOST"', '"SOIL HEALTH"'],
            prompts,
        )
        self.assertIn('"COMPOST","SOIL HEALTH"', prompt)
        self.assertIn("Compost improves soil health.", prompt)

    def test_entity_names_for_chunk_preserves_ordered_unique_names(self) -> None:
        names = entity_names_for_chunk(
            {
                "entities": [
                    {"entity_name": '"COMPOST"'},
                    {"entity_name": '"SOIL HEALTH"'},
                    {"entity_name": '"COMPOST"'},
                ]
            }
        )
        self.assertEqual(names, ['"COMPOST"', '"SOIL HEALTH"'])

    def test_parse_relation_batch_output(self) -> None:
        prompts = resolve_prompts("ultra_lean")
        text = (
            '("relationship"<|>"Compost"<|>"Soil Health"<|>"Compost improves soil health."<|>8)'
            "<|COMPLETE|>"
        )
        relations = parse_relations(text, prompts, "chunk-1")
        self.assertEqual(relations[0]["src_id"], '"COMPOST"')
        self.assertEqual(relations[0]["tgt_id"], '"SOIL HEALTH"')
        self.assertEqual(relations[0]["weight"], 8.0)

    def test_convert_relation_batch_rows_with_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            metadata = tmp / "metadata.jsonl"
            output = tmp / "output.jsonl"
            metadata.write_text(
                json.dumps(
                    {
                        "custom_id": "index_relation|chunk-1",
                        "chunk_id": "chunk-1",
                        "full_doc_id": "doc-1",
                        "chunk_order_index": 0,
                        "entity_count": 2,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            output.write_text(
                json.dumps(
                    {
                        "custom_id": "index_relation|chunk-1",
                        "response": {
                            "status_code": 200,
                            "body": {
                                "choices": [
                                    {
                                        "message": {
                                            "content": (
                                                '("relationship"<|>"Compost"<|>"Soil Health"<|>"Compost improves soil health."<|>8)'
                                                "<|COMPLETE|>"
                                            )
                                        }
                                    }
                                ]
                            },
                        },
                        "error": None,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            rows = convert_relation_rows(output, load_metadata(metadata), resolve_prompts("ultra_lean"))
        self.assertEqual(rows[0]["relation_count"], 1)
        self.assertIsNone(rows[0]["error"])

    def test_assemble_base_graph_merges_duplicate_nodes_and_edges(self) -> None:
        entity_rows = [
            {
                "error": None,
                "entities": [
                    {
                        "entity_name": '"COMPOST"',
                        "entity_type": '"CONCEPT"',
                        "description": "Organic matter.",
                        "source_id": "chunk-1",
                    },
                    {
                        "entity_name": '"COMPOST"',
                        "entity_type": '"CONCEPT"',
                        "description": "Soil amendment.",
                        "source_id": "chunk-2",
                    },
                    {
                        "entity_name": '"SOIL HEALTH"',
                        "entity_type": '"CONCEPT"',
                        "description": "Soil quality.",
                        "source_id": "chunk-1",
                    },
                ],
            }
        ]
        relation_rows = [
            {
                "error": None,
                "relations": [
                    {
                        "src_id": '"COMPOST"',
                        "tgt_id": '"SOIL HEALTH"',
                        "description": "Compost improves soil health.",
                        "weight": 8.0,
                        "source_id": "chunk-1",
                    },
                    {
                        "src_id": '"SOIL HEALTH"',
                        "tgt_id": '"COMPOST"',
                        "description": "Soil health benefits from compost.",
                        "weight": 3.0,
                        "source_id": "chunk-2",
                    },
                ],
            }
        ]
        nodes = merge_entity_nodes(entity_rows)
        edges = merge_relation_edges(relation_rows)
        graph = build_graph(nodes, edges)
        self.assertEqual(graph.number_of_nodes(), 2)
        self.assertEqual(graph.number_of_edges(), 1)
        self.assertIn("Organic matter.", nodes['"COMPOST"']["description"])
        self.assertIn("chunk-2", nodes['"COMPOST"']["source_id"])
        edge = graph.edges[('"COMPOST"', '"SOIL HEALTH"')]
        self.assertEqual(edge["weight"], 11.0)

    def test_write_base_graph_outputs_graphml(self) -> None:
        nodes = {
            '"COMPOST"': {
                "entity_type": '"CONCEPT"',
                "description": "Organic matter.",
                "source_id": "chunk-1",
            },
            '"SOIL HEALTH"': {
                "entity_type": '"CONCEPT"',
                "description": "Soil quality.",
                "source_id": "chunk-1",
            },
        }
        edges = {
            ('"COMPOST"', '"SOIL HEALTH"'): {
                "description": "Compost improves soil health.",
                "source_id": "chunk-1",
                "weight": 8.0,
                "order": 1,
            }
        }
        graph = build_graph(nodes, edges)
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            write_outputs(output_dir, graph, "base.graphml", nodes, edges, [], [])
            loaded = nx.read_graphml(output_dir / "base.graphml")
            summary = json.loads((output_dir / "summary.json").read_text())
        self.assertEqual(loaded.number_of_edges(), 1)
        self.assertEqual(summary["nodes"], 2)

    def test_retry_helper_selects_length_rows_and_updates_cap(self) -> None:
        source = {
            "custom_id": "row-1",
            "body": {"model": "gpt-5.4-mini", "max_completion_tokens": 2048},
        }
        output = {
            "custom_id": "row-1",
            "response": {
                "status_code": 200,
                "body": {"choices": [{"finish_reason": "length"}]},
            },
            "error": None,
        }
        self.assertTrue(is_retryable_batch_row(output))
        updated = body_with_completion_cap(source, 4096, "max_completion_tokens")
        self.assertEqual(updated["body"]["max_completion_tokens"], 4096)
        self.assertNotIn("max_tokens", updated["body"])

    def test_cluster_summary_prompt_uses_clustered_graph_nodes(self) -> None:
        graph = nx.Graph()
        graph.add_node(
            '"COMPOST"',
            description="Compost is organic matter.",
            clusters=json.dumps([{"level": 0, "cluster": 1}]),
        )
        graph.add_node(
            '"SOIL"',
            description="Soil is improved by compost.",
            clusters=json.dumps([{"level": 0, "cluster": 1}]),
        )
        clusters = graph_clusters(graph)
        self.assertEqual(clusters[(0, "1")], ['"COMPOST"', '"SOIL"'])
        prompt = build_cluster_prompt(graph, clusters[(0, "1")], resolve_prompts("ultra_lean"), 12000)
        self.assertIn("Compost is organic matter.", prompt)

    def test_import_cluster_summary_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            metadata = tmp / "metadata.jsonl"
            output = tmp / "output.jsonl"
            metadata.write_text(
                json.dumps(
                    {
                        "custom_id": "cluster_summary|0|1",
                        "level": 0,
                        "cluster_id": "1",
                        "nodes": ['"COMPOST"'],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            output.write_text(
                json.dumps(
                    {
                        "custom_id": "cluster_summary|0|1",
                        "response": {
                            "status_code": 200,
                            "body": {
                                "choices": [
                                    {
                                        "message": {
                                            "content": (
                                                '("entity"<|>"Soil Improvement"<|>"concept"<|>"Compost improves soil.")'
                                                "<|COMPLETE|>"
                                            )
                                        }
                                    }
                                ]
                            },
                        },
                        "error": None,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            rows = convert_cluster_rows(output, load_metadata(metadata), resolve_prompts("ultra_lean"))
        self.assertEqual(rows[0]["entity_count"], 1)
        self.assertIsNone(rows[0]["parse_error"])

    def test_import_community_report_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            metadata = tmp / "metadata.jsonl"
            output = tmp / "output.jsonl"
            metadata.write_text(
                json.dumps(
                    {
                        "custom_id": "community_report|1",
                        "community_id": "1",
                        "level": 0,
                        "community": {
                            "level": 0,
                            "title": "Cluster 1",
                            "edges": [],
                            "nodes": ['"COMPOST"'],
                            "chunk_ids": ["chunk-1"],
                            "occurrence": 1.0,
                            "sub_communities": [],
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            output.write_text(
                json.dumps(
                    {
                        "custom_id": "community_report|1",
                        "response": {
                            "status_code": 200,
                            "body": {
                                "choices": [
                                    {
                                        "message": {
                                            "content": json.dumps(
                                                {
                                                    "title": "Compost",
                                                    "summary": "Compost community.",
                                                    "rating": 5,
                                                    "findings": [
                                                        {
                                                            "summary": "Soil",
                                                            "explanation": "Compost helps soil.",
                                                        }
                                                    ],
                                                }
                                            )
                                        }
                                    }
                                ]
                            },
                        },
                        "error": None,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            rows = convert_report_rows(output, load_metadata(metadata))
        self.assertIn("# Compost", rows[0]["report"]["report_string"])
        self.assertIsNone(rows[0]["parse_error"])

    def test_apply_cluster_summaries_adds_summary_nodes_and_edges(self) -> None:
        graph = nx.Graph()
        graph.add_node('"COMPOST"', entity_type='"CONCEPT"', description="Compost.", source_id="chunk-1")
        rows = [
            {
                "level": 0,
                "cluster_id": "1",
                "error": None,
                "parse_error": None,
                "entities": [
                    {
                        "entity_name": '"SOIL IMPROVEMENT"',
                        "entity_type": '"CONCEPT"',
                        "description": "Soil improvement summarizes compost effects.",
                    }
                ],
                "relations": [
                    {
                        "src_id": '"COMPOST"',
                        "tgt_id": '"SOIL IMPROVEMENT"',
                        "description": "Compost supports soil improvement.",
                        "weight": 8.0,
                    }
                ],
            }
        ]
        stats = apply_summaries(graph, rows)
        self.assertEqual(stats["added_nodes"], 1)
        self.assertEqual(stats["added_edges"], 1)
        self.assertTrue(graph.has_edge('"COMPOST"', '"SOIL IMPROVEMENT"'))

    def test_embedding_batch_request_shape_and_entity_ids(self) -> None:
        graph = nx.Graph()
        graph.add_node('"COMPOST"', description="Compost is organic matter.")
        items = entity_items(graph)
        self.assertEqual(items[0]["namespace"], "entities")
        self.assertEqual(items[0]["meta"]["entity_name"], '"COMPOST"')
        body = embedding_request_body("text-embedding-3-small", items[0]["content"])
        self.assertEqual(body["model"], "text-embedding-3-small")
        self.assertEqual(body["encoding_format"], "float")

    def test_batch_validation_rejects_endpoint_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            batch_file = Path(tmpdir) / "requests.jsonl"
            batch_file.write_text(
                json.dumps(
                    {
                        "custom_id": "embedding|chunks|chunk-1",
                        "method": "POST",
                        "url": "/v1/embeddings",
                        "body": {
                            "model": "text-embedding-3-small",
                            "input": "Compost improves soil.",
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "do not match selected endpoint"):
                validate_batch_file(batch_file, endpoint="/v1/chat/completions")
            stats = validate_batch_file(batch_file, endpoint="/v1/embeddings")
        self.assertEqual(stats["urls"], ["/v1/embeddings"])

    def test_import_embedding_batch_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            metadata = tmp / "metadata.jsonl"
            output = tmp / "output.jsonl"
            metadata.write_text(
                json.dumps(
                    {
                        "custom_id": "embedding|chunks|chunk-1",
                        "namespace": "chunks",
                        "id": "chunk-1",
                        "meta": {},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            output.write_text(
                json.dumps(
                    {
                        "custom_id": "embedding|chunks|chunk-1",
                        "response": {
                            "status_code": 200,
                            "body": {"data": [{"embedding": [0.1, 0.2, 0.3]}]},
                        },
                        "error": None,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            rows = convert_embedding_rows(output, load_metadata(metadata))
        self.assertEqual(rows[0]["namespace"], "chunks")
        self.assertEqual(rows[0]["embedding"], [0.1, 0.2, 0.3])
        self.assertIsNone(rows[0]["error"])

    def test_write_precomputed_vector_stores_outputs_nanovdb_files(self) -> None:
        graph = nx.Graph()
        graph.add_node('"COMPOST"', description="Compost is organic matter.")
        entity_id = entity_items(graph)[0]["id"]
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            entity_embeddings = tmp / "entities_embeddings.jsonl"
            chunk_embeddings = tmp / "chunks_embeddings.jsonl"
            entity_embeddings.write_text(
                json.dumps(
                    {
                        "namespace": "entities",
                        "id": entity_id,
                        "embedding": [0.1, 0.2, 0.3],
                        "error": None,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            chunk_embeddings.write_text(
                json.dumps(
                    {
                        "namespace": "chunks",
                        "id": "chunk-1",
                        "embedding": [0.3, 0.2, 0.1],
                        "error": None,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            args = Namespace(
                entity_embeddings_jsonl=str(entity_embeddings),
                chunk_embeddings_jsonl=str(chunk_embeddings),
            )
            write_precomputed_vector_stores(
                graph,
                {"chunk-1": {"content": "Compost improves soil."}},
                args,
                tmp,
            )
            entities_vdb = json.loads((tmp / "vdb_entities.json").read_text())
            chunks_vdb = json.loads((tmp / "vdb_chunks.json").read_text())
        self.assertEqual(entities_vdb["embedding_dim"], 3)
        self.assertEqual(chunks_vdb["embedding_dim"], 3)
        self.assertEqual(entities_vdb["data"][0]["entity_name"], '"COMPOST"')


if __name__ == "__main__":
    unittest.main()
