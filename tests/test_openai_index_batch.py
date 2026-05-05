import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()
