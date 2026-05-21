import csv
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from eval.answer_generation_benchmark import compact_debug, existing_keys
from eval.build_human_annotation_packet import (
    anonymized_answers,
    choose_queries,
    load_answers as load_human_answers,
)
from eval.eval_utils import (
    batch_custom_id,
    build_answer_messages,
    estimate_chat_tokens,
)
from eval.export_openai_batch_requests import request_body
from eval.export_openai_batch_requests import retrieval_trace
from eval.export_openai_judge_batch_requests import judge_custom_id
from eval.export_openai_judge_batch_requests import request_body as judge_request_body
from eval.import_openai_batch_answers import convert_rows as convert_answer_batch_rows
from eval.import_openai_batch_answers import load_context_metadata
from eval.import_openai_batch_judgments import convert_rows as convert_judge_batch_rows
from eval.import_openai_batch_judgments import load_metadata as load_judge_metadata
from eval.pairwise_answer_judge import (
    build_prompt,
    map_winner,
    parse_judge_json,
    write_summary,
)
from eval.summarize_retrieval_traces import flatten_trace, summarize


class AnswerLevelEvaluationTests(unittest.TestCase):
    def test_answer_generation_resume_keys_and_debug_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            answers_path = Path(tmpdir) / "answers.jsonl"
            answers_path.write_text(
                json.dumps({"query_id": "7", "variant": "hi"}) + "\n",
                encoding="utf-8",
            )
            self.assertEqual(existing_keys(answers_path), {("7", "hi")})

        debug = compact_debug(
            {
                "bridge_strategy": "minmax_budgeted",
                "local_entities": ["A", "B"],
                "communities": [(0, "demo")],
                "bridge_edges": [{"source": "A", "target": "B"}],
                "bridge_budget_fallbacks": 1,
                "bridge_budget_stopped": True,
            }
        )
        self.assertEqual(debug["bridge_strategy"], "minmax_budgeted")
        self.assertEqual(debug["local_entity_count"], 2)
        self.assertTrue(debug["bridge_budget_stopped"])

    def test_pairwise_prompt_parser_and_summary(self) -> None:
        prompt = build_prompt("What is soil health?", "A", "B")
        self.assertIn("Comprehensiveness", prompt)
        self.assertIn("Output only valid JSON", prompt)

        parsed, error = parse_judge_json(
            'prefix {"Overall Winner": {"Winner": "Answer 2", "Explanation": "better"}} suffix'
        )
        self.assertIsNone(error)
        self.assertEqual(parsed["Overall Winner"]["Winner"], "Answer 2")
        self.assertEqual(map_winner("Answer 2", ["hi", "hi_minmax"]), "hi_minmax")
        malformed, error = parse_judge_json("not json")
        self.assertIsNone(malformed)
        self.assertIsNotNone(error)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            write_summary(
                output_dir,
                [
                    {
                        "variant_a": "hi",
                        "variant_b": "hi_minmax",
                        "overall_winner": "hi_minmax",
                        "parse_error": None,
                        "latency_seconds": 1.0,
                    },
                    {
                        "variant_a": "hi",
                        "variant_b": "hi_minmax",
                        "overall_winner": "Tie",
                        "parse_error": None,
                        "latency_seconds": 3.0,
                    },
                ],
            )
            rows = list(csv.DictReader((output_dir / "judge_summary.csv").open()))
        self.assertEqual(rows[0]["variant"], "hi_minmax")
        self.assertEqual(float(rows[0]["win_rate"]), 0.5)

    def test_openai_batch_request_shape(self) -> None:
        messages = build_answer_messages(
            "What is compost?",
            "-----Backgrounds-----\ncompost context",
            "Short answer",
        )
        body = request_body(
            Namespace(
                model="gpt-5.4-mini",
                answer_max_tokens=256,
                temperature=None,
                completion_token_param="max_completion_tokens",
            ),
            messages,
        )
        request = {
            "custom_id": batch_custom_id("q1", "hi_minmax_budgeted"),
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": body,
        }
        self.assertEqual(request["method"], "POST")
        self.assertEqual(request["url"], "/v1/chat/completions")
        self.assertEqual(request["body"]["model"], "gpt-5.4-mini")
        self.assertEqual(request["body"]["max_completion_tokens"], 256)
        self.assertNotIn("max_tokens", request["body"])
        self.assertGreater(estimate_chat_tokens(messages), 0)

    def test_retrieval_trace_preserves_path_and_selection_details(self) -> None:
        row = {
            "custom_id": "answer|0|hi_rerank_weighted",
            "query_id": 0,
            "query": "How does compost affect soil?",
            "variant": "hi_rerank_weighted",
            "model": "gpt-5.4-mini",
            "input_tokens": 100,
            "context_tokens": 80,
            "latency_seconds": 1.2,
            "error": None,
        }
        trace = retrieval_trace(
            row,
            context="context",
            debug={
                "mode": "hi_rerank_weighted",
                "bridge_strategy": "query_weighted",
                "local_rerank_strategy": "fastembed_late_interaction",
                "local_entities": ["COMPOST", "SOIL"],
                "communities": [(1, "Soil health")],
                "bridge_edges": [{"source": "COMPOST", "target": "SOIL"}],
                "bridge_path_nodes": 2,
                "bridge_path_edges": 1,
                "mean_edge_score": 0.8,
                "bridge_path_decisions": [
                    {"source": "COMPOST", "target": "SOIL", "decision": "query_weighted"}
                ],
            },
            budget_debug={"context_input_tokens": 100, "context_within_budget": True},
        )
        self.assertEqual(trace["retrieval"]["local_entity_count"], 2)
        self.assertEqual(trace["retrieval"]["community_count"], 1)
        self.assertEqual(trace["retrieval"]["bridge_path_edges"], 1)
        self.assertEqual(trace["retrieval"]["bridge_path_decisions"][0]["decision"], "query_weighted")

        flat = flatten_trace(trace)
        summary = summarize([flat])
        self.assertEqual(summary["hi_rerank_weighted"]["rows"], 1)
        self.assertEqual(summary["hi_rerank_weighted"]["mean_bridge_path_edges"], 1)

    def test_openai_answer_batch_import(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            contexts = tmp / "contexts.jsonl"
            output = tmp / "batch_output.jsonl"
            contexts.write_text(
                json.dumps(
                    {
                        "custom_id": "answer|0|hi",
                        "query_id": 0,
                        "query": "Question",
                        "variant": "hi",
                        "context_debug": {"context_input_tokens": 123},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            output.write_text(
                json.dumps(
                    {
                        "custom_id": "answer|0|hi",
                        "response": {
                            "status_code": 200,
                            "body": {
                                "id": "resp_1",
                                "choices": [{"message": {"content": "Answer text"}}],
                                "usage": {"prompt_tokens": 10, "completion_tokens": 2},
                            },
                        },
                        "error": None,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            rows = convert_answer_batch_rows(output, load_context_metadata(contexts))
        self.assertEqual(rows[0]["answer"], "Answer text")
        self.assertEqual(rows[0]["variant"], "hi")
        self.assertIsNone(rows[0]["error"])

    def test_openai_judge_batch_request_uses_completion_tokens(self) -> None:
        body = judge_request_body(
            Namespace(
                model="gpt-5.4-mini",
                max_tokens=512,
                temperature=None,
                completion_token_param="max_completion_tokens",
            ),
            "Question",
            "Answer A",
            "Answer B",
        )
        self.assertEqual(body["max_completion_tokens"], 512)
        self.assertNotIn("max_tokens", body)

    def test_openai_judge_batch_import(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            custom_id = judge_custom_id("0", "hi", "hi_minmax_budgeted", "variant_first")
            metadata = tmp / "judge_meta.jsonl"
            output = tmp / "judge_output.jsonl"
            metadata.write_text(
                json.dumps(
                    {
                        "custom_id": custom_id,
                        "query_id": "0",
                        "query": "Question",
                        "variant_a": "hi",
                        "variant_b": "hi_minmax_budgeted",
                        "answer_order": "variant_first",
                        "order": ["hi_minmax_budgeted", "hi"],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            output.write_text(
                json.dumps(
                    {
                        "custom_id": custom_id,
                        "response": {
                            "status_code": 200,
                            "body": {
                                "choices": [
                                    {
                                        "message": {
                                            "content": json.dumps(
                                                {
                                                    "Comprehensiveness": {"Winner": "Answer 1", "Explanation": ""},
                                                    "Diversity": {"Winner": "Tie", "Explanation": ""},
                                                    "Empowerment": {"Winner": "Answer 1", "Explanation": ""},
                                                    "Overall Winner": {"Winner": "Answer 1", "Explanation": "better"},
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
            rows = convert_judge_batch_rows(output, load_judge_metadata(metadata))
        self.assertEqual(rows[0]["overall_winner"], "hi_minmax_budgeted")
        self.assertIsNone(rows[0]["parse_error"])

    def test_human_packet_sampling_and_anonymization(self) -> None:
        variants = ["hi", "hi_weighted", "hi_minmax", "hi_rerank_weighted"]
        with tempfile.TemporaryDirectory() as tmpdir:
            answers_path = Path(tmpdir) / "answers.jsonl"
            with answers_path.open("w", encoding="utf-8") as handle:
                for query_id in range(6):
                    for variant in variants:
                        handle.write(
                            json.dumps(
                                {
                                    "query_id": str(query_id),
                                    "query": f"Question {query_id}",
                                    "variant": variant,
                                    "answer": f"{variant} answer {query_id}",
                                    "error": None,
                                }
                            )
                            + "\n"
                        )
            metrics_path = Path(tmpdir) / "metrics.csv"
            with metrics_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "variant",
                        "bridge_vs_query_overlap",
                        "bridge_edge_count",
                    ],
                )
                writer.writeheader()
                for variant in variants:
                    for query_id in range(6):
                        writer.writerow(
                            {
                                "variant": variant,
                                "bridge_vs_query_overlap": query_id if variant == "hi_minmax" else 0,
                                "bridge_edge_count": query_id * 2 if variant == "hi_rerank_weighted" else query_id,
                            }
                        )
            answers = load_human_answers(answers_path, variants)
            selected = choose_queries(
                Namespace(
                    query_count=6,
                    seed=3,
                    proxy_metrics=str(metrics_path),
                ),
                answers,
            )
        self.assertEqual(len(selected), 6)
        self.assertTrue(any(bucket == "minmax_proxy_win" for _qid, bucket in selected))

        labels, answer_text = anonymized_answers("0", answers["0"], variants, seed=3)
        self.assertEqual(set(labels.keys()), {"Answer A", "Answer B", "Answer C", "Answer D"})
        self.assertEqual(set(answer_text.keys()), set(labels.keys()))
        self.assertEqual(set(labels.values()), set(variants))


if __name__ == "__main__":
    unittest.main()
