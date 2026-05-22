import csv
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from eval.answer_generation_benchmark import compact_debug, existing_keys
from eval.build_human_annotation_packet import (
    anonymized_answers,
    choose_queries,
    load_answers as load_human_answers,
)
from eval.eval_utils import (
    batch_custom_id,
    build_answer_messages,
    build_context_with_budget,
    estimate_chat_tokens,
    make_query_param,
)
from eval.export_openai_batch_requests import request_body
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

        mcts_debug = compact_debug(
            {
                "bridge_strategy": "mcts",
                "bridge_path_decisions": [
                    {
                        "decision": "mcts",
                        "iterations": 32,
                        "successful_rollouts": 6,
                        "candidate_hops": 2,
                        "candidate_radius_expanded": False,
                        "candidate_nodes": 41,
                        "candidate_edges": 58,
                    },
                    {
                        "decision": "mcts_weighted_fallback",
                        "iterations": 12,
                        "successful_rollouts": 0,
                        "candidate_hops": 3,
                        "candidate_radius_expanded": True,
                        "candidate_nodes": 15,
                        "candidate_edges": 16,
                    },
                    {
                        "decision": "mcts_no_path",
                        "iterations": 4,
                        "successful_rollouts": 0,
                        "candidate_hops": 4,
                        "candidate_radius_expanded": True,
                        "candidate_nodes": 9,
                        "candidate_edges": 7,
                    },
                ],
            }
        )
        self.assertEqual(len(mcts_debug["bridge_path_decisions"]), 3)
        self.assertEqual(mcts_debug["mcts_total_segments"], 3)
        self.assertEqual(mcts_debug["mcts_segments"], 3)
        self.assertEqual(mcts_debug["mcts_success_count"], 1)
        self.assertEqual(mcts_debug["mcts_weighted_fallback_count"], 1)
        self.assertEqual(mcts_debug["mcts_no_path_count"], 1)
        self.assertTrue(mcts_debug["mcts_any_success"])
        self.assertTrue(mcts_debug["mcts_any_no_path"])
        self.assertEqual(mcts_debug["mcts_iterations"], 48)
        self.assertEqual(mcts_debug["mcts_successful_rollouts"], 6)
        self.assertEqual(mcts_debug["mcts_candidate_hops_max"], 4)
        self.assertAlmostEqual(mcts_debug["mcts_candidate_hops_mean"], 3.0)
        self.assertEqual(mcts_debug["mcts_radius_expanded_count"], 2)
        self.assertEqual(mcts_debug["mcts_candidate_nodes_max"], 41)
        self.assertTrue(mcts_debug["mcts_used_weighted_fallback"])

    def test_hi_mcts_query_param_and_budget_loop(self) -> None:
        args = SimpleNamespace(
            top_k=12,
            top_m=6,
            response_type="Multiple Paragraphs",
            min_section_budget=100,
            max_token_for_local_context=1000,
            max_token_for_bridge_knowledge=900,
            max_token_for_community_report=800,
            max_token_for_text_unit=700,
            text_unit_snippet_chars=1200,
            text_unit_snippet_strategy="query_overlap",
            max_budget_attempts=3,
            budget_shrink_factor=0.5,
            max_input_tokens=400,
        )
        param = make_query_param("hi_mcts", args, only_need_context=True)
        self.assertEqual(param.mode, "hi_mcts")
        self.assertTrue(param.only_need_context)

        class _FakeGraph:
            def __init__(self) -> None:
                self.calls = 0

            def query(self, query, param):
                self.calls += 1
                param.debug_info["bridge_strategy"] = "mcts"
                param.debug_info["bridge_path_decisions"] = [
                    {
                        "decision": "mcts",
                        "iterations": 9,
                        "successful_rollouts": 2,
                        "candidate_nodes": 8,
                        "candidate_edges": 11,
                    }
                ]
                size = 1200 if self.calls == 1 else 10
                return "context " * size

        context, final_param, budget_debug = build_context_with_budget(
            _FakeGraph(),
            "Why bridge?",
            "hi_mcts",
            args,
        )
        self.assertLessEqual(budget_debug["context_input_tokens"], args.max_input_tokens)
        self.assertGreaterEqual(budget_debug["retrieval_time_seconds"], 0.0)
        self.assertIn("retrieval_seconds", budget_debug["context_budget_attempts"][0])
        self.assertEqual(final_param.mode, "hi_mcts")
        self.assertEqual(final_param.debug_info["bridge_strategy"], "mcts")
        self.assertIn("context", context)

    def test_build_context_with_budget_tracks_retrieval_time(self) -> None:
        args = SimpleNamespace(
            top_k=12,
            top_m=6,
            response_type="Multiple Paragraphs",
            min_section_budget=100,
            max_token_for_local_context=1000,
            max_token_for_bridge_knowledge=900,
            max_token_for_community_report=800,
            max_token_for_text_unit=700,
            text_unit_snippet_chars=1200,
            text_unit_snippet_strategy="query_overlap",
            max_budget_attempts=2,
            budget_shrink_factor=0.5,
            max_input_tokens=999999,
        )

        class _FakeGraph:
            def query(self, query, param):
                return "tiny context"

        with patch("eval.eval_utils.time.perf_counter", side_effect=[10.0, 10.4]):
            _context, _param, budget_debug = build_context_with_budget(
                _FakeGraph(),
                "Why bridge?",
                "hi_mcts",
                args,
            )

        self.assertAlmostEqual(budget_debug["retrieval_time_seconds"], 0.4)
        self.assertAlmostEqual(
            budget_debug["context_budget_attempts"][0]["retrieval_seconds"], 0.4
        )

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
