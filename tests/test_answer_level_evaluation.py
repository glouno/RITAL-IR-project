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
