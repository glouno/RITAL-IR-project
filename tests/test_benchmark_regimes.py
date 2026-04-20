import unittest
from argparse import Namespace

from eval.benchmark_regimes import (
    build_preflight_summary,
    make_run_record,
    parse_stage_max_token_overrides,
)
from hirag.regimes import expand_benchmark_variants


class BenchmarkRegimeTests(unittest.TestCase):
    def test_benchmark_variant_expansion(self) -> None:
        variants = expand_benchmark_variants(["baseline", "lean"])
        self.assertEqual([variant["name"] for variant in variants], ["baseline", "lean"])
        self.assertEqual(variants[0]["prompt_regime"], "baseline")
        self.assertEqual(variants[1]["entity_extract_max_gleaning"], 0)

    def test_stage_override_parser(self) -> None:
        overrides = parse_stage_max_token_overrides(
            ["entity_extract=256", "community_report=512"]
        )
        self.assertEqual(overrides["entity_extract"], 256)
        self.assertEqual(overrides["community_report"], 512)

    def test_run_record_contains_expected_schema(self) -> None:
        record = make_run_record(
            variant_name="lean",
            prompt_regime="lean",
            entity_extract_max_gleaning=0,
            context_count=2,
            graph_dir=__import__("pathlib").Path("/tmp/demo"),
            indexing_seconds=1.5,
            telemetry_payload={"summary": {"cache_hits": 0, "stages": {"entity_extract": {}}}},
            graph_metrics={
                "chunk_count": 10,
                "entity_count": 20,
                "relation_count": 30,
                "community_count": 4,
            },
            best_model_max_async=8,
            cheap_model_max_async=8,
            community_report_input_max_tokens=8192,
            cluster_summary_input_max_tokens=6144,
            preflight={"variant": "lean"},
        )
        for key in (
            "variant",
            "prompt_regime",
            "entity_extract_max_gleaning",
            "context_count",
            "graph_dir",
            "indexing_seconds",
            "chunk_count",
            "entity_count",
            "relation_count",
            "community_count",
            "cache_hit_count",
            "best_model_max_async",
            "cheap_model_max_async",
            "community_report_input_max_tokens",
            "cluster_summary_input_max_tokens",
            "per_stage",
            "preflight",
            "telemetry",
            "smoke_queries",
        ):
            self.assertIn(key, record)

    def test_preflight_summary_shape(self) -> None:
        args = Namespace(
            best_model_max_async=8,
            cheap_model_max_async=8,
            community_report_input_max_tokens=8192,
            cluster_summary_input_max_tokens=6144,
        )
        variant = {"name": "lean", "prompt_regime": "lean", "entity_extract_max_gleaning": 0}
        preflight = build_preflight_summary(
            variant=variant,
            args=args,
            context_input=["short context"],
            model_max_context=12288,
            stage_max_token_overrides={},
        )
        self.assertEqual(preflight["variant"], "lean")
        self.assertEqual(preflight["best_model_max_async"], 8)
        self.assertIn("entity_extract", preflight["stages"])
        self.assertIn("community_report", preflight["stages"])


if __name__ == "__main__":
    unittest.main()
