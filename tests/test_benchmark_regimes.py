import unittest

from eval.benchmark_regimes import make_run_record, parse_stage_max_token_overrides
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
            "per_stage",
            "telemetry",
            "smoke_queries",
        ):
            self.assertIn(key, record)


if __name__ == "__main__":
    unittest.main()
