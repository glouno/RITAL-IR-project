import unittest

from hirag import prompt
from hirag.prompt_baseline import PROMPTS as BASELINE_PROMPTS
from hirag.prompt_registry import get_prompt_bundle
from hirag.regimes import resolve_stage_max_tokens


REQUIRED_PROMPT_KEYS = {
    "community_report",
    "entity_extraction",
    "hi_entity_extraction",
    "hi_relation_extraction",
    "summarize_entity_descriptions",
    "entiti_continue_extraction",
    "entiti_if_loop_extraction",
    "summary_clusters",
    "DEFAULT_ENTITY_TYPES",
    "META_ENTITY_TYPES",
    "DEFAULT_TUPLE_DELIMITER",
    "DEFAULT_RECORD_DELIMITER",
    "DEFAULT_COMPLETION_DELIMITER",
    "local_rag_response",
    "naive_rag_response",
    "fail_response",
    "process_tickers",
    "default_text_separator",
}


class PromptRegimeTests(unittest.TestCase):
    def test_prompt_registry_returns_complete_bundles(self) -> None:
        baseline_keys = set(BASELINE_PROMPTS.keys())
        for regime in ("baseline", "lean", "ultra_lean"):
            bundle = get_prompt_bundle(regime)
            self.assertEqual(set(bundle.keys()), baseline_keys)
            self.assertTrue(REQUIRED_PROMPT_KEYS.issubset(bundle.keys()))

    def test_prompt_wrapper_resolves_baseline_bundle(self) -> None:
        self.assertEqual(prompt.PROMPTS, BASELINE_PROMPTS)

    def test_stage_config_defaults_apply_per_regime(self) -> None:
        baseline = resolve_stage_max_tokens(
            "baseline",
            best_model_max_token_size=32768,
            entity_summary_to_max_tokens=500,
        )
        lean = resolve_stage_max_tokens(
            "lean",
            best_model_max_token_size=32768,
            entity_summary_to_max_tokens=500,
        )
        ultra = resolve_stage_max_tokens(
            "ultra_lean",
            best_model_max_token_size=32768,
            entity_summary_to_max_tokens=500,
        )

        self.assertEqual(baseline["entity_extract"], 32768)
        self.assertEqual(baseline["entity_merge_summary"], 500)
        self.assertEqual(lean["entity_extract"], 512)
        self.assertEqual(lean["community_report"], 512)
        self.assertEqual(ultra["entity_extract"], 256)
        self.assertEqual(ultra["community_report"], 256)


if __name__ == "__main__":
    unittest.main()

