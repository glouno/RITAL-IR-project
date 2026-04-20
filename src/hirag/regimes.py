from copy import deepcopy
from typing import Any

from .prompt_registry import get_prompt_bundle, prompt_regimes


PROMPT_REGIME_CHOICES = prompt_regimes()
STAGE_NAMES = (
    "entity_extract",
    "entity_glean_check",
    "entity_glean_continue",
    "relation_extract",
    "relation_glean_check",
    "relation_glean_continue",
    "cluster_summary",
    "community_report",
    "entity_merge_summary",
    "query_answer",
)

DEFAULT_GLEANING_BY_REGIME = {
    "baseline": 1,
    "lean": 0,
    "ultra_lean": 0,
}

REGIME_STAGE_MAX_TOKENS = {
    "lean": {
        "entity_extract": 512,
        "entity_glean_check": 8,
        "entity_glean_continue": 256,
        "relation_extract": 512,
        "relation_glean_check": 8,
        "relation_glean_continue": 256,
        "cluster_summary": 256,
        "community_report": 512,
        "entity_merge_summary": 256,
        "query_answer": 512,
    },
    "ultra_lean": {
        "entity_extract": 256,
        "entity_glean_check": 8,
        "entity_glean_continue": 128,
        "relation_extract": 384,
        "relation_glean_check": 8,
        "relation_glean_continue": 192,
        "cluster_summary": 192,
        "community_report": 256,
        "entity_merge_summary": 160,
        "query_answer": 384,
    },
}

BENCHMARK_VARIANTS = {
    "baseline": {"prompt_regime": "baseline", "entity_extract_max_gleaning": 1},
    "baseline_no_glean": {"prompt_regime": "baseline", "entity_extract_max_gleaning": 0},
    "lean": {"prompt_regime": "lean", "entity_extract_max_gleaning": 0},
    "ultra_lean": {"prompt_regime": "ultra_lean", "entity_extract_max_gleaning": 0},
}


def resolve_prompt_regime(prompt_regime: str | None) -> str:
    regime = (prompt_regime or "baseline").strip()
    if regime not in PROMPT_REGIME_CHOICES:
        raise ValueError(f"Unknown prompt regime: {regime}")
    return regime


def resolve_prompts(prompt_regime: str) -> dict[str, object]:
    return get_prompt_bundle(resolve_prompt_regime(prompt_regime))


def resolve_entity_extract_max_gleaning(
    prompt_regime: str, explicit_value: int | None
) -> int:
    if explicit_value is not None:
        return explicit_value
    return DEFAULT_GLEANING_BY_REGIME[resolve_prompt_regime(prompt_regime)]


def resolve_stage_max_tokens(
    prompt_regime: str,
    *,
    best_model_max_token_size: int,
    entity_summary_to_max_tokens: int,
    overrides: dict[str, int] | None = None,
) -> dict[str, int]:
    regime = resolve_prompt_regime(prompt_regime)
    if regime == "baseline":
        resolved = {
            "entity_extract": best_model_max_token_size,
            "entity_glean_check": 8,
            "entity_glean_continue": best_model_max_token_size,
            "relation_extract": best_model_max_token_size,
            "relation_glean_check": 8,
            "relation_glean_continue": best_model_max_token_size,
            "cluster_summary": best_model_max_token_size,
            "community_report": best_model_max_token_size,
            "entity_merge_summary": entity_summary_to_max_tokens,
            "query_answer": best_model_max_token_size,
        }
    else:
        resolved = deepcopy(REGIME_STAGE_MAX_TOKENS[regime])

    if overrides:
        for stage, value in overrides.items():
            if stage not in STAGE_NAMES:
                raise ValueError(f"Unknown stage name: {stage}")
            resolved[stage] = value
    return resolved


def expand_benchmark_variants(names: list[str] | None) -> list[dict[str, Any]]:
    variant_names = names or list(BENCHMARK_VARIANTS.keys())
    resolved = []
    for name in variant_names:
        if name not in BENCHMARK_VARIANTS:
            raise ValueError(f"Unknown benchmark variant: {name}")
        resolved.append({"name": name, **deepcopy(BENCHMARK_VARIANTS[name])})
    return resolved

