from copy import deepcopy

from .prompt_baseline import PROMPTS as BASELINE_PROMPTS
from .prompt_concise import build_lean_prompts, build_ultra_lean_prompts


_PROMPT_BUNDLES = {
    "baseline": BASELINE_PROMPTS,
    "lean": build_lean_prompts(),
    "ultra_lean": build_ultra_lean_prompts(),
}


def get_prompt_bundle(name: str) -> dict[str, object]:
    if name not in _PROMPT_BUNDLES:
        raise ValueError(f"Unknown prompt regime: {name}")
    return deepcopy(_PROMPT_BUNDLES[name])


def prompt_regimes() -> tuple[str, ...]:
    return tuple(_PROMPT_BUNDLES.keys())

