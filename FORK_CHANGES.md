# Fork Change Log

This file tracks the meaningful changes made in this fork relative to the cloned HiRAG base.

Rule for future changes:

- Whenever we make a repo-level change that affects runtime behavior, datasets, prompts, evaluation, infra assumptions, or developer workflow, add a dated entry here.
- Keep entries short and grouped by change set, not by every tiny edit.
- Prefer linking to the main touched files and describing the user-visible or experiment-visible effect.

## Base Fork Point

### 2026-04-20

Starting point in this repo history:

- Base imported HiRAG code and assets
- Added local project packaging and `uv` workflow
- Added local vLLM + FastEmbed entrypoint around HiRAG in top-level `main.py`
- Added pre-extracted `*_unique_contexts.json` datasets used for indexing experiments
- Added `AGENTS.md` to document the active runtime path and dataset usage

Relevant files:

- [main.py](/home/paulbeglin/projects/RITAL-IR-project/main.py)
- [pyproject.toml](/home/paulbeglin/projects/RITAL-IR-project/pyproject.toml)
- [eval/datasets/agriculture/agriculture_unique_contexts.json](/home/paulbeglin/projects/RITAL-IR-project/eval/datasets/agriculture/agriculture_unique_contexts.json)
- [eval/datasets/cs/cs_unique_contexts.json](/home/paulbeglin/projects/RITAL-IR-project/eval/datasets/cs/cs_unique_contexts.json)
- [eval/datasets/legal/legal_unique_contexts.json](/home/paulbeglin/projects/RITAL-IR-project/eval/datasets/legal/legal_unique_contexts.json)
- [eval/datasets/mix/mix_unique_contexts.json](/home/paulbeglin/projects/RITAL-IR-project/eval/datasets/mix/mix_unique_contexts.json)
- [AGENTS.md](/home/paulbeglin/projects/RITAL-IR-project/AGENTS.md)

## Concise Regimes And Benchmarking

### 2026-04-20

Added a parallel concise-prompt track while preserving the original baseline behavior.

What changed:

- Moved the original prompt set into a baseline module and kept `prompt.py` as a baseline-compatible wrapper
- Added two alternative prompt regimes:
  - `lean`
  - `ultra_lean`
- Added prompt-regime selection and regime defaults for gleaning and stage token caps
- Added per-stage `max_tokens` handling and stage labels across extraction, clustering, community reporting, and query answering
- Added telemetry capture for vLLM calls with prompt/completion/total tokens, latency, cache-hit status, and stage labels
- Added a dedicated indexing benchmark runner to compare:
  - `baseline`
  - `baseline_no_glean`
  - `lean`
  - `ultra_lean`
- Added targeted tests for prompt bundles, regime resolution, telemetry, CLI parsing, and benchmark variant expansion
- Added CLI/env support for:
  - `PROMPT_REGIME`
  - `ENTITY_EXTRACT_MAX_GLEANING`
  - `TELEMETRY_OUTPUT`
  - `BENCHMARK_LABEL`
- Added output-token clamping against model context budget in the local vLLM wrapper to avoid invalid `max_tokens` requests on long prompts

Relevant files:

- [main.py](/home/paulbeglin/projects/RITAL-IR-project/main.py)
- [src/hirag/hirag.py](/home/paulbeglin/projects/RITAL-IR-project/src/hirag/hirag.py)
- [src/hirag/_op.py](/home/paulbeglin/projects/RITAL-IR-project/src/hirag/_op.py)
- [src/hirag/_cluster_utils.py](/home/paulbeglin/projects/RITAL-IR-project/src/hirag/_cluster_utils.py)
- [src/hirag/_llm.py](/home/paulbeglin/projects/RITAL-IR-project/src/hirag/_llm.py)
- [src/hirag/prompt.py](/home/paulbeglin/projects/RITAL-IR-project/src/hirag/prompt.py)
- [src/hirag/prompt_baseline.py](/home/paulbeglin/projects/RITAL-IR-project/src/hirag/prompt_baseline.py)
- [src/hirag/prompt_concise.py](/home/paulbeglin/projects/RITAL-IR-project/src/hirag/prompt_concise.py)
- [src/hirag/prompt_registry.py](/home/paulbeglin/projects/RITAL-IR-project/src/hirag/prompt_registry.py)
- [src/hirag/regimes.py](/home/paulbeglin/projects/RITAL-IR-project/src/hirag/regimes.py)
- [src/hirag/telemetry.py](/home/paulbeglin/projects/RITAL-IR-project/src/hirag/telemetry.py)
- [eval/benchmark_regimes.py](/home/paulbeglin/projects/RITAL-IR-project/eval/benchmark_regimes.py)
- [tests/test_prompt_regimes.py](/home/paulbeglin/projects/RITAL-IR-project/tests/test_prompt_regimes.py)
- [tests/test_main_runtime.py](/home/paulbeglin/projects/RITAL-IR-project/tests/test_main_runtime.py)
- [tests/test_benchmark_regimes.py](/home/paulbeglin/projects/RITAL-IR-project/tests/test_benchmark_regimes.py)
- [`.env.example`](/home/paulbeglin/projects/RITAL-IR-project/.env.example)

Validation done for this change set:

- targeted unit tests for the new regime/telemetry/benchmark code
- syntax/compile pass
- benchmark smoke run on the tiny test dataset through the real local vLLM path until extraction, relation building, clustering, and community-report startup were confirmed

## Throughput Preflight And Async Controls

### 2026-04-20

Added the runtime controls needed for safe overnight throughput runs against a reduced vLLM context ceiling.

What changed:

- Exposed explicit client-side LLM concurrency controls in the single-run and benchmark CLIs:
  - `BEST_MODEL_MAX_ASYNC`
  - `CHEAP_MODEL_MAX_ASYNC`
- Added dedicated input packing caps for the long later stages:
  - `COMMUNITY_REPORT_INPUT_MAX_TOKENS`
  - `CLUSTER_SUMMARY_INPUT_MAX_TOKENS`
- Wired community-report packing to use its dedicated cap instead of the full model token ceiling
- Added cluster-summary input truncation before prompt construction
- Extended the benchmark harness with:
  - preflight-only mode
  - async-limit reporting
  - input-cap reporting
  - stage-size safety estimates against the discovered vLLM context window
- Extended runtime and benchmark JSON outputs so overnight runs retain the exact concurrency and packing settings used

Relevant files:

- [main.py](/home/paulbeglin/projects/RITAL-IR-project/main.py)
- [src/hirag/hirag.py](/home/paulbeglin/projects/RITAL-IR-project/src/hirag/hirag.py)
- [src/hirag/_op.py](/home/paulbeglin/projects/RITAL-IR-project/src/hirag/_op.py)
- [src/hirag/_cluster_utils.py](/home/paulbeglin/projects/RITAL-IR-project/src/hirag/_cluster_utils.py)
- [eval/benchmark_regimes.py](/home/paulbeglin/projects/RITAL-IR-project/eval/benchmark_regimes.py)
- [tests/test_main_runtime.py](/home/paulbeglin/projects/RITAL-IR-project/tests/test_main_runtime.py)
- [tests/test_benchmark_regimes.py](/home/paulbeglin/projects/RITAL-IR-project/tests/test_benchmark_regimes.py)
- [`.env.example`](/home/paulbeglin/projects/RITAL-IR-project/.env.example)
