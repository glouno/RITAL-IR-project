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

## Context Budget Hardening And Baseline Recovery

### 2026-04-21

Hardened local chat token budgeting after overnight long-run failures at the model context boundary.

What changed:

- Increased client-side response-token safety margin in the local OpenAI-compatible wrapper from `256` to `1024`
- This specifically targets borderline failures where server-side token counting exceeded client estimates by a small margin
- Result: long community-report calls should degrade gracefully by shrinking `max_tokens` instead of crashing the whole run on 12288-limit overflow

Relevant files:

- [main.py](/home/paulbeglin/projects/RITAL-IR-project/main.py)

## Concise Community Prompt Format Fix

### 2026-04-22

Fixed a runtime crash during lean/ultra community report generation caused by Python `str.format` interpreting literal JSON braces in concise prompt templates.

What changed:

- Escaped literal JSON braces in concise community report prompt examples (`{{` / `}}`)
- Prevents `KeyError` on keys such as `"title"` during `community_report_prompt.format(input_text=...)`
- Enables lean/ultra runs to continue through community-report generation instead of aborting near the end

Relevant files:

- [src/hirag/prompt_concise.py](/home/paulbeglin/projects/RITAL-IR-project/src/hirag/prompt_concise.py)

## Curated Agriculture Graph Export

### 2026-04-23

Added a tracked, commit-ready export of the completed agriculture benchmark artifacts so results can be shared from GitHub without relying on ignored `.runs/` paths.

What changed:

- Added `artifacts/agriculture_graphs_2026-04-23/` with strategy-explicit GraphML filenames:
  - `agriculture_baseline_no_glean.graphml`
  - `agriculture_lean.graphml`
  - `agriculture_ultra_lean.graphml`
- Added curated JSON exports for community reports per strategy
- Added the available benchmark summary JSON (`ultra_lean`) and run metadata logs
- Added a folder-level README and manifest to map exported files back to source run directories

Why this matters:

- `.runs/` is gitignored and cannot be shared directly through normal commits
- The curated folder keeps the most useful review artifacts (graph structure + reports + summary metadata) while excluding heavier internal caches/vector stores

Relevant files:

- [artifacts/agriculture_graphs_2026-04-23/README.md](/home/paulbeglin/projects/RITAL-IR-project/artifacts/agriculture_graphs_2026-04-23/README.md)
- [artifacts/agriculture_graphs_2026-04-23/manifest.json](/home/paulbeglin/projects/RITAL-IR-project/artifacts/agriculture_graphs_2026-04-23/manifest.json)
- [FORK_CHANGES.md](/home/paulbeglin/projects/RITAL-IR-project/FORK_CHANGES.md)

## Neo4j Team Visualization Runbook

### 2026-04-23

Added teammate-facing documentation for exploring exported HiRAG graphs through Neo4j from personal laptops.

What changed:

- Added a step-by-step runbook in `docs/` for:
  - connecting to an existing Neo4j instance through SSH tunnel
  - understanding rootless Podman per-user visibility limits
  - spinning up an isolated local Neo4j Podman container on alternate ports
  - importing GraphML with APOC
  - querying large graphs with bounded Cypher patterns to avoid browser freezes

Why this matters:

- makes graph exploration reproducible for teammates without relying on one user's container session
- prevents accidental data collisions with other project databases (for example PMIND)

Relevant files:

- [docs/neo4j_graph_visualization_guide.md](/home/paulbeglin/projects/RITAL-IR-project/docs/neo4j_graph_visualization_guide.md)

## Hierarchical Neo4j Graph Artifacts

### 2026-04-25

Added generated graph artifacts that explicitly materialize hierarchy for Neo4j/Bloom exploration.

What changed:

- Added exporter script to build hierarchical GraphML variants from curated outputs:
  - reads original variant GraphML + community report JSON
  - preserves entity graph edges as `RELATES_TO`
  - adds explicit `Community` nodes
  - adds `IN_COMMUNITY` edges (entity -> community)
  - adds `HAS_SUBCOMMUNITY` edges (community -> subcommunity)
  - adds readability helper fields (`description_clean`, split segment JSON fields)
- Generated hierarchical artifacts for all three agriculture variants under `artifacts/.../neo4j_hierarchical/`
- Added import/query helper Cypher and folder README for direct Neo4j usage

Why this matters:

- the original graph stores hierarchy hints as serialized text (`clusters`) but not as explicit topology
- Bloom/Neo4j can now perform level-based drilldown and hierarchy navigation directly through graph relationships

Relevant files:

- [eval/build_hierarchical_graphml_artifacts.py](/home/paulbeglin/projects/RITAL-IR-project/eval/build_hierarchical_graphml_artifacts.py)
- [artifacts/agriculture_graphs_2026-04-23/neo4j_hierarchical/hierarchical_manifest.json](/home/paulbeglin/projects/RITAL-IR-project/artifacts/agriculture_graphs_2026-04-23/neo4j_hierarchical/hierarchical_manifest.json)
- [artifacts/agriculture_graphs_2026-04-23/neo4j_hierarchical/neo4j_import_queries.cypher](/home/paulbeglin/projects/RITAL-IR-project/artifacts/agriculture_graphs_2026-04-23/neo4j_hierarchical/neo4j_import_queries.cypher)
- [artifacts/agriculture_graphs_2026-04-23/neo4j_hierarchical/README.md](/home/paulbeglin/projects/RITAL-IR-project/artifacts/agriculture_graphs_2026-04-23/neo4j_hierarchical/README.md)

## Research Next-Step Notes

### 2026-05-04

Added planning notes for the next HiRAG research direction based on the paper, active `src/hirag` implementation, previous agriculture runs, and colleague suggestions.

What changed:

- Added a ranked next-step plan centered on retrieval-first experiments before more expensive indexing runs
- Captured the current implementation gap: bridge paths are seeded by query-relevant entities but path selection itself is unweighted/topological
- Proposed query-conditioned bridge retrieval, local retrieval reranking, entropy/confidence-based second-pass clustering, and GMM hardening
- Added a focused retrieval experiment note with concrete variant designs and metrics

Why this matters:

- gives the team a durable trace of the current reasoning before implementing experimental variants
- keeps the next implementation loop narrow enough to use existing graph artifacts instead of rerunning many-hour indexing jobs

Relevant files:

- [docs/NEXT_STEPS.md](/home/paulbeglin/projects/RITAL-IR-project/docs/NEXT_STEPS.md)
- [docs/RETRIEVAL_EXPERIMENTS.md](/home/paulbeglin/projects/RITAL-IR-project/docs/RETRIEVAL_EXPERIMENTS.md)

## Retrieval Branch Experiments

### 2026-05-04

Implemented runnable retrieval experiments on the active `main.py` + `src/hirag/` path without requiring a new indexing run.

What changed:

- Added experimental query modes:
  - `hi_weighted`
  - `hi_minmax`
  - `hi_rerank`
  - `hi_rerank_weighted`
- Added query preset fields for bridge strategy, local reranking strategy, rerank candidate counts, and bridge-cost weights.
- Refactored hierarchical bridge/context construction into shared helpers while preserving baseline `hi` behavior aside from deterministic key-entity ordering.
- Added NetworkX query-weighted Dijkstra bridge paths.
- Added minmax bridge paths that avoid one very bad high-cost edge when possible.
- Added FastEmbed late-interaction local entity reranking with dense-retrieval fallback.
- Added deterministic retrieval-context benchmark outputs for contexts, metrics, summaries, and bridge coverage diagnostics.
- Hardened GMM clustering against duplicate/collapsed embeddings and covariance failures.
- Added cluster-balance diagnostics and bounded second-pass split-plan prototyping for oversized/high-entropy communities.

Validation:

- `uv run pytest tests/test_main_runtime.py tests/test_retrieval.py tests/test_hierarchy.py tests/test_hirag_retrieval_experiments.py tests/test_cluster_balance.py`
- `uv run python eval/retrieval_context_benchmark.py --working-dir .runs/2026-04-22-agri-resume2/graphs/benchmark_ultra_lean_run1_20260422_204326 --query-file eval/datasets/agriculture/agriculture_query.jsonl --query-limit 10 --output-dir .runs/retrieval_eval/dev_retrieval_smoke`
- `uv run python eval/analyze_cluster_balance.py --graphml artifacts/agriculture_graphs_2026-04-23/graphs/agriculture_ultra_lean.graphml --community-reports artifacts/agriculture_graphs_2026-04-23/metrics/agriculture_ultra_lean_community_reports.json --output-dir .runs/cluster_balance/dev_retrieval_smoke`

Relevant files:

- [main.py](/home/paulbeglin/projects/RITAL-IR-project/main.py)
- [src/hirag/base.py](/home/paulbeglin/projects/RITAL-IR-project/src/hirag/base.py)
- [src/hirag/hirag.py](/home/paulbeglin/projects/RITAL-IR-project/src/hirag/hirag.py)
- [src/hirag/_op.py](/home/paulbeglin/projects/RITAL-IR-project/src/hirag/_op.py)
- [src/hirag/_cluster_utils.py](/home/paulbeglin/projects/RITAL-IR-project/src/hirag/_cluster_utils.py)
- [eval/retrieval_context_benchmark.py](/home/paulbeglin/projects/RITAL-IR-project/eval/retrieval_context_benchmark.py)
- [eval/analyze_cluster_balance.py](/home/paulbeglin/projects/RITAL-IR-project/eval/analyze_cluster_balance.py)
- [docs/NEXT_STEPS.md](/home/paulbeglin/projects/RITAL-IR-project/docs/NEXT_STEPS.md)
- [docs/RETRIEVAL_EXPERIMENTS.md](/home/paulbeglin/projects/RITAL-IR-project/docs/RETRIEVAL_EXPERIMENTS.md)

## Answer-Level Retrieval Evaluation

### 2026-05-05

Added the next evaluation layer for the retrieval branch and the first budget-controlled bridge variant.

What changed:

- Added `hi_minmax_budgeted`, which uses query-weighted minmax bridge paths with per-path and total bridge edge budgets.
- Added disk persistence for weighted bridge edge embeddings under `.runs/edge_embedding_cache/`.
- Added an answer generation benchmark that runs saved HiRAG graphs against selected query modes and writes resumable `answers.jsonl`.
- Added an OpenAI-compatible pairwise judge harness with swapped answer order and paper-style criteria: comprehensiveness, diversity, empowerment, and overall winner.
- Added a blind human annotation packet builder for human comparison rounds.
- Added answer-level unit tests for resume keys, judge prompt/parsing/summary, human anonymization, and sampling buckets.
- Increased the local vLLM answer-call safety buffer to avoid server-side context-window rejections on long graph prompts.

Validation:

- `uv run pytest tests/test_main_runtime.py tests/test_retrieval.py tests/test_hierarchy.py tests/test_hirag_retrieval_experiments.py tests/test_cluster_balance.py tests/test_answer_level_evaluation.py`
- `uv run python eval/retrieval_context_benchmark.py --working-dir .runs/2026-04-22-agri-resume2/graphs/benchmark_ultra_lean_run1_20260422_204326 --query-file eval/datasets/agriculture/agriculture_query.jsonl --query-limit 2 --variants hi hi_minmax_budgeted --output-dir .runs/retrieval_eval/dev_budgeted_smoke`
- `uv run python eval/answer_generation_benchmark.py --working-dir .runs/2026-04-22-agri-resume2/graphs/benchmark_ultra_lean_run1_20260422_204326 --query-file eval/datasets/agriculture/agriculture_query.jsonl --query-limit 1 --variants hi hi_minmax_budgeted --output-dir .runs/answer_eval/dev_smoke --overwrite --chat-model nvidia/Gemma-4-31B-IT-NVFP4`
- `uv run python eval/pairwise_answer_judge.py --answers .runs/answer_eval/dev_smoke/answers.jsonl --baseline hi --variants hi_minmax_budgeted --judge-base-url http://127.0.0.1:8000/v1 --judge-api-key EMPTY --judge-model nvidia/Gemma-4-31B-IT-NVFP4 --output-dir .runs/answer_eval/dev_smoke_judge --max-concurrency 1 --overwrite`
- `uv run python eval/build_human_annotation_packet.py --answers .runs/answer_eval/dev_smoke/answers.jsonl --judge-results .runs/answer_eval/dev_smoke_judge/judge_results.jsonl --query-count 1 --variants hi hi_minmax_budgeted --output-dir .runs/human_eval/dev_smoke`

Relevant files:

- [eval/answer_generation_benchmark.py](/home/paulbeglin/projects/RITAL-IR-project/eval/answer_generation_benchmark.py)
- [eval/pairwise_answer_judge.py](/home/paulbeglin/projects/RITAL-IR-project/eval/pairwise_answer_judge.py)
- [eval/build_human_annotation_packet.py](/home/paulbeglin/projects/RITAL-IR-project/eval/build_human_annotation_packet.py)
- [eval/eval_utils.py](/home/paulbeglin/projects/RITAL-IR-project/eval/eval_utils.py)
- [src/hirag/_op.py](/home/paulbeglin/projects/RITAL-IR-project/src/hirag/_op.py)
- [docs/RETRIEVAL_EXPERIMENTS.md](/home/paulbeglin/projects/RITAL-IR-project/docs/RETRIEVAL_EXPERIMENTS.md)
- [docs/RETRIEVAL_RESULTS_AND_NEXT_IDEAS.md](/home/paulbeglin/projects/RITAL-IR-project/docs/RETRIEVAL_RESULTS_AND_NEXT_IDEAS.md)

## OpenAI Batch Export And Compact Answer Budgets

### 2026-05-05

Added a reliable batch-export route and tightened answer-level context construction.

What changed:

- Added OpenAI Batch API JSONL export for answer generation requests.
- Added compact answer-context construction that builds context first, estimates chat prompt tokens, and shrinks budgets before generation/export.
- Added explicit answer token cap and request timeout to the local answer harness.
- Added bounded source-document snippets so compact contexts keep source evidence instead of dropping full chunks.
- Added a retrieval budget sweep script for comparing tiny/small/medium/large answer-context regimes.

Validation:

- `uv run pytest tests/test_main_runtime.py tests/test_answer_level_evaluation.py tests/test_hirag_retrieval_experiments.py`
- `uv run python eval/export_openai_batch_requests.py --working-dir .runs/2026-04-22-agri-resume2/graphs/benchmark_ultra_lean_run1_20260422_204326 --query-file eval/datasets/agriculture/agriculture_query.jsonl --query-limit 2 --variants hi hi_minmax_budgeted --model gpt-5.4-mini --output-dir .runs/openai_batch/dev_snippet_smoke --overwrite --include-contexts`
- `uv run python eval/answer_generation_benchmark.py --working-dir .runs/2026-04-22-agri-resume2/graphs/benchmark_ultra_lean_run1_20260422_204326 --query-file eval/datasets/agriculture/agriculture_query.jsonl --query-limit 1 --variants hi hi_minmax_budgeted --output-dir .runs/answer_eval/dev_compact_smoke --overwrite --chat-model nvidia/Gemma-4-31B-IT-NVFP4 --base-url http://127.0.0.1:8000/v1 --api-key EMPTY --request-timeout-seconds 120`
- `uv run python eval/retrieval_budget_sweep.py --working-dir .runs/2026-04-22-agri-resume2/graphs/benchmark_ultra_lean_run1_20260422_204326 --query-file eval/datasets/agriculture/agriculture_query.jsonl --query-limit 3 --variants hi hi_minmax_budgeted hi_rerank_weighted --regimes tiny small medium --output-dir .runs/retrieval_budget_sweep/dev_snippet_smoke`

Relevant files:

- [eval/export_openai_batch_requests.py](/home/paulbeglin/projects/RITAL-IR-project/eval/export_openai_batch_requests.py)
- [eval/retrieval_budget_sweep.py](/home/paulbeglin/projects/RITAL-IR-project/eval/retrieval_budget_sweep.py)
- [eval/eval_utils.py](/home/paulbeglin/projects/RITAL-IR-project/eval/eval_utils.py)
- [eval/answer_generation_benchmark.py](/home/paulbeglin/projects/RITAL-IR-project/eval/answer_generation_benchmark.py)
- [src/hirag/base.py](/home/paulbeglin/projects/RITAL-IR-project/src/hirag/base.py)
- [src/hirag/_op.py](/home/paulbeglin/projects/RITAL-IR-project/src/hirag/_op.py)

## OpenAI Batch Result Pipeline

### 2026-05-05

Added the downstream batch utilities needed after answer generation.

What changed:

- Added importer for OpenAI answer-batch outputs into the repo's normal `answers.jsonl` schema.
- Added exporter for pairwise judge requests as OpenAI Batch JSONL.
- Added importer for judge-batch outputs into `judge_results.jsonl`, `judge_summary.csv`, and `judge_summary.md`.
- Added tests with synthetic OpenAI Batch output rows.

Validation:

- `uv run pytest tests/test_answer_level_evaluation.py tests/test_main_runtime.py`
- `uv run python eval/export_openai_judge_batch_requests.py --answers .runs/answer_eval/dev_compact_smoke/answers.jsonl --baseline hi --variants hi_minmax_budgeted --model gpt-5.4-mini --output-dir .runs/answer_eval/dev_judge_batch_smoke`

Relevant files:

- [eval/import_openai_batch_answers.py](/home/paulbeglin/projects/RITAL-IR-project/eval/import_openai_batch_answers.py)
- [eval/export_openai_judge_batch_requests.py](/home/paulbeglin/projects/RITAL-IR-project/eval/export_openai_judge_batch_requests.py)
- [eval/import_openai_batch_judgments.py](/home/paulbeglin/projects/RITAL-IR-project/eval/import_openai_batch_judgments.py)
- [eval/pairwise_answer_judge.py](/home/paulbeglin/projects/RITAL-IR-project/eval/pairwise_answer_judge.py)

## Query-Aware Source Snippets

### 2026-05-05

Added a compact retrieval improvement for answer-eval and OpenAI Batch exports.

What changed:

- Added `QueryParam.text_unit_snippet_strategy` with `prefix` and `query_overlap` options.
- Kept raw HiRAG behavior compatible by defaulting `QueryParam` to `prefix`.
- Defaulted answer-eval/export contexts to `query_overlap`, so bounded source snippets are selected from the most query-overlapping sentence/window inside each retrieved chunk instead of always truncating from the chunk prefix.
- Added focused unit coverage for query-overlap source snippet selection.

Smoke observation:

- On the first agriculture query with `hi`, query-overlap snippets reduced the exported answer prompt estimate from 4,403 to 3,799 input tokens while keeping source evidence from the same retrieved chunks.

Validation:

- `uv run pytest tests/test_hirag_retrieval_experiments.py tests/test_answer_level_evaluation.py`
- `uv run python eval/export_openai_batch_requests.py --working-dir .runs/2026-04-22-agri-resume2/graphs/benchmark_ultra_lean_run1_20260422_204326 --query-file eval/datasets/agriculture/agriculture_query.jsonl --query-limit 1 --variants hi --model gpt-5.4-mini --output-dir .runs/openai_batch/dev_query_snippet_smoke --overwrite --include-contexts --text-unit-snippet-strategy query_overlap`
- `uv run python eval/export_openai_batch_requests.py --working-dir .runs/2026-04-22-agri-resume2/graphs/benchmark_ultra_lean_run1_20260422_204326 --query-file eval/datasets/agriculture/agriculture_query.jsonl --query-limit 1 --variants hi --model gpt-5.4-mini --output-dir .runs/openai_batch/dev_prefix_snippet_smoke --overwrite --include-contexts --text-unit-snippet-strategy prefix`

Relevant files:

- [src/hirag/base.py](/home/paulbeglin/projects/RITAL-IR-project/src/hirag/base.py)
- [src/hirag/_op.py](/home/paulbeglin/projects/RITAL-IR-project/src/hirag/_op.py)
- [eval/eval_utils.py](/home/paulbeglin/projects/RITAL-IR-project/eval/eval_utils.py)
- [eval/retrieval_budget_sweep.py](/home/paulbeglin/projects/RITAL-IR-project/eval/retrieval_budget_sweep.py)
