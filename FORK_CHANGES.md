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

## OpenAI Batch Completion Token Parameter

### 2026-05-05

Fixed OpenAI Batch request exports for GPT-5.x Chat Completions models.

What changed:

- Replaced the default Batch output-token field from legacy `max_tokens` to `max_completion_tokens`.
- Added `--completion-token-param {max_completion_tokens,max_tokens,none}` to both answer and judge batch exporters.
- Updated tests and docs so GPT-5.4/GPT-5.4-mini Batch JSONL files do not include unsupported `max_tokens`.

Reason:

- The first submitted OpenAI Batch returned `400 unsupported_parameter` for every row: `max_tokens` is not supported with this model; use `max_completion_tokens` instead.

Relevant files:

- [eval/export_openai_batch_requests.py](/home/paulbeglin/projects/RITAL-IR-project/eval/export_openai_batch_requests.py)
- [eval/export_openai_judge_batch_requests.py](/home/paulbeglin/projects/RITAL-IR-project/eval/export_openai_judge_batch_requests.py)
- [tests/test_answer_level_evaluation.py](/home/paulbeglin/projects/RITAL-IR-project/tests/test_answer_level_evaluation.py)

## OpenAI Batch Indexing Prototype

### 2026-05-05

Added the first reusable pieces for graph-construction batching.

What changed:

- Added an entity-extraction Batch exporter for HiRAG indexing chunks.
- Added an importer/parser for entity-extraction Batch outputs.
- Added a stage plan to the exporter manifest that documents which graph-indexing stages are batchable immediately and which stages are sequentially dependent on prior outputs.
- Added tests for request shape, prompt construction, stage planning, and entity-output parsing.

Current scope:

- This only covers the first graph-indexing LLM stage: `entity_extract`.
- The next stage, `relation_extract`, requires imported per-chunk entities from the first batch before its requests can be built.
- Later `cluster_summary` and `community_report` stages are batchable within each layer/level, but layers/levels are sequential.

Smoke:

- `uv run python eval/export_openai_index_entity_batch.py --context-file eval/datasets/agriculture/agriculture_unique_contexts.json --model gpt-5.4-mini --output-dir .runs/openai_index_batch/agriculture_entity_extract_smoke --prompt-regime ultra_lean --overwrite`
- Result: 1,756 entity-extraction requests for agriculture, 12 source documents, about 2.52M estimated input tokens, using `max_completion_tokens`.
- A local JSONL validator confirmed 1,756 unique request IDs and no legacy `max_tokens` fields.

Validation:

- `uv run pytest tests/test_openai_index_batch.py tests/test_answer_level_evaluation.py`

Relevant files:

- [eval/export_openai_index_entity_batch.py](/home/paulbeglin/projects/RITAL-IR-project/eval/export_openai_index_entity_batch.py)
- [eval/import_openai_index_entity_batch.py](/home/paulbeglin/projects/RITAL-IR-project/eval/import_openai_index_entity_batch.py)
- [tests/test_openai_index_batch.py](/home/paulbeglin/projects/RITAL-IR-project/tests/test_openai_index_batch.py)

## OpenAI Batch Indexing Relation Stage

### 2026-05-05

Added the second reusable graph-construction Batch stage.

What changed:

- Added a relation-extraction Batch exporter that consumes imported per-chunk entity results.
- Added a relation Batch importer/parser that converts OpenAI Batch output into parsed per-chunk relationships.
- Added tests for relation prompt construction, request shape, ordered entity-name handling, and output parsing.

Current indexing Batch chain:

1. `export_openai_index_entity_batch.py`
2. OpenAI Batch entity extraction
3. `import_openai_index_entity_batch.py`
4. `export_openai_index_relation_batch.py`
5. OpenAI Batch relation extraction
6. `import_openai_index_relation_batch.py`

Smoke:

- Exported 2 synthetic relation-extraction requests against real agriculture chunks using synthetic imported entities.
- Local JSONL validation confirmed unique request IDs, `/v1/chat/completions`, `max_completion_tokens`, and no legacy `max_tokens`.

Validation:

- `uv run pytest tests/test_openai_index_batch.py`

Relevant files:

- [eval/export_openai_index_relation_batch.py](/home/paulbeglin/projects/RITAL-IR-project/eval/export_openai_index_relation_batch.py)
- [eval/import_openai_index_relation_batch.py](/home/paulbeglin/projects/RITAL-IR-project/eval/import_openai_index_relation_batch.py)
- [tests/test_openai_index_batch.py](/home/paulbeglin/projects/RITAL-IR-project/tests/test_openai_index_batch.py)

## OpenAI Batch Base Graph Assembly

### 2026-05-05

Added an offline assembly step for imported OpenAI Batch graph-indexing outputs.

What changed:

- Added `assemble_openai_index_base_graph.py`, which merges parsed entity and relation JSONL files into a base GraphML artifact.
- The assembler deduplicates entity nodes, combines descriptions/source chunk IDs with `GRAPH_FIELD_SEP`, sums duplicate edge weights, and writes `merged_nodes.json`, `merged_edges.json`, `summary.json`, and `summary.md`.
- Added tests for duplicate node/edge merging and GraphML output.

Current scope:

- This produces the flat base graph after entity and relation extraction.
- It does not yet run hierarchical GMM summaries, Leiden clustering, community reports, or vector DB population.

Smoke:

- Assembled a synthetic two-node/one-edge graph into `.runs/openai_index_batch/base_graph_assemble_smoke/base_graph.graphml`.

Validation:

- `uv run pytest tests/test_openai_index_batch.py`

Relevant files:

- [eval/assemble_openai_index_base_graph.py](/home/paulbeglin/projects/RITAL-IR-project/eval/assemble_openai_index_base_graph.py)
- [tests/test_openai_index_batch.py](/home/paulbeglin/projects/RITAL-IR-project/tests/test_openai_index_batch.py)

## Final Evaluation Batch Prep

### 2026-05-10

Prepared the final answer-level evaluation loop for the project presentation.

What changed:

- Fixed OpenAI Batch judge import summaries so imported rows with `latency_seconds: null` do not crash summary generation.
- Imported the two successful agriculture answer Batch outputs into a complete q30 answer set:
  - 30 queries
  - `hi`, `hi_minmax_budgeted`, and `hi_rerank_weighted`
  - 90 successful answers and no response errors
- Imported the available partial agriculture judge Batch output and generated a missing-judge Batch request file for the remaining 74 comparisons.
- Added a compact final-evaluation working summary under the curated agriculture artifact folder, including retrieval proxy metrics, answer token/cost metrics, partial judge win rates, and qualitative examples.
- Generated GPT-5.4 Mini Mix judge request JSONL files from the existing Mix evaluation request files so Mix can be judged with the same current model family.
- Brought `Fiche_Rendu_RI.pdf` onto the retrieval branch from the newer concise remote checkpoint.

Relevant files:

- [eval/pairwise_answer_judge.py](/home/paulbeglin/projects/RITAL-IR-project/eval/pairwise_answer_judge.py)
- [artifacts/agriculture_graphs_2026-04-23/final_evaluation/summary.md](/home/paulbeglin/projects/RITAL-IR-project/artifacts/agriculture_graphs_2026-04-23/final_evaluation/summary.md)
- [Fiche_Rendu_RI.pdf](/home/paulbeglin/projects/RITAL-IR-project/Fiche_Rendu_RI.pdf)

## Mix Batch Graph To Answer Batch

### 2026-05-11

Materialized the GPT-5.4 Mini Mix Batch-indexed relationship graph into a
queryable HiRAG workdir and exported fresh answer-generation requests.

What changed:

- Added `materialize_openai_index_hirag_workdir.py`, which rebuilds HiRAG's
  local storage files from a Batch-assembled GraphML graph and the original
  unique-context JSON.
- The materializer regenerates full-doc and text-chunk KV stores, runs the
  existing Leiden clustering, creates extractive community reports, and
  populates FastEmbed entity/chunk vector stores.
- Materialized the final Mix Batch graph into `.runs/openai_index_batch/mix_hirag_workdir_gpt54_mini_final`.
- Exported 90 fresh GPT-5.4 Mini answer Batch requests for 30 Mix queries across
  `hi`, `hi_minmax_budgeted`, and `hi_rerank_weighted`.
- Added concise final-presentation notes covering the paper, weak points,
  fork improvements, agriculture evidence, and the Mix evaluation handoff.

Current Mix materialized graph:

- 61 docs
- 579 chunks
- 17,575 graph nodes
- 18,588 graph edges
- 12,572 clustered nodes
- 2,759 extractive community reports

Validation:

- `uv run python eval/materialize_openai_index_hirag_workdir.py --context-file eval/datasets/mix/mix_unique_contexts.json --graphml artifacts/mix_batch_requests_2026-05-10/OpenAI_batch_indexing_graphs/mix_base_graph_gpt54_mini_final/base_graph.graphml --output-dir .runs/openai_index_batch/mix_hirag_workdir_gpt54_mini_final --overwrite --embedding-batch-num 64 --fastembed-cache-path .runs/fastembed_cache`
- `uv run python eval/export_openai_batch_requests.py --working-dir .runs/openai_index_batch/mix_hirag_workdir_gpt54_mini_final --query-file eval/datasets/mix/mix.jsonl --query-limit 30 --variants hi hi_minmax_budgeted hi_rerank_weighted --model gpt-5.4-mini --output-dir artifacts/mix_batch_requests_2026-05-10/OpenAI_batch_answer_requests/mix_q30_answers_gpt54_mini_from_batch_graph --include-contexts --answer-max-tokens 512 --completion-token-param max_completion_tokens`

Relevant files:

- [eval/materialize_openai_index_hirag_workdir.py](/home/paulbeglin/projects/RITAL-IR-project/eval/materialize_openai_index_hirag_workdir.py)
- [artifacts/mix_batch_requests_2026-05-10/OpenAI_batch_answer_requests/mix_q30_answers_gpt54_mini_from_batch_graph/manifest.json](/home/paulbeglin/projects/RITAL-IR-project/artifacts/mix_batch_requests_2026-05-10/OpenAI_batch_answer_requests/mix_q30_answers_gpt54_mini_from_batch_graph/manifest.json)
- [artifacts/final_project_presentation_notes_2026-05-11.md](/home/paulbeglin/projects/RITAL-IR-project/artifacts/final_project_presentation_notes_2026-05-11.md)

### 2026-05-11 Mix Answer Batch Import

Imported the completed Mix q30 answer-generation Batch output and generated the
actual judge Batch requests.

What changed:

- Imported 90 GPT-5.4 Mini Mix answers with 0 errors.
- Generated 120 pairwise judge requests:
  - 30 queries
  - `hi_minmax_budgeted` vs `hi`
  - `hi_rerank_weighted` vs `hi`
  - two swapped answer orders per comparison

Answer-batch usage:

- `hi`: 108,851 total tokens
- `hi_minmax_budgeted`: 115,361 total tokens
- `hi_rerank_weighted`: 103,187 total tokens

Notable retrieval/cost signal:

- `hi_rerank_weighted` used the smallest answer prompt budget on Mix
  (mean context input tokens 3,255.9) and fewer bridge edges on average.
- `hi_minmax_budgeted` used the largest answer prompt budget
  (mean context input tokens 3,692.1) and more bridge edges on average.
- Judge results are still pending; the answer batch is complete, but the judge
  batch still needs to be submitted and imported.

Relevant files:

- [artifacts/mix_batch_requests_2026-05-10/final_answer_import/mix_q30_answers_gpt54_mini_from_batch_graph/answers.jsonl](/home/paulbeglin/projects/RITAL-IR-project/artifacts/mix_batch_requests_2026-05-10/final_answer_import/mix_q30_answers_gpt54_mini_from_batch_graph/answers.jsonl)
- [artifacts/mix_batch_requests_2026-05-10/OpenAI_batch_judge_requests/mix_q30_judge_gpt54_mini_from_batch_graph/judge_requests.jsonl](/home/paulbeglin/projects/RITAL-IR-project/artifacts/mix_batch_requests_2026-05-10/OpenAI_batch_judge_requests/mix_q30_judge_gpt54_mini_from_batch_graph/judge_requests.jsonl)

### 2026-05-11 Batch Submission Helper

Added a generic OpenAI Batch launcher for repo-generated JSONL files.

What changed:

- Added `run_openai_batch_file.py`.
- The script validates JSONL row count, duplicate `custom_id`s, and per-file
  Batch limits before upload.
- It uploads with `purpose="batch"`, creates a Batch job, optionally polls until
  completion, and downloads output/error files.
- API keys are read from `OPENAI_API_KEY` after optionally loading `.env`; keys
  should not be passed in command arguments or chat.
- Added `OPENAI_API_KEY=` to `.env.example` and created an ignored local `.env`
  placeholder for the active workspace.

Relevant file:

- [eval/run_openai_batch_file.py](/home/paulbeglin/projects/RITAL-IR-project/eval/run_openai_batch_file.py)

### 2026-05-11 Mix Judge Import

Imported the completed Mix q30 judge Batch output.

What changed:

- Imported 120 judge rows with 0 parse errors.
- Added swapped-order agreement artifacts for interpreting judge reliability.
- Updated final presentation notes with the Mix judge result.

Raw swapped-order judge result:

- `hi_minmax_budgeted` vs `hi`: 17 wins, 40 losses, 3 ties, win rate 28.3%.
- `hi_rerank_weighted` vs `hi`: 33 wins, 27 losses, 0 ties, win rate 55.0%.

Stricter swapped-order agreement:

- `hi_minmax_budgeted`: 14/30 query pairs agreed, strict variant win rate 6.7%.
- `hi_rerank_weighted`: 21/30 query pairs agreed, strict variant win rate 40.0%.

Takeaway:

- Mix supports the same practical story as agriculture: rerank-weighted retrieval
  is the cleaner improvement signal, while the minmax-budgeted variant is more
  brittle on this cross-domain graph.

Relevant files:

- [artifacts/mix_batch_requests_2026-05-10/final_judge_import/mix_q30_judge_gpt54_mini_from_batch_graph/judge_summary.md](/home/paulbeglin/projects/RITAL-IR-project/artifacts/mix_batch_requests_2026-05-10/final_judge_import/mix_q30_judge_gpt54_mini_from_batch_graph/judge_summary.md)
- [artifacts/mix_batch_requests_2026-05-10/final_judge_import/mix_q30_judge_gpt54_mini_from_batch_graph/swapped_order_agreement.md](/home/paulbeglin/projects/RITAL-IR-project/artifacts/mix_batch_requests_2026-05-10/final_judge_import/mix_q30_judge_gpt54_mini_from_batch_graph/swapped_order_agreement.md)

## Final Agriculture Judge Completion And Analysis

### 2026-05-21

Finished the remaining agriculture q30 judge evaluation and consolidated the
final project results.

What changed:

- Submitted the prepared 74-row missing agriculture judge Batch file with the
  repo Batch launcher.
- Imported the completed output and merged it with the earlier 46 imported
  judge rows into a full 120-row agriculture judge result set.
- Added full agriculture swapped-order agreement artifacts.
- Added a final consolidated analysis folder comparing Agriculture and Mix
  answer-level judge results, strict swapped-order agreement, token cost, graph
  construction status, and final limitations.
- Updated the presentation notes to remove stale "pending submission" language
  and center the final claim around `hi_rerank_weighted`.

Final answer-level judge signal:

- Agriculture:
  - `hi_minmax_budgeted` vs `hi`: 30 wins, 30 losses, 0 ties, raw win rate 50.0%.
  - `hi_rerank_weighted` vs `hi`: 35 wins, 25 losses, 0 ties, raw win rate 58.3%.
- Mix:
  - `hi_minmax_budgeted` vs `hi`: 17 wins, 40 losses, 3 ties, raw win rate 28.3%.
  - `hi_rerank_weighted` vs `hi`: 33 wins, 27 losses, 0 ties, raw win rate 55.0%.

Takeaway:

- `hi_rerank_weighted` is the cleaner final improvement: positive raw judge win
  rate on both final datasets, stronger swapped-order reliability than
  `hi_minmax_budgeted`, and lower answer-token usage than baseline.

Relevant files:

- [artifacts/final_analysis_2026-05-21/summary.md](/home/paulbeglin/projects/RITAL-IR-project/artifacts/final_analysis_2026-05-21/summary.md)
- [artifacts/agriculture_graphs_2026-04-23/final_evaluation/full_q30_judge_summary.md](/home/paulbeglin/projects/RITAL-IR-project/artifacts/agriculture_graphs_2026-04-23/final_evaluation/full_q30_judge_summary.md)
- [artifacts/agriculture_graphs_2026-04-23/final_evaluation/swapped_order_agreement.md](/home/paulbeglin/projects/RITAL-IR-project/artifacts/agriculture_graphs_2026-04-23/final_evaluation/swapped_order_agreement.md)
- [artifacts/final_project_presentation_notes_2026-05-11.md](/home/paulbeglin/projects/RITAL-IR-project/artifacts/final_project_presentation_notes_2026-05-11.md)

## Full OpenAI Batch HiRAG Run Preparation

### 2026-05-21

Prepared the tooling needed to rerun Agriculture and Mix with a homogeneous
OpenAI Batch pipeline instead of the local vLLM-constrained path.

What changed:

- Added generic retry-request export for OpenAI Batch rows with API errors,
  HTTP errors, or `finish_reason=length`.
- Added Batch exporters/importers for:
  - cluster summaries
  - community reports
- Added a graph application step that inserts parsed cluster-summary entities
  and relations back into GraphML before final community reporting.
- Extended workdir materialization so final runs can:
  - consume imported LLM community reports instead of extractive reports
  - use OpenAI embeddings through `text-embedding-3-small`
  - consume precomputed Batch embedding JSONL files and write `NanoVectorDB`
    stores directly
  - preserve existing cluster annotations with `--skip-clustering`
- Added Batch exporters/importers for `text-embedding-3-small` embeddings.
- Added a cost-summary utility for collecting Batch usage across a run folder.
- Added a full OpenAI HiRAG runbook with Phase 1 Agriculture/Mix commands,
  token caps, screen launch patterns, retry policy, and Go/No-Go criteria before
  CS/Legal.

Validation:

- `uv run pytest tests/test_openai_index_batch.py`
- `uv run python -m py_compile` on the new Batch/materialization scripts
- `--help` smoke checks for the new CLI scripts

Relevant files:

- [docs/OPENAI_FULL_HIRAG_RUNBOOK.md](/home/paulbeglin/projects/RITAL-IR-project/docs/OPENAI_FULL_HIRAG_RUNBOOK.md)
- [eval/export_openai_batch_retry_requests.py](/home/paulbeglin/projects/RITAL-IR-project/eval/export_openai_batch_retry_requests.py)
- [eval/export_openai_index_cluster_summary_batch.py](/home/paulbeglin/projects/RITAL-IR-project/eval/export_openai_index_cluster_summary_batch.py)
- [eval/export_openai_index_community_report_batch.py](/home/paulbeglin/projects/RITAL-IR-project/eval/export_openai_index_community_report_batch.py)
- [eval/materialize_openai_index_hirag_workdir.py](/home/paulbeglin/projects/RITAL-IR-project/eval/materialize_openai_index_hirag_workdir.py)

## Dynamic Graph Traversal Extension Note

### 2026-05-21

Documented a proposed next research extension for controlled dynamic graph
traversal on top of `hi_rerank_weighted`.

What changed:

- Summarized the current local/global/bridge retrieval hyperparameters.
- Clarified that local retrieval uses vector seed entities plus one-hop graph
  evidence, not a configurable graph radius.
- Described a Level 2 dynamic community traversal design with typed tools and
  budgeted community selection.
- Described a Level 3 agentic graph traversal design with constrained graph
  tools instead of free-form Cypher.
- Recommended dynamic community and path selection as the most defensible next
  improvement after the full OpenAI Batch reruns.

Relevant file:

- [docs/DYNAMIC_GRAPH_TRAVERSAL_EXTENSION.md](/home/paulbeglin/projects/RITAL-IR-project/docs/DYNAMIC_GRAPH_TRAVERSAL_EXTENSION.md)

## Retrieval Trace Capture For Answer Runs

### 2026-05-21

Extended answer Batch export so final OpenAI answer runs preserve detailed
retrieval traces for later benchmarking.

What changed:

- `eval/export_openai_batch_requests.py` now writes `retrieval_traces.jsonl`
  for every query/variant, including selected entities, communities, bridge
  edges, bridge path decisions, path length metrics, context sections, and
  budget attempts.
- Added `eval/summarize_retrieval_traces.py` to produce CSV/JSON/Markdown
  summaries grouped by retrieval variant.
- Updated the OpenAI full-run runbook to keep and summarize these traces after
  answer request export.

Validation:

- `uv run pytest tests/test_answer_level_evaluation.py tests/test_hirag_retrieval_experiments.py tests/test_cluster_balance.py`
- `uv run python -m py_compile eval/export_openai_batch_requests.py eval/summarize_retrieval_traces.py`

Relevant files:

- [eval/export_openai_batch_requests.py](/home/paulbeglin/projects/RITAL-IR-project/eval/export_openai_batch_requests.py)
- [eval/summarize_retrieval_traces.py](/home/paulbeglin/projects/RITAL-IR-project/eval/summarize_retrieval_traces.py)
- [docs/OPENAI_FULL_HIRAG_RUNBOOK.md](/home/paulbeglin/projects/RITAL-IR-project/docs/OPENAI_FULL_HIRAG_RUNBOOK.md)

## Run Inventory And Artifact Organization

### 2026-05-21

Added a central run index and a reserved artifact destination for the full
OpenAI-only HiRAG reruns.

What changed:

- Added `artifacts/RUNS_INDEX_2026-05-21.md` to classify active, final,
  legacy, superseded, smoke, and cache run folders.
- Reserved `artifacts/openai_full_hirag_2026-05/` as the clean destination for
  validated Agriculture and Mix full OpenAI rerun outputs.
- Updated the shared artifact inventory to point collaborators to the run index
  and future full OpenAI artifact destination.

Safety note:

- `.runs/openai_full_hirag_2026-05/` remains untouched while the active Batch
  `screen` sessions are running.

Relevant files:

- [artifacts/RUNS_INDEX_2026-05-21.md](/home/paulbeglin/projects/RITAL-IR-project/artifacts/RUNS_INDEX_2026-05-21.md)
- [artifacts/openai_full_hirag_2026-05/README.md](/home/paulbeglin/projects/RITAL-IR-project/artifacts/openai_full_hirag_2026-05/README.md)

## Direct API Micro-Retry Helper

### 2026-05-22

Added a secondary live-API execution path for tiny OpenAI retry files while
keeping Batch API as the primary long-run path.

What changed:

- Added `eval/run_openai_requests_file_direct.py`, which reads Batch-style
  JSONL requests and calls the normal OpenAI API directly for:
  - `/v1/chat/completions`
  - `/v1/embeddings`
- The helper writes Batch-compatible output JSONL so existing importers can
  parse direct results without special-case code.
- Added endpoint guardrails to reject mixed or unsupported request files.
- Added tests for direct chat dispatch, embedding dispatch, and endpoint
  validation.

Why this matters:

- For micro-retries with only a handful of rows, direct API can avoid waiting
  for Batch scheduling/finalization.
- The full Agriculture/Mix rerun still uses Batch as the default path for
  substantial stages and cost efficiency.

Relevant files:

- [eval/run_openai_requests_file_direct.py](/home/paulbeglin/projects/RITAL-IR-project/eval/run_openai_requests_file_direct.py)
- [tests/test_openai_index_batch.py](/home/paulbeglin/projects/RITAL-IR-project/tests/test_openai_index_batch.py)
