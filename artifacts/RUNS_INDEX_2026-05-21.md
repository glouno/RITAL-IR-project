# Runs Index

Date: 2026-05-21

This index classifies the current run folders so collaborators know what to
read, what is final, what is legacy, and what should not be touched while Batch
jobs are active.

## Status And Action Vocabulary

Statuses:

- `final`: final shared project evidence.
- `legacy_useful`: older or intermediate evidence worth sharing for analysis.
- `active`: currently used by running jobs or ongoing staged work.
- `superseded`: useful historically, but already represented by cleaner shared artifacts.
- `smoke`: local smoke tests or tiny validation runs.
- `cache`: generated cache data, not experiment evidence.
- `legacy_local_graph_runs`: old local graph runs; keep until coverage is verified.

Actions:

- `keep`: keep and continue sharing.
- `promote`: copy selected outputs into `artifacts/`.
- `archive_later`: compress or move out of repo if disk pressure matters.
- `delete_later`: safe deletion candidate after active runs complete and artifacts are promoted.
- `do_not_touch`: do not move, delete, or rewrite.

## Active Runs

| Run | Path | Status | Dataset | Model | Variants | Size | Important Artifacts | Action |
|---|---|---|---|---|---|---:|---|---|
| Full OpenAI HiRAG Phase 1 | `.runs/openai_full_hirag_2026-05/` | `active` | Agriculture, Mix | `gpt-5.4-mini`, `text-embedding-3-small` | Planned: `hi`, `naive`, `hi_nobridge`, `hi_rerank_weighted`, `hi_minmax_budgeted` | ~29M now | Batch manifests, staged requests/results, future final workdirs | `do_not_touch` |
| Agriculture full OpenAI entity Batch | `.runs/openai_full_hirag_2026-05/agriculture/` | `active` | Agriculture | `gpt-5.4-mini` | N/A indexing stage | included above | `02_entity_results/raw_batch/batch_run_manifest.json` | `do_not_touch` |
| Mix full OpenAI entity Batch | `.runs/openai_full_hirag_2026-05/mix/` | `active` | Mix | `gpt-5.4-mini` | N/A indexing stage | included above | `02_entity_results/raw_batch/batch_run_manifest.json` | `do_not_touch` |

Current Batch snapshot when this index was written:

- Agriculture entity Batch: `batch_6a0f1b1a65d88190b1869bdab3d9def2`, `in_progress`, `0/1756`.
- Mix entity Batch: `batch_6a0f1b18f070819086f3e16305beb860`, `in_progress`, `494/579`.

## Shared Final Artifacts

| Run | Path | Status | Dataset | Model | Variants | Size | Important Artifacts | Action |
|---|---|---|---|---|---|---:|---|---|
| Final cross-dataset analysis | `artifacts/final_analysis_2026-05-21/` | `final` | Agriculture, Mix | mixed final eval | `hi`, `hi_minmax_budgeted`, `hi_rerank_weighted` | ~20K | `summary.md`, `answer_costs.csv`, `judge_results.csv` | `keep` |
| Agriculture graph and final eval package | `artifacts/agriculture_graphs_2026-04-23/` | `final` | Agriculture | local vLLM/FastEmbed + OpenAI judges | `hi`, `hi_minmax_budgeted`, `hi_rerank_weighted` plus legacy graph variants | ~178M | GraphML, visualizations, community reports, final answers/judges | `keep` |
| Mix Batch graph and final eval package | `artifacts/mix_batch_requests_2026-05-10/` | `final` | Mix | `gpt-5.4-mini` for Batch/eval | `hi`, `hi_minmax_budgeted`, `hi_rerank_weighted` | ~35M | Batch graph, final answers, contexts, judges, summaries | `keep` |
| Legacy answer evaluation package | `artifacts/legacy_answer_eval_2026-05-21/` | `legacy_useful` | Agriculture | `gpt-5.4`, `gpt-5.4-mini` | `hi`, `hi_minmax_budgeted`, `hi_rerank_weighted`, partial dev variants | ~7.2M | Legacy contexts, answer requests, answers, judge results | `keep` |

## Useful Local Runs Not Fully Shared

| Run | Path | Status | Dataset | Model | Variants | Size | Important Artifacts | Action |
|---|---|---|---|---|---|---:|---|---|
| Agriculture fixed answer requests | `.runs/openai_batch/agriculture_q30_gpt54_fixed/` | `legacy_useful` | Agriculture | `gpt-5.4` | `hi`, `hi_minmax_budgeted`, `hi_rerank_weighted` | ~3.6M | `contexts.jsonl`, `answer_requests.jsonl`, `manifest.json`; promoted to `artifacts/legacy_answer_eval_2026-05-21/` | `keep` |
| Agriculture mini fixed answer requests | `.runs/openai_batch/agriculture_q30_gpt54_mini_fixed/` | `legacy_useful` | Agriculture | `gpt-5.4-mini` | `hi`, `hi_minmax_budgeted`, `hi_rerank_weighted` | ~3.6M | `contexts.jsonl`, `answer_requests.jsonl`, `manifest.json`; promoted to `artifacts/legacy_answer_eval_2026-05-21/` | `keep` |
| Agriculture answer imports and judges | `.runs/answer_eval/agriculture_q30_gpt54_mini_*` | `legacy_useful` | Agriculture | `gpt-5.4-mini` | `hi`, `hi_minmax_budgeted`, `hi_rerank_weighted` | ~2.3M total answer_eval | `answers.jsonl`, `judge_results.jsonl`, summaries; final imports promoted to `artifacts/legacy_answer_eval_2026-05-21/` | `keep` |
| Retrieval q50 context benchmark | `.runs/retrieval_eval/dev_retrieval_q50/` | `legacy_useful` | Agriculture | local embedding/rerank stack | multiple retrieval variants | ~48M context file, small summaries | `metrics.csv`, `summary.md`, `summary.json` | `promote_summaries_only` |
| Retrieval budget sweeps | `.runs/retrieval_budget_sweep/` | `legacy_useful` | Agriculture/dev | local retrieval stack | budget configs | ~36K | `budget_sweep.csv`, `summary.csv`, `summary.md` | `promote_summaries_only` |
| Cluster balance diagnostics | `.runs/cluster_balance/dev_retrieval_smoke/` | `legacy_useful` | Agriculture/dev | local graph diagnostics | N/A | ~1.2M | `community_balance.csv`, `split_plan.json` | `promote_light_files_only` |

## Superseded Or Cleanup Candidates

| Run | Path | Status | Dataset | Model | Variants | Size | Important Artifacts | Action |
|---|---|---|---|---|---|---:|---|---|
| FastEmbed model cache | `.runs/fastembed_cache/` | `cache` | N/A | FastEmbed | N/A | ~369M | cached models only | `delete_later` |
| Query-weighted edge embedding cache | `.runs/edge_embedding_cache/` | `cache` | Agriculture/Mix graphs | FastEmbed embedding cache | N/A | ~273M | cache JSON files | `delete_later` |
| Mix OpenAI index scratch | `.runs/openai_index_batch/mix_*` | `superseded` | Mix | `gpt-5.4-mini` | indexing stages | part of ~164M | duplicated by `artifacts/mix_batch_requests_2026-05-10/` | `archive_later` |
| OpenAI Batch smoke tests | `.runs/openai_batch_smoke/` | `smoke` | toy | `gpt-5.4-mini` | N/A | ~32K | smoke request/output | `delete_later` |
| Full OpenAI smoke tests | `.runs/openai_full_hirag_smoke/` | `smoke` | test | `text-embedding-3-small` | N/A | ~44K | embedding smoke files | `delete_later` |
| Dev answer smoke tests | `.runs/answer_eval/dev_*`, `.runs/openai_batch/dev_*` | `smoke` | dev | mixed | dev variants | small | smoke contexts/answers/judges | `delete_later` |
| Local Agriculture graph runs | `.runs/2026-04-20-*`, `.runs/2026-04-21-*`, `.runs/2026-04-22-*`, `.runs/20260424_002918_agri_baseline_glean/` | `legacy_local_graph_runs` | Agriculture | local vLLM/FastEmbed | graph build variants | ~178M combined | local graph workdirs/logs; verify coverage by `artifacts/agriculture_graphs_2026-04-23/` before archiving | `archive_later` |

## Promotion Rules For New Full OpenAI Runs

When `.runs/openai_full_hirag_2026-05/` completes and passes validation, promote
only cleaned analysis artifacts into:

```text
artifacts/openai_full_hirag_2026-05/
  agriculture/
  mix/
```

Promote:

- `cost_summary.json`
- `graph_summary.md`
- `quality_summary.md`
- `materialization_summary.md`
- final `answers.jsonl`
- `retrieval_traces.jsonl`
- `retrieval_trace_summary.*`
- final `judge_results.jsonl`
- `judge_summary.*`
- `swapped_order_agreement.*`
- graph summaries and optionally GraphML if size is acceptable

Do not promote by default:

- embedding caches
- NanoVectorDB workdirs
- raw Batch outputs unless audit provenance is required
- failed retries that were superseded
- smoke tests

## Safety Checklist Before Deletion

Run these commands before any future cleanup:

```bash
screen -ls
find .runs/openai_full_hirag_2026-05 -name batch_run_manifest.json -print
git status --short --ignored
du -sh .runs/* artifacts/*
```

Never delete or move:

- any path referenced by an active `screen`;
- any path referenced by a current Batch manifest;
- any `.runs/openai_full_hirag_2026-05/` path until the run is complete and promoted;
- any file that is the only copy of a final answer, judge, trace, graph summary, or cost summary.
