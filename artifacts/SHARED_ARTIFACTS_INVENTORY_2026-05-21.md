# Shared Artifacts Inventory

Date: 2026-05-21

This note separates what is already shared in git from useful local `.runs/`
outputs that may be worth promoting into `artifacts/` for collaborators.

## Current Situation

- `artifacts/` is tracked by git and is currently about 213 MB.
- `.runs/` is gitignored and is currently about 1.1 GB.
- No files under `.runs/openai_batch`, `.runs/answer_eval`,
  `.runs/retrieval_eval`, `.runs/cluster_balance`, or
  `.runs/retrieval_budget_sweep` are tracked by git.
- The most important final Agriculture and Mix answer/judge artifacts are
  already tracked in `artifacts/`.

## Already Shared And Useful

### Final Cross-Dataset Analysis

- `artifacts/final_analysis_2026-05-21/summary.md`
- `artifacts/final_analysis_2026-05-21/answer_costs.csv`
- `artifacts/final_analysis_2026-05-21/judge_results.csv`

These are the best lightweight entry point for collaborators.

### Agriculture Final Evaluation

- `artifacts/agriculture_graphs_2026-04-23/final_evaluation/agriculture_q30_answers_gpt54_mini.jsonl`
- `artifacts/agriculture_graphs_2026-04-23/final_evaluation/agriculture_q30_full_judge_results_gpt54_mini.jsonl`
- `artifacts/agriculture_graphs_2026-04-23/final_evaluation/full_q30_judge_summary.md`
- `artifacts/agriculture_graphs_2026-04-23/final_evaluation/full_q30_judge_win_rates.csv`
- `artifacts/agriculture_graphs_2026-04-23/final_evaluation/swapped_order_agreement.md`
- `artifacts/agriculture_graphs_2026-04-23/final_evaluation/swapped_order_agreement.csv`
- `artifacts/agriculture_graphs_2026-04-23/final_evaluation/q30_answer_costs.csv`
- `artifacts/agriculture_graphs_2026-04-23/final_evaluation/q50_retrieval_proxy.csv`
- `artifacts/agriculture_graphs_2026-04-23/final_evaluation/qualitative_examples.json`

These are enough to review Agriculture answers, judge outcomes, and retrieval
proxy metrics.

### Mix Final Evaluation And Batch Graph

- `artifacts/mix_batch_requests_2026-05-10/final_answer_import/mix_q30_answers_gpt54_mini_from_batch_graph/answers.jsonl`
- `artifacts/mix_batch_requests_2026-05-10/final_judge_import/mix_q30_judge_gpt54_mini_from_batch_graph/judge_results.jsonl`
- `artifacts/mix_batch_requests_2026-05-10/final_judge_import/mix_q30_judge_gpt54_mini_from_batch_graph/judge_summary.csv`
- `artifacts/mix_batch_requests_2026-05-10/final_judge_import/mix_q30_judge_gpt54_mini_from_batch_graph/judge_summary.md`
- `artifacts/mix_batch_requests_2026-05-10/final_judge_import/mix_q30_judge_gpt54_mini_from_batch_graph/swapped_order_agreement.csv`
- `artifacts/mix_batch_requests_2026-05-10/final_judge_import/mix_q30_judge_gpt54_mini_from_batch_graph/swapped_order_agreement.md`
- `artifacts/mix_batch_requests_2026-05-10/OpenAI_batch_answer_requests/mix_q30_answers_gpt54_mini_from_batch_graph/contexts.jsonl`
- `artifacts/mix_batch_requests_2026-05-10/OpenAI_batch_answer_requests/mix_q30_answers_gpt54_mini_from_batch_graph/manifest.json`
- `artifacts/mix_batch_requests_2026-05-10/OpenAI_batch_indexing_graphs/mix_base_graph_gpt54_mini_final/base_graph.graphml`
- `artifacts/mix_batch_requests_2026-05-10/OpenAI_batch_indexing_graphs/mix_base_graph_gpt54_mini_final/summary.md`
- `artifacts/mix_batch_requests_2026-05-10/OpenAI_batch_indexing_workdirs/mix_hirag_workdir_gpt54_mini_final/summary.md`

These are enough to review Mix answers, contexts, judge outcomes, and graph size.

## Useful Local Artifacts Not Shared Yet

These are in `.runs/` only.

### Agriculture Answer Request Contexts

Candidate directories:

- `.runs/openai_batch/agriculture_q30_gpt54_mini_v2/`
- `.runs/openai_batch/agriculture_q30_gpt54_mini_fixed/`
- `.runs/openai_batch/agriculture_q30_gpt54_mini/`

Useful files:

- `contexts.jsonl`
- `manifest.json`
- possibly `answer_requests.jsonl` if collaborators need exact prompts.

Recommendation:

- Do not share all duplicate attempts.
- Share only the exact request context directory that corresponds to the final
  imported Agriculture answers. The final imported answers are already tracked,
  but the exact request contexts are not obviously present in `artifacts/`.
- If uncertain which attempt is final, prefer not to promote yet; the new full
  OpenAI rerun will generate cleaner `retrieval_traces.jsonl`.

### Retrieval Context Benchmarks

Candidate directories:

- `.runs/retrieval_eval/dev_retrieval_q50/`
- `.runs/retrieval_eval/dev_retrieval_smoke/`
- `.runs/retrieval_eval/dev_budgeted_smoke/`

Useful files:

- `metrics.csv`
- `summary.md`
- `summary.json`
- selected `contexts.jsonl` only if collaborators need raw context inspection.

Recommendation:

- Promote `metrics.csv`, `summary.md`, and `summary.json` from
  `dev_retrieval_q50`.
- Avoid committing `dev_retrieval_q50/contexts.jsonl` by default because it is
  about 48 MB and can be regenerated. If raw contexts are needed, compress or
  share externally.

### Retrieval Budget Sweep

Candidate directories:

- `.runs/retrieval_budget_sweep/dev_smoke/`
- `.runs/retrieval_budget_sweep/dev_snippet_smoke/`

Useful files:

- `budget_sweep.csv`
- `summary.csv`
- `summary.md`

Recommendation:

- Promote these lightweight CSV/Markdown summaries if we want collaborators to
  inspect why answer contexts were budgeted down.

### Cluster Balance Diagnostics

Candidate directory:

- `.runs/cluster_balance/dev_retrieval_smoke/`

Useful files:

- `community_balance.csv`
- `split_plan.json`

Recommendation:

- Promote `community_balance.csv` and `split_plan.json` if we discuss graph
  construction limitations or community-size skew.
- Avoid committing `oversized_communities.json` by default; it is larger and
  mostly diagnostic.

## What Not To Share By Default

- `.runs/fastembed_cache/` and `.runs/edge_embedding_cache/`: caches, not
  experiment evidence.
- `.runs/openai_index_batch/*` duplicates that already exist in
  `artifacts/mix_batch_requests_2026-05-10/`.
- Old smoke directories such as `.runs/openai_batch/dev_smoke/` unless a test
  case is needed.
- Raw Batch request files for every failed or superseded attempt; they are large
  and confusing unless tied to a final result.
- Full local workdirs unless collaborators need to run queries offline. Prefer
  graph summaries, GraphML, answer traces, and generated reports.

## Recommended Sharing Package Now

Create a small collaborator-facing package under:

```text
artifacts/collaboration_package_2026-05-21/
```

Suggested contents:

- `README.md` explaining what each file is.
- Copy or link summary files from `artifacts/final_analysis_2026-05-21/`.
- Copy Agriculture final answers/judges/summaries.
- Copy Mix final answers/judges/contexts/summaries.
- Promote lightweight retrieval benchmark summaries:
  - `.runs/retrieval_eval/dev_retrieval_q50/metrics.csv`
  - `.runs/retrieval_eval/dev_retrieval_q50/summary.md`
  - `.runs/retrieval_eval/dev_retrieval_q50/summary.json`
  - `.runs/retrieval_budget_sweep/dev_snippet_smoke/summary.csv`
  - `.runs/retrieval_budget_sweep/dev_snippet_smoke/summary.md`
  - `.runs/cluster_balance/dev_retrieval_smoke/community_balance.csv`
  - `.runs/cluster_balance/dev_retrieval_smoke/split_plan.json`

This package would be small enough to review and would avoid duplicating the
full 1.1 GB `.runs/` tree.

An initial legacy answer-evaluation package has been created under:

```text
artifacts/legacy_answer_eval_2026-05-21/
```

It intentionally keeps previous answer/context/judge artifacts separate from
both the final summary folders and the new full OpenAI-only reruns.

## Recommended Sharing Package After Full OpenAI Reruns

For the new Agriculture and Mix full OpenAI reruns, share the following per
dataset:

```text
artifacts/openai_full_hirag_2026-05/<dataset>/
  graph_summary.md
  quality_summary.md
  cost_summary.json
  05_base_graph/summary.md
  10_hirag_workdir/materialization_summary.md
  11_answers/imported/answers.jsonl
  11_answers/requests/retrieval_traces.jsonl
  11_answers/retrieval_trace_summary/retrieval_trace_metrics.csv
  11_answers/retrieval_trace_summary/retrieval_trace_summary.md
  12_judges/imported/judge_results.jsonl
  12_judges/imported/judge_summary.csv
  12_judges/imported/swapped_order_agreement.csv
```

These files are the right level of evidence for collaborators: answers,
retrieval traces, judge results, graph quality, and cost. Raw Batch request and
output files can stay in `.runs/` unless someone needs audit-level provenance.
