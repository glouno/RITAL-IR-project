# Legacy Answer Evaluation Artifacts

Date: 2026-05-21

This folder promotes selected files from gitignored `.runs/` directories so
collaborators can inspect answer prompts, contexts, answers, and judge outputs
from the previous Agriculture evaluation work.

These are **legacy/intermediate artifacts**. They are useful for analysis, but
they are not the new full OpenAI-only reconstruction currently running under:

```text
.runs/openai_full_hirag_2026-05/
```

When those new runs are complete and validated, their cleaned outputs should be
copied into:

```text
artifacts/openai_full_hirag_2026-05/
```

## Contents

### `agriculture_q30_gpt54_fixed_requests/`

Source:

```text
.runs/openai_batch/agriculture_q30_gpt54_fixed/
```

Contains:

- `answer_requests.jsonl`: exact OpenAI Batch answer-generation requests.
- `contexts.jsonl`: resolved HiRAG contexts and compact retrieval debug for 30
  Agriculture questions x 3 variants.
- `manifest.json`: request count, variants, model, and token estimates.

Notes:

- Model: `gpt-5.4`.
- Variants: `hi`, `hi_minmax_budgeted`, `hi_rerank_weighted`.
- 90 requests, 0 context export errors.

### `agriculture_q30_gpt54_mini_fixed_requests/`

Source:

```text
.runs/openai_batch/agriculture_q30_gpt54_mini_fixed/
```

Contains the equivalent request/context/manifest files for `gpt-5.4-mini`.

Notes:

- Model: `gpt-5.4-mini`.
- Variants: `hi`, `hi_minmax_budgeted`, `hi_rerank_weighted`.
- 90 requests, 0 context export errors.
- This is the most relevant legacy request/context set for comparison with the
  current `gpt-5.4-mini` full OpenAI rerun plan.

### `agriculture_q30_gpt54_mini_final_import/`

Source:

```text
.runs/answer_eval/agriculture_q30_gpt54_mini_final_import/
```

Contains:

- `answers.jsonl`: imported answer outputs.
- `summary.json`
- `summary.md`

Notes:

- 90 answer rows.
- 30 Agriculture questions x 3 variants.
- 0 import errors.
- This duplicates the tracked final Agriculture answer file in
  `artifacts/agriculture_graphs_2026-04-23/final_evaluation/`, but keeps the
  `.runs/answer_eval` provenance visible for collaborators.

### `agriculture_q30_gpt54_mini_full_judge_import/`

Source:

```text
.runs/answer_eval/agriculture_q30_gpt54_mini_full_judge_import/
```

Contains:

- `judge_results.jsonl`
- `judge_summary.csv`
- `judge_summary.md`

Notes:

- 120 judge rows.
- 0 import errors.
- This corresponds to the completed Agriculture q30 full judge import.

### `agriculture_q30_local_legacy_partial/`

Source:

```text
.runs/answer_eval/agriculture_q30/
```

Contains:

- `answers.jsonl`
- `run.log`

Notes:

- Local legacy partial run.
- 21 answer rows.
- Includes errors and experimental variants such as `hi_weighted` and
  `hi_minmax`.
- Keep this separate from final results; it is useful for debugging evolution,
  not for headline evaluation.

## How To Use

For answer analysis, start with:

```text
agriculture_q30_gpt54_mini_final_import/answers.jsonl
agriculture_q30_gpt54_mini_full_judge_import/judge_results.jsonl
```

For retrieval-context analysis, start with:

```text
agriculture_q30_gpt54_mini_fixed_requests/contexts.jsonl
```

For exact prompt reconstruction, use:

```text
agriculture_q30_gpt54_mini_fixed_requests/answer_requests.jsonl
```

For final project-level numbers, prefer:

```text
artifacts/final_analysis_2026-05-21/
artifacts/agriculture_graphs_2026-04-23/final_evaluation/
artifacts/mix_batch_requests_2026-05-10/
```

## Recommended Future Structure

Keep three artifact categories separate:

- `artifacts/final_analysis_YYYY-MM-DD/`: final tables and conclusions.
- `artifacts/legacy_answer_eval_YYYY-MM-DD/`: useful historical/intermediate
  answer evaluation evidence.
- `artifacts/openai_full_hirag_YYYY-MM/`: cleaned outputs from the new full
  OpenAI reconstruction runs.

This avoids mixing old vLLM/FastEmbed-era evidence with the new homogeneous
OpenAI Batch reproduction.
