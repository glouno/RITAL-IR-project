# Full OpenAI HiRAG Artifacts

This directory contains cleaned, shareable outputs from the full OpenAI
Agriculture + Mix rerun.

This is the new OpenAI-integrated run:

- `gpt-5.4-mini` for answers and judges;
- `text-embedding-3-small` for vector stores;
- OpenAI embeddings for query-time retrieval and weighted bridge edge scoring;
- 30 questions per dataset;
- 5 answer variants: `hi`, `naive`, `hi_nobridge`,
  `hi_rerank_weighted`, `hi_minmax_budgeted`.

Start here:

- `summary/FINAL_OPENAI_FULL_HIRAG_RESULTS_2026-05-22.md`
- `summary/judge_win_rates.csv`
- `summary/retrieval_metrics.csv`
- `summary/swapped_order_agreement.csv`

## Layout

```text
artifacts/openai_full_hirag_2026-05/
  summary/
    FINAL_OPENAI_FULL_HIRAG_RESULTS_2026-05-22.md
    judge_win_rates.csv
    retrieval_metrics.csv
    answer_judge_costs.csv
    swapped_order_agreement.csv

  agriculture/
    answers/
      answers.jsonl
      summary.*
      batch_run_manifest.json
    retrieval/
      contexts.jsonl
      retrieval_traces.jsonl
      retrieval_trace_metrics.csv
      retrieval_trace_summary.*
    judges/
      judge_results.jsonl
      judge_summary.*
      batch_run_manifest.json

  mix/
    same structure
```

## What To Use For Analysis

- `answers/answers.jsonl`: generated answers for qualitative inspection.
- `retrieval/contexts.jsonl`: exact context sent to answer generation.
- `retrieval/retrieval_traces.jsonl`: rich trace data including selected
  entities, communities, bridge edges, graph path lengths, budget attempts, and
  context sections.
- `retrieval/retrieval_trace_metrics.csv`: flat per-question metrics for
  plotting and comparisons.
- `judges/judge_results.jsonl`: pairwise judge outputs with both answer orders.
- `judges/judge_summary.csv`: per-variant win/loss/tie summary against `hi`.

## Not Included By Default

- embedding caches;
- full NanoVectorDB workdirs;
- raw Batch outputs unless audit-level provenance is needed;
- superseded retry attempts;
- smoke tests.
