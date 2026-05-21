# Full OpenAI HiRAG Artifacts

This directory is reserved for cleaned, shareable outputs from the full
OpenAI-only HiRAG reruns.

Active staging runs currently write to:

```text
.runs/openai_full_hirag_2026-05/agriculture/
.runs/openai_full_hirag_2026-05/mix/
```

Do not move or delete those `.runs/` directories while their `screen` sessions
are active. Once each dataset is complete and validated, copy only curated
outputs here.

## Intended Layout

```text
artifacts/openai_full_hirag_2026-05/
  agriculture/
    graph_summary.md
    quality_summary.md
    cost_summary.json
    10_hirag_workdir/materialization_summary.md
    11_answers/imported/answers.jsonl
    11_answers/requests/retrieval_traces.jsonl
    11_answers/retrieval_trace_summary/retrieval_trace_metrics.csv
    11_answers/retrieval_trace_summary/retrieval_trace_summary.md
    12_judges/imported/judge_results.jsonl
    12_judges/imported/judge_summary.csv
    12_judges/imported/swapped_order_agreement.csv

  mix/
    same structure
```

## Promotion Rules

Promote:

- final answers, judges, retrieval traces, summaries, and cost files;
- graph/workdir summaries;
- GraphML only when size is acceptable and collaborators need graph inspection.

Do not promote by default:

- embedding caches;
- full NanoVectorDB workdirs;
- raw Batch outputs unless audit-level provenance is needed;
- superseded retry attempts;
- smoke tests.
