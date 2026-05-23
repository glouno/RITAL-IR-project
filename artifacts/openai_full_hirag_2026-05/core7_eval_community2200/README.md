# Core 7 Evaluation With Community Report Budget 2200

Date: 2026-05-23

This run is a follow-up to `core7_eval` after diagnosing that graph modes were effectively losing community/global context because `max_token_for_community_report=1000` was too low.

## Intentional Differences From `core7_eval`

- `max_token_for_community_report=2200` during answer context export.
- Runtime text-unit lookup filters `cluster-*` provenance ids so cluster-summary provenance is not treated as missing text chunks.
- Same datasets: Agriculture and Mix.
- Same Core 7 variants: `naive`, `hi`, `hi_weighted`, `hi_minmax`, `hi_minmax_budgeted`, `hi_rerank_weighted`, `hi_mcts`.
- Same answer model: `gpt-5.4-mini`.
- Same embedding model/provider: OpenAI `text-embedding-3-small`.

## Local Run Root

```text
.runs/openai_full_hirag_2026-05/core7_eval_community2200/
```

## Smoke Result

One-query smoke exports for Agriculture and Mix confirmed that graph modes now retrieve `1` community report instead of `0`, while `naive` remains unchanged.

## Promotion Policy

After answer and judge batches finish, promote final shareable outputs here:

```text
artifacts/openai_full_hirag_2026-05/core7_eval_community2200/
```

Keep this folder separate from the original `core7_eval` artifacts so comparisons remain auditable.
