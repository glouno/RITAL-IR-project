# Final OpenAI Full HiRAG Results: Agriculture + Mix

These tables come from the new OpenAI-integrated run under `.runs/openai_full_hirag_2026-05/`: `gpt-5.4-mini` for answer generation and judging, `text-embedding-3-small` for vector stores, and OpenAI embeddings for query-time retrieval / weighted bridge edge scoring. This is separate from older legacy answer-eval artifacts.

## Shared Artifacts

- `agriculture/answers/answers.jsonl` and `mix/answers/answers.jsonl`: final generated answers, 30 questions x 5 variants.
- `*/retrieval/contexts.jsonl`: exact retrieved context passed into answer generation.
- `*/retrieval/retrieval_traces.jsonl`: detailed retrieval traces with entities, communities, bridge edges, path lengths, budget attempts, and context sections.
- `*/retrieval/retrieval_trace_metrics.csv`: flattened per-query/per-variant retrieval metrics.
- `*/judges/judge_results.jsonl`: pairwise LLM-as-judge outputs with swapped answer order.
- `summary/*.csv`: consolidated tables for analysis.

## Judge Win Rates Against `hi`

Each variant is judged pairwise against `hi` for 30 questions, with both answer orders. Thus each variant has 60 valid comparisons per dataset. A win rate of `0.60` means the variant was preferred over `hi` in 60% of these pairwise judgments; it does not mean factual accuracy is 60%.

| Dataset | Variant | Wins | Losses | Ties | Win Rate |
|---|---|---:|---:|---:|---:|
| agriculture | `naive` | 52 | 7 | 1 | 0.8667 |
| agriculture | `hi_nobridge` | 38 | 20 | 2 | 0.6333 |
| agriculture | `hi_minmax_budgeted` | 27 | 31 | 2 | 0.4500 |
| agriculture | `hi_rerank_weighted` | 27 | 33 | 0 | 0.4500 |
| mix | `naive` | 49 | 11 | 0 | 0.8167 |
| mix | `hi_nobridge` | 36 | 24 | 0 | 0.6000 |
| mix | `hi_rerank_weighted` | 32 | 28 | 0 | 0.5333 |
| mix | `hi_minmax_budgeted` | 28 | 32 | 0 | 0.4667 |

## Retrieval Metrics

These metrics describe what retrieval selected before answer generation. `bridge_path_edges` approximates graph traversal distance along the selected bridge path. `naive` has no graph entities/path because it retrieves text chunks directly.

| Dataset | Variant | Mean Context Tokens | Mean Entities | Mean Communities | Mean Bridge Edges | Mean Bridge Path Edges | Mean Edge Score | Out-of-Budget Rows |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| agriculture | `hi` | 5039.87 | 11.47 | 0.00 | 15.33 | 11.87 | n/a | 0 |
| agriculture | `hi_minmax_budgeted` | 5141.00 | 11.47 | 0.00 | 17.60 | 16.40 | 0.5683 | 0 |
| agriculture | `hi_nobridge` | 5825.30 | 0.00 | 0.00 | 0.00 | n/a | n/a | 0 |
| agriculture | `hi_rerank_weighted` | 5800.83 | 11.00 | 0.03 | 14.80 | 9.87 | 0.5581 | 0 |
| agriculture | `naive` | 6016.47 | 0.00 | 0.00 | 0.00 | n/a | n/a | 0 |
| mix | `hi` | 3173.83 | 12.00 | 0.00 | 12.60 | 12.43 | n/a | 0 |
| mix | `hi_minmax_budgeted` | 3205.73 | 12.00 | 0.00 | 12.93 | 13.87 | 0.5515 | 0 |
| mix | `hi_nobridge` | 4124.17 | 0.00 | 0.00 | 0.00 | n/a | n/a | 0 |
| mix | `hi_rerank_weighted` | 3597.03 | 12.00 | 0.00 | 12.63 | 11.47 | 0.5510 | 0 |
| mix | `naive` | 6614.27 | 0.00 | 0.00 | 0.00 | n/a | n/a | 0 |

## Answer And Judge Costs

Costs use the project Batch estimate for `gpt-5.4-mini`: `$0.1875 / 1M input tokens` and `$1.125 / 1M output tokens`. Edge/query embeddings costs are not included in this small answer/judge table.

| Dataset | Stage | Completed | Failed | Input Tokens | Output Tokens | Estimated Cost |
|---|---|---:|---:|---:|---:|---:|
| agriculture | answers | 150 | 0 | 870417 | 37807 | $0.2057 |
| agriculture | judges | 240 | 0 | 182198 | 61659 | $0.1035 |
| mix | answers | 150 | 0 | 657761 | 33723 | $0.1613 |
| mix | judges | 240 | 0 | 171308 | 61654 | $0.1015 |

## Swapped Order Agreement

For each question/variant pair, we judged both `hi` first and variant first. Agreement means the same system won regardless of ordering. Low agreement indicates judge sensitivity or very close answer quality.

| Dataset | Variant | Query Pairs | Agree | Disagree | Agreement Rate |
|---|---|---:|---:|---:|---:|
| agriculture | `hi_minmax_budgeted` | 30 | 18 | 12 | 0.6000 |
| agriculture | `hi_nobridge` | 30 | 21 | 9 | 0.7000 |
| agriculture | `hi_rerank_weighted` | 30 | 23 | 7 | 0.7667 |
| agriculture | `naive` | 30 | 25 | 5 | 0.8333 |
| mix | `hi_minmax_budgeted` | 30 | 16 | 14 | 0.5333 |
| mix | `hi_nobridge` | 30 | 20 | 10 | 0.6667 |
| mix | `hi_rerank_weighted` | 30 | 16 | 14 | 0.5333 |
| mix | `naive` | 30 | 27 | 3 | 0.9000 |

## Interpretation

- `naive` wins strongly on both datasets. This likely reflects the judge preference for broader, denser answer context: `naive` uses many raw text chunks and often produces comprehensive answers, even though it does not use the graph.
- `hi_nobridge` beats `hi` on both datasets. This suggests the current bridge context can add noise or distracting path information for these benchmark questions.
- `hi_rerank_weighted` is mixed: it beats `hi` on Mix but loses on Agriculture. The weighted/reranked traversal may help when graph paths are cleaner, but it is not robustly better yet.
- `hi_minmax_budgeted` underperforms `hi` in both datasets. Its path constraint may be too conservative or may select semantically coherent but answer-irrelevant bridge paths.
- These are LLM-as-judge preferences, not ground-truth QA accuracy. The strongest next analysis is manual inspection of cases where `naive` beats graph modes and where `hi_nobridge` beats `hi`.
