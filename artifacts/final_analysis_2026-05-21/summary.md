# Final HiRAG Analysis

Date: 2026-05-21

## Executive Takeaway

We have enough data to finalize around a retrieval-layer improvement story rather than a full four-dataset reproduction. Agriculture and Mix both have complete q30 answer sets for `hi`, `hi_minmax_budgeted`, and `hi_rerank_weighted`. Mix also has a complete OpenAI Batch-indexed graph. The most defensible improvement is `hi_rerank_weighted`: it is cheaper than baseline in answer-token usage and is the only variant with a positive raw judge win rate on both datasets. `hi_minmax_budgeted` improves bridge relevance proxies, but its answer-level behavior is unstable and more expensive.

## Final Answer-Level Judge Results

| Dataset | Variant vs `hi` | Valid Comparisons | Wins | Losses | Ties | Raw Win Rate | Swapped Agreement | Strict Variant Win Rate |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| agriculture | `hi_minmax_budgeted` | 60 | 30 | 30 | 0 | 50.0% | 40.0% | 20.0% |
| agriculture | `hi_rerank_weighted` | 60 | 35 | 25 | 0 | 58.3% | 76.7% | 46.7% |
| mix | `hi_minmax_budgeted` | 60 | 17 | 40 | 3 | 28.3% | 46.7% | 6.7% |
| mix | `hi_rerank_weighted` | 60 | 33 | 27 | 0 | 55.0% | 70.0% | 40.0% |

Interpretation: raw win rate counts both swapped answer orders as separate judge rows. Strict variant win rate only counts a query as won when both swapped-order judgments agree on the variant, so it is intentionally conservative.

## Answer Cost And Retrieval Footprint

| Dataset | Variant | Answers | Mean Context Tokens | Mean Bridge Edges | Mean Path Edges | Actual Total Tokens |
|---|---|---:|---:|---:|---:|---:|
| agriculture | `hi` | 30 | 4934.5 | 62.6 | 81.4 | 152986 |
| agriculture | `hi_minmax_budgeted` | 30 | 5616.1 | 81.7 | 119.3 | 172923 |
| agriculture | `hi_rerank_weighted` | 30 | 4751.6 | 55.7 | 68.7 | 148158 |
| mix | `hi` | 30 | 3460.3 | 24.6 | 38.4 | 108851 |
| mix | `hi_minmax_budgeted` | 30 | 3692.1 | 31.1 | 49.5 | 115361 |
| mix | `hi_rerank_weighted` | 30 | 3255.9 | 18.6 | 29.5 | 103187 |

Cost interpretation: `hi_rerank_weighted` is the compact practical variant. It used fewer total answer tokens than baseline on both agriculture and Mix, while `hi_minmax_budgeted` consistently used more bridge/context budget.

## Graph Construction Status

| Dataset | Graph Status | Evaluation Status | Notes |
|---|---|---|---|
| Agriculture | Local vLLM/FastEmbed HiRAG graphs completed for baseline/no-glean, lean, ultra-lean | q30 answers complete; q30 judges complete | Main retrieval-development graph; LLM community reports available from local runs |
| Mix | OpenAI Batch entity/relation graph completed and materialized | q30 answers complete; q30 judges complete | 579 chunks, 17,575 nodes, 18,588 edges, 2,759 extractive community reports |
| CS | Not run for final graph construction | Not evaluated in final loop | Deferred |
| Legal | Not run for final graph construction | Not evaluated in final loop | Deferred |

## Final Analysis

HiRAG remains a strong structure for multi-hop retrieval because it combines local entity evidence, global community summaries, and bridge paths. The weakness we observed is that the bridge can be too topology-driven: once seed entities are selected, shortest-path traversal can add semantically weak edges simply because they are graph-close.

`hi_minmax_budgeted` validates the idea that query-conditioned bridge path scoring can improve bridge relevance, especially in agriculture retrieval proxy metrics, but answer-level evaluation shows that stronger bridge relevance is not automatically better generation. The variant often spends more context budget, and on Mix its judge performance is poor.

`hi_rerank_weighted` is the best final improvement. It changes the seed evidence and uses query-weighted bridge retrieval while keeping contexts compact. It has the strongest cross-dataset answer-level signal: 58.3% raw win rate on agriculture, 55.0% raw win rate on Mix, and lower total answer-token usage than baseline on both.

The final claim should therefore be modest and sharp: we do not fully reproduce HiRAG across all paper datasets, but we identify and test a concrete weakness in HiRetrieval, implement query-aware retrieval variants, and show that reranking plus weighted bridge retrieval is a practical improvement path.

## Remaining Limitations

- Agriculture full judging is now complete, but the final evidence is still q30 per dataset, not the full paper-scale evaluation.
- Mix community reports are extractive in the materialized Batch workdir, not LLM-written summaries, so Mix is best treated as evidence for retrieval mechanics and answer behavior, not a perfect paper-faithful HiRAG reproduction.
- Swapped-order judge disagreement is non-trivial, especially for `hi_minmax_budgeted`; the strict agreement table should be shown alongside raw win rates.
- CS and Legal graph construction were intentionally deferred to avoid spending more Batch/local runtime after the main claim was already testable.
