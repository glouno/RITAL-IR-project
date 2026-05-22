# Final Project Presentation Notes

## Core Story

HiRAG's hierarchy is useful, but the original bridge retrieval is too
topology-driven: shortest paths can be cheap to compute but can include weakly
relevant bridge edges. This fork keeps the HiRAG graph idea and improves the
retrieval layer with query-conditioned bridge selection, budgeted minmax bridge
paths, and optional reranking for local entity evidence.

## What The Paper Does

- Builds an entity/relation graph from document chunks.
- Clusters that graph into hierarchical communities.
- Generates community reports for global context.
- Answers questions using local entity evidence, global community reports, and
  bridge paths through the graph.

## Weak Points We Targeted

- Bridge paths can be selected mainly by graph topology rather than query
  relevance.
- Retrieved contexts can become very large and expensive.
- Local entity retrieval can surface semantically weak neighbors.
- The pipeline is expensive to run end to end, especially graph construction.

## Improvements In This Fork

- `hi_minmax_budgeted`: query-weighted minmax bridge paths with per-path and
  total bridge edge budgets.
- `hi_rerank_weighted`: query-weighted bridge retrieval plus local reranking.
- Batch API utilities for answer generation, judging, entity extraction, and
  relation extraction.
- A Mix OpenAI Batch graph-construction path using GPT-5.4 Mini.

## Current Evidence

Agriculture is now complete for the final answer-level loop:

- 30 queries x 3 variants = 90 successful GPT-5.4 Mini answers.
- 120 GPT-5.4 Mini judge rows imported with 0 parse errors.
- Raw swapped-order judge results:
  - `hi_minmax_budgeted` vs `hi`: 30 wins, 30 losses, 0 ties, win rate 50.0%.
  - `hi_rerank_weighted` vs `hi`: 35 wins, 25 losses, 0 ties, win rate 58.3%.
- Stricter swapped-order agreement:
  - `hi_minmax_budgeted`: 12/30 query pairs agreed, strict variant win rate 20.0%.
  - `hi_rerank_weighted`: 23/30 query pairs agreed, strict variant win rate 46.7%.
- Retrieval proxy improved most for minmax-style bridge selection:
  - `hi`: q50 overlap 0.6285.
  - `hi_minmax`: q50 overlap 0.6677.
  - `hi_weighted`: q50 overlap 0.6455.

Mix is also complete for the final answer-level loop:

- Batch-indexed relationship graph: 17,575 nodes and 18,588 edges.
- Materialized queryable workdir: 579 chunks and 2,759 extractive community
  reports.
- 90 successful GPT-5.4 Mini answers.
- 120 GPT-5.4 Mini judge rows imported with 0 parse errors.
- Mix q30 judge results are now available:
  - `hi_minmax_budgeted` vs `hi`: raw swapped-order win rate 28.3%.
  - `hi_rerank_weighted` vs `hi`: raw swapped-order win rate 55.0%.
  - Swapped-order agreement is much stronger for `hi_rerank_weighted` (70.0%)
    than `hi_minmax_budgeted` (46.7%).

## Slide / Video Structure

1. Problem: Graph RAG helps multi-hop QA, but graph traversal can retrieve
   irrelevant bridge context.
2. Paper recap: HiRAG combines local entities, bridge paths, and global
   communities.
3. Reproduction: local vLLM/FastEmbed runtime, agriculture graph, Batch API
   answer and judge loop.
4. Weakness found: bridge path relevance and context budget control.
5. Our changes: weighted bridge retrieval, minmax-budgeted paths, reranking,
   OpenAI Batch indexing/evaluation tooling.
6. Results: agriculture and Mix q30 answer-level judge tables.
7. Mix extension: larger cross-domain batch-indexed graph with complete answer
   and judge evaluation.
8. Honest limitations: Mix community reports are extractive rather than
   LLM-written in the materialized workdir, and judge swapped-order disagreement
   means we should emphasize `hi_rerank_weighted` as the cleaner Mix signal.

## Final Claim

The cleanest final claim is not that every query-aware bridge variant wins. It
is that HiRAG's bridge retrieval can be improved by making retrieval more
query-aware, but the practical variant is the one that controls context budget:
`hi_rerank_weighted`.

Across Agriculture and Mix, `hi_rerank_weighted` is the only experimental
variant with positive raw judge win rates on both datasets, stronger
swapped-order reliability than `hi_minmax_budgeted`, and lower total answer
token usage than baseline.

The final consolidated tables are in
`artifacts/final_analysis_2026-05-21/summary.md`.
