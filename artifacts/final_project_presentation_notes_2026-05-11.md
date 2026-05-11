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

Agriculture is the completed answer-level result set:

- 30 queries x 3 variants = 90 successful GPT-5.4 Mini answers.
- Partial judge results currently available:
  - `hi_minmax_budgeted` vs `hi`: 9 wins, 11 losses, 0 ties, win rate 45.0%.
  - `hi_rerank_weighted` vs `hi`: 15 wins, 11 losses, 0 ties, win rate 57.7%.
- Retrieval proxy improved most for minmax-style bridge selection:
  - `hi`: q50 overlap 0.6285.
  - `hi_minmax`: q50 overlap 0.6677.
  - `hi_weighted`: q50 overlap 0.6455.

Mix is now ready for answer-level evaluation:

- Batch-indexed relationship graph: 17,575 nodes and 18,588 edges.
- Materialized queryable workdir: 579 chunks and 2,759 extractive community
  reports.
- Fresh q30 answer Batch file: 90 requests for `hi`, `hi_minmax_budgeted`, and
  `hi_rerank_weighted` with GPT-5.4 Mini.
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
6. Results: agriculture retrieval proxy and partial judge win rates.
7. Mix extension: larger cross-domain batch-indexed graph and q30 answer batch
   ready to submit.
8. Honest limitations: Mix community reports are extractive rather than
   LLM-written in the materialized workdir, and judge swapped-order disagreement
   means we should emphasize `hi_rerank_weighted` as the cleaner Mix signal.

## Next Submission Steps

Submit:

- `artifacts/mix_batch_requests_2026-05-10/OpenAI_batch_answer_requests/mix_q30_answers_gpt54_mini_from_batch_graph/answer_requests.jsonl`

After completion:

```bash
uv run python eval/import_openai_batch_answers.py \
  --batch-output /path/to/openai_answer_batch_output.jsonl \
  --contexts artifacts/mix_batch_requests_2026-05-10/OpenAI_batch_answer_requests/mix_q30_answers_gpt54_mini_from_batch_graph/contexts.jsonl \
  --output-dir artifacts/mix_batch_requests_2026-05-10/final_answer_import

uv run python eval/export_openai_judge_batch_requests.py \
  --answers artifacts/mix_batch_requests_2026-05-10/final_answer_import/answers.jsonl \
  --baseline hi \
  --variants hi_minmax_budgeted hi_rerank_weighted \
  --model gpt-5.4-mini \
  --output-dir artifacts/mix_batch_requests_2026-05-10/OpenAI_batch_judge_requests/mix_q30_judge_gpt54_mini_from_batch_graph
```
