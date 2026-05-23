# Retrieval Thesis Slide Artifacts

Purpose: provide slide-ready statistics for the thesis that better graph traversal alone is not enough; final context composition matters.

## Key Stats

| Dataset | Variant | Context tokens | Source evidence share | Graph abstraction share | Bridge edges | Retrieval time | LLM win rate vs HiRAG |
|---|---|---:|---:|---:|---:|---:|---:|
| agriculture | Naive | 6016 | 100.0% | 0.0% | 0.0 | 0.51s | 80.0% |
| agriculture | HiRAG | 5040 | 21.0% | 79.0% | 15.3 | 0.22s | - |
| agriculture | Minmax budgeted | 5141 | 20.5% | 79.5% | 17.6 | 5.02s | 48.3% |
| agriculture | MCTS | 4990 | 21.1% | 78.9% | 14.9 | 2.04s | 50.0% |
| mix | Naive | 6614 | 100.0% | 0.0% | 0.0 | 0.41s | 75.0% |
| mix | HiRAG | 3174 | 33.9% | 66.1% | 12.6 | 0.16s | - |
| mix | Minmax budgeted | 3206 | 33.6% | 66.4% | 12.9 | 2.43s | 56.7% |
| mix | MCTS | 3159 | 34.0% | 66.0% | 12.3 | 0.96s | 45.0% |

## How To Use These Slides

1. Show `graph_retrieval_structure_agriculture.png` to prove the variants really change graph traversal.
2. Show `context_composition_mix.png` or `context_composition_agriculture.png` to show why this is not sufficient: graph modes allocate much less final context to direct source evidence. These plots use absolute context size in tokens, not normalized percentages, so Naive's larger prompt is visible.
3. Show `source_evidence_share_vs_win_rate.png` to connect evidence share with end-to-end judge preference.

Suggested speaking line:

> Our variants do change graph traversal: they select different bridge paths and pay different retrieval costs. But the final prompt still contains much less direct source evidence than naive RAG. For QA tasks, this evidence composition matters more than graph structure alone.

Caveat: source evidence share is computed from final context sections, so it is a prompt-composition metric, not a ground-truth relevance metric.
