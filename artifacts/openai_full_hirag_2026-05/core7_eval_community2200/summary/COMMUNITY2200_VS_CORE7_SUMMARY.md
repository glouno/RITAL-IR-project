# Community2200 vs Core7 Summary

Date: 2026-05-23

This compares the original Core 7 run against the rerun with `max_token_for_community_report=2200` and the `cluster-*` text-chunk lookup fix.

## Main Result

- Agriculture: graph variants generally improve modestly, but `naive` becomes even stronger by raw LLM judge win rate (`0.800 -> 0.883`).
- Mix: `naive` drops from `0.750 -> 0.667`, but all graph variants also drop versus `hi`; community reports did not produce a clear graph-mode win.
- Both judge imports have `0` API failures and `0` parse errors.

## Raw Win Rate vs `hi`

| Dataset | Variant | Core7 | Community2200 | Delta | Strict W/L/T/Disagree | Mean communities | Mean ctx tokens |
|---|---|---:|---:|---:|---|---:|---:|
| agriculture | naive | 0.800 | 0.883 | +0.083 | 26/3/0/1 | 0.00 | 6016 |
| agriculture | hi_weighted | 0.467 | 0.550 | +0.083 | 11/8/0/11 | 0.93 | 6184 |
| agriculture | hi_minmax | 0.383 | 0.450 | +0.067 | 9/12/0/9 | 0.93 | 6269 |
| agriculture | hi_minmax_budgeted | 0.483 | 0.533 | +0.050 | 9/7/0/14 | 0.93 | 6269 |
| agriculture | hi_rerank_weighted | 0.400 | 0.433 | +0.033 | 10/14/0/6 | 0.87 | 6448 |
| agriculture | hi_mcts | 0.500 | 0.533 | +0.033 | 10/8/0/12 | 0.93 | 6220 |
| mix | naive | 0.750 | 0.667 | -0.083 | 15/5/0/10 | 0.00 | 6614 |
| mix | hi_weighted | 0.550 | 0.400 | -0.150 | 7/13/0/10 | 1.10 | 4551 |
| mix | hi_minmax | 0.517 | 0.433 | -0.083 | 7/11/0/12 | 1.10 | 4577 |
| mix | hi_minmax_budgeted | 0.567 | 0.383 | -0.183 | 6/13/0/11 | 1.10 | 4577 |
| mix | hi_rerank_weighted | 0.500 | 0.417 | -0.083 | 7/12/0/11 | 1.10 | 4974 |
| mix | hi_mcts | 0.450 | 0.483 | +0.033 | 9/10/0/11 | 1.10 | 4573 |

## Interpretation

Raising the community-report budget fixed the mechanical bug: graph modes now include community/global context frequently instead of almost never. But this alone does not solve the main retrieval problem. The answer model still often needs exact source passages, and the graph context spends many tokens on summaries/entities/path structure rather than direct evidence.

Next priority: test a hybrid graph+source variant that keeps graph traversal but guarantees a larger raw source-evidence budget, instead of only adding community reports.
