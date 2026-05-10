# Final Evaluation Working Summary

Status: agriculture q30 answers are complete; judge results are partial until the 74-request missing judge batch is run.

## q50 Retrieval Proxy

| Variant | Query Overlap | Local Recall | Global Recall | Context Tokens | Latency s |
|---|---:|---:|---:|---:|---:|
| `hi` | 0.6285 | 0.6456 | 0.3601 | 29826.6 | 0.0569 |
| `hi_minmax` | 0.6677 | 0.6752 | 0.3930 | 32411.5 | 9.2689 |
| `hi_rerank` | 0.6354 | 0.6046 | 0.3594 | 29790.0 | 0.6931 |
| `hi_rerank_weighted` | 0.6443 | 0.6100 | 0.3598 | 29886.7 | 0.9844 |
| `hi_weighted` | 0.6455 | 0.6465 | 0.3571 | 29898.0 | 3.0468 |

## q30 Answer Cost, GPT-5.4 Mini

| Variant | Mean Input Tokens | Mean Context Tokens | Mean Bridge Edges | Actual Total Tokens |
|---|---:|---:|---:|---:|
| `hi` | 4934.5 | 4639.9 | 62.6 | 152986 |
| `hi_minmax_budgeted` | 5616.1 | 5309.2 | 81.7 | 172923 |
| `hi_rerank_weighted` | 4751.6 | 4460.2 | 55.7 | 148158 |

## Partial q30 Judge Results

| Variant vs `hi` | Valid Comparisons | Wins | Losses | Ties | Win Rate |
|---|---:|---:|---:|---:|---:|
| `hi_minmax_budgeted` | 20 | 9 | 11 | 0 | 0.4500 |
| `hi_rerank_weighted` | 26 | 15 | 11 | 0 | 0.5769 |

## Current Claim

HiRAG’s hierarchy is useful, but bridge retrieval is still too topology-driven. Query-conditioned bridge path selection improves bridge relevance; budgeted retrieval and reranking are needed to control cost and answer quality.

## Files

- `agriculture_q30_answers_gpt54_mini.jsonl`
- `agriculture_q30_partial_judge_results_gpt54_mini.jsonl`
- `q50_retrieval_proxy.csv`
- `q30_answer_costs.csv`
- `partial_q30_judge_win_rates.csv`
- `qualitative_examples.json`
- `agriculture_q30_full_judge_batch_manifest.json`
- `agriculture_q30_missing_judge_batch_manifest.json`
