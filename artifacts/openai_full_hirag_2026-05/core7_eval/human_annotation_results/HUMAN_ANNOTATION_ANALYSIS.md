# Human Annotation Analysis

Date: 2026-05-23

Input file: `RAW_annotation_results - annotations.csv`.

## Coverage

- Human annotation rows: 60.
- Annotators: vlad=20, sitara=20, milos=20
- Pair distribution: 20 `hi` vs `naive`, 20 `hi` vs `hi_minmax_budgeted`, 20 `hi` vs `hi_mcts`.
- The packet is a sampled subset of the Core 7 LLM-judged run, not the full LLM-judge set.

## Slide Table

| Variante | Win rate humain vs HiRAG | Densité du span utile | Gain moyen de préférence | Gain moyen answer-question |
|---|---:|---:|---:|---:|
| HiRAG classique | - | 28.7% | - | - |
| Naive | 65.0% | 32.0% | +0.30 | -0.03 |
| Minmax budgeted | 45.0% | 32.1% | -0.10 | -0.03 |
| MCTS | 52.5% | 30.4% | +0.05 | +0.00 |

Metric definitions:

- Win rate humain vs HiRAG: pairwise preference against `hi`, with ties counted as 0.5 wins.
- Densité du span utile: highlighted useful span length divided by full answer length, averaged per variant.
- Gain moyen de préférence: average pairwise score against `hi`, where win=+1, tie=0, loss=-1.
- Gain moyen answer-question: difference in answer-question score against `hi`, where yes=1, partially=0.5, no=0.

## Human vs LLM Judge On Same Human-Annotated Items

| Pair | Human W/L/T for variant | Human win rate | LLM strict W/L/T/Disagree for variant |
|---|---:|---:|---:|
| Naive vs HiRAG | 10/4/6 | 65.0% | 11/3/0/6 |
| Minmax budgeted vs HiRAG | 5/7/8 | 45.0% | 8/5/0/7 |
| MCTS vs HiRAG | 7/6/7 | 52.5% | 5/8/0/7 |

## Interpretation

- Humans strongly prefer `naive` over classic HiRAG on the sampled rows, matching the broad LLM-judge direction.
- Humans rate `hi_minmax_budgeted` slightly below HiRAG on preference (`45.0%` half-credit win rate), although many rows are ties.
- Humans rate `hi_mcts` as a near tie / very slight preference gain over HiRAG (`52.5%` half-credit win rate, `+0.05` gain).
- Useful span density is not a pure quality metric: a lower density can mean a verbose answer with a small useful part, while a higher density can mean a more concise answer. For slides, use it as an evidence compactness signal, not as standalone answer quality.

## Files Exported

- `human_annotation_variant_summary.csv`: slide table values.
- `human_vs_llm_pair_summary.csv`: pair-level human vs LLM summary.
- `human_annotation_pairwise_vs_hi.csv`: one row per human pair with variant mapping and LLM outcome.
- `human_annotation_enriched_long.csv`: one row per answer with span density and answer-question score.
