# Slide Table With Human And LLM Judge

| Variante | Win rate humain | Densite du span utile | Gain moyen humain | LLM win rate strict |
|---|---:|---:|---:|---:|
| HiRAG classique | - | 28.7% | - | - |
| Naive | 65.0% | 32.0% | +0.30 | 78.6% |
| Minmax budgeted | 45.0% | 32.1% | -0.10 | 61.5% |
| MCTS | 52.5% | 30.4% | +0.05 | 38.5% |

Notes:

- Human win rate counts ties as 0.5 wins against classic HiRAG.
- LLM win rate strict uses only swapped-order-consistent LLM judgments on the same human-annotated rows.
- Useful span density is useful highlighted span length divided by answer length.
