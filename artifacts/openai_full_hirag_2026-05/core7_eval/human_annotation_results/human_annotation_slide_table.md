# Human Annotation Slide Table

| Variante | Win rate | Densité du span utile | Gain moyen |
|---|---:|---:|---:|
| HiRAG classique | - | 28.7% | - |
| Naive | 65.0% | 32.0% | +0.30 |
| Minmax budgeted | 45.0% | 32.1% | -0.10 |
| MCTS | 52.5% | 30.4% | +0.05 |

Definitions:

- Win rate: human pairwise preference against HiRAG classique, ties counted as 0.5.
- Densité du span utile: useful highlighted span length divided by answer length.
- Gain moyen: mean human preference gain vs HiRAG classique, where win=+1, tie=0, loss=-1.
