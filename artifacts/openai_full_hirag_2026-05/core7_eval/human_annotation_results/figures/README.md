# Human Annotation Figures

Generated from `human_annotation_variant_summary.csv` and `human_vs_llm_pair_summary.csv`.

Files:

- `human_vs_llm_win_rates.png`: grouped bars comparing human win rate and strict LLM win rate vs classic HiRAG.
- `human_vs_llm_decision_breakdown.png`: stacked decisions showing variant wins, HiRAG wins, ties, and LLM swapped-order disagreements.
- `span_density_vs_preference_gain.png`: useful span density against human preference gain.
- `human_answer_question_scores.png`: human yes/partial/no answer-question score per variant.
- `slide_table_with_llm.md` and `.csv`: slide-ready table including human and LLM win rates.

Recommended slide narrative:

- Naive is the only clearly preferred variant by both humans and the LLM judge.
- Minmax budgeted and MCTS are close to HiRAG under human annotation.
- LLM judge has substantial swapped-order disagreement on close comparisons, so use it as a high-throughput proxy rather than an oracle.

Swapped-order disagreement means the same answer pair was judged twice, once as A/B and once as B/A, and the LLM did not pick the same underlying variant after mapping the labels back. This is a judge-stability warning, not a separate model variant.
