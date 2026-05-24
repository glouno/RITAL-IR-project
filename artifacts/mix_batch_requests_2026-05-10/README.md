# Mix Batch-Indexed HiRAG Artifacts

This folder keeps the final, inspectable outputs from the Mix batch-indexed
HiRAG run. Superseded request JSONL files, raw Batch API downloads, retry
payloads, and temporary workdir summaries were removed from the submission
cleanup.

## Relationship Graph

- `OpenAI_batch_indexing_graphs/mix_base_graph_gpt54_mini_final/base_graph.graphml`
- 17,575 nodes
- 18,588 edges
- 23,874 parsed entities
- 19,676 parsed relations

## Imported Answers

- `final_answer_import/mix_q30_answers_gpt54_mini_from_batch_graph/answers.jsonl`
- 90 answers imported successfully
- 30 Mix queries across `hi`, `hi_minmax_budgeted`, and
  `hi_rerank_weighted`

## Imported Judge Results

- `final_judge_import/mix_q30_judge_gpt54_mini_from_batch_graph/judge_results.jsonl`
- 120 pairwise judge rows imported successfully
- 0 parse errors

Raw swapped-order win rates:

- `hi_minmax_budgeted`: 17 wins, 40 losses, 3 ties, win rate 28.3%
- `hi_rerank_weighted`: 33 wins, 27 losses, 0 ties, win rate 55.0%

Strict swapped-order agreement:

- `hi_minmax_budgeted`: 14/30 query pairs agree, strict variant win rate 6.7%
- `hi_rerank_weighted`: 21/30 query pairs agree, strict variant win rate 40.0%
