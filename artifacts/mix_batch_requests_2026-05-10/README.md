# Mix GPT-5.4 Mini Batch Requests

This folder contains OpenAI Batch judge/evaluation request JSONL files regenerated from the existing Mix evaluation request files with:

- `model`: `gpt-5.4-mini`
- output token parameter: `max_completion_tokens`

These first files are evaluation/judge requests over existing Mix answer files.

Files:

- `mix_eval_gpt54_mini.jsonl`: 20 requests
- `mix_eval_hi_gpt54_mini.jsonl`: 260 requests
- `mix_eval_hi_naiveR_hi_gpt54_mini.jsonl`: 260 requests

## Fresh Batch-Indexed HiRAG Run

The OpenAI Batch indexing chain has now produced a Mix relationship graph and a
queryable HiRAG working directory materialized from that graph.

Relationship graph:

- `OpenAI_batch_indexing_graphs/mix_base_graph_gpt54_mini_final/base_graph.graphml`
- 17,575 nodes
- 18,588 edges
- 23,874 parsed entities
- 19,676 parsed relations

Runnable workdir summary:

- `OpenAI_batch_indexing_workdirs/mix_hirag_workdir_gpt54_mini_final/summary.md`
- 61 docs
- 579 chunks
- 2,759 extractive community reports
- 12,572 clustered nodes

Fresh answer-generation Batch requests:

- `OpenAI_batch_answer_requests/mix_q30_answers_gpt54_mini_from_batch_graph/answer_requests.jsonl`
- `OpenAI_batch_answer_requests/mix_q30_answers_gpt54_mini_from_batch_graph/contexts.jsonl`
- 90 requests: 30 Mix queries x `hi`, `hi_minmax_budgeted`, `hi_rerank_weighted`
- model: `gpt-5.4-mini`

Submit the answer file to OpenAI Batch with endpoint `/v1/chat/completions` and
`completion_window="24h"`. After the answer batch completes, import it with
`eval/import_openai_batch_answers.py`, then generate the pairwise judge batch
with `eval/export_openai_judge_batch_requests.py`.
