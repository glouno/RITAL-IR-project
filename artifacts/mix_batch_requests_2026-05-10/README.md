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

Imported answer-generation output:

- `final_answer_import/mix_q30_answers_gpt54_mini_from_batch_graph/answers.jsonl`
- 90 answers imported successfully
- 0 answer errors

Fresh judge Batch requests:

- `OpenAI_batch_judge_requests/mix_q30_judge_gpt54_mini_from_batch_graph/judge_requests.jsonl`
- 120 requests: 30 queries x 2 comparisons x 2 swapped answer orders
- model: `gpt-5.4-mini`

The judge file was submitted to OpenAI Batch and imported successfully.

Imported judge output:

- `final_judge_import/mix_q30_judge_gpt54_mini_from_batch_graph/judge_results.jsonl`
- 120 judge rows imported successfully
- 0 parse errors
- Raw swapped-order win rates:
  - `hi_minmax_budgeted`: 17 wins, 40 losses, 3 ties, win rate 28.3%
  - `hi_rerank_weighted`: 33 wins, 27 losses, 0 ties, win rate 55.0%
- Strict swapped-order agreement:
  - `hi_minmax_budgeted`: 14/30 query pairs agree, strict variant win rate 6.7%
  - `hi_rerank_weighted`: 21/30 query pairs agree, strict variant win rate 40.0%

Convenience launcher used for this kind of Batch submission:

```bash
export OPENAI_API_KEY=...
uv run python eval/run_openai_batch_file.py \
  --batch-file artifacts/mix_batch_requests_2026-05-10/OpenAI_batch_judge_requests/mix_q30_judge_gpt54_mini_from_batch_graph/judge_requests.jsonl \
  --output-dir artifacts/mix_batch_requests_2026-05-10/OpenAI_batch_judge_results/mix_q30_judge_gpt54_mini_from_batch_graph \
  --description mix_q30_judge_gpt54_mini_from_batch_graph \
  --poll-interval-seconds 120
```

This uploads the JSONL, creates the Batch, waits for a terminal status, and
downloads output/error JSONL files.

For SSH persistence, launch the same command inside `screen`:

```bash
screen -dmS rital_mix_judge_batch_$(date +%Y%m%d_%H%M%S) bash -lc '
cd /home/paulbeglin/projects/RITAL-IR-project &&
uv run python eval/run_openai_batch_file.py \
  --batch-file artifacts/mix_batch_requests_2026-05-10/OpenAI_batch_judge_requests/mix_q30_judge_gpt54_mini_from_batch_graph/judge_requests.jsonl \
  --output-dir artifacts/mix_batch_requests_2026-05-10/OpenAI_batch_judge_results/mix_q30_judge_gpt54_mini_from_batch_graph \
  --description mix_q30_judge_gpt54_mini_from_batch_graph \
  --poll-interval-seconds 120
'
```
