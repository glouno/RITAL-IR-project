# Core 7 HiRAG Variant Evaluation Runbook

Date: 2026-05-23

This runbook tracks the post-merge Core 7 evaluation on `master`, after merging
`dev/retrieval` and `dev/mcts`.

## Scope

Datasets:

- Agriculture: `.runs/openai_full_hirag_2026-05/agriculture/10_hirag_workdir_final`
- Mix: `.runs/openai_full_hirag_2026-05/mix/10_hirag_workdir_final`

Variants:

- `naive`
- `hi`
- `hi_weighted`
- `hi_minmax`
- `hi_minmax_budgeted`
- `hi_rerank_weighted`
- `hi_mcts`

`hi` is the fork baseline HiRAG mode. It uses the baseline unweighted bridge
retrieval path, but final answer exports still use this fork's budgeted
query-aware source snippets. It is therefore not a byte-for-byte upstream
GitHub HiRAG prompt baseline.

## Preflight

Run targeted tests:

```bash
uv run python -m unittest \
  tests.test_hirag_retrieval_experiments \
  tests.test_answer_level_evaluation
```

The answer/export path reads `.env` through `eval/eval_utils.py`, so
`OPENAI_API_KEY` must be available there or in the environment.

## Edge Embedding Cache Warmup

The weighted, minmax, budgeted, rerank-weighted, and MCTS variants need
query-weighted edge costs. To avoid slow live edge embedding during answer
export, warm the edge cache through OpenAI Batch first.

Export requests:

```bash
uv run python eval/export_openai_edge_embedding_batch.py \
  --working-dir .runs/openai_full_hirag_2026-05/agriculture/10_hirag_workdir_final \
  --output-dir .runs/openai_full_hirag_2026-05/core7_eval/agriculture/00_edge_embedding_requests \
  --model text-embedding-3-small \
  --embed-dim 1536 \
  --cache-root .runs/edge_embedding_cache \
  --overwrite

uv run python eval/export_openai_edge_embedding_batch.py \
  --working-dir .runs/openai_full_hirag_2026-05/mix/10_hirag_workdir_final \
  --output-dir .runs/openai_full_hirag_2026-05/core7_eval/mix/00_edge_embedding_requests \
  --model text-embedding-3-small \
  --embed-dim 1536 \
  --cache-root .runs/edge_embedding_cache \
  --overwrite
```

Validate each request shard with `--endpoint /v1/embeddings`:

```bash
uv run python eval/run_openai_batch_file.py \
  --batch-file <edge_embedding_requests_NNN.jsonl> \
  --endpoint /v1/embeddings \
  --validate-only
```

Launch each shard in `screen`:

```bash
screen -dmS rital_core7_<dataset>_edgeemb_<shard>_$(date +%Y%m%d_%H%M%S) bash -lc '
cd /home/paulbeglin/projects/RITAL-IR-project &&
uv run python eval/run_openai_batch_file.py \
  --batch-file <REQUESTS.jsonl> \
  --output-dir <RAW_BATCH_DIR> \
  --endpoint /v1/embeddings \
  --description <dataset>_core7_edge_embeddings_<shard> \
  --poll-interval-seconds 120
'
```

When all shards finish, import them into the target edge cache:

```bash
uv run python eval/import_openai_edge_embedding_batch.py \
  --batch-output <RAW_BATCH_DIR_0>/<OUTPUT.jsonl> [<RAW_BATCH_DIR_1>/<OUTPUT.jsonl>] \
  --metadata <REQUEST_DIR>/edge_embedding_metadata_000.jsonl [<REQUEST_DIR>/edge_embedding_metadata_001.jsonl] \
  --manifest <REQUEST_DIR>/manifest.json \
  --summary-dir <DATASET_DIR>/01_edge_embedding_results/imported \
  --merge-existing
```

The post-edge continuation can also be run as one guarded step:

```bash
screen -dmS rital_core7_after_edge_$(date +%Y%m%d_%H%M%S) bash -lc '
cd /home/paulbeglin/projects/RITAL-IR-project &&
scripts/run_core7_after_edge_embeddings.sh
'
```

It waits for all edge Batch outputs, imports Agriculture and Mix edge caches,
runs one-query smokes, verifies that all Core 7 variants traced successfully
and that `hi_mcts` exposes MCTS bridge metadata, validates full answer request
JSONL files, and launches the Agriculture/Mix answer Batch screens.

## Answer Export And Batch

After edge caches are imported, run one-query smoke exports and summarize traces:

```bash
uv run python eval/export_openai_batch_requests.py \
  --working-dir <WORKDIR> \
  --query-file <QUERY_FILE> \
  --query-limit 1 \
  --variants naive hi hi_weighted hi_minmax hi_minmax_budgeted hi_rerank_weighted hi_mcts \
  --model gpt-5.4-mini \
  --output-dir <DATASET_DIR>/smoke_answer_export \
  --overwrite \
  --include-contexts \
  --embedding-provider openai \
  --embed-model text-embedding-3-small \
  --embed-dim 1536 \
  --answer-max-tokens 512 \
  --completion-token-param max_completion_tokens

uv run python eval/summarize_retrieval_traces.py \
  --traces <DATASET_DIR>/smoke_answer_export/retrieval_traces.jsonl \
  --output-dir <DATASET_DIR>/smoke_answer_export/trace_summary
```

Full export uses `--query-limit 30` and the same variants. Validate the JSONL
with:

```bash
uv run python eval/run_openai_batch_file.py \
  --batch-file <DATASET_DIR>/answer_export/answer_requests.jsonl \
  --validate-only
```

Then launch answer batches in `screen`.

## Judge And Human Evaluation

Judge against `hi` with swapped order:

```bash
uv run python eval/export_openai_judge_batch_requests.py \
  --answers <answers.jsonl> \
  --baseline hi \
  --variants naive hi_weighted hi_minmax hi_minmax_budgeted hi_rerank_weighted hi_mcts \
  --model gpt-5.4-mini \
  --output-dir <DATASET_DIR>/judge_requests
```

Validate and launch the resulting `judge_requests.jsonl` with
`eval/run_openai_batch_file.py`, then import the Batch output:

```bash
uv run python eval/import_openai_batch_judgments.py \
  --batch-output <JUDGE_RAW_BATCH_DIR>/<OUTPUT.jsonl> \
  --metadata <DATASET_DIR>/judge_requests/judge_request_metadata.jsonl \
  --output-dir <DATASET_DIR>/judge_results
```

After imported judge results exist, choose `best_custom` by strict swapped-order
win rate among `hi_weighted`, `hi_minmax`, `hi_minmax_budgeted`, and
`hi_rerank_weighted`, then generate a 60-row pairwise human packet:

```bash
uv run python eval/build_human_annotation_packet.py \
  --answers <combined_answers.jsonl> \
  --packet-format pairwise \
  --pairs hi:naive hi:hi_mcts hi:<best_custom> \
  --rows-per-pair 20 \
  --output-dir artifacts/openai_full_hirag_2026-05/core7_eval/human_annotation_packet
```

Human fields:

- `does_answer_a_answer_question`: yes / partially / no
- `does_answer_b_answer_question`: yes / partially / no
- `preferred_answer`: A / B / tie / neither
- `useful_span_a`
- `useful_span_b`
- `notes`
