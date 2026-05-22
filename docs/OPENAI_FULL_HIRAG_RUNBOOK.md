# OpenAI Full HiRAG Batch Runbook

This runbook is for the full OpenAI-only rerun path:

- `gpt-5.4-mini` for indexing, community reports, answers, and judges.
- `text-embedding-3-small` for vector stores.
- Embeddings can be generated through OpenAI Batch and then materialized as
  precomputed `NanoVectorDB` files.
- High `max_completion_tokens` caps with targeted retries.
- Phase 1 datasets: `agriculture`, then `mix`.
- Phase 2 datasets: `cs`, then `legal`, only after Phase 1 cost/quality checks pass.

## Directory Layout

Use one root per campaign:

```text
.runs/openai_full_hirag_2026-05/
  agriculture/
    01_entity_requests/
    02_entity_results/
    03_relation_requests/
    04_relation_results/
    05_base_graph/
    06_cluster_summary_requests/
    07_cluster_summary_results/
    08_community_report_requests/
    09_community_report_results/
    10_hirag_workdir/
    11_answers/
    12_judges/
    cost_summary.json
```

## Token Caps

| Stage | Initial | Retry 1 | Retry 2 |
|---|---:|---:|---:|
| Entity extraction | 2048 | 4096 | 8192 |
| Relation extraction | 4096 | 8192 | 12000 |
| Cluster summary | 2048 | 4096 | 8192 |
| Community report | 2048 | 4096 | 8192 |
| Answer generation | 1024 | 2048 | - |
| Judge | 1024 | 2048 | - |

## Screen Launch Pattern

Validate first:

```bash
uv run python eval/run_openai_batch_file.py \
  --batch-file <REQUESTS.jsonl> \
  --validate-only
```

Launch:

```bash
screen -dmS rital_<dataset>_<stage>_$(date +%Y%m%d_%H%M%S) bash -lc '
cd /home/paulbeglin/projects/RITAL-IR-project &&
uv run python eval/run_openai_batch_file.py \
  --batch-file <REQUESTS.jsonl> \
  --output-dir <RESULT_DIR> \
  --description <dataset>_<stage>_gpt54_mini_full \
  --poll-interval-seconds 120
'
```

Monitor:

```bash
screen -ls
screen -r <session_name>
tail -f <RESULT_DIR>/batch_run_manifest.json
```

## Optional Direct API Micro-Retries

Batch remains the default path for cost and reproducibility. For very small
retry files, for example fewer than 10 rows, the live API helper can avoid
waiting for Batch scheduling while preserving the same request body and output
shape used by the Batch importers.

Validate the file first:

```bash
uv run python eval/run_openai_requests_file_direct.py \
  --batch-file <RETRY_REQUESTS.jsonl> \
  --validate-only
```

Run direct API calls:

```bash
uv run python eval/run_openai_requests_file_direct.py \
  --batch-file <RETRY_REQUESTS.jsonl> \
  --output-file <DIRECT_OUTPUT.jsonl> \
  --overwrite
```

The resulting `<DIRECT_OUTPUT.jsonl>` can be passed to the same importer as a
normal Batch output file. Use this only as a secondary convenience path for
micro-retries; keep large extraction, summary, report, answer, judge, and
embedding stages on Batch.

## Phase 1 Commands

Set variables:

```bash
ROOT=.runs/openai_full_hirag_2026-05
DATASET=agriculture
CONTEXT=eval/datasets/agriculture/agriculture_unique_contexts.json
MODEL=gpt-5.4-mini
```

Export entity requests:

```bash
uv run python eval/export_openai_index_entity_batch.py \
  --context-file "$CONTEXT" \
  --model "$MODEL" \
  --prompt-regime baseline \
  --max-completion-tokens 2048 \
  --output-dir "$ROOT/$DATASET/01_entity_requests" \
  --overwrite
```

After the Batch completes, import:

```bash
uv run python eval/import_openai_index_entity_batch.py \
  --batch-output "$ROOT/$DATASET/02_entity_results/raw_batch/<BATCH>_output.jsonl" \
  --metadata "$ROOT/$DATASET/01_entity_requests/entity_extract_metadata.jsonl" \
  --prompt-regime baseline \
  --output-dir "$ROOT/$DATASET/02_entity_results/imported"
```

Export retries if needed:

```bash
uv run python eval/export_openai_batch_retry_requests.py \
  --source-requests "$ROOT/$DATASET/01_entity_requests/entity_extract_requests.jsonl" \
  --batch-output "$ROOT/$DATASET/02_entity_results/raw_batch/<BATCH>_output.jsonl" \
  --max-completion-tokens 4096 \
  --output-dir "$ROOT/$DATASET/02_entity_results/retry_4096"
```

Export relation requests:

```bash
uv run python eval/export_openai_index_relation_batch.py \
  --context-file "$CONTEXT" \
  --entity-results "$ROOT/$DATASET/02_entity_results/imported/entity_extract_results.jsonl" \
  --model "$MODEL" \
  --prompt-regime baseline \
  --max-completion-tokens 4096 \
  --output-dir "$ROOT/$DATASET/03_relation_requests" \
  --overwrite
```

Assemble graph after relation import:

```bash
uv run python eval/assemble_openai_index_base_graph.py \
  --entity-results "$ROOT/$DATASET/02_entity_results/imported/entity_extract_results.jsonl" \
  --relation-results "$ROOT/$DATASET/04_relation_results/imported/relation_extract_results.jsonl" \
  --output-dir "$ROOT/$DATASET/05_base_graph"
```

Materialize clustered graph with temporary extractive reports so communities exist:

```bash
uv run python eval/materialize_openai_index_hirag_workdir.py \
  --context-file "$CONTEXT" \
  --graphml "$ROOT/$DATASET/05_base_graph/base_graph.graphml" \
  --output-dir "$ROOT/$DATASET/10_hirag_workdir_pre_reports" \
  --embedding-provider fastembed \
  --overwrite
```

Optional cluster-summary inspection batch:

```bash
uv run python eval/export_openai_index_cluster_summary_batch.py \
  --graphml "$ROOT/$DATASET/10_hirag_workdir_pre_reports/graph_chunk_entity_relation.graphml" \
  --model "$MODEL" \
  --prompt-regime baseline \
  --input-max-tokens 12000 \
  --max-completion-tokens 2048 \
  --output-dir "$ROOT/$DATASET/06_cluster_summary_requests" \
  --overwrite
```

After the cluster-summary Batch completes:

```bash
uv run python eval/import_openai_index_cluster_summary_batch.py \
  --batch-output "$ROOT/$DATASET/07_cluster_summary_results/raw_batch/<BATCH>_output.jsonl" \
  --metadata "$ROOT/$DATASET/06_cluster_summary_requests/cluster_summary_metadata.jsonl" \
  --prompt-regime baseline \
  --output-dir "$ROOT/$DATASET/07_cluster_summary_results/imported"
```

Apply parsed summary entities/relations to the graph:

```bash
uv run python eval/apply_openai_index_cluster_summaries.py \
  --graphml "$ROOT/$DATASET/10_hirag_workdir_pre_reports/graph_chunk_entity_relation.graphml" \
  --cluster-summary-results "$ROOT/$DATASET/07_cluster_summary_results/imported/cluster_summary_results.jsonl" \
  --output-dir "$ROOT/$DATASET/05_base_graph_with_summaries"
```

Re-cluster the summary-augmented graph before community reports:

```bash
uv run python eval/materialize_openai_index_hirag_workdir.py \
  --context-file "$CONTEXT" \
  --graphml "$ROOT/$DATASET/05_base_graph_with_summaries/graph_with_cluster_summaries.graphml" \
  --output-dir "$ROOT/$DATASET/10_hirag_workdir_pre_reports_with_summaries" \
  --embedding-provider fastembed \
  --overwrite
```

Export community report requests from the clustered graph:

```bash
uv run python eval/export_openai_index_community_report_batch.py \
  --graphml "$ROOT/$DATASET/10_hirag_workdir_pre_reports_with_summaries/graph_chunk_entity_relation.graphml" \
  --model "$MODEL" \
  --prompt-regime baseline \
  --input-max-tokens 12000 \
  --max-completion-tokens 2048 \
  --output-dir "$ROOT/$DATASET/08_community_report_requests" \
  --overwrite
```

For the closest match to the original sequential report generation, export one
community level at a time with `--level <N>` and pass the previous imported
`kv_store_community_reports.json` back as `--existing-reports`. The all-level
command above is acceptable for a first Phase 1 validation run when there are no
large truncated communities.

Import community reports:

```bash
uv run python eval/import_openai_index_community_report_batch.py \
  --batch-output "$ROOT/$DATASET/09_community_report_results/raw_batch/<BATCH>_output.jsonl" \
  --metadata "$ROOT/$DATASET/08_community_report_requests/community_report_metadata.jsonl" \
  --output-dir "$ROOT/$DATASET/09_community_report_results/imported"
```

Fallback materialization with direct embeddings API:

```bash
uv run python eval/materialize_openai_index_hirag_workdir.py \
  --context-file "$CONTEXT" \
  --graphml "$ROOT/$DATASET/10_hirag_workdir_pre_reports_with_summaries/graph_chunk_entity_relation.graphml" \
  --community-reports-json "$ROOT/$DATASET/09_community_report_results/imported/kv_store_community_reports.json" \
  --embedding-provider openai \
  --embedding-batch-num 64 \
  --skip-clustering \
  --output-dir "$ROOT/$DATASET/10_hirag_workdir" \
  --overwrite
```

Prefer the fully Batch-priced embedding path below for the final Phase 1 runs.
Export embeddings before final materialization:

```bash
uv run python eval/export_openai_embedding_batch.py \
  --context-file "$CONTEXT" \
  --graphml "$ROOT/$DATASET/10_hirag_workdir_pre_reports_with_summaries/graph_chunk_entity_relation.graphml" \
  --model text-embedding-3-small \
  --output-dir "$ROOT/$DATASET/10_embeddings/requests" \
  --overwrite
```

Launch the embedding Batch with the embeddings endpoint:

```bash
screen -dmS "rital_${DATASET}_embeddings_$(date +%Y%m%d_%H%M%S)" bash -lc "
cd /home/paulbeglin/projects/RITAL-IR-project &&
uv run python eval/run_openai_batch_file.py \
  --endpoint /v1/embeddings \
  --batch-file '$ROOT/$DATASET/10_embeddings/requests/embedding_requests.jsonl' \
  --output-dir '$ROOT/$DATASET/10_embeddings/raw_batch' \
  --description '${DATASET}_embeddings_text_embedding_3_small_full' \
  --poll-interval-seconds 120
"
```

After the embedding Batch completes:

```bash
uv run python eval/import_openai_embedding_batch.py \
  --batch-output "$ROOT/$DATASET/10_embeddings/raw_batch/<BATCH>_output.jsonl" \
  --metadata "$ROOT/$DATASET/10_embeddings/requests/embedding_metadata.jsonl" \
  --output-dir "$ROOT/$DATASET/10_embeddings/imported"
```

Then materialize using the precomputed embeddings:

```bash
uv run python eval/materialize_openai_index_hirag_workdir.py \
  --context-file "$CONTEXT" \
  --graphml "$ROOT/$DATASET/10_hirag_workdir_pre_reports_with_summaries/graph_chunk_entity_relation.graphml" \
  --community-reports-json "$ROOT/$DATASET/09_community_report_results/imported/kv_store_community_reports.json" \
  --embedding-provider precomputed \
  --entity-embeddings-jsonl "$ROOT/$DATASET/10_embeddings/imported/entities_embeddings.jsonl" \
  --chunk-embeddings-jsonl "$ROOT/$DATASET/10_embeddings/imported/chunks_embeddings.jsonl" \
  --skip-clustering \
  --output-dir "$ROOT/$DATASET/10_hirag_workdir" \
  --overwrite
```

Summarize cost:

```bash
uv run python eval/summarize_openai_full_hirag_run.py \
  --run-dir "$ROOT/$DATASET"
```

Generate answer requests after the final workdir exists:

```bash
uv run python eval/export_openai_batch_requests.py \
  --working-dir "$ROOT/$DATASET/10_hirag_workdir" \
  --query-file "eval/datasets/$DATASET/${DATASET}_query.jsonl" \
  --query-limit 30 \
  --variants hi naive hi_nobridge hi_rerank_weighted hi_minmax_budgeted \
  --model "$MODEL" \
  --answer-max-tokens 1024 \
  --output-dir "$ROOT/$DATASET/11_answers/requests" \
  --include-contexts \
  --completion-token-param max_completion_tokens
```

This also writes `retrieval_traces.jsonl`, which should be kept with the final
artifacts. It contains the selected entities, communities, bridge edges, bridge
path decisions, token budgets, and context sections for each query/variant.

Summarize retrieval traces:

```bash
uv run python eval/summarize_retrieval_traces.py \
  --traces "$ROOT/$DATASET/11_answers/requests/retrieval_traces.jsonl" \
  --output-dir "$ROOT/$DATASET/11_answers/retrieval_trace_summary"
```

For `mix`, use `eval/datasets/mix/mix.jsonl` as the query file. After importing
answers, export judges:

```bash
uv run python eval/export_openai_judge_batch_requests.py \
  --answers "$ROOT/$DATASET/11_answers/imported/answers.jsonl" \
  --baseline hi \
  --variants naive hi_nobridge hi_rerank_weighted hi_minmax_budgeted \
  --model "$MODEL" \
  --max-tokens 1024 \
  --output-dir "$ROOT/$DATASET/12_judges/requests"
```

Repeat with:

```bash
DATASET=mix
CONTEXT=eval/datasets/mix/mix_unique_contexts.json
```

## Go / No-Go Before CS and Legal

- 0 final API errors after retries.
- Less than 1% final parse errors.
- Entity/relation retry rate below 30%.
- Community reports are `llm_batch`, not `extractive_from_batch_graph`.
- Actual cost is within 2x of estimate.
