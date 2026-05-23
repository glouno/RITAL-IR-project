#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/paulbeglin/projects/RITAL-IR-project"
RUN_ROOT=".runs/openai_full_hirag_2026-05/core7_eval_community2200"
COMMUNITY_REPORT_BUDGET=2200
VARIANTS=(naive hi hi_weighted hi_minmax hi_minmax_budgeted hi_rerank_weighted hi_mcts)

cd "$ROOT"

summarize_trace() {
  local trace_path="$1"
  local out_dir="$2"
  uv run python eval/summarize_retrieval_traces.py \
    --traces "$trace_path" \
    --output-dir "$out_dir"
}

export_answers() {
  local dataset="$1"
  local workdir="$2"
  local query_file="$3"
  local query_limit="$4"
  local output_dir="$5"
  local answer_max_tokens="$6"

  uv run python eval/export_openai_batch_requests.py \
    --working-dir "$workdir" \
    --query-file "$query_file" \
    --query-limit "$query_limit" \
    --variants "${VARIANTS[@]}" \
    --model gpt-5.4-mini \
    --output-dir "$output_dir" \
    --overwrite \
    --include-contexts \
    --embedding-provider openai \
    --embed-model text-embedding-3-small \
    --embed-dim 1536 \
    --answer-max-tokens "$answer_max_tokens" \
    --max-token-for-community-report "$COMMUNITY_REPORT_BUDGET" \
    --completion-token-param max_completion_tokens

  summarize_trace "$output_dir/retrieval_traces.jsonl" "$output_dir/trace_summary"

  uv run python eval/run_openai_batch_file.py \
    --batch-file "$output_dir/answer_requests.jsonl" \
    --validate-only
}

launch_answer_batch() {
  local dataset="$1"
  local request_dir="$2"
  local result_dir="$3"
  local session_name="rital_core7_comm2200_${dataset}_answers_$(date +%Y%m%d_%H%M%S)"

  mkdir -p "$result_dir"
  screen -dmS "$session_name" bash -lc "
cd $ROOT &&
uv run python eval/run_openai_batch_file.py \
  --batch-file $request_dir/answer_requests.jsonl \
  --output-dir $result_dir/raw_batch \
  --description ${dataset}_core7_comm2200_answers_gpt54_mini \
  --poll-interval-seconds 120
"
  echo "Launched $session_name"
}

AGRI_DIR="$RUN_ROOT/agriculture"
MIX_DIR="$RUN_ROOT/mix"

export_answers \
  agriculture \
  ".runs/openai_full_hirag_2026-05/agriculture/10_hirag_workdir_final" \
  "eval/datasets/agriculture/agriculture_query.jsonl" \
  30 \
  "$AGRI_DIR/02_answer_requests" \
  1024

export_answers \
  mix \
  ".runs/openai_full_hirag_2026-05/mix/10_hirag_workdir_final" \
  30 \
  "$MIX_DIR/02_answer_requests" \
  1024

launch_answer_batch agriculture "$AGRI_DIR/02_answer_requests" "$AGRI_DIR/03_answer_results"
launch_answer_batch mix "$MIX_DIR/02_answer_requests" "$MIX_DIR/03_answer_results"

cat <<EOF
Core 7 community2200 answer batches launched.

Run root:
  $RUN_ROOT

Intentional differences from core7_eval:
  - max_token_for_community_report=$COMMUNITY_REPORT_BUDGET
  - cluster-* provenance ids are ignored during text-chunk lookup

Next after answer batch completion:
  1. import answers from 03_answer_results/raw_batch
  2. export/import pairwise judges vs hi
  3. promote artifacts under artifacts/openai_full_hirag_2026-05/core7_eval_community2200/
EOF
