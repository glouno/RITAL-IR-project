#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/paulbeglin/projects/RITAL-IR-project"
RUN_ROOT=".runs/openai_full_hirag_2026-05/core7_eval"
VARIANTS=(naive hi hi_weighted hi_minmax hi_minmax_budgeted hi_rerank_weighted hi_mcts)

cd "$ROOT"

wait_for_batch_output() {
  local raw_dir="$1"
  local label="$2"
  local manifest="$raw_dir/batch_run_manifest.json"

  while true; do
    if [[ ! -f "$manifest" ]]; then
      echo "[$(date --iso-8601=seconds)] $label: waiting for manifest $manifest" >&2
      sleep 60
      continue
    fi

    local status
    status="$(jq -r '.latest_batch.status // "unknown"' "$manifest")"
    local completed failed total output_path
    completed="$(jq -r '.latest_batch.request_counts.completed // 0' "$manifest")"
    failed="$(jq -r '.latest_batch.request_counts.failed // 0' "$manifest")"
    total="$(jq -r '.latest_batch.request_counts.total // 0' "$manifest")"
    output_path="$(jq -r '.output_path // empty' "$manifest")"
    echo "[$(date --iso-8601=seconds)] $label: status=$status completed=$completed failed=$failed total=$total" >&2

    if [[ "$status" == "completed" && -n "$output_path" && -s "$output_path" ]]; then
      printf '%s\n' "$output_path"
      return 0
    fi
    if [[ "$status" =~ ^(failed|expired|cancelled)$ ]]; then
      echo "$label ended with terminal non-success status: $status" >&2
      return 1
    fi
    sleep 120
  done
}

summarize_trace() {
  local trace_path="$1"
  local out_dir="$2"
  uv run python eval/summarize_retrieval_traces.py \
    --traces "$trace_path" \
    --output-dir "$out_dir"
}

verify_smoke_trace() {
  local trace_path="$1"
  uv run python - "$trace_path" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
variants = {row.get("variant") for row in rows}
required = {
    "naive",
    "hi",
    "hi_weighted",
    "hi_minmax",
    "hi_minmax_budgeted",
    "hi_rerank_weighted",
    "hi_mcts",
}

missing = sorted(required - variants)
errors = [row for row in rows if row.get("error")]
if missing:
    raise SystemExit(f"smoke trace missing variants: {missing}")
if errors:
    raise SystemExit(f"smoke trace has errors: {[row.get('custom_id') for row in errors]}")

mcts_rows = [row for row in rows if row.get("variant") == "hi_mcts"]
if not mcts_rows:
    raise SystemExit("smoke trace has no hi_mcts row")
mcts_retrieval = mcts_rows[0].get("retrieval", {})
if mcts_retrieval.get("bridge_strategy") != "mcts":
    raise SystemExit(f"hi_mcts bridge_strategy is {mcts_retrieval.get('bridge_strategy')!r}, expected 'mcts'")
if "bridge_path_decisions" not in mcts_retrieval:
    raise SystemExit("hi_mcts trace does not expose bridge_path_decisions")

for row in rows:
    variant = row.get("variant")
    if variant not in {"hi", "hi_weighted", "hi_minmax", "hi_minmax_budgeted", "hi_rerank_weighted", "hi_mcts"}:
        continue
    retrieval = row.get("retrieval", {})
    if "bridge_path_edges" not in retrieval:
        raise SystemExit(f"{variant} trace does not expose bridge_path_edges")

print(f"Smoke trace verified: {len(rows)} rows, variants={sorted(variants)}")
PY
}

runtime_edge_cache_file() {
  local manifest="$1"
  uv run python - "$manifest" <<'PY'
import json
import sys
from pathlib import Path

from hirag._op import _hash_text

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
cache_file = manifest["cache_file"]
request_model_hash = _hash_text(str(manifest["model"]))[:12]
runtime_model_hash = _hash_text("wait_func")[:12]
print(cache_file.replace(f"_{request_model_hash}_", f"_{runtime_model_hash}_"))
PY
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
    --completion-token-param max_completion_tokens

  summarize_trace "$output_dir/retrieval_traces.jsonl" "$output_dir/trace_summary"
  if [[ "$query_limit" == "1" ]]; then
    verify_smoke_trace "$output_dir/retrieval_traces.jsonl"
  fi

  uv run python eval/run_openai_batch_file.py \
    --batch-file "$output_dir/answer_requests.jsonl" \
    --validate-only
}

launch_answer_batch() {
  local dataset="$1"
  local request_dir="$2"
  local result_dir="$3"
  local session_name="rital_core7_${dataset}_answers_$(date +%Y%m%d_%H%M%S)"

  screen -dmS "$session_name" bash -lc "
cd $ROOT &&
uv run python eval/run_openai_batch_file.py \
  --batch-file $request_dir/answer_requests.jsonl \
  --output-dir $result_dir/raw_batch \
  --description ${dataset}_core7_answers_gpt54_mini \
  --poll-interval-seconds 120
"
  echo "Launched $session_name"
}

AGRI_DIR="$RUN_ROOT/agriculture"
MIX_DIR="$RUN_ROOT/mix"

AGRI_EDGE_000="$(wait_for_batch_output "$AGRI_DIR/01_edge_embedding_results/raw_batch_000" "agriculture edge shard 000")"
AGRI_EDGE_001="$(wait_for_batch_output "$AGRI_DIR/01_edge_embedding_results/raw_batch_001" "agriculture edge shard 001")"
MIX_EDGE_000="$(wait_for_batch_output "$MIX_DIR/01_edge_embedding_results/raw_batch_000" "mix edge shard 000")"

AGRI_RUNTIME_EDGE_CACHE="$(runtime_edge_cache_file "$AGRI_DIR/00_edge_embedding_requests/manifest.json")"
MIX_RUNTIME_EDGE_CACHE="$(runtime_edge_cache_file "$MIX_DIR/00_edge_embedding_requests/manifest.json")"

uv run python eval/import_openai_edge_embedding_batch.py \
  --batch-output "$AGRI_EDGE_000" "$AGRI_EDGE_001" \
  --metadata \
    "$AGRI_DIR/00_edge_embedding_requests/edge_embedding_metadata_000.jsonl" \
    "$AGRI_DIR/00_edge_embedding_requests/edge_embedding_metadata_001.jsonl" \
  --manifest "$AGRI_DIR/00_edge_embedding_requests/manifest.json" \
  --cache-file "$AGRI_RUNTIME_EDGE_CACHE" \
  --summary-dir "$AGRI_DIR/01_edge_embedding_results/imported" \
  --merge-existing

uv run python eval/import_openai_edge_embedding_batch.py \
  --batch-output "$MIX_EDGE_000" \
  --metadata "$MIX_DIR/00_edge_embedding_requests/edge_embedding_metadata_000.jsonl" \
  --manifest "$MIX_DIR/00_edge_embedding_requests/manifest.json" \
  --cache-file "$MIX_RUNTIME_EDGE_CACHE" \
  --summary-dir "$MIX_DIR/01_edge_embedding_results/imported" \
  --merge-existing

export_answers \
  agriculture \
  ".runs/openai_full_hirag_2026-05/agriculture/10_hirag_workdir_final" \
  "eval/datasets/agriculture/agriculture_query.jsonl" \
  1 \
  "$AGRI_DIR/smoke_answer_export" \
  512

export_answers \
  mix \
  ".runs/openai_full_hirag_2026-05/mix/10_hirag_workdir_final" \
  "eval/datasets/mix/mix.jsonl" \
  1 \
  "$MIX_DIR/smoke_answer_export" \
  512

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
  "eval/datasets/mix/mix.jsonl" \
  30 \
  "$MIX_DIR/02_answer_requests" \
  1024

launch_answer_batch agriculture "$AGRI_DIR/02_answer_requests" "$AGRI_DIR/03_answer_results"
launch_answer_batch mix "$MIX_DIR/02_answer_requests" "$MIX_DIR/03_answer_results"

echo "Core 7 answer batches launched. Next: import answers after screens finish, then export/import judge batches."
