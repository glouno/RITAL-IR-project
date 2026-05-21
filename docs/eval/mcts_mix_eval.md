# MCTS Mix Eval

Use local vLLM only. Do not point any command at hosted OpenAI.

## 1. Check local vLLM

```bash
curl -sS http://127.0.0.1:8000/v1/models
```

If this fails, start vLLM first. Keep `--base-url http://127.0.0.1:8000/v1` in all commands below.

## 2. Materialize runnable Mix workdir from tracked artifact graph

Use this if `.runs/openai_index_batch/mix_hirag_workdir_gpt54_mini_final` missing.

```bash
uv run python eval/materialize_openai_index_hirag_workdir.py \
  --context-file eval/datasets/mix/mix_unique_contexts.json \
  --graphml artifacts/mix_batch_requests_2026-05-10/OpenAI_batch_indexing_graphs/mix_base_graph_gpt54_mini_final/base_graph.graphml \
  --output-dir .runs/openai_index_batch/mix_hirag_workdir_gpt54_mini_final \
  --overwrite \
  --embed-model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 \
  --embed-dim 384 \
  --max-token-size 8192 \
  --embedding-batch-num 64 \
  --embedding-func-max-async 8 \
  --fastembed-cache-path .runs/fastembed_cache
```

Expected artifact summary:

- graph: `artifacts/mix_batch_requests_2026-05-10/OpenAI_batch_indexing_graphs/mix_base_graph_gpt54_mini_final/base_graph.graphml`
- docs: 61
- chunks: 579
- nodes: 17575
- edges: 18588

## 3. Retrieval smoke: `hi_mcts`

Context-only smoke. No answer generation.

```bash
uv run python main.py \
  --working-dir .runs/openai_index_batch/mix_hirag_workdir_gpt54_mini_final \
  --query-mode hi_mcts \
  --query "What is main objection Mary has to poem The Witch of Atlas?" \
  --base-url http://127.0.0.1:8000/v1 \
  --api-key EMPTY \
  --chat-model nvidia/Gemma-4-31B-IT-NVFP4 \
  --embed-model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 \
  --embed-dim 384 \
  --max-token-size 8192 \
  --fastembed-cache-path .runs/fastembed_cache \
```

If you want raw retrieval context only, use small Python harness:

```bash
uv run python - <<'PY'
from hirag import QueryParam
from eval.eval_utils import build_fastembed_hirag
from main import create_vllm_chat_model
from types import SimpleNamespace

runtime = {
    "base_url": "http://127.0.0.1:8000/v1",
    "api_key": "EMPTY",
    "chat_model": "YOUR_LOCAL_MODEL",
    "model_max_context": 8192,
}
llm = create_vllm_chat_model(runtime)
args = SimpleNamespace(
    working_dir=".runs/openai_index_batch/mix_hirag_workdir_gpt54_mini_final",
    embed_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    embed_dim=384,
    max_token_size=8192,
    embedding_batch_num=6,
    embedding_func_max_async=8,
    fastembed_cache_path=".runs/fastembed_cache",
    fastembed_threads=None,
    fastembed_parallel=None,
    reranker_model="answerdotai/answerai-colbert-small-v1",
    reranker_cache_path=".runs/fastembed_cache",
    local_rerank_top_n=100,
    edge_embedding_cache_path=".runs/edge_embedding_cache",
    disable_edge_embedding_cache=False,
    answer_max_tokens=256,
)
graph = build_fastembed_hirag(args, llm_func=llm, enable_llm_cache=False)
param = QueryParam(mode="hi_mcts", only_need_context=True)
context = graph.query("What is main objection Mary has to poem The Witch of Atlas?", param=param)
print(context[:2000])
print(param.debug_info)
PY
```

## 4. Generation eval on Mix with `hi_mcts`

Runs answer generation through local vLLM only.

```bash
uv run python eval/answer_generation_benchmark.py \
  --working-dir .runs/openai_index_batch/mix_hirag_workdir_gpt54_mini_final \
  --query-file eval/datasets/mix/mix.jsonl \
  --query-limit 100 \
  --variants hi_mcts \
  --output-dir .runs/answer_eval/mix_mcts_eval \
  --overwrite \
  --base-url http://127.0.0.1:8000/v1 \
  --api-key EMPTY \
  --chat-model nvidia/Gemma-4-31B-IT-NVFP4 \
  --model-max-context 8192 \
  --request-timeout-seconds 180 \
  --embed-model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 \
  --embed-dim 384 \
  --max-token-size 8192 \
  --embedding-batch-num 6 \
  --embedding-func-max-async 8 \
  --fastembed-cache-path .runs/fastembed_cache \
  --reranker-cache-path .runs/fastembed_cache \
  --edge-embedding-cache-path .runs/edge_embedding_cache \
  --top-k 12 \
  --top-m 6 \
  --max-input-tokens 9000 \
  --answer-max-tokens 512
```

Output:

- `.runs/answer_eval/mix_mcts_eval/answers.jsonl`

Look at:

- `variant == "hi_mcts"`
- `context_debug.bridge_strategy == "mcts"`
- `context_debug.mcts_iterations`
- `context_debug.mcts_successful_rollouts`
- `context_debug.mcts_candidate_nodes_max`
- `context_debug.mcts_used_weighted_fallback`

## 5. Unit tests

```bash
uv run python -m unittest \
  tests.test_hirag_retrieval_experiments \
  tests.test_answer_level_evaluation
```

## 6. Notes

- `mix_eval.jsonl` is judge-request file. Do not use for answer generation.
- Use `eval/datasets/mix/mix.jsonl` for query input.
- `hi_mcts` currently sits on top of query-weighted edge scoring + weighted fallback.
- If vLLM down, unit tests still run. Live generation eval will not.
