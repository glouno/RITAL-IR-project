"""Shared helpers for local HiRAG evaluation scripts."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import tiktoken

from hirag import QueryParam
from hirag import HiRAG
from hirag._storage import NetworkXStorage


REPO_ROOT = Path(__file__).resolve().parents[1]
MAIN_SPEC = importlib.util.spec_from_file_location("hirag_repo_main", REPO_ROOT / "main.py")
if MAIN_SPEC is None or MAIN_SPEC.loader is None:
    raise RuntimeError("Unable to load top-level main.py")
REPO_MAIN = importlib.util.module_from_spec(MAIN_SPEC)
MAIN_SPEC.loader.exec_module(REPO_MAIN)

DEFAULT_WORKING_DIR = (
    ".runs/2026-04-22-agri-resume2/graphs/"
    "benchmark_ultra_lean_run1_20260422_204326"
)
DEFAULT_QUERY_FILE = "eval/datasets/agriculture/agriculture_query.jsonl"
DEFAULT_ANSWER_VARIANTS = ["hi", "hi_minmax_budgeted", "hi_rerank_weighted"]
DEFAULT_RESPONSE_TYPE = "Multiple Paragraphs"
DEFAULT_TOP_K = 12
DEFAULT_TOP_M = 6
DEFAULT_CONTEXT_BUDGET = 1000
DEFAULT_TEXT_UNIT_BUDGET = 6000
DEFAULT_TEXT_UNIT_SNIPPET_CHARS = 1200
DEFAULT_ANSWER_MAX_TOKENS = 512
DEFAULT_MAX_INPUT_TOKENS = 9000


def load_queries(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    queries: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            data = json.loads(line)
            query = data.get("query") or data.get("input") or data.get("question")
            if isinstance(query, str) and query.strip():
                queries.append(
                    {
                        "query_id": data.get("query_id", index),
                        "query": query.strip(),
                    }
                )
            if limit is not None and len(queries) >= limit:
                break
    return queries


def build_fastembed_hirag(args: Any, *, llm_func: Any, enable_llm_cache: bool) -> HiRAG:
    if args.fastembed_cache_path:
        Path(args.fastembed_cache_path).mkdir(parents=True, exist_ok=True)
    if args.reranker_cache_path:
        Path(args.reranker_cache_path).mkdir(parents=True, exist_ok=True)
    runtime = {
        "embed_model": args.embed_model,
        "embed_dim": args.embed_dim,
        "max_token_size": args.max_token_size,
        "fastembed_threads": args.fastembed_threads,
        "fastembed_parallel": args.fastembed_parallel,
        "fastembed_cache_path": args.fastembed_cache_path,
    }
    embedding_func = REPO_MAIN.create_fastembed_embedding(runtime)
    stage_max_tokens = {}
    if hasattr(args, "answer_max_tokens") and args.answer_max_tokens:
        stage_max_tokens["query_answer"] = args.answer_max_tokens
    return HiRAG(
        working_dir=args.working_dir,
        enable_llm_cache=enable_llm_cache,
        embedding_func=embedding_func,
        best_model_func=llm_func,
        cheap_model_func=llm_func,
        graph_storage_cls=NetworkXStorage,
        enable_naive_rag=True,
        embedding_batch_num=args.embedding_batch_num,
        embedding_func_max_async=args.embedding_func_max_async,
        reranker_model=args.reranker_model,
        reranker_cache_path=args.reranker_cache_path or args.fastembed_cache_path,
        local_rerank_top_n=args.local_rerank_top_n,
        edge_embedding_cache_path=args.edge_embedding_cache_path,
        disable_edge_embedding_cache=args.disable_edge_embedding_cache,
        stage_max_tokens=stage_max_tokens,
    )


def add_runtime_args(parser: Any) -> None:
    parser.add_argument("--working-dir", default=DEFAULT_WORKING_DIR)
    parser.add_argument("--embed-model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    parser.add_argument("--embed-dim", type=int, default=384)
    parser.add_argument("--max-token-size", type=int, default=8192)
    parser.add_argument("--embedding-batch-num", type=int, default=6)
    parser.add_argument("--embedding-func-max-async", type=int, default=8)
    parser.add_argument("--fastembed-cache-path", default=".runs/fastembed_cache")
    parser.add_argument("--fastembed-threads", type=int, default=None)
    parser.add_argument("--fastembed-parallel", type=int, default=None)
    parser.add_argument("--reranker-model", default="answerdotai/answerai-colbert-small-v1")
    parser.add_argument("--reranker-cache-path", default=".runs/fastembed_cache")
    parser.add_argument("--local-rerank-top-n", type=int, default=100)
    parser.add_argument("--edge-embedding-cache-path", default=".runs/edge_embedding_cache")
    parser.add_argument("--disable-edge-embedding-cache", action="store_true")


def add_answer_context_args(parser: Any) -> None:
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--top-m", type=int, default=DEFAULT_TOP_M)
    parser.add_argument("--response-type", default=DEFAULT_RESPONSE_TYPE)
    parser.add_argument("--answer-max-tokens", type=int, default=DEFAULT_ANSWER_MAX_TOKENS)
    parser.add_argument("--max-input-tokens", type=int, default=DEFAULT_MAX_INPUT_TOKENS)
    parser.add_argument("--max-budget-attempts", type=int, default=5)
    parser.add_argument("--budget-shrink-factor", type=float, default=0.7)
    parser.add_argument("--min-section-budget", type=int, default=250)
    parser.add_argument("--max-token-for-local-context", type=int, default=DEFAULT_CONTEXT_BUDGET)
    parser.add_argument("--max-token-for-bridge-knowledge", type=int, default=DEFAULT_CONTEXT_BUDGET)
    parser.add_argument("--max-token-for-community-report", type=int, default=DEFAULT_CONTEXT_BUDGET)
    parser.add_argument("--max-token-for-text-unit", type=int, default=DEFAULT_TEXT_UNIT_BUDGET)
    parser.add_argument("--text-unit-snippet-chars", type=int, default=DEFAULT_TEXT_UNIT_SNIPPET_CHARS)


def make_query_param(
    variant: str,
    args: Any,
    *,
    only_need_context: bool,
    budget_scale: float = 1.0,
) -> QueryParam:
    def scaled(value: int) -> int:
        return max(int(args.min_section_budget), int(value * budget_scale))

    top_k = max(4, int(args.top_k * budget_scale))
    top_m = max(3, int(args.top_m * budget_scale))
    return QueryParam(
        mode=variant,
        only_need_context=only_need_context,
        top_k=top_k,
        top_m=top_m,
        response_type=args.response_type,
        max_token_for_local_context=scaled(args.max_token_for_local_context),
        max_token_for_bridge_knowledge=scaled(args.max_token_for_bridge_knowledge),
        max_token_for_community_report=scaled(args.max_token_for_community_report),
        max_token_for_text_unit=scaled(args.max_token_for_text_unit),
        text_unit_snippet_chars=args.text_unit_snippet_chars,
    )


def build_answer_messages(
    query: str,
    context: str,
    response_type: str,
    prompts: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    prompts = prompts or REPO_MAIN.resolve_prompts("baseline")
    system_prompt = prompts["local_rag_response"].format(
        context_data=context,
        response_type=response_type,
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query},
    ]


def estimate_chat_tokens(messages: list[dict[str, str]]) -> int:
    encoder = tiktoken.get_encoding("cl100k_base")
    return sum(len(encoder.encode(message["content"])) + 4 for message in messages) + 2


def build_context_with_budget(
    graph: HiRAG,
    query: str,
    variant: str,
    args: Any,
) -> tuple[str, QueryParam, dict[str, Any]]:
    last_context = ""
    last_param: QueryParam | None = None
    attempts = []
    for attempt in range(max(1, args.max_budget_attempts)):
        scale = args.budget_shrink_factor ** attempt
        param = make_query_param(
            variant,
            args,
            only_need_context=True,
            budget_scale=scale,
        )
        context = graph.query(query, param=param)
        messages = build_answer_messages(query, context, args.response_type)
        input_tokens = estimate_chat_tokens(messages)
        attempts.append(
            {
                "attempt": attempt + 1,
                "scale": scale,
                "top_k": param.top_k,
                "top_m": param.top_m,
                "section_budgets": {
                    "local": param.max_token_for_local_context,
                    "bridge": param.max_token_for_bridge_knowledge,
                    "global": param.max_token_for_community_report,
                    "text": param.max_token_for_text_unit,
                },
                "input_tokens": input_tokens,
            }
        )
        last_context = context
        last_param = param
        if input_tokens <= args.max_input_tokens:
            break
    if last_param is None:
        raise RuntimeError("No context budget attempt was executed")
    budget_debug = {
        "context_input_tokens": attempts[-1]["input_tokens"],
        "context_budget_attempts": attempts,
        "context_within_budget": attempts[-1]["input_tokens"] <= args.max_input_tokens,
    }
    return last_context, last_param, budget_debug


def batch_custom_id(query_id: Any, variant: str) -> str:
    safe_query_id = str(query_id).replace("|", "_").replace(" ", "_")
    safe_variant = str(variant).replace("|", "_").replace(" ", "_")
    return f"answer|{safe_query_id}|{safe_variant}"
