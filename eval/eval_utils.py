"""Shared helpers for local HiRAG evaluation scripts."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

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
