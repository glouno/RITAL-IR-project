#!/usr/bin/env python3
"""Deterministic context benchmark for HiRAG retrieval variants."""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from hirag import HiRAG, QueryParam
from hirag._op import encode_string_by_tiktoken
from hirag._storage import NetworkXStorage

import importlib.util


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
DEFAULT_VARIANTS = ["hi", "hi_weighted", "hi_minmax", "hi_rerank", "hi_rerank_weighted"]
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how",
    "in", "is", "it", "of", "on", "or", "that", "the", "to", "what", "why",
    "with", "does", "do", "can", "according",
}


async def _dummy_llm(*_args, **_kwargs) -> str:
    return ""


def parse_args() -> argparse.Namespace:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description="Benchmark HiRAG retrieval contexts")
    parser.add_argument("--working-dir", default=DEFAULT_WORKING_DIR)
    parser.add_argument("--query-file", default="eval/datasets/agriculture/agriculture_query.jsonl")
    parser.add_argument("--query-limit", type=int, default=10)
    parser.add_argument("--variants", nargs="*", default=DEFAULT_VARIANTS)
    parser.add_argument("--output-dir", default=f".runs/retrieval_eval/{timestamp}")
    parser.add_argument("--embed-model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    parser.add_argument("--embed-dim", type=int, default=384)
    parser.add_argument("--max-token-size", type=int, default=8192)
    parser.add_argument("--fastembed-cache-path", default=".runs/fastembed_cache")
    parser.add_argument("--fastembed-threads", type=int, default=None)
    parser.add_argument("--fastembed-parallel", type=int, default=None)
    parser.add_argument("--reranker-model", default="answerdotai/answerai-colbert-small-v1")
    parser.add_argument("--reranker-cache-path", default=".runs/fastembed_cache")
    parser.add_argument("--local-rerank-top-n", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--top-m", type=int, default=10)
    return parser.parse_args()


def load_queries(path: Path, limit: int) -> list[str]:
    queries: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            data = json.loads(line)
            query = data.get("query") or data.get("input") or data.get("question")
            if isinstance(query, str) and query.strip():
                queries.append(query.strip())
            if len(queries) >= limit:
                break
    return queries


def tokenize(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if token not in STOPWORDS and len(token) > 1
    }


def recall(source: set[str], target: set[str]) -> float:
    if not target:
        return 0.0
    return len(source & target) / len(target)


def coverage_metrics(query: str, debug: dict[str, Any]) -> dict[str, float]:
    bridge_text = debug.get("bridge_context_text") or " ".join(
        edge.get("description", "") for edge in debug.get("bridge_edges", [])
    )
    local_text = debug.get("local_context_text") or " ".join(debug.get("local_entities", []))
    global_text = debug.get("global_context_text") or " ".join(
        title for _level, title in debug.get("communities", [])
    )
    bridge_tokens = tokenize(bridge_text)
    return {
        "bridge_vs_local_recall": recall(bridge_tokens, tokenize(local_text)),
        "bridge_vs_global_recall": recall(bridge_tokens, tokenize(global_text)),
        "bridge_vs_query_overlap": recall(bridge_tokens, tokenize(query)),
    }


def build_graph(args: argparse.Namespace) -> HiRAG:
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
        enable_llm_cache=False,
        embedding_func=embedding_func,
        best_model_func=_dummy_llm,
        cheap_model_func=_dummy_llm,
        graph_storage_cls=NetworkXStorage,
        enable_naive_rag=True,
        embedding_batch_num=6,
        embedding_func_max_async=8,
        reranker_model=args.reranker_model,
        reranker_cache_path=args.reranker_cache_path or args.fastembed_cache_path,
        local_rerank_top_n=args.local_rerank_top_n,
    )


def write_summary(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["variant"]].append(row)
    summary = {}
    md_lines = ["# Retrieval Context Benchmark", ""]
    for variant, items in grouped.items():
        summary[variant] = {}
        md_lines.append(f"## {variant}")
        for key in [
            "latency_seconds",
            "context_tokens",
            "bridge_edge_count",
            "bridge_path_edges",
            "mean_edge_score",
            "max_edge_cost",
            "bridge_vs_local_recall",
            "bridge_vs_global_recall",
            "bridge_vs_query_overlap",
        ]:
            values = [item[key] for item in items if item.get(key) not in (None, "")]
            if values:
                summary[variant][key] = statistics.mean(values)
                md_lines.append(f"- {key}: {summary[variant][key]:.4f}")
        md_lines.append("")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "summary.md").write_text("\n".join(md_lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    graph = build_graph(args)
    queries = load_queries(Path(args.query_file), args.query_limit)

    context_path = output_dir / "contexts.jsonl"
    metrics_path = output_dir / "metrics.csv"
    rows: list[dict[str, Any]] = []
    with context_path.open("w", encoding="utf-8") as context_handle:
        for query in queries:
            for variant in args.variants:
                param = QueryParam(
                    mode=variant,
                    only_need_context=True,
                    top_k=args.top_k,
                    top_m=args.top_m,
                )
                start = time.perf_counter()
                context = graph.query(query, param=param)
                elapsed = time.perf_counter() - start
                debug = dict(param.debug_info)
                coverage = coverage_metrics(query, debug)
                row = {
                    "query": query,
                    "variant": variant,
                    "latency_seconds": elapsed,
                    "context_tokens": len(encode_string_by_tiktoken(context or "", "gpt-4o")),
                    "local_entity_count": len(debug.get("local_entities", [])),
                    "community_count": len(debug.get("communities", [])),
                    "bridge_edge_count": len(debug.get("bridge_edges", [])),
                    "bridge_path_edges": debug.get("bridge_path_edges", 0),
                    "mean_edge_score": debug.get("mean_edge_score"),
                    "max_edge_cost": debug.get("max_edge_cost"),
                    **coverage,
                }
                rows.append(row)
                context_handle.write(
                    json.dumps(
                        {
                            "query": query,
                            "variant": variant,
                            "context": context,
                            "selected_entities": debug.get("local_entities", []),
                            "communities": debug.get("communities", []),
                            "bridge_edges": debug.get("bridge_edges", []),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    with metrics_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    write_summary(output_dir, rows)
    print(f"Wrote retrieval benchmark outputs to {output_dir}")


if __name__ == "__main__":
    main()
