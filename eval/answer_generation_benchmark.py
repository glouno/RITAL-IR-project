#!/usr/bin/env python3
"""Generate answers for HiRAG retrieval variants on a fixed query set."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from hirag import QueryParam

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.eval_utils import (
    DEFAULT_QUERY_FILE,
    DEFAULT_WORKING_DIR,
    REPO_MAIN,
    add_runtime_args,
    build_fastembed_hirag,
    load_queries,
)


DEFAULT_VARIANTS = ["hi", "hi_weighted", "hi_minmax", "hi_rerank_weighted"]


def parse_args() -> argparse.Namespace:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description="Generate answer-level outputs for HiRAG variants")
    parser.add_argument("--query-file", default=DEFAULT_QUERY_FILE)
    parser.add_argument("--query-limit", type=int, default=30)
    parser.add_argument("--variants", nargs="*", default=DEFAULT_VARIANTS)
    parser.add_argument("--output-dir", default=f".runs/answer_eval/{timestamp}")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--chat-model", default=None)
    parser.add_argument("--model-max-context", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--top-m", type=int, default=10)
    parser.add_argument("--response-type", default="Multiple Paragraphs")
    parser.add_argument("--max-token-for-local-context", type=int, default=2000)
    parser.add_argument("--max-token-for-bridge-knowledge", type=int, default=2500)
    parser.add_argument("--max-token-for-community-report", type=int, default=2500)
    parser.add_argument("--max-token-for-text-unit", type=int, default=2500)
    parser.add_argument("--llm-cache", action="store_true")
    add_runtime_args(parser)
    return parser.parse_args()


def existing_keys(path: Path) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    keys = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            data = json.loads(line)
            keys.add((str(data.get("query_id")), str(data.get("variant"))))
    return keys


def compact_debug(debug: dict[str, Any]) -> dict[str, Any]:
    return {
        "bridge_strategy": debug.get("bridge_strategy"),
        "local_rerank_strategy": debug.get("local_rerank_strategy"),
        "local_entity_count": len(debug.get("local_entities", [])),
        "community_count": len(debug.get("communities", [])),
        "bridge_edge_count": len(debug.get("bridge_edges", [])),
        "bridge_path_edges": debug.get("bridge_path_edges"),
        "mean_edge_score": debug.get("mean_edge_score"),
        "max_edge_cost": debug.get("max_edge_cost"),
        "bridge_budget_fallbacks": debug.get("bridge_budget_fallbacks", 0),
        "bridge_budget_stopped": debug.get("bridge_budget_stopped", False),
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    answers_path = output_dir / "answers.jsonl"
    done = set() if args.overwrite else existing_keys(answers_path)

    chat_model = args.chat_model
    model_max_context = args.model_max_context
    if not chat_model or not model_max_context:
        discovered_model, discovered_context = REPO_MAIN.discover_local_chat_model_info(
            args.base_url, args.api_key
        )
        chat_model = chat_model or discovered_model
        model_max_context = model_max_context or discovered_context
    if not chat_model:
        raise ValueError("No chat model configured; pass --chat-model")
    runtime = {
        "base_url": args.base_url,
        "api_key": args.api_key,
        "chat_model": chat_model,
        "model_max_context": model_max_context,
    }
    llm_func = REPO_MAIN.create_vllm_chat_model(runtime)
    graph = build_fastembed_hirag(args, llm_func=llm_func, enable_llm_cache=args.llm_cache)
    queries = load_queries(Path(args.query_file), args.query_limit)

    mode = "w" if args.overwrite else "a"
    with answers_path.open(mode, encoding="utf-8") as handle:
        for query_item in queries:
            for variant in args.variants:
                key = (str(query_item["query_id"]), variant)
                if key in done:
                    continue
                param = QueryParam(
                    mode=variant,
                    only_need_context=False,
                    top_k=args.top_k,
                    top_m=args.top_m,
                    response_type=args.response_type,
                    max_token_for_local_context=args.max_token_for_local_context,
                    max_token_for_bridge_knowledge=args.max_token_for_bridge_knowledge,
                    max_token_for_community_report=args.max_token_for_community_report,
                    max_token_for_text_unit=args.max_token_for_text_unit,
                )
                start = time.perf_counter()
                error = None
                answer = ""
                try:
                    answer = graph.query(query_item["query"], param=param)
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                elapsed = time.perf_counter() - start
                record = {
                    "query_id": query_item["query_id"],
                    "query": query_item["query"],
                    "variant": variant,
                    "answer": answer,
                    "latency_seconds": elapsed,
                    "context_debug": compact_debug(dict(param.debug_info)),
                    "error": error,
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
    print(f"Wrote answers to {answers_path}")


if __name__ == "__main__":
    main()
