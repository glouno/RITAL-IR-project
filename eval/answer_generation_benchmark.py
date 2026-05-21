#!/usr/bin/env python3
"""Generate answers for HiRAG retrieval variants on a fixed query set."""

from __future__ import annotations

import argparse
import asyncio
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
    DEFAULT_ANSWER_MAX_TOKENS,
    DEFAULT_ANSWER_VARIANTS,
    DEFAULT_QUERY_FILE,
    DEFAULT_WORKING_DIR,
    REPO_MAIN,
    add_answer_context_args,
    add_runtime_args,
    build_answer_messages,
    build_context_with_budget,
    build_fastembed_hirag,
    load_queries,
)


DEFAULT_VARIANTS = DEFAULT_ANSWER_VARIANTS


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
    parser.add_argument("--request-timeout-seconds", type=float, default=180.0)
    add_answer_context_args(parser)
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
    decisions = debug.get("bridge_path_decisions", []) or []
    mcts_steps = [
        item for item in decisions if str(item.get("decision", "")).startswith("mcts")
    ]
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
        "mcts_segments": len(mcts_steps),
        "mcts_used_weighted_fallback": any(
            item.get("decision") == "mcts_weighted_fallback" for item in mcts_steps
        ),
        "mcts_iterations": sum(int(item.get("iterations", 0) or 0) for item in mcts_steps),
        "mcts_successful_rollouts": sum(
            int(item.get("successful_rollouts", 0) or 0) for item in mcts_steps
        ),
        "mcts_candidate_nodes_max": max(
            [int(item.get("candidate_nodes", 0) or 0) for item in mcts_steps],
            default=0,
        ),
        "mcts_candidate_edges_max": max(
            [int(item.get("candidate_edges", 0) or 0) for item in mcts_steps],
            default=0,
        ),
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
        "request_timeout": args.request_timeout_seconds,
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
                start = time.perf_counter()
                error = None
                answer = ""
                param = QueryParam(mode=variant)
                budget_debug: dict[str, Any] = {}
                try:
                    context, param, budget_debug = build_context_with_budget(
                        graph,
                        query_item["query"],
                        variant,
                        args,
                    )
                    messages = build_answer_messages(
                        query_item["query"],
                        context,
                        args.response_type,
                    )

                    async def _call_answer() -> str:
                        return await llm_func(
                            messages[-1]["content"],
                            system_prompt=messages[0]["content"],
                            stage="query_answer",
                            max_tokens=args.answer_max_tokens or DEFAULT_ANSWER_MAX_TOKENS,
                        )

                    answer = asyncio.run(
                        asyncio.wait_for(
                            _call_answer(),
                            timeout=args.request_timeout_seconds,
                        )
                    )
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                elapsed = time.perf_counter() - start
                record = {
                    "query_id": query_item["query_id"],
                    "query": query_item["query"],
                    "variant": variant,
                    "answer": answer,
                    "latency_seconds": elapsed,
                    "context_debug": {
                        **compact_debug(dict(param.debug_info)),
                        **budget_debug,
                    },
                    "error": error,
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
    print(f"Wrote answers to {answers_path}")


if __name__ == "__main__":
    main()
