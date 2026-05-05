#!/usr/bin/env python3
"""Export HiRAG answer-generation requests as OpenAI Batch API JSONL."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from hirag._op import encode_string_by_tiktoken

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.answer_generation_benchmark import compact_debug
from eval.eval_utils import (
    DEFAULT_ANSWER_MAX_TOKENS,
    DEFAULT_ANSWER_VARIANTS,
    DEFAULT_QUERY_FILE,
    REPO_MAIN,
    add_answer_context_args,
    add_runtime_args,
    batch_custom_id,
    build_answer_messages,
    build_context_with_budget,
    build_fastembed_hirag,
    estimate_chat_tokens,
    load_queries,
)


def parse_args() -> argparse.Namespace:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(
        description="Create OpenAI Batch API JSONL requests from HiRAG contexts"
    )
    parser.add_argument("--query-file", default=DEFAULT_QUERY_FILE)
    parser.add_argument("--query-limit", type=int, default=30)
    parser.add_argument("--variants", nargs="*", default=DEFAULT_ANSWER_VARIANTS)
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument("--output-dir", default=f".runs/openai_batch/{timestamp}")
    parser.add_argument("--batch-file-name", default="answer_requests.jsonl")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument(
        "--completion-token-param",
        choices=["max_completion_tokens", "max_tokens", "none"],
        default="max_completion_tokens",
        help=(
            "Output-token limit parameter to emit. GPT-5.x Chat Completions "
            "requires max_completion_tokens; older OpenAI-compatible backends "
            "may require max_tokens."
        ),
    )
    parser.add_argument(
        "--include-contexts",
        action="store_true",
        help="Also write resolved context text to contexts.jsonl for inspection",
    )
    add_answer_context_args(parser)
    add_runtime_args(parser)
    return parser.parse_args()


async def _dummy_llm(*_args: Any, **_kwargs: Any) -> str:
    return ""


def existing_custom_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    custom_ids = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                custom_ids.add(str(json.loads(line).get("custom_id")))
    return custom_ids


def request_body(args: argparse.Namespace, messages: list[dict[str, str]]) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": args.model,
        "messages": messages,
    }
    if args.completion_token_param != "none":
        body[args.completion_token_param] = (
            args.answer_max_tokens or DEFAULT_ANSWER_MAX_TOKENS
        )
    if args.temperature is not None:
        body["temperature"] = args.temperature
    return body


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    batch_path = output_dir / args.batch_file_name
    contexts_path = output_dir / "contexts.jsonl"
    manifest_path = output_dir / "manifest.json"
    if batch_path.exists() and not args.overwrite:
        done = existing_custom_ids(batch_path)
        mode = "a"
    else:
        done = set()
        mode = "w"

    graph = build_fastembed_hirag(args, llm_func=_dummy_llm, enable_llm_cache=False)
    queries = load_queries(Path(args.query_file), args.query_limit)
    rows: list[dict[str, Any]] = []
    total_input_tokens = 0
    total_context_tokens = 0
    written = 0
    skipped = 0

    context_mode = "a" if mode == "a" and contexts_path.exists() else "w"
    with batch_path.open(mode, encoding="utf-8") as batch_handle, contexts_path.open(
        context_mode, encoding="utf-8"
    ) as context_handle:
        for query_item in queries:
            for variant in args.variants:
                custom_id = batch_custom_id(query_item["query_id"], variant)
                if custom_id in done:
                    skipped += 1
                    continue
                start = time.perf_counter()
                error = None
                context = ""
                param = None
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
                    input_tokens = estimate_chat_tokens(messages)
                    context_tokens = len(encode_string_by_tiktoken(context))
                    request = {
                        "custom_id": custom_id,
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": request_body(args, messages),
                    }
                    batch_handle.write(json.dumps(request, ensure_ascii=False) + "\n")
                    total_input_tokens += input_tokens
                    total_context_tokens += context_tokens
                    written += 1
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    input_tokens = 0
                    context_tokens = 0
                row = {
                    "custom_id": custom_id,
                    "query_id": query_item["query_id"],
                    "query": query_item["query"],
                    "variant": variant,
                    "model": args.model,
                    "input_tokens": input_tokens,
                    "context_tokens": context_tokens,
                    "latency_seconds": time.perf_counter() - start,
                    "context_debug": {
                        **(compact_debug(dict(param.debug_info)) if param else {}),
                        **budget_debug,
                    },
                    "error": error,
                }
                rows.append(row)
                if args.include_contexts:
                    context_handle.write(
                        json.dumps(
                            {
                                **row,
                                "context": context,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                batch_handle.flush()
                context_handle.flush()

    # Batch API offers a 50% discount relative to synchronous API pricing.
    price_table = {
        "gpt-5.4": {"input_per_m": 1.25 * 0.5, "output_per_m": 7.50 * 0.5},
        "gpt-5.4-mini": {"input_per_m": 0.375 * 0.5, "output_per_m": 2.25 * 0.5},
    }
    expected_output_tokens = written * (args.answer_max_tokens or DEFAULT_ANSWER_MAX_TOKENS)
    price = price_table.get(args.model)
    estimated_cost = None
    if price:
        estimated_cost = (
            total_input_tokens / 1_000_000 * price["input_per_m"]
            + expected_output_tokens / 1_000_000 * price["output_per_m"]
        )
    manifest = {
        "batch_file": str(batch_path),
        "contexts_file": str(contexts_path) if args.include_contexts else None,
        "model": args.model,
        "request_count": written,
        "skipped_existing": skipped,
        "query_count": len(queries),
        "variants": args.variants,
        "total_input_tokens_estimate": total_input_tokens,
        "mean_input_tokens_estimate": total_input_tokens / written if written else 0,
        "total_context_tokens_estimate": total_context_tokens,
        "max_output_tokens_per_request": args.answer_max_tokens,
        "completion_token_param": args.completion_token_param,
        "expected_output_tokens_upper_bound": expected_output_tokens,
        "estimated_batch_cost_usd": estimated_cost,
        "errors": [row for row in rows if row["error"]],
        "upload_hint": (
            "Upload with purpose='batch', then create a batch with "
            "endpoint='/v1/chat/completions' and completion_window='24h'."
        ),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {written} batch requests to {batch_path}")
    print(f"Wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
