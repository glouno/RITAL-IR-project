#!/usr/bin/env python3
"""Export pairwise answer-judge requests as OpenAI Batch API JSONL."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.pairwise_answer_judge import build_prompt, load_answers


def parse_args() -> argparse.Namespace:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description="Create OpenAI Batch JSONL for pairwise judging")
    parser.add_argument("--answers", required=True)
    parser.add_argument("--baseline", default="hi")
    parser.add_argument("--variants", nargs="*", default=["hi_minmax_budgeted", "hi_rerank_weighted"])
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument("--output-dir", default=f".runs/answer_eval/judge_batch_{timestamp}")
    parser.add_argument("--batch-file-name", default="judge_requests.jsonl")
    parser.add_argument("--max-tokens", type=int, default=1024)
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
    return parser.parse_args()


def judge_custom_id(query_id: str, baseline: str, variant: str, answer_order: str) -> str:
    safe_query_id = str(query_id).replace("|", "_").replace(" ", "_")
    return f"judge|{safe_query_id}|{baseline}|{variant}|{answer_order}"


def request_body(args: argparse.Namespace, query: str, answer1: str, answer2: str) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": "You are a careful answer evaluator. Return only JSON."},
            {"role": "user", "content": build_prompt(query, answer1, answer2)},
        ],
    }
    if args.completion_token_param != "none":
        body[args.completion_token_param] = args.max_tokens
    if args.temperature is not None:
        body["temperature"] = args.temperature
    return body


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    answers = load_answers(Path(args.answers))
    query_ids = sorted({query_id for query_id, _variant in answers}, key=lambda x: int(x) if x.isdigit() else x)
    batch_path = output_dir / args.batch_file_name
    metadata_path = output_dir / "judge_request_metadata.jsonl"
    manifest_path = output_dir / "manifest.json"
    written = 0
    skipped = 0
    with batch_path.open("w", encoding="utf-8") as batch_handle, metadata_path.open("w", encoding="utf-8") as meta_handle:
        for query_id in query_ids:
            baseline_record = answers.get((query_id, args.baseline))
            if not baseline_record:
                continue
            for variant in args.variants:
                variant_record = answers.get((query_id, variant))
                if not variant_record:
                    skipped += 2
                    continue
                orders = [
                    ("baseline_first", [args.baseline, variant]),
                    ("variant_first", [variant, args.baseline]),
                ]
                by_variant = {args.baseline: baseline_record, variant: variant_record}
                for order_name, order in orders:
                    custom_id = judge_custom_id(query_id, args.baseline, variant, order_name)
                    request = {
                        "custom_id": custom_id,
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": request_body(
                            args,
                            baseline_record["query"],
                            by_variant[order[0]]["answer"],
                            by_variant[order[1]]["answer"],
                        ),
                    }
                    metadata = {
                        "custom_id": custom_id,
                        "query_id": query_id,
                        "query": baseline_record["query"],
                        "variant_a": args.baseline,
                        "variant_b": variant,
                        "answer_order": order_name,
                        "order": order,
                    }
                    batch_handle.write(json.dumps(request, ensure_ascii=False) + "\n")
                    meta_handle.write(json.dumps(metadata, ensure_ascii=False) + "\n")
                    written += 1
    manifest = {
        "batch_file": str(batch_path),
        "metadata_file": str(metadata_path),
        "answers_file": args.answers,
        "model": args.model,
        "request_count": written,
        "skipped_missing_answer_requests": skipped,
        "completion_token_param": args.completion_token_param,
        "baseline": args.baseline,
        "variants": args.variants,
        "upload_hint": (
            "Upload with purpose='batch', then create a batch with "
            "endpoint='/v1/chat/completions' and completion_window='24h'."
        ),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {written} judge batch requests to {batch_path}")


if __name__ == "__main__":
    main()
