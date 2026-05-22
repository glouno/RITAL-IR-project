#!/usr/bin/env python3
"""Create retry JSONL requests from failed/truncated OpenAI Batch rows."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.openai_batch_retry_utils import (
    body_with_completion_cap,
    is_retryable_batch_row,
    load_jsonl,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export retry requests for retryable OpenAI Batch rows.")
    parser.add_argument("--source-requests", required=True)
    parser.add_argument("--batch-output", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-file-name", default="retry_requests.jsonl")
    parser.add_argument("--max-completion-tokens", type=int, required=True)
    parser.add_argument(
        "--completion-token-param",
        choices=["max_completion_tokens", "max_tokens", "none"],
        default="max_completion_tokens",
    )
    parser.add_argument("--reason", default="Retry rows with API error, HTTP error, or finish_reason=length.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_by_id = {str(row["custom_id"]): row for row in load_jsonl(Path(args.source_requests))}
    retry_ids = [
        str(row.get("custom_id"))
        for row in load_jsonl(Path(args.batch_output))
        if is_retryable_batch_row(row)
    ]
    retry_rows = [
        body_with_completion_cap(source_by_id[custom_id], args.max_completion_tokens, args.completion_token_param)
        for custom_id in retry_ids
        if custom_id in source_by_id
    ]
    output_dir = Path(args.output_dir)
    batch_path = output_dir / args.batch_file_name
    write_jsonl(batch_path, retry_rows)
    manifest = {
        "source_requests": args.source_requests,
        "batch_output": args.batch_output,
        "batch_file": str(batch_path),
        "request_count": len(retry_rows),
        "max_completion_tokens_per_request": args.max_completion_tokens,
        "completion_token_param": args.completion_token_param,
        "reason": args.reason,
        "missing_source_request_count": len(retry_ids) - len(retry_rows),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {len(retry_rows)} retry requests to {batch_path}")


if __name__ == "__main__":
    main()
