#!/usr/bin/env python3
"""Run a small OpenAI Batch-style JSONL file through the live API.

This is intentionally a secondary path for tiny retries. It preserves the
Batch output JSONL shape so existing import/parsing scripts can be reused.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from openai import OpenAI

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.run_openai_batch_file import as_dict, load_env_file, validate_batch_file


SUPPORTED_ENDPOINTS = {"/v1/chat/completions", "/v1/embeddings"}


def parse_args() -> argparse.Namespace:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(
        description=(
            "Execute a small Batch-style JSONL request file with live OpenAI API calls "
            "and write Batch-compatible JSONL output."
        )
    )
    parser.add_argument("--batch-file", required=True)
    parser.add_argument("--endpoint", default=None, help="Optional endpoint assertion.")
    parser.add_argument("--output-file", default=None)
    parser.add_argument("--output-dir", default=f".runs/openai_direct_submit/{timestamp}")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--request-interval-seconds", type=float, default=0.0)
    parser.add_argument("--max-requests", type=int, default=0, help="0 means no limit.")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def iter_batch_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def endpoint_for_rows(rows: list[dict[str, Any]], endpoint: str | None) -> str:
    urls = {str(row.get("url", "")) for row in rows}
    if endpoint is not None:
        if urls != {endpoint}:
            raise ValueError(f"request urls {sorted(urls)} do not match selected endpoint {endpoint!r}")
        selected = endpoint
    elif len(urls) == 1:
        selected = next(iter(urls))
    else:
        raise ValueError(f"request file must contain exactly one endpoint, got {sorted(urls)}")
    if selected not in SUPPORTED_ENDPOINTS:
        raise ValueError(f"unsupported endpoint {selected!r}; supported: {sorted(SUPPORTED_ENDPOINTS)}")
    return selected


def call_live_api(client: OpenAI, endpoint: str, body: dict[str, Any]) -> Any:
    if endpoint == "/v1/chat/completions":
        return client.chat.completions.create(**body)
    if endpoint == "/v1/embeddings":
        return client.embeddings.create(**body)
    raise ValueError(f"unsupported endpoint {endpoint!r}")


def error_dict(exc: Exception) -> dict[str, Any]:
    return {
        "type": exc.__class__.__name__,
        "message": str(exc),
        "status_code": getattr(exc, "status_code", None),
        "code": getattr(exc, "code", None),
        "body": getattr(exc, "body", None),
    }


def make_success_row(source_row: dict[str, Any], response: Any) -> dict[str, Any]:
    return {
        "custom_id": source_row.get("custom_id"),
        "response": {
            "status_code": 200,
            "body": as_dict(response),
        },
        "error": None,
    }


def make_error_row(source_row: dict[str, Any], exc: Exception) -> dict[str, Any]:
    err = error_dict(exc)
    status_code = err.get("status_code")
    return {
        "custom_id": source_row.get("custom_id"),
        "response": {
            "status_code": status_code,
            "body": {"error": err},
        },
        "error": err,
    }


def run_rows(
    client: OpenAI,
    rows: list[dict[str, Any]],
    endpoint: str,
    *,
    request_interval_seconds: float = 0.0,
    fail_fast: bool = False,
) -> list[dict[str, Any]]:
    outputs: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        try:
            response = call_live_api(client, endpoint, row.get("body") or {})
            outputs.append(make_success_row(row, response))
        except Exception as exc:
            outputs.append(make_error_row(row, exc))
            if fail_fast:
                raise
        if request_interval_seconds > 0 and index < len(rows) - 1:
            time.sleep(request_interval_seconds)
    return outputs


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    load_env_file(Path(args.env_file))

    batch_file = Path(args.batch_file)
    file_stats = validate_batch_file(batch_file, endpoint=args.endpoint)
    rows = iter_batch_rows(batch_file)
    endpoint = endpoint_for_rows(rows, args.endpoint)
    if args.max_requests:
        rows = rows[: args.max_requests]

    if args.validate_only:
        print(json.dumps({**file_stats, "selected_endpoint": endpoint}, indent=2))
        return

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("Set OPENAI_API_KEY in the environment or .env; do not put API keys in command arguments.")

    output_file = Path(args.output_file) if args.output_file else Path(args.output_dir) / "direct_output.jsonl"
    if output_file.exists() and not args.overwrite:
        raise FileExistsError(f"{output_file} exists; pass --overwrite to replace it")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    client = OpenAI()
    outputs = run_rows(
        client,
        rows,
        endpoint,
        request_interval_seconds=args.request_interval_seconds,
        fail_fast=args.fail_fast,
    )
    write_jsonl(output_file, outputs)
    summary = {
        "batch_file": str(batch_file),
        "endpoint": endpoint,
        "output_file": str(output_file),
        "rows": len(outputs),
        "ok": sum(1 for row in outputs if not row.get("error")),
        "errors": sum(1 for row in outputs if row.get("error")),
        "file_stats": file_stats,
        "mode": "direct_api_secondary_path",
    }
    (output_file.parent / "direct_run_manifest.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
