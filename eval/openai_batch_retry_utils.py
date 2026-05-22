#!/usr/bin/env python3
"""Shared helpers for OpenAI Batch retry/export/import tooling."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def batch_finish_reason(batch_row: dict[str, Any]) -> str | None:
    response = batch_row.get("response") or {}
    body = response.get("body") or {}
    try:
        return body["choices"][0].get("finish_reason")
    except (KeyError, IndexError, TypeError):
        return None


def batch_usage(batch_row: dict[str, Any]) -> dict[str, int]:
    response = batch_row.get("response") or {}
    body = response.get("body") or {}
    usage = body.get("usage") or {}
    return {
        "prompt_tokens": int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0),
        "completion_tokens": int(usage.get("completion_tokens") or usage.get("output_tokens") or 0),
        "total_tokens": int(usage.get("total_tokens") or 0),
    }


def is_retryable_batch_row(batch_row: dict[str, Any]) -> bool:
    if batch_row.get("error"):
        return True
    response = batch_row.get("response") or {}
    status_code = response.get("status_code")
    if status_code and int(status_code) >= 400:
        return True
    return batch_finish_reason(batch_row) == "length"


def body_with_completion_cap(row: dict[str, Any], cap: int, token_param: str) -> dict[str, Any]:
    updated = json.loads(json.dumps(row))
    body = updated.setdefault("body", {})
    body.pop("max_tokens", None)
    body.pop("max_completion_tokens", None)
    if token_param != "none":
        body[token_param] = cap
    return updated
