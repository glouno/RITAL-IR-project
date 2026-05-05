#!/usr/bin/env python3
"""Convert OpenAI Batch answer outputs into the repo's answers.jsonl schema."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description="Import OpenAI Batch answer results")
    parser.add_argument("--batch-output", required=True)
    parser.add_argument("--contexts", required=True)
    parser.add_argument("--output-dir", default=f".runs/answer_eval/openai_batch_{timestamp}")
    parser.add_argument("--answers-file-name", default="answers.jsonl")
    return parser.parse_args()


def parse_answer_custom_id(custom_id: str) -> tuple[str, str]:
    parts = custom_id.split("|")
    if len(parts) >= 3 and parts[0] == "answer":
        return parts[1], "|".join(parts[2:])
    return custom_id, "unknown"


def load_context_metadata(path: Path) -> dict[str, dict[str, Any]]:
    metadata = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            data = json.loads(line)
            metadata[str(data["custom_id"])] = data
    return metadata


def extract_batch_response_text(row: dict[str, Any]) -> tuple[str, str | None, dict[str, Any] | None]:
    if row.get("error"):
        return "", json.dumps(row["error"], ensure_ascii=False), None
    response = row.get("response") or {}
    status_code = response.get("status_code")
    body = response.get("body") or {}
    if status_code and int(status_code) >= 400:
        return "", json.dumps(body, ensure_ascii=False), body
    try:
        content = body["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError) as exc:
        return "", f"{type(exc).__name__}: unable to extract answer content", body
    return content, None, body


def convert_rows(batch_output: Path, context_metadata: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    records = []
    with batch_output.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            custom_id = str(row.get("custom_id", ""))
            query_id, variant = parse_answer_custom_id(custom_id)
            meta = context_metadata.get(custom_id, {})
            answer, error, body = extract_batch_response_text(row)
            usage = body.get("usage", {}) if isinstance(body, dict) else {}
            records.append(
                {
                    "query_id": meta.get("query_id", query_id),
                    "query": meta.get("query", ""),
                    "variant": meta.get("variant", variant),
                    "answer": answer,
                    "latency_seconds": None,
                    "context_debug": meta.get("context_debug", {}),
                    "error": error,
                    "source": "openai_batch",
                    "custom_id": custom_id,
                    "response_id": body.get("id") if isinstance(body, dict) else None,
                    "usage": usage,
                }
            )
    records.sort(key=lambda item: (int(item["query_id"]) if str(item["query_id"]).isdigit() else str(item["query_id"]), item["variant"]))
    return records


def write_summary(output_dir: Path, records: list[dict[str, Any]]) -> None:
    by_variant: dict[str, Counter] = {}
    for record in records:
        counter = by_variant.setdefault(record["variant"], Counter())
        counter["rows"] += 1
        counter["errors" if record.get("error") else "ok"] += 1
    summary = {
        "rows": len(records),
        "ok": sum(1 for record in records if not record.get("error")),
        "errors": sum(1 for record in records if record.get("error")),
        "by_variant": {variant: dict(counter) for variant, counter in sorted(by_variant.items())},
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    md = ["# Imported OpenAI Batch Answers", ""]
    md.append(f"- rows: {summary['rows']}")
    md.append(f"- ok/errors: {summary['ok']}/{summary['errors']}")
    for variant, counter in summary["by_variant"].items():
        md.append(f"- {variant}: {counter.get('ok', 0)} ok, {counter.get('errors', 0)} errors")
    (output_dir / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = load_context_metadata(Path(args.contexts))
    records = convert_rows(Path(args.batch_output), metadata)
    answers_path = output_dir / args.answers_file_name
    with answers_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    write_summary(output_dir, records)
    print(f"Wrote imported answers to {answers_path}")


if __name__ == "__main__":
    main()
