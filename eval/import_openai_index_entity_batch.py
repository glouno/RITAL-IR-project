#!/usr/bin/env python3
"""Import OpenAI Batch entity-extraction outputs into parsed HiRAG chunk entities."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hirag._utils import clean_str, split_string_by_multi_markers
from hirag.regimes import resolve_prompts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert OpenAI Batch entity-extraction results into parsed JSONL."
    )
    parser.add_argument("--batch-output", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prompt-regime", choices=["baseline", "lean", "ultra_lean"], default="ultra_lean")
    return parser.parse_args()


def load_metadata(path: Path) -> dict[str, dict[str, Any]]:
    data: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                data[str(row["custom_id"])] = row
    return data


def extract_response_text(row: dict[str, Any]) -> tuple[str, str | None, dict[str, Any] | None]:
    if row.get("error"):
        return "", str(row["error"]), row.get("response", {}).get("body")
    response = row.get("response") or {}
    status_code = response.get("status_code")
    body = response.get("body") or {}
    if status_code and int(status_code) >= 400:
        error = body.get("error", body)
        return "", json.dumps(error, ensure_ascii=False), body
    try:
        return str(body["choices"][0]["message"].get("content") or ""), None, body
    except Exception as exc:
        return "", f"{type(exc).__name__}: unable to extract response text", body


def parse_entities(text: str, prompts: dict[str, Any], chunk_id: str) -> list[dict[str, Any]]:
    records = split_string_by_multi_markers(
        text,
        [
            prompts["DEFAULT_RECORD_DELIMITER"],
            prompts["DEFAULT_COMPLETION_DELIMITER"],
        ],
    )
    entities: list[dict[str, Any]] = []
    for record in records:
        match = re.search(r"\((.*)\)", record)
        if match is None:
            continue
        attrs = split_string_by_multi_markers(
            match.group(1),
            [prompts["DEFAULT_TUPLE_DELIMITER"]],
        )
        if len(attrs) < 4 or attrs[0] != '"entity"':
            continue
        entity_name = clean_str(attrs[1].upper())
        if not entity_name.strip():
            continue
        entities.append(
            {
                "entity_name": entity_name,
                "entity_type": clean_str(attrs[2].upper()),
                "description": clean_str(attrs[3]),
                "source_id": chunk_id,
            }
        )
    return entities


def convert_rows(
    batch_output: Path,
    metadata: dict[str, dict[str, Any]],
    prompts: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with batch_output.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            raw = json.loads(line)
            custom_id = str(raw.get("custom_id"))
            meta = metadata.get(custom_id, {})
            chunk_id = str(meta.get("chunk_id") or custom_id.replace("index_entity|", ""))
            response_text, error, body = extract_response_text(raw)
            entities = [] if error else parse_entities(response_text, prompts, chunk_id)
            rows.append(
                {
                    "custom_id": custom_id,
                    "chunk_id": chunk_id,
                    "full_doc_id": meta.get("full_doc_id"),
                    "chunk_order_index": meta.get("chunk_order_index"),
                    "entity_count": len(entities),
                    "entities": entities,
                    "raw_response": response_text,
                    "usage": (body or {}).get("usage"),
                    "error": error,
                }
            )
    return rows


def write_summary(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    ok_rows = [row for row in rows if not row.get("error")]
    type_counts = Counter(
        entity.get("entity_type", "UNKNOWN")
        for row in ok_rows
        for entity in row.get("entities", [])
    )
    summary = {
        "rows": len(rows),
        "ok": len(ok_rows),
        "errors": len(rows) - len(ok_rows),
        "total_entities": sum(int(row.get("entity_count", 0)) for row in ok_rows),
        "mean_entities_per_chunk": (
            sum(int(row.get("entity_count", 0)) for row in ok_rows) / len(ok_rows)
            if ok_rows
            else 0
        ),
        "top_entity_types": type_counts.most_common(20),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    md = [
        "# Entity Batch Import Summary",
        "",
        f"- rows: {summary['rows']}",
        f"- ok/errors: {summary['ok']}/{summary['errors']}",
        f"- total entities: {summary['total_entities']}",
        f"- mean entities per chunk: {summary['mean_entities_per_chunk']:.2f}",
        "",
        "## Top Entity Types",
        "",
    ]
    for entity_type, count in summary["top_entity_types"]:
        md.append(f"- `{entity_type}`: {count}")
    (output_dir / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prompts = resolve_prompts(args.prompt_regime)
    rows = convert_rows(
        Path(args.batch_output),
        load_metadata(Path(args.metadata)),
        prompts,
    )
    with (output_dir / "entity_extract_results.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_summary(output_dir, rows)
    print(f"Wrote {len(rows)} parsed rows to {output_dir / 'entity_extract_results.jsonl'}")


if __name__ == "__main__":
    main()
