#!/usr/bin/env python3
"""Import OpenAI Batch relation-extraction outputs into parsed HiRAG chunk relations."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.import_openai_index_entity_batch import extract_response_text, load_metadata
from hirag._utils import clean_str, is_float_regex, split_string_by_multi_markers
from hirag.regimes import resolve_prompts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert OpenAI Batch relation-extraction results into parsed JSONL."
    )
    parser.add_argument("--batch-output", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prompt-regime", choices=["baseline", "lean", "ultra_lean"], default="ultra_lean")
    return parser.parse_args()


def parse_relations(text: str, prompts: dict[str, Any], chunk_id: str) -> list[dict[str, Any]]:
    records = split_string_by_multi_markers(
        text,
        [
            prompts["DEFAULT_RECORD_DELIMITER"],
            prompts["DEFAULT_COMPLETION_DELIMITER"],
        ],
    )
    relations: list[dict[str, Any]] = []
    for record in records:
        match = re.search(r"\((.*)\)", record)
        if match is None:
            continue
        attrs = split_string_by_multi_markers(
            match.group(1),
            [prompts["DEFAULT_TUPLE_DELIMITER"]],
        )
        if len(attrs) < 5 or attrs[0] != '"relationship"':
            continue
        source = clean_str(attrs[1].upper())
        target = clean_str(attrs[2].upper())
        if not source.strip() or not target.strip():
            continue
        relations.append(
            {
                "src_id": source,
                "tgt_id": target,
                "description": clean_str(attrs[3]),
                "weight": float(attrs[-1]) if is_float_regex(attrs[-1]) else 1.0,
                "source_id": chunk_id,
            }
        )
    return relations


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
            chunk_id = str(meta.get("chunk_id") or custom_id.replace("index_relation|", ""))
            response_text, error, body = extract_response_text(raw)
            relations = [] if error else parse_relations(response_text, prompts, chunk_id)
            rows.append(
                {
                    "custom_id": custom_id,
                    "chunk_id": chunk_id,
                    "full_doc_id": meta.get("full_doc_id"),
                    "chunk_order_index": meta.get("chunk_order_index"),
                    "input_entity_count": meta.get("entity_count"),
                    "relation_count": len(relations),
                    "relations": relations,
                    "raw_response": response_text,
                    "usage": (body or {}).get("usage"),
                    "error": error,
                }
            )
    return rows


def write_summary(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    ok_rows = [row for row in rows if not row.get("error")]
    total_relations = sum(int(row.get("relation_count", 0)) for row in ok_rows)
    summary = {
        "rows": len(rows),
        "ok": len(ok_rows),
        "errors": len(rows) - len(ok_rows),
        "total_relations": total_relations,
        "mean_relations_per_chunk": total_relations / len(ok_rows) if ok_rows else 0,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    md = [
        "# Relation Batch Import Summary",
        "",
        f"- rows: {summary['rows']}",
        f"- ok/errors: {summary['ok']}/{summary['errors']}",
        f"- total relations: {summary['total_relations']}",
        f"- mean relations per chunk: {summary['mean_relations_per_chunk']:.2f}",
    ]
    (output_dir / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = convert_rows(
        Path(args.batch_output),
        load_metadata(Path(args.metadata)),
        resolve_prompts(args.prompt_regime),
    )
    with (output_dir / "relation_extract_results.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_summary(output_dir, rows)
    print(f"Wrote {len(rows)} parsed rows to {output_dir / 'relation_extract_results.jsonl'}")


if __name__ == "__main__":
    main()
