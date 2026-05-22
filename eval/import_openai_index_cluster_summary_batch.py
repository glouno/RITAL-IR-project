#!/usr/bin/env python3
"""Import OpenAI Batch cluster-summary outputs into parsed entity/relation rows."""

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

from eval.import_openai_index_entity_batch import extract_response_text, load_metadata, parse_entities
from eval.import_openai_index_relation_batch import parse_relations
from hirag.regimes import resolve_prompts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert OpenAI Batch cluster summaries into parsed JSONL.")
    parser.add_argument("--batch-output", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prompt-regime", choices=["baseline", "lean", "ultra_lean"], default="baseline")
    return parser.parse_args()


def convert_rows(batch_output: Path, metadata: dict[str, dict[str, Any]], prompts: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with batch_output.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            raw = json.loads(line)
            custom_id = str(raw.get("custom_id"))
            meta = metadata.get(custom_id, {})
            response_text, error, body = extract_response_text(raw)
            source_id = f"cluster-{meta.get('level')}-{meta.get('cluster_id')}"
            entities = [] if error else parse_entities(response_text, prompts, source_id)
            relations = [] if error else parse_relations(response_text, prompts, source_id)
            rows.append(
                {
                    "custom_id": custom_id,
                    "level": meta.get("level"),
                    "cluster_id": meta.get("cluster_id"),
                    "source_nodes": meta.get("nodes", []),
                    "entity_count": len(entities),
                    "relation_count": len(relations),
                    "entities": entities,
                    "relations": relations,
                    "raw_response": response_text,
                    "usage": (body or {}).get("usage"),
                    "error": error,
                    "parse_error": None if error or entities else "no summary entity parsed",
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = convert_rows(Path(args.batch_output), load_metadata(Path(args.metadata)), resolve_prompts(args.prompt_regime))
    with (output_dir / "cluster_summary_results.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "rows": len(rows),
        "ok": sum(1 for row in rows if not row.get("error") and not row.get("parse_error")),
        "errors": sum(1 for row in rows if row.get("error")),
        "parse_errors": sum(1 for row in rows if row.get("parse_error")),
        "total_entities": sum(row.get("entity_count", 0) for row in rows if not row.get("error")),
        "total_relations": sum(row.get("relation_count", 0) for row in rows if not row.get("error")),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "summary.md").write_text(
        "\n".join(
            [
                "# Cluster Summary Batch Import Summary",
                "",
                f"- rows: {summary['rows']}",
                f"- ok/errors/parse_errors: {summary['ok']}/{summary['errors']}/{summary['parse_errors']}",
                f"- total summary entities: {summary['total_entities']}",
                f"- total summary relations: {summary['total_relations']}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(rows)} cluster summary rows to {output_dir}")


if __name__ == "__main__":
    main()
