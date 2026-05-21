#!/usr/bin/env python3
"""Import OpenAI Batch community reports into HiRAG community report storage JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.import_openai_index_entity_batch import extract_response_text, load_metadata
from hirag._op import _community_report_json_to_str
from hirag._utils import convert_response_to_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert OpenAI Batch community report outputs into JSON storage.")
    parser.add_argument("--batch-output", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--merge-existing", default=None)
    return parser.parse_args()


def convert_rows(batch_output: Path, metadata: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with batch_output.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            raw = json.loads(line)
            custom_id = str(raw.get("custom_id"))
            meta = metadata.get(custom_id, {})
            response_text, error, body = extract_response_text(raw)
            parsed = {} if error else convert_response_to_json(response_text)
            parse_error = None if parsed else "unable to parse community report JSON"
            community = meta.get("community") or {}
            report_string = _community_report_json_to_str(parsed) if parsed else ""
            rows.append(
                {
                    "custom_id": custom_id,
                    "community_id": str(meta.get("community_id") or custom_id.replace("community_report|", "")),
                    "level": meta.get("level"),
                    "report": {
                        "report_string": report_string,
                        "report_json": parsed,
                        **community,
                    },
                    "raw_response": response_text,
                    "usage": (body or {}).get("usage"),
                    "error": error,
                    "parse_error": parse_error if not error else None,
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = convert_rows(Path(args.batch_output), load_metadata(Path(args.metadata)))
    reports: dict[str, Any] = {}
    if args.merge_existing:
        reports.update(json.loads(Path(args.merge_existing).read_text(encoding="utf-8")))
    for row in rows:
        if not row.get("error") and not row.get("parse_error"):
            reports[row["community_id"]] = row["report"]
    with (output_dir / "community_report_results.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (output_dir / "kv_store_community_reports.json").write_text(json.dumps(reports, indent=2, ensure_ascii=False), encoding="utf-8")
    summary = {
        "rows": len(rows),
        "ok": sum(1 for row in rows if not row.get("error") and not row.get("parse_error")),
        "errors": sum(1 for row in rows if row.get("error")),
        "parse_errors": sum(1 for row in rows if row.get("parse_error")),
        "merged_report_count": len(reports),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "summary.md").write_text(
        "\n".join(
            [
                "# Community Report Batch Import Summary",
                "",
                f"- rows: {summary['rows']}",
                f"- ok/errors/parse_errors: {summary['ok']}/{summary['errors']}/{summary['parse_errors']}",
                f"- merged reports: {summary['merged_report_count']}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {summary['ok']} parsed community reports to {output_dir}")


if __name__ == "__main__":
    main()
