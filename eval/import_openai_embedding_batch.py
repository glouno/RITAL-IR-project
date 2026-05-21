#!/usr/bin/env python3
"""Import OpenAI Batch embedding outputs into precomputed embedding JSONL files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.import_openai_index_entity_batch import load_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert OpenAI Batch embedding output into namespace JSONL files.")
    parser.add_argument("--batch-output", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def extract_embedding(row: dict[str, Any]) -> tuple[list[float] | None, str | None, dict[str, Any] | None]:
    if row.get("error"):
        return None, json.dumps(row["error"], ensure_ascii=False), None
    response = row.get("response") or {}
    body = response.get("body") or {}
    status_code = response.get("status_code")
    if status_code and int(status_code) >= 400:
        return None, json.dumps(body.get("error", body), ensure_ascii=False), body
    try:
        return list(body["data"][0]["embedding"]), None, body
    except (KeyError, IndexError, TypeError) as exc:
        return None, f"{type(exc).__name__}: unable to extract embedding", body


def convert_rows(batch_output: Path, metadata: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with batch_output.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            raw = json.loads(line)
            custom_id = str(raw.get("custom_id"))
            meta = metadata.get(custom_id, {})
            embedding, error, body = extract_embedding(raw)
            rows.append(
                {
                    "custom_id": custom_id,
                    "namespace": meta.get("namespace"),
                    "id": meta.get("id"),
                    "meta": meta.get("meta", {}),
                    "embedding": embedding,
                    "usage": (body or {}).get("usage"),
                    "error": error,
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = convert_rows(Path(args.batch_output), load_metadata(Path(args.metadata)))
    by_namespace: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("namespace"):
            by_namespace.setdefault(str(row["namespace"]), []).append(row)
    with (output_dir / "embedding_results.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    for namespace, namespace_rows in by_namespace.items():
        with (output_dir / f"{namespace}_embeddings.jsonl").open("w", encoding="utf-8") as handle:
            for row in namespace_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "rows": len(rows),
        "ok": sum(1 for row in rows if not row.get("error")),
        "errors": sum(1 for row in rows if row.get("error")),
        "by_namespace": {
            namespace: {
                "rows": len(namespace_rows),
                "ok": sum(1 for row in namespace_rows if not row.get("error")),
            }
            for namespace, namespace_rows in sorted(by_namespace.items())
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote imported embeddings to {output_dir}")


if __name__ == "__main__":
    main()
