#!/usr/bin/env python3
"""Import OpenAI Batch edge embeddings into HiRAG bridge edge cache JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.import_openai_embedding_batch import extract_embedding
from eval.import_openai_index_entity_batch import load_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert OpenAI Batch edge embedding outputs into a HiRAG edge cache."
    )
    parser.add_argument("--batch-output", nargs="+", required=True)
    parser.add_argument("--metadata", nargs="+", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--cache-file", default=None)
    parser.add_argument("--summary-dir", default=None)
    parser.add_argument("--merge-existing", action="store_true")
    return parser.parse_args()


def load_all_metadata(paths: list[Path]) -> dict[str, dict[str, Any]]:
    metadata: dict[str, dict[str, Any]] = {}
    for path in paths:
        loaded = load_metadata(path)
        overlap = set(metadata) & set(loaded)
        if overlap:
            raise ValueError(f"Duplicate metadata custom_id values found in {path}")
        metadata.update(loaded)
    return metadata


def convert_rows(
    batch_outputs: list[Path],
    metadata: dict[str, dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    edge_embeddings: dict[str, dict[str, Any]] = {}
    errors: list[dict[str, Any]] = []
    for batch_output in batch_outputs:
        with batch_output.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                raw = json.loads(line)
                custom_id = str(raw.get("custom_id"))
                meta = metadata.get(custom_id, {})
                embedding, error, body = extract_embedding(raw)
                if error:
                    errors.append(
                        {
                            "custom_id": custom_id,
                            "cache_key": meta.get("cache_key"),
                            "error": error,
                            "usage": (body or {}).get("usage"),
                        }
                    )
                    continue
                edge_embeddings[str(meta["cache_key"])] = {
                    "text_hash": meta["text_hash"],
                    "embedded_text_hash": meta["embedded_text_hash"],
                    "embedding": embedding,
                }
    return edge_embeddings, errors


def load_existing_cache(cache_file: Path) -> dict[str, Any]:
    if not cache_file.exists():
        return {"edge_embeddings": {}}
    return json.loads(cache_file.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    cache_file = Path(args.cache_file or manifest["cache_file"])
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    metadata = load_all_metadata([Path(path) for path in args.metadata])
    edge_embeddings, errors = convert_rows(
        [Path(path) for path in args.batch_output],
        metadata,
    )
    cache = load_existing_cache(cache_file) if args.merge_existing else {"edge_embeddings": {}}
    cache.setdefault("edge_embeddings", {}).update(edge_embeddings)
    cache["graph_edges"] = manifest.get("edge_count")
    cache["model"] = manifest.get("model")
    cache["embedding_dim"] = manifest.get("embed_dim")
    cache["max_item_tokens"] = manifest.get("max_edge_tokens")
    cache["source"] = "openai_batch_edge_embedding"
    cache_file.write_text(json.dumps(cache), encoding="utf-8")

    summary = {
        "cache_file": str(cache_file),
        "metadata_rows": len(metadata),
        "imported_embeddings": len(edge_embeddings),
        "cache_embeddings": len(cache.get("edge_embeddings", {})),
        "errors": len(errors),
        "error_rows": errors,
    }
    summary_dir = Path(args.summary_dir) if args.summary_dir else cache_file.parent
    summary_dir.mkdir(parents=True, exist_ok=True)
    (summary_dir / "edge_embedding_import_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Wrote {len(edge_embeddings)} edge embeddings to {cache_file}")
    if errors:
        print(f"Warning: {len(errors)} rows had errors; see edge_embedding_import_summary.json")


if __name__ == "__main__":
    main()
