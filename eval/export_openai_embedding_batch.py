#!/usr/bin/env python3
"""Export entity/chunk embeddings as OpenAI Batch JSONL requests."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import networkx as nx

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.materialize_openai_index_hirag_workdir import build_docs_and_chunks, load_contexts
from hirag._utils import compute_mdhash_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create OpenAI Batch requests for HiRAG vector-store embeddings.")
    parser.add_argument("--context-file", required=True)
    parser.add_argument("--graphml", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="text-embedding-3-small")
    parser.add_argument("--chunk-token-size", type=int, default=1200)
    parser.add_argument("--chunk-overlap-token-size", type=int, default=100)
    parser.add_argument("--namespaces", nargs="+", choices=["entities", "chunks"], default=["entities", "chunks"])
    parser.add_argument("--batch-file-name", default="embedding_requests.jsonl")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def request_body(model: str, text: str) -> dict[str, Any]:
    return {"model": model, "input": text, "encoding_format": "float"}


def entity_items(graph: nx.Graph) -> list[dict[str, Any]]:
    return [
        {
            "namespace": "entities",
            "id": compute_mdhash_id(str(node_id), prefix="ent-"),
            "content": str(node_id) + str(data.get("description", "")),
            "meta": {"entity_name": str(node_id)},
        }
        for node_id, data in sorted(graph.nodes(data=True), key=lambda item: str(item[0]))
    ]


def chunk_items(chunks: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "namespace": "chunks",
            "id": chunk_id,
            "content": str(chunk.get("content", "")),
            "meta": {},
        }
        for chunk_id, chunk in sorted(chunks.items())
    ]


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    batch_path = output_dir / args.batch_file_name
    metadata_path = output_dir / "embedding_metadata.jsonl"
    if batch_path.exists() and not args.overwrite:
        raise FileExistsError(f"{batch_path} exists; pass --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)
    graph = nx.read_graphml(args.graphml)
    _docs, chunks = build_docs_and_chunks(
        load_contexts(Path(args.context_file)),
        chunk_token_size=args.chunk_token_size,
        chunk_overlap_token_size=args.chunk_overlap_token_size,
    )
    items: list[dict[str, Any]] = []
    if "entities" in args.namespaces:
        items.extend(entity_items(graph))
    if "chunks" in args.namespaces:
        items.extend(chunk_items(chunks))
    with batch_path.open("w", encoding="utf-8") as batch_handle, metadata_path.open("w", encoding="utf-8") as meta_handle:
        for item in items:
            custom_id = f"embedding|{item['namespace']}|{item['id']}"
            batch_handle.write(
                json.dumps(
                    {
                        "custom_id": custom_id,
                        "method": "POST",
                        "url": "/v1/embeddings",
                        "body": request_body(args.model, item["content"]),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            meta_handle.write(json.dumps({"custom_id": custom_id, **item}, ensure_ascii=False) + "\n")
    manifest = {
        "context_file": args.context_file,
        "graphml": args.graphml,
        "batch_file": str(batch_path),
        "metadata_file": str(metadata_path),
        "model": args.model,
        "namespaces": args.namespaces,
        "request_count": len(items),
        "entity_count": graph.number_of_nodes() if "entities" in args.namespaces else 0,
        "chunk_count": len(chunks) if "chunks" in args.namespaces else 0,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {len(items)} embedding requests to {batch_path}")


if __name__ == "__main__":
    main()
