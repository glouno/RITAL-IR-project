#!/usr/bin/env python3
"""Export HiRAG bridge edge embeddings as OpenAI Batch requests."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import networkx as nx

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hirag._op import (
    _EDGE_EMBEDDING_MAX_BATCH_TOKENS,
    _EDGE_EMBEDDING_MAX_INPUT_TOKENS,
    _edge_embedding_cache_file,
    _edge_description_text,
    _edge_key,
    _hash_text,
    _truncate_edge_embedding_text,
)
from hirag._llm import openai_embedding


DEFAULT_MAX_FILE_BYTES = 180 * 1024 * 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create OpenAI Batch embedding requests for query-weighted bridge edge caches."
    )
    parser.add_argument("--working-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="text-embedding-3-small")
    parser.add_argument("--embed-dim", type=int, default=1536)
    parser.add_argument(
        "--runtime-cache-model",
        default=None,
        help=(
            "Optional model identity for the HiRAG runtime cache filename. "
            "Leave unset for OpenAI eval runtime, which keys edge caches by "
            "the decorated embedding function name."
        ),
    )
    parser.add_argument("--cache-root", default=None)
    parser.add_argument("--graphml", default=None)
    parser.add_argument("--max-edge-tokens", type=int, default=_EDGE_EMBEDDING_MAX_INPUT_TOKENS)
    parser.add_argument("--max-requests-per-file", type=int, default=45_000)
    parser.add_argument("--max-file-bytes", type=int, default=DEFAULT_MAX_FILE_BYTES)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def graphml_path(args: argparse.Namespace) -> Path:
    if args.graphml:
        return Path(args.graphml)
    return Path(args.working_dir) / "graph_chunk_entity_relation.graphml"


def request_body(model: str, text: str) -> dict[str, Any]:
    return {"model": model, "input": text, "encoding_format": "float"}


def edge_cache_path(
    graph: nx.Graph,
    graphml: Path,
    *,
    cache_root: Path,
    model: str,
    embed_dim: int,
    runtime_cache_model: str | None = None,
) -> Path:
    # HiRAG wraps embedding_func with limit_async_func_call at runtime, so the
    # cache key sees the decorated callable name, not the underlying provider.
    embedding_func = SimpleNamespace(__name__="wait_func", embedding_dim=embed_dim)
    storage = SimpleNamespace(_graphml_xml_file=str(graphml))
    global_config: dict[str, Any] = {
        "edge_embedding_cache_path": str(cache_root),
        "embedding_func": embedding_func,
    }
    if runtime_cache_model:
        global_config["embed_model"] = runtime_cache_model
    return _edge_embedding_cache_file(
        graph,
        storage,
        global_config,
    )


def edge_items(
    graph: nx.Graph,
    *,
    model: str,
    max_edge_tokens: int,
) -> list[dict[str, Any]]:
    items = []
    for index, (source, target, data) in enumerate(
        sorted(graph.edges(data=True), key=lambda item: _edge_key(item[0], item[1]))
    ):
        edge_key = _edge_key(str(source), str(target))
        edge_text = _edge_description_text(graph, source, target, data)
        bounded_text = _truncate_edge_embedding_text(edge_text, max_edge_tokens)
        stable_id = _hash_text(f"{edge_key[0]}|{edge_key[1]}|{_hash_text(edge_text)}")[:24]
        items.append(
            {
                "custom_id": f"edge_embedding|{index:08d}|{stable_id}",
                "source": edge_key[0],
                "target": edge_key[1],
                "cache_key": "|".join(edge_key),
                "text_hash": _hash_text(edge_text),
                "embedded_text_hash": _hash_text(bounded_text),
                "content": bounded_text,
                "model": model,
            }
        )
    return items


def write_shards(
    items: list[dict[str, Any]],
    output_dir: Path,
    *,
    model: str,
    max_requests_per_file: int,
    max_file_bytes: int,
) -> list[dict[str, Any]]:
    shards: list[dict[str, Any]] = []
    shard_index = 0
    batch_handle = None
    metadata_handle = None
    batch_path: Path | None = None
    metadata_path: Path | None = None
    shard_rows = 0
    shard_bytes = 0

    def close_shard() -> None:
        nonlocal batch_handle, metadata_handle, batch_path, metadata_path, shard_rows, shard_bytes
        if batch_handle is None or metadata_handle is None:
            return
        batch_handle.close()
        metadata_handle.close()
        shards.append(
            {
                "batch_file": str(batch_path),
                "metadata_file": str(metadata_path),
                "request_count": shard_rows,
                "size_bytes": shard_bytes,
            }
        )
        batch_handle = None
        metadata_handle = None
        batch_path = None
        metadata_path = None
        shard_rows = 0
        shard_bytes = 0

    def open_shard() -> None:
        nonlocal batch_handle, metadata_handle, batch_path, metadata_path, shard_index
        batch_path = output_dir / f"edge_embedding_requests_{shard_index:03d}.jsonl"
        metadata_path = output_dir / f"edge_embedding_metadata_{shard_index:03d}.jsonl"
        batch_handle = batch_path.open("w", encoding="utf-8")
        metadata_handle = metadata_path.open("w", encoding="utf-8")
        shard_index += 1

    for item in items:
        request = {
            "custom_id": item["custom_id"],
            "method": "POST",
            "url": "/v1/embeddings",
            "body": request_body(model, item["content"]),
        }
        request_line = json.dumps(request, ensure_ascii=False) + "\n"
        metadata_line = json.dumps(
            {key: value for key, value in item.items() if key != "content"},
            ensure_ascii=False,
        ) + "\n"
        projected_bytes = shard_bytes + len(request_line.encode("utf-8"))
        if (
            batch_handle is not None
            and (
                shard_rows >= max_requests_per_file
                or projected_bytes > max_file_bytes
            )
        ):
            close_shard()
        if batch_handle is None:
            open_shard()
        assert batch_handle is not None
        assert metadata_handle is not None
        batch_handle.write(request_line)
        metadata_handle.write(metadata_line)
        shard_rows += 1
        shard_bytes += len(request_line.encode("utf-8"))
    close_shard()
    return shards


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{output_dir} exists; pass --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)
    graphml = graphml_path(args)
    graph = nx.read_graphml(graphml)
    cache_root = Path(args.cache_root) if args.cache_root else output_dir / "edge_embedding_cache"
    cache_path = edge_cache_path(
        graph,
        graphml,
        cache_root=cache_root,
        model=args.model,
        embed_dim=args.embed_dim,
        runtime_cache_model=args.runtime_cache_model,
    )
    items = edge_items(graph, model=args.model, max_edge_tokens=args.max_edge_tokens)
    shards = write_shards(
        items,
        output_dir,
        model=args.model,
        max_requests_per_file=args.max_requests_per_file,
        max_file_bytes=args.max_file_bytes,
    )
    manifest = {
        "working_dir": args.working_dir,
        "graphml": str(graphml),
        "model": args.model,
        "embed_dim": args.embed_dim,
        "cache_root": str(cache_root),
        "cache_file": str(cache_path),
        "runtime_cache_model": args.runtime_cache_model,
        "max_edge_tokens": args.max_edge_tokens,
        "runtime_max_batch_tokens": _EDGE_EMBEDDING_MAX_BATCH_TOKENS,
        "edge_count": graph.number_of_edges(),
        "request_count": len(items),
        "shards": shards,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {len(items)} edge embedding requests across {len(shards)} shard(s)")
    print(f"Target cache file: {cache_path}")


if __name__ == "__main__":
    main()
