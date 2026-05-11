#!/usr/bin/env python3
"""Materialize a runnable HiRAG workdir from OpenAI Batch indexing artifacts.

This is a bridge utility for the staged Batch API indexing path. The batch
pipeline gives us extracted entities/relations and a flat GraphML graph; this
script rebuilds the local storage files expected by `HiRAG.query`.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import networkx as nx

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hirag._op import get_chunks
from hirag._storage import NanoVectorDBStorage, NetworkXStorage
from hirag._utils import compute_mdhash_id


MAIN_SPEC = importlib.util.spec_from_file_location("hirag_repo_main", REPO_ROOT / "main.py")
if MAIN_SPEC is None or MAIN_SPEC.loader is None:
    raise RuntimeError("Unable to load top-level main.py")
REPO_MAIN = importlib.util.module_from_spec(MAIN_SPEC)
MAIN_SPEC.loader.exec_module(REPO_MAIN)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a queryable HiRAG working directory from Batch graph artifacts"
    )
    parser.add_argument("--context-file", required=True)
    parser.add_argument("--graphml", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--chunk-token-size", type=int, default=1200)
    parser.add_argument("--chunk-overlap-token-size", type=int, default=100)
    parser.add_argument("--max-graph-cluster-size", type=int, default=10)
    parser.add_argument("--graph-cluster-seed", type=int, default=0xDEADBEEF)
    parser.add_argument("--embed-model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    parser.add_argument("--embed-dim", type=int, default=384)
    parser.add_argument("--max-token-size", type=int, default=8192)
    parser.add_argument("--embedding-batch-num", type=int, default=64)
    parser.add_argument("--embedding-func-max-async", type=int, default=8)
    parser.add_argument("--fastembed-cache-path", default=".runs/fastembed_cache")
    parser.add_argument("--fastembed-threads", type=int, default=None)
    parser.add_argument("--fastembed-parallel", type=int, default=None)
    return parser.parse_args()


def load_contexts(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not all(isinstance(item, str) for item in data):
        raise ValueError(f"{path} must be a JSON list of strings")
    return [item.strip() for item in data if item.strip()]


def build_docs_and_chunks(
    contexts: list[str],
    *,
    chunk_token_size: int,
    chunk_overlap_token_size: int,
) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, Any]]]:
    docs = {
        compute_mdhash_id(content, prefix="doc-"): {"content": content}
        for content in contexts
    }
    chunks = get_chunks(
        docs,
        overlap_token_size=chunk_overlap_token_size,
        max_token_size=chunk_token_size,
    )
    return docs, chunks


def storage_config(args: argparse.Namespace, working_dir: Path) -> dict[str, Any]:
    return {
        "working_dir": str(working_dir),
        "max_graph_cluster_size": args.max_graph_cluster_size,
        "graph_cluster_seed": args.graph_cluster_seed,
        "embedding_batch_num": args.embedding_batch_num,
    }


async def cluster_graph(
    graph: nx.Graph,
    args: argparse.Namespace,
    output_dir: Path,
) -> nx.Graph:
    storage = NetworkXStorage(
        namespace="chunk_entity_relation",
        global_config=storage_config(args, output_dir),
    )
    storage._graph = graph
    await storage.clustering("leiden")
    return storage._graph


def first_sentences(text: str, limit: int = 3) -> list[str]:
    parts = [part.strip() for part in text.replace("\n", " ").split(".")]
    sentences = [part + "." for part in parts if part]
    return sentences[:limit]


def report_string(report_json: dict[str, Any]) -> str:
    findings = report_json.get("findings", [])
    sections = "\n\n".join(
        f"## {item['summary']}\n\n{item['explanation']}" for item in findings
    )
    return f"# {report_json['title']}\n\n{report_json['summary']}\n\n{sections}".strip()


async def build_extractive_community_reports(
    graph: nx.Graph,
    args: argparse.Namespace,
    output_dir: Path,
) -> dict[str, dict[str, Any]]:
    storage = NetworkXStorage(
        namespace="chunk_entity_relation",
        global_config=storage_config(args, output_dir),
    )
    storage._graph = graph
    schema = await storage.community_schema()
    reports: dict[str, dict[str, Any]] = {}
    for key, community in schema.items():
        nodes = list(community["nodes"])
        ranked_nodes = sorted(
            nodes,
            key=lambda node: graph.degree(node) if graph.has_node(node) else 0,
            reverse=True,
        )
        node_findings = []
        for node in ranked_nodes[:5]:
            data = graph.nodes[node]
            desc = " ".join(first_sentences(str(data.get("description", "")), 2))
            node_findings.append(f"{node}: {desc}".strip())
        edge_findings = []
        for source, target in community["edges"][:5]:
            if graph.has_edge(source, target):
                edge = graph.edges[source, target]
                edge_findings.append(
                    f"{source} - {target}: {edge.get('description', '')}".strip()
                )
        title = f"Cluster {key}"
        summary = (
            f"Extractive report for {len(nodes)} entities and "
            f"{len(community['edges'])} relationships."
        )
        report_json = {
            "title": title,
            "summary": summary,
            "rating": 5,
            "findings": [
                {
                    "summary": "Central entities",
                    "explanation": "\n".join(node_findings) or "No central entities available.",
                },
                {
                    "summary": "Representative relationships",
                    "explanation": "\n".join(edge_findings) or "No representative relationships available.",
                },
            ],
        }
        reports[str(key)] = {
            "report_string": report_string(report_json),
            "report_json": report_json,
            **community,
        }
    return reports


async def write_vector_stores(
    graph: nx.Graph,
    chunks: dict[str, dict[str, Any]],
    args: argparse.Namespace,
    output_dir: Path,
) -> None:
    if args.fastembed_cache_path:
        Path(args.fastembed_cache_path).mkdir(parents=True, exist_ok=True)
    embedding_func = REPO_MAIN.create_fastembed_embedding(
        {
            "embed_model": args.embed_model,
            "embed_dim": args.embed_dim,
            "max_token_size": args.max_token_size,
            "fastembed_threads": args.fastembed_threads,
            "fastembed_parallel": args.fastembed_parallel,
            "fastembed_cache_path": args.fastembed_cache_path,
        }
    )
    config = storage_config(args, output_dir)
    entities_vdb = NanoVectorDBStorage(
        namespace="entities",
        global_config=config,
        embedding_func=embedding_func,
        meta_fields={"entity_name"},
    )
    chunks_vdb = NanoVectorDBStorage(
        namespace="chunks",
        global_config=config,
        embedding_func=embedding_func,
        meta_fields=set(),
    )
    entity_rows = {
        compute_mdhash_id(str(node_id), prefix="ent-"): {
            "content": str(node_id) + str(data.get("description", "")),
            "entity_name": str(node_id),
        }
        for node_id, data in graph.nodes(data=True)
    }
    await entities_vdb.upsert(entity_rows)
    await chunks_vdb.upsert(chunks)
    await entities_vdb.index_done_callback()
    await chunks_vdb.index_done_callback()


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_summary(
    output_dir: Path,
    docs: dict[str, Any],
    chunks: dict[str, Any],
    graph: nx.Graph,
    reports: dict[str, Any],
) -> None:
    clustered_nodes = sum(1 for _node, data in graph.nodes(data=True) if data.get("clusters"))
    levels = Counter()
    for report in reports.values():
        levels[str(report.get("level"))] += 1
    summary = {
        "working_dir": str(output_dir),
        "docs": len(docs),
        "chunks": len(chunks),
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "clustered_nodes": clustered_nodes,
        "community_reports": len(reports),
        "community_levels": dict(sorted(levels.items())),
        "community_report_mode": "extractive_from_batch_graph",
    }
    write_json(output_dir / "materialization_summary.json", summary)
    lines = [
        "# Mix HiRAG Workdir Materialization",
        "",
        f"- working dir: `{output_dir}`",
        f"- docs: {summary['docs']}",
        f"- chunks: {summary['chunks']}",
        f"- graph nodes: {summary['nodes']}",
        f"- graph edges: {summary['edges']}",
        f"- clustered nodes: {summary['clustered_nodes']}",
        f"- community reports: {summary['community_reports']}",
        f"- community report mode: `{summary['community_report_mode']}`",
    ]
    (output_dir / "materialization_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


async def amain() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"{output_dir} exists; pass --overwrite to replace it")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    contexts = load_contexts(Path(args.context_file))
    docs, chunks = build_docs_and_chunks(
        contexts,
        chunk_token_size=args.chunk_token_size,
        chunk_overlap_token_size=args.chunk_overlap_token_size,
    )
    graph = nx.read_graphml(args.graphml)
    graph = await cluster_graph(graph, args, output_dir)
    reports = await build_extractive_community_reports(graph, args, output_dir)

    write_json(output_dir / "kv_store_full_docs.json", docs)
    write_json(output_dir / "kv_store_text_chunks.json", chunks)
    write_json(output_dir / "kv_store_community_reports.json", reports)
    nx.write_graphml(graph, output_dir / "graph_chunk_entity_relation.graphml")
    await write_vector_stores(graph, chunks, args, output_dir)
    write_summary(output_dir, docs, chunks, graph, reports)
    print(f"Wrote runnable HiRAG workdir to {output_dir}")


def main() -> None:
    asyncio.run(amain())


if __name__ == "__main__":
    main()
