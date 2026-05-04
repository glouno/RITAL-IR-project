#!/usr/bin/env python3
"""Analyze oversized or high-entropy HiRAG communities and propose local splits."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np

from hirag._cluster_utils import GMM_cluster


def parse_args() -> argparse.Namespace:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description="Analyze HiRAG community balance")
    parser.add_argument("--graphml", required=True)
    parser.add_argument("--community-reports", required=True)
    parser.add_argument("--output-dir", default=f".runs/cluster_balance/{timestamp}")
    parser.add_argument("--max-community-nodes", type=int, default=100)
    parser.add_argument("--entropy-threshold", type=float, default=0.8)
    parser.add_argument("--embed-model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    parser.add_argument("--fastembed-cache-path", default=".runs/fastembed_cache")
    parser.add_argument("--max-split-communities", type=int, default=1)
    parser.add_argument("--max-split-members", type=int, default=100)
    return parser.parse_args()


def normalized_entropy(values: list[str]) -> float:
    if not values:
        return 0.0
    counts = Counter(values)
    if len(counts) <= 1:
        return 0.0
    total = sum(counts.values())
    entropy = -sum((count / total) * math.log(count / total) for count in counts.values())
    return entropy / math.log(len(counts))


def split_sep(value: Any) -> list[str]:
    text = "" if value is None else str(value)
    return [part.strip() for part in text.split("<SEP>") if part.strip()]


def entity_text(graph: nx.Graph, node_id: str) -> str:
    data = graph.nodes[node_id]
    return "\n".join(
        [
            str(node_id),
            f"type: {data.get('entity_type', 'UNKNOWN')}",
            f"description: {data.get('description', '')}",
        ]
    )


def load_reports(path: Path) -> dict[str, dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("community reports must be a JSON object")
    return data


def top_counts(values: list[str], limit: int = 5) -> list[dict[str, Any]]:
    return [
        {"value": value, "count": count}
        for value, count in Counter(values).most_common(limit)
    ]


def analyze_communities(
    graph: nx.Graph,
    reports: dict[str, dict],
    *,
    max_nodes: int,
    entropy_threshold: float,
) -> list[dict[str, Any]]:
    rows = []
    for community_id, report in reports.items():
        nodes = [node for node in report.get("nodes", []) if graph.has_node(node)]
        subgraph = graph.subgraph(nodes)
        entity_types = [str(graph.nodes[node].get("entity_type", "UNKNOWN")) for node in nodes]
        chunk_ids = set()
        for node in nodes:
            chunk_ids.update(split_sep(graph.nodes[node].get("source_id", "")))
        node_count = len(nodes)
        edge_count = subgraph.number_of_edges()
        density = nx.density(subgraph) if node_count > 1 else 0.0
        entropy = normalized_entropy(entity_types)
        should_split = node_count > max_nodes or entropy >= entropy_threshold
        rows.append(
            {
                "community_id": community_id,
                "level": report.get("level"),
                "title": report.get("title"),
                "node_count": node_count,
                "edge_count": edge_count,
                "edge_density": density,
                "entity_type_entropy": entropy,
                "source_chunk_count": len(chunk_ids),
                "top_entity_types": json.dumps(top_counts(entity_types), ensure_ascii=False),
                "should_split": should_split,
            }
        )
    return rows


def build_split_plan(
    graph: nx.Graph,
    rows: list[dict[str, Any]],
    reports: dict[str, dict],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    candidates = [
        row for row in rows if row["should_split"] and int(row["node_count"]) > 2
    ]
    candidates.sort(
        key=lambda row: (int(row["node_count"]), float(row["entity_type_entropy"])),
        reverse=True,
    )
    candidates = candidates[: args.max_split_communities]
    if not candidates:
        return []

    from fastembed import TextEmbedding

    embedder = TextEmbedding(model_name=args.embed_model, cache_dir=args.fastembed_cache_path)
    split_plan = []
    for row in candidates:
        all_nodes = [
            node for node in reports[row["community_id"]].get("nodes", []) if graph.has_node(node)
        ]
        nodes = all_nodes[: args.max_split_members]
        texts = [entity_text(graph, node) for node in nodes]
        try:
            embeddings = np.asarray(list(embedder.embed(texts)), dtype=np.float64)
            labels, cluster_count = GMM_cluster(embeddings, threshold=0.1, reg_covar=1e-6)
        except Exception as exc:
            split_plan.append(
                {
                    "community_id": row["community_id"],
                    "error": str(exc),
                    "original_candidate_members": len(all_nodes),
                    "embedded_members": len(nodes),
                    "proposed_cluster_count": 0,
                    "clusters": [],
                }
            )
            continue

        cluster_members: dict[int, list[str]] = {}
        for node, node_labels in zip(nodes, labels):
            label = int(node_labels[0]) if len(node_labels) else 0
            cluster_members.setdefault(label, []).append(node)
        clusters = []
        for label, members in sorted(cluster_members.items()):
            entity_types = [str(graph.nodes[node].get("entity_type", "UNKNOWN")) for node in members]
            clusters.append(
                {
                    "label": label,
                    "size": len(members),
                    "entity_type_entropy": normalized_entropy(entity_types),
                    "representative_entities": members[:10],
                    "top_entity_types": top_counts(entity_types),
                }
            )
        split_plan.append(
            {
                "community_id": row["community_id"],
                "original_node_count": row["node_count"],
                "original_entity_type_entropy": row["entity_type_entropy"],
                "original_candidate_members": len(all_nodes),
                "embedded_members": len(nodes),
                "truncated_members": len(all_nodes) > len(nodes),
                "proposed_cluster_count": cluster_count,
                "clusters": clusters,
            }
        )
    return split_plan


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.fastembed_cache_path:
        Path(args.fastembed_cache_path).mkdir(parents=True, exist_ok=True)
    graph = nx.read_graphml(args.graphml)
    reports = load_reports(Path(args.community_reports))
    rows = analyze_communities(
        graph,
        reports,
        max_nodes=args.max_community_nodes,
        entropy_threshold=args.entropy_threshold,
    )
    with (output_dir / "community_balance.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    oversized = [row for row in rows if row["should_split"]]
    (output_dir / "oversized_communities.json").write_text(
        json.dumps(oversized, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    split_plan = build_split_plan(graph, rows, reports, args)
    (output_dir / "split_plan.json").write_text(
        json.dumps(split_plan, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Wrote cluster balance outputs to {output_dir}")


if __name__ == "__main__":
    main()
