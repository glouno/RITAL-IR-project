#!/usr/bin/env python3
"""Apply parsed cluster-summary entities/relations to a GraphML graph."""

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

from hirag.prompt import GRAPH_FIELD_SEP


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Add cluster-summary nodes/edges to a HiRAG GraphML graph.")
    parser.add_argument("--graphml", required=True)
    parser.add_argument("--cluster-summary-results", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--graph-file-name", default="graph_with_cluster_summaries.graphml")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def append_sep(existing: Any, value: Any) -> str:
    values = [str(item).strip() for item in str(existing or "").split(GRAPH_FIELD_SEP) if str(item).strip()]
    new_value = str(value or "").strip()
    if new_value and new_value not in values:
        values.append(new_value)
    return GRAPH_FIELD_SEP.join(values)


def apply_summaries(graph: nx.Graph, rows: list[dict[str, Any]]) -> dict[str, int]:
    added_nodes = 0
    added_edges = 0
    for row in rows:
        if row.get("error") or row.get("parse_error"):
            continue
        source_id = f"cluster-{row.get('level')}-{row.get('cluster_id')}"
        for entity in row.get("entities", []):
            name = str(entity.get("entity_name", "")).strip()
            if not name:
                continue
            if graph.has_node(name):
                graph.nodes[name]["description"] = append_sep(graph.nodes[name].get("description"), entity.get("description"))
                graph.nodes[name]["source_id"] = append_sep(graph.nodes[name].get("source_id"), source_id)
            else:
                graph.add_node(
                    name,
                    entity_type=str(entity.get("entity_type", '"SUMMARY"')),
                    description=str(entity.get("description", "")),
                    source_id=source_id,
                    summary_level=str(row.get("level")),
                    summary_cluster=str(row.get("cluster_id")),
                )
                added_nodes += 1
        for relation in row.get("relations", []):
            source = str(relation.get("src_id", "")).strip()
            target = str(relation.get("tgt_id", "")).strip()
            if not source or not target:
                continue
            for node in (source, target):
                if not graph.has_node(node):
                    graph.add_node(node, entity_type='"UNKNOWN"', description="", source_id=source_id)
                    added_nodes += 1
            if graph.has_edge(source, target):
                graph.edges[source, target]["description"] = append_sep(graph.edges[source, target].get("description"), relation.get("description"))
                graph.edges[source, target]["source_id"] = append_sep(graph.edges[source, target].get("source_id"), source_id)
                graph.edges[source, target]["weight"] = float(graph.edges[source, target].get("weight", 0) or 0) + float(relation.get("weight", 1.0) or 1.0)
            else:
                graph.add_edge(
                    source,
                    target,
                    description=str(relation.get("description", "")),
                    weight=float(relation.get("weight", 1.0) or 1.0),
                    source_id=source_id,
                    order=1,
                    summary_level=str(row.get("level")),
                    summary_cluster=str(row.get("cluster_id")),
                )
                added_edges += 1
    return {"added_nodes": added_nodes, "added_edges": added_edges}


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    graph = nx.read_graphml(args.graphml)
    stats = apply_summaries(graph, load_jsonl(Path(args.cluster_summary_results)))
    graph_path = output_dir / args.graph_file_name
    nx.write_graphml(graph, graph_path)
    summary = {
        "input_graphml": args.graphml,
        "cluster_summary_results": args.cluster_summary_results,
        "graph_file": str(graph_path),
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        **stats,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "summary.md").write_text(
        "\n".join(
            [
                "# Cluster Summary Graph Application",
                "",
                f"- graph: `{graph_path}`",
                f"- nodes: {summary['nodes']}",
                f"- edges: {summary['edges']}",
                f"- added nodes: {summary['added_nodes']}",
                f"- added edges: {summary['added_edges']}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote graph with cluster summaries to {graph_path}")


if __name__ == "__main__":
    main()
