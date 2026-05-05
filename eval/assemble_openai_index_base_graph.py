#!/usr/bin/env python3
"""Assemble a base HiRAG graph from imported OpenAI Batch entity/relation stages."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import networkx as nx

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hirag.prompt import GRAPH_FIELD_SEP


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge imported OpenAI Batch entity/relation extraction results into "
            "a base GraphML artifact. This assembles the flat graph only; hierarchy "
            "and community reports remain later stages."
        )
    )
    parser.add_argument("--entity-results", required=True)
    parser.add_argument("--relation-results", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--graph-file-name", default="base_graph.graphml")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def ordered_join(values: list[Any]) -> str:
    seen: dict[str, None] = {}
    for value in values:
        text = str(value).strip()
        if text:
            seen.setdefault(text, None)
    return GRAPH_FIELD_SEP.join(seen.keys())


def merge_entity_nodes(entity_rows: list[dict[str, Any]]) -> dict[str, dict[str, str]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in entity_rows:
        if row.get("error"):
            continue
        for entity in row.get("entities", []):
            name = str(entity.get("entity_name", "")).strip()
            if name:
                grouped[name].append(entity)

    nodes: dict[str, dict[str, str]] = {}
    for name, items in grouped.items():
        type_counts = Counter(str(item.get("entity_type", '"UNKNOWN"')) for item in items)
        entity_type = type_counts.most_common(1)[0][0] if type_counts else '"UNKNOWN"'
        nodes[name] = {
            "entity_type": entity_type,
            "description": ordered_join([item.get("description", "") for item in items]),
            "source_id": ordered_join([item.get("source_id", "") for item in items]),
        }
    return nodes


def merge_relation_edges(relation_rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in relation_rows:
        if row.get("error"):
            continue
        for relation in row.get("relations", []):
            source = str(relation.get("src_id", "")).strip()
            target = str(relation.get("tgt_id", "")).strip()
            if source and target:
                grouped[tuple(sorted((source, target)))].append(relation)

    edges: dict[tuple[str, str], dict[str, Any]] = {}
    for key, items in grouped.items():
        edges[key] = {
            "weight": sum(float(item.get("weight", 1.0) or 1.0) for item in items),
            "description": ordered_join([item.get("description", "") for item in items]),
            "source_id": ordered_join([item.get("source_id", "") for item in items]),
            "order": min(int(item.get("order", 1) or 1) for item in items),
        }
    return edges


def build_graph(
    nodes: dict[str, dict[str, str]],
    edges: dict[tuple[str, str], dict[str, Any]],
) -> nx.Graph:
    graph = nx.Graph()
    for node_id, attrs in nodes.items():
        graph.add_node(node_id, **attrs)
    for (source, target), attrs in edges.items():
        for node_id in (source, target):
            if not graph.has_node(node_id):
                graph.add_node(
                    node_id,
                    entity_type='"UNKNOWN"',
                    description=str(attrs.get("description", "")),
                    source_id=str(attrs.get("source_id", "")),
                )
        graph.add_edge(source, target, **attrs)
    return graph


def write_outputs(
    output_dir: Path,
    graph: nx.Graph,
    graph_file_name: str,
    nodes: dict[str, dict[str, str]],
    edges: dict[tuple[str, str], dict[str, Any]],
    entity_rows: list[dict[str, Any]],
    relation_rows: list[dict[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    graph_path = output_dir / graph_file_name
    nx.write_graphml(graph, graph_path)
    node_rows = [
        {"entity_name": node_id, **attrs}
        for node_id, attrs in sorted(nodes.items(), key=lambda item: item[0])
    ]
    edge_rows = [
        {"src_id": source, "tgt_id": target, **attrs}
        for (source, target), attrs in sorted(edges.items(), key=lambda item: item[0])
    ]
    (output_dir / "merged_nodes.json").write_text(
        json.dumps(node_rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "merged_edges.json").write_text(
        json.dumps(edge_rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    summary = {
        "graph_file": str(graph_path),
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "entity_result_rows": len(entity_rows),
        "relation_result_rows": len(relation_rows),
        "entity_result_errors": sum(1 for row in entity_rows if row.get("error")),
        "relation_result_errors": sum(1 for row in relation_rows if row.get("error")),
        "parsed_entities": sum(len(row.get("entities", [])) for row in entity_rows if not row.get("error")),
        "parsed_relations": sum(len(row.get("relations", [])) for row in relation_rows if not row.get("error")),
        "unknown_nodes_added_from_relations": sum(
            1 for _node_id, attrs in graph.nodes(data=True) if attrs.get("entity_type") == '"UNKNOWN"'
        ),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "summary.md").write_text(
        "\n".join(
            [
                "# Base Graph Assembly Summary",
                "",
                f"- graph: `{graph_path}`",
                f"- nodes: {summary['nodes']}",
                f"- edges: {summary['edges']}",
                f"- parsed entities: {summary['parsed_entities']}",
                f"- parsed relations: {summary['parsed_relations']}",
                f"- unknown nodes added from relations: {summary['unknown_nodes_added_from_relations']}",
                f"- entity row errors: {summary['entity_result_errors']}",
                f"- relation row errors: {summary['relation_result_errors']}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    entity_rows = load_jsonl(Path(args.entity_results))
    relation_rows = load_jsonl(Path(args.relation_results))
    nodes = merge_entity_nodes(entity_rows)
    edges = merge_relation_edges(relation_rows)
    graph = build_graph(nodes, edges)
    write_outputs(
        Path(args.output_dir),
        graph,
        args.graph_file_name,
        nodes,
        edges,
        entity_rows,
        relation_rows,
    )
    print(f"Wrote base graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")


if __name__ == "__main__":
    main()
