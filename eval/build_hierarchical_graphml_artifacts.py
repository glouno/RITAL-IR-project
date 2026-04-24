#!/usr/bin/env python3
"""Build Neo4j-friendly hierarchical GraphML artifacts from curated run outputs.

This script augments the exported variant graph with explicit community hierarchy:
- Adds :Community-like nodes (id prefix `community:`)
- Adds IN_COMMUNITY edges from entities to communities
- Adds HAS_SUBCOMMUNITY edges between community levels
- Preserves original entity/entity edges as RELATES_TO
- Adds clean text fields where <SEP> was used as internal separators
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import networkx as nx


def _clean_sep(value: Any) -> str:
    text = "" if value is None else str(value)
    if "<SEP>" in text:
        return text.split("<SEP>", 1)[0].strip()
    return text.strip()


def _split_sep(value: Any) -> list[str]:
    text = "" if value is None else str(value)
    if not text:
        return []
    return [part.strip() for part in text.split("<SEP>") if part.strip()]


def _community_node_id(cid: int) -> str:
    return f"community:{cid}"


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _build_variant_graph(
    *,
    variant_name: str,
    graphml_path: Path,
    community_reports_path: Path,
    out_path: Path,
) -> dict[str, Any]:
    base_graph = nx.read_graphml(graphml_path)
    is_multi = isinstance(base_graph, (nx.MultiGraph, nx.MultiDiGraph))

    graph = nx.MultiGraph() if is_multi else nx.Graph()
    graph.graph["variant"] = variant_name
    graph.graph["source_graphml"] = str(graphml_path)
    graph.graph["community_reports"] = str(community_reports_path)
    graph.graph["schema_version"] = "hierarchical_v1"

    # 1) Add original entity nodes with normalized convenience fields.
    for node_id, data in base_graph.nodes(data=True):
        entity_type = str(data.get("entity_type", "UNKNOWN"))
        clean_name = str(node_id).strip().strip('"')
        desc = str(data.get("description", ""))
        src = str(data.get("source_id", ""))

        graph.add_node(
            node_id,
            kind="Entity",
            labels="Entity",
            name=clean_name,
            entity_type=entity_type,
            description=desc,
            description_clean=_clean_sep(desc),
            description_segments_json=json.dumps(_split_sep(desc), ensure_ascii=False),
            source_id=src,
            source_segments_json=json.dumps(_split_sep(src), ensure_ascii=False),
            clusters=str(data.get("clusters", "")),
        )

    # 2) Add original edges as RELATES_TO.
    for source, target, data in base_graph.edges(data=True):
        desc = str(data.get("description", ""))
        src_id = str(data.get("source_id", ""))
        edge_attrs = {
            "kind": "RELATES_TO",
            "type": "RELATES_TO",
            "description": desc,
            "description_clean": _clean_sep(desc),
            "description_segments_json": json.dumps(_split_sep(desc), ensure_ascii=False),
            "source_id": src_id,
            "source_segments_json": json.dumps(_split_sep(src_id), ensure_ascii=False),
            "order": int(data.get("order", 1)) if str(data.get("order", "")).isdigit() else 1,
        }
        w = _safe_float(data.get("weight"))
        if w is not None:
            edge_attrs["weight"] = w

        if is_multi:
            graph.add_edge(source, target, **edge_attrs)
        else:
            # For simple graphs, keep the strongest relation if duplicates appear.
            if graph.has_edge(source, target):
                old = graph[source][target]
                old_w = _safe_float(old.get("weight"))
                if old_w is None or (w is not None and w > old_w):
                    graph[source][target].update(edge_attrs)
            else:
                graph.add_edge(source, target, **edge_attrs)

    # 3) Add community nodes and hierarchy edges from reports.
    reports_raw = _load_json(community_reports_path)
    if not isinstance(reports_raw, dict):
        raise ValueError(f"Expected dict community report file: {community_reports_path}")

    community_levels: dict[int, int] = {}
    in_community_count = 0
    has_subcommunity_count = 0

    for cid_raw, payload in reports_raw.items():
        if not isinstance(payload, dict):
            continue
        cid = int(cid_raw)
        cnode = _community_node_id(cid)
        level = int(payload.get("level", -1))
        community_levels[cid] = level

        report_json = payload.get("report_json")
        if not isinstance(report_json, dict):
            report_json = {}

        findings = report_json.get("findings")
        findings_count = len(findings) if isinstance(findings, list) else 0

        graph.add_node(
            cnode,
            kind="Community",
            labels="Community",
            community_id=cid,
            level=level,
            title=str(payload.get("title") or report_json.get("title") or f"Community {cid}"),
            summary=str(report_json.get("summary", "")),
            occurrence=float(_safe_float(payload.get("occurrence")) or 0.0),
            findings_count=findings_count,
            rating=float(_safe_float(report_json.get("rating")) or 0.0),
            rating_explanation=str(report_json.get("rating_explanation", "")),
            sub_communities_json=json.dumps(payload.get("sub_communities", []), ensure_ascii=False),
            chunk_ids_json=json.dumps(payload.get("chunk_ids", []), ensure_ascii=False),
        )

        for ent in payload.get("nodes", []) or []:
            if graph.has_node(ent):
                graph.add_edge(
                    ent,
                    cnode,
                    kind="IN_COMMUNITY",
                    type="IN_COMMUNITY",
                    level=level,
                    source="community_report",
                )
                in_community_count += 1

    for cid_raw, payload in reports_raw.items():
        if not isinstance(payload, dict):
            continue
        parent_id = int(cid_raw)
        parent_node = _community_node_id(parent_id)
        for child in payload.get("sub_communities", []) or []:
            try:
                child_id = int(child)
            except Exception:
                continue
            child_node = _community_node_id(child_id)
            if graph.has_node(parent_node) and graph.has_node(child_node):
                graph.add_edge(
                    parent_node,
                    child_node,
                    kind="HAS_SUBCOMMUNITY",
                    type="HAS_SUBCOMMUNITY",
                    parent_level=community_levels.get(parent_id, -1),
                    child_level=community_levels.get(child_id, -1),
                    source="community_report",
                )
                has_subcommunity_count += 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(graph, out_path)

    entity_nodes = sum(1 for _, d in graph.nodes(data=True) if d.get("kind") == "Entity")
    community_nodes = sum(1 for _, d in graph.nodes(data=True) if d.get("kind") == "Community")
    rel_edges = sum(1 for _, _, d in graph.edges(data=True) if d.get("kind") == "RELATES_TO")

    return {
        "variant": variant_name,
        "output_graphml": str(out_path),
        "entity_nodes": entity_nodes,
        "community_nodes": community_nodes,
        "total_nodes": graph.number_of_nodes(),
        "relates_to_edges": rel_edges,
        "in_community_edges": in_community_count,
        "has_subcommunity_edges": has_subcommunity_count,
        "total_edges": graph.number_of_edges(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build hierarchical GraphML artifacts for Neo4j/Bloom")
    parser.add_argument(
        "--artifact-root",
        default="artifacts/agriculture_graphs_2026-04-23",
        help="Curated artifact root containing manifest.json",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory under artifact root (default: neo4j_hierarchical)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifact_root = Path(args.artifact_root).resolve()
    out_dir = artifact_root / (args.output_dir or "neo4j_hierarchical")

    manifest_path = artifact_root / "manifest.json"
    manifest = _load_json(manifest_path)
    variants = manifest.get("variants", [])

    runs: list[dict[str, Any]] = []
    for variant in variants:
        name = variant["name"]
        graphml_path = artifact_root / variant["graphml"]
        reports_path = artifact_root / variant["community_reports"]
        out_graph = out_dir / f"agriculture_{name}_hierarchical.graphml"
        summary = _build_variant_graph(
            variant_name=name,
            graphml_path=graphml_path,
            community_reports_path=reports_path,
            out_path=out_graph,
        )
        runs.append(summary)
        print(json.dumps(summary, ensure_ascii=False))

    summary_path = out_dir / "hierarchical_manifest.json"
    summary = {
        "source_manifest": str(manifest_path),
        "output_dir": str(out_dir),
        "runs": runs,
        "notes": [
            "RELATES_TO edges come from original GraphML edges",
            "IN_COMMUNITY and HAS_SUBCOMMUNITY come from community_reports.json",
            "description_clean strips internal <SEP> for readability",
        ],
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
