#!/usr/bin/env python3
"""Summarize detailed HiRAG retrieval traces into CSV and Markdown metrics."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize retrieval_traces.jsonl files.")
    parser.add_argument("--traces", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_traces(path: Path) -> list[dict[str, Any]]:
    traces: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                traces.append(json.loads(line))
    return traces


def flatten_trace(trace: dict[str, Any]) -> dict[str, Any]:
    retrieval = trace.get("retrieval") or {}
    budgets = trace.get("budgets") or {}
    attempts = budgets.get("context_budget_attempts") or []
    return {
        "custom_id": trace.get("custom_id"),
        "query_id": trace.get("query_id"),
        "variant": trace.get("variant"),
        "input_tokens": trace.get("input_tokens"),
        "context_tokens": trace.get("context_tokens"),
        "latency_seconds": trace.get("latency_seconds"),
        "local_entity_count": retrieval.get("local_entity_count", 0),
        "community_count": retrieval.get("community_count", 0),
        "bridge_edge_count": retrieval.get("bridge_edge_count", 0),
        "bridge_path_nodes": retrieval.get("bridge_path_nodes"),
        "bridge_path_edges": retrieval.get("bridge_path_edges"),
        "mean_edge_cost": retrieval.get("mean_edge_cost"),
        "max_edge_cost": retrieval.get("max_edge_cost"),
        "mean_edge_score": retrieval.get("mean_edge_score"),
        "bridge_budget_fallbacks": retrieval.get("bridge_budget_fallbacks", 0),
        "bridge_budget_stopped": retrieval.get("bridge_budget_stopped", False),
        "context_input_tokens": budgets.get("context_input_tokens"),
        "context_within_budget": budgets.get("context_within_budget"),
        "budget_attempt_count": len(attempts),
        "error": trace.get("error"),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("variant"))].append(row)
    summary: dict[str, dict[str, Any]] = {}
    metric_keys = [
        "input_tokens",
        "context_tokens",
        "latency_seconds",
        "local_entity_count",
        "community_count",
        "bridge_edge_count",
        "bridge_path_nodes",
        "bridge_path_edges",
        "mean_edge_cost",
        "max_edge_cost",
        "mean_edge_score",
        "bridge_budget_fallbacks",
        "context_input_tokens",
        "budget_attempt_count",
    ]
    for variant, items in sorted(grouped.items()):
        entry: dict[str, Any] = {
            "rows": len(items),
            "errors": sum(1 for item in items if item.get("error")),
            "budget_stopped_rows": sum(
                1 for item in items if item.get("bridge_budget_stopped") is True
            ),
            "out_of_budget_rows": sum(
                1 for item in items if item.get("context_within_budget") is False
            ),
        }
        for key in metric_keys:
            values = [
                converted
                for item in items
                if (converted := as_float(item.get(key))) is not None
            ]
            entry[f"mean_{key}"] = mean(values)
        summary[variant] = entry
    return summary


def write_markdown(path: Path, summary: dict[str, dict[str, Any]]) -> None:
    lines = ["# Retrieval Trace Summary", ""]
    for variant, metrics in summary.items():
        lines.append(f"## {variant}")
        lines.append(f"- rows: {metrics['rows']}")
        lines.append(f"- errors: {metrics['errors']}")
        lines.append(f"- mean context tokens: {metrics.get('mean_context_tokens')}")
        lines.append(f"- mean local entities: {metrics.get('mean_local_entity_count')}")
        lines.append(f"- mean communities: {metrics.get('mean_community_count')}")
        lines.append(f"- mean bridge edges: {metrics.get('mean_bridge_edge_count')}")
        lines.append(f"- mean bridge path edges: {metrics.get('mean_bridge_path_edges')}")
        lines.append(f"- mean edge score: {metrics.get('mean_mean_edge_score')}")
        lines.append(f"- budget stopped rows: {metrics['budget_stopped_rows']}")
        lines.append(f"- out-of-budget rows: {metrics['out_of_budget_rows']}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [flatten_trace(trace) for trace in load_traces(Path(args.traces))]
    csv_path = output_dir / "retrieval_trace_metrics.csv"
    if rows:
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    summary = summarize(rows)
    (output_dir / "retrieval_trace_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_markdown(output_dir / "retrieval_trace_summary.md", summary)
    print(f"Wrote retrieval trace summary to {output_dir}")


if __name__ == "__main__":
    main()
