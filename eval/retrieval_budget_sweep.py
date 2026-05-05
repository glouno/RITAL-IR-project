#!/usr/bin/env python3
"""Sweep answer-generation context budgets for HiRAG variants."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.answer_generation_benchmark import compact_debug
from eval.eval_utils import (
    DEFAULT_ANSWER_VARIANTS,
    DEFAULT_QUERY_FILE,
    add_runtime_args,
    build_context_with_budget,
    build_fastembed_hirag,
    load_queries,
)


def parse_args() -> argparse.Namespace:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description="Sweep compact retrieval budgets")
    parser.add_argument("--query-file", default=DEFAULT_QUERY_FILE)
    parser.add_argument("--query-limit", type=int, default=10)
    parser.add_argument("--variants", nargs="*", default=DEFAULT_ANSWER_VARIANTS)
    parser.add_argument(
        "--regimes",
        nargs="*",
        default=["tiny", "small", "medium"],
        choices=["tiny", "small", "medium", "large"],
    )
    parser.add_argument("--output-dir", default=f".runs/retrieval_budget_sweep/{timestamp}")
    parser.add_argument("--response-type", default="Multiple Paragraphs")
    parser.add_argument("--answer-max-tokens", type=int, default=512)
    parser.add_argument("--text-unit-snippet-chars", type=int, default=1200)
    parser.add_argument("--max-budget-attempts", type=int, default=1)
    parser.add_argument("--budget-shrink-factor", type=float, default=0.7)
    add_runtime_args(parser)
    return parser.parse_args()


REGIMES = {
    "tiny": {"top_k": 8, "top_m": 4, "section_budget": 600, "text_budget": 3500, "max_input_tokens": 6000},
    "small": {"top_k": 12, "top_m": 6, "section_budget": 1000, "text_budget": 6000, "max_input_tokens": 9000},
    "medium": {"top_k": 16, "top_m": 8, "section_budget": 1500, "text_budget": 8000, "max_input_tokens": 11000},
    "large": {"top_k": 20, "top_m": 10, "section_budget": 2500, "text_budget": 10000, "max_input_tokens": 12000},
}


async def _dummy_llm(*_args: Any, **_kwargs: Any) -> str:
    return ""


def apply_regime(args: argparse.Namespace, regime: str) -> argparse.Namespace:
    values = REGIMES[regime]
    copy = argparse.Namespace(**vars(args))
    copy.top_k = values["top_k"]
    copy.top_m = values["top_m"]
    copy.max_token_for_local_context = values["section_budget"]
    copy.max_token_for_bridge_knowledge = values["section_budget"]
    copy.max_token_for_community_report = values["section_budget"]
    copy.max_token_for_text_unit = values["text_budget"]
    copy.max_input_tokens = values["max_input_tokens"]
    copy.min_section_budget = 200
    copy.text_unit_snippet_chars = getattr(args, "text_unit_snippet_chars", 1200)
    copy.text_unit_snippet_strategy = getattr(args, "text_unit_snippet_strategy", "query_overlap")
    return copy


def write_summary(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["regime"], row["variant"]), []).append(row)
    summary_rows = []
    for (regime, variant), items in sorted(grouped.items()):
        input_tokens = [item["input_tokens"] for item in items]
        context_tokens = [item["context_tokens"] for item in items]
        summary_rows.append(
            {
                "regime": regime,
                "variant": variant,
                "rows": len(items),
                "within_budget_rate": sum(item["within_budget"] for item in items) / len(items),
                "mean_input_tokens": sum(input_tokens) / len(input_tokens),
                "max_input_tokens": max(input_tokens),
                "mean_context_tokens": sum(context_tokens) / len(context_tokens),
                "mean_bridge_edges": sum(item["bridge_edge_count"] for item in items) / len(items),
                "mean_bridge_path_edges": sum(item["bridge_path_edges"] for item in items) / len(items),
            }
        )
    with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()) if summary_rows else ["regime"])
        writer.writeheader()
        writer.writerows(summary_rows)
    md = ["# Retrieval Budget Sweep", ""]
    for row in summary_rows:
        md.append(
            "- {regime}/{variant}: within={within_budget_rate:.2f}, "
            "mean_input={mean_input_tokens:.0f}, max_input={max_input_tokens:.0f}, "
            "mean_bridge_edges={mean_bridge_edges:.1f}".format(**row)
        )
    (output_dir / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    graph = build_fastembed_hirag(args, llm_func=_dummy_llm, enable_llm_cache=False)
    queries = load_queries(Path(args.query_file), args.query_limit)
    rows = []
    rows_path = output_dir / "budget_sweep.csv"
    with rows_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "regime",
            "query_id",
            "variant",
            "input_tokens",
            "context_tokens",
            "within_budget",
            "latency_seconds",
            "local_entity_count",
            "community_count",
            "bridge_edge_count",
            "bridge_path_edges",
            "budget_attempts",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for regime in args.regimes:
            regime_args = apply_regime(args, regime)
            for query_item in queries:
                for variant in args.variants:
                    start = time.perf_counter()
                    context, param, budget_debug = build_context_with_budget(
                        graph,
                        query_item["query"],
                        variant,
                        regime_args,
                    )
                    debug = compact_debug(dict(param.debug_info))
                    row = {
                        "regime": regime,
                        "query_id": query_item["query_id"],
                        "variant": variant,
                        "input_tokens": budget_debug["context_input_tokens"],
                        "context_tokens": len(context.split()),
                        "within_budget": budget_debug["context_within_budget"],
                        "latency_seconds": time.perf_counter() - start,
                        "local_entity_count": debug["local_entity_count"],
                        "community_count": debug["community_count"],
                        "bridge_edge_count": debug["bridge_edge_count"],
                        "bridge_path_edges": debug["bridge_path_edges"] or 0,
                        "budget_attempts": len(budget_debug["context_budget_attempts"]),
                    }
                    rows.append(row)
                    writer.writerow(row)
                    handle.flush()
    write_summary(output_dir, rows)
    print(f"Wrote sweep outputs to {output_dir}")


if __name__ == "__main__":
    main()
