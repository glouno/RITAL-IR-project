#!/usr/bin/env python3
"""Summarize OpenAI Batch usage/cost for a full HiRAG run directory."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.openai_batch_retry_utils import batch_usage, load_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write cost_summary.json for an OpenAI full HiRAG run folder.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-file", default=None)
    parser.add_argument("--batch-input-price-per-mtok", type=float, default=0.375)
    parser.add_argument("--batch-output-price-per-mtok", type=float, default=2.25)
    parser.add_argument("--embedding-price-per-mtok", type=float, default=0.01)
    return parser.parse_args()


def usage_from_manifest(path: Path) -> dict[str, int]:
    data = json.loads(path.read_text(encoding="utf-8"))
    usage = ((data.get("latest_batch") or {}).get("usage") or {})
    return {
        "input_tokens": int(usage.get("input_tokens") or 0),
        "output_tokens": int(usage.get("output_tokens") or 0),
        "total_tokens": int(usage.get("total_tokens") or 0),
    }


def usage_from_batch_output(path: Path) -> dict[str, int]:
    totals = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    for row in load_jsonl(path):
        usage = batch_usage(row)
        totals["input_tokens"] += usage["prompt_tokens"]
        totals["output_tokens"] += usage["completion_tokens"]
        totals["total_tokens"] += usage["total_tokens"]
    return totals


def cost(input_tokens: int, output_tokens: int, input_price: float, output_price: float) -> float:
    return input_tokens / 1_000_000 * input_price + output_tokens / 1_000_000 * output_price


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    stages: list[dict[str, Any]] = []
    seen_outputs: set[Path] = set()
    for manifest in sorted(run_dir.rglob("batch_run_manifest.json")):
        usage = usage_from_manifest(manifest)
        stages.append({"source": str(manifest), **usage})
        data = json.loads(manifest.read_text(encoding="utf-8"))
        if data.get("output_path"):
            seen_outputs.add((manifest.parent / Path(data["output_path"]).name).resolve())
    for output in sorted(run_dir.rglob("*_output.jsonl")):
        if output.resolve() in seen_outputs:
            continue
        usage = usage_from_batch_output(output)
        if usage["total_tokens"]:
            stages.append({"source": str(output), **usage})
    total_input = sum(stage["input_tokens"] for stage in stages)
    total_output = sum(stage["output_tokens"] for stage in stages)
    summary = {
        "run_dir": str(run_dir),
        "stage_count": len(stages),
        "input_tokens": total_input,
        "output_tokens": total_output,
        "total_tokens": total_input + total_output,
        "estimated_batch_chat_cost_usd": cost(
            total_input,
            total_output,
            args.batch_input_price_per_mtok,
            args.batch_output_price_per_mtok,
        ),
        "prices_per_mtok": {
            "batch_input": args.batch_input_price_per_mtok,
            "batch_output": args.batch_output_price_per_mtok,
            "embedding": args.embedding_price_per_mtok,
        },
        "stages": stages,
    }
    output_file = Path(args.output_file) if args.output_file else run_dir / "cost_summary.json"
    output_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote cost summary to {output_file}")


if __name__ == "__main__":
    main()
