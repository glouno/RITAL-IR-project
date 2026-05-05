#!/usr/bin/env python3
"""Build blind human annotation packets from generated answer files."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_VARIANTS = ["hi", "hi_weighted", "hi_minmax", "hi_rerank_weighted"]


def parse_args() -> argparse.Namespace:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description="Build anonymized human answer annotation packets")
    parser.add_argument("--answers", required=True)
    parser.add_argument("--judge-results", default=None)
    parser.add_argument("--proxy-metrics", default=".runs/retrieval_eval/dev_retrieval_q50/metrics.csv")
    parser.add_argument("--query-count", type=int, default=30)
    parser.add_argument("--variants", nargs="*", default=DEFAULT_VARIANTS)
    parser.add_argument("--output-dir", default=f".runs/human_eval/{timestamp}")
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def load_answers(path: Path, variants: list[str]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = defaultdict(dict)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            data = json.loads(line)
            if data.get("error") or data.get("variant") not in variants:
                continue
            grouped[str(data["query_id"])][data["variant"]] = data
    return {
        query_id: by_variant
        for query_id, by_variant in grouped.items()
        if all(variant in by_variant for variant in variants)
    }


def load_proxy_buckets(path: Path, available_query_ids: set[str]) -> dict[str, list[str]]:
    if not path.exists():
        return {}
    import pandas as pd

    metrics = pd.read_csv(path)
    metrics["query_id"] = metrics.groupby("variant").cumcount().astype(str)
    baseline = metrics[metrics.variant == "hi"].set_index("query_id")
    minmax = metrics[metrics.variant == "hi_minmax"].set_index("query_id")
    rerank = metrics[metrics.variant == "hi_rerank_weighted"].set_index("query_id")
    buckets: dict[str, list[str]] = {}
    if not minmax.empty:
        joined = minmax.join(baseline, lsuffix="", rsuffix="_hi")
        joined["minmax_delta"] = (
            joined["bridge_vs_query_overlap"] - joined["bridge_vs_query_overlap_hi"]
        )
        ids = [str(idx) for idx in joined.sort_values("minmax_delta", ascending=False).index if str(idx) in available_query_ids]
        buckets["minmax_proxy_win"] = ids
        ids = [str(idx) for idx in joined.sort_values("minmax_delta", ascending=True).index if str(idx) in available_query_ids]
        buckets["minmax_proxy_loss"] = ids
    if not rerank.empty:
        joined = rerank.join(baseline, lsuffix="", rsuffix="_hi")
        joined["rerank_bridge_delta"] = (
            joined["bridge_edge_count"] - joined["bridge_edge_count_hi"]
        ).abs()
        ids = [str(idx) for idx in joined.sort_values("rerank_bridge_delta", ascending=False).index if str(idx) in available_query_ids]
        buckets["rerank_changed_bridge"] = ids
    return buckets


def choose_queries(args: argparse.Namespace, answers: dict[str, dict[str, Any]]) -> list[tuple[str, str]]:
    rng = random.Random(args.seed)
    available = set(answers.keys())
    buckets = load_proxy_buckets(Path(args.proxy_metrics), available)
    selected: list[tuple[str, str]] = []
    used: set[str] = set()

    def take(bucket: str, count: int) -> None:
        for query_id in buckets.get(bucket, []):
            if query_id in used:
                continue
            selected.append((query_id, bucket))
            used.add(query_id)
            if sum(1 for _qid, name in selected if name == bucket) >= count:
                break

    if args.query_count >= 30:
        take("minmax_proxy_win", 10)
        take("minmax_proxy_loss", 5)
        take("rerank_changed_bridge", 5)
    else:
        take("minmax_proxy_win", max(1, args.query_count // 3))
        take("minmax_proxy_loss", max(1, args.query_count // 6))
        take("rerank_changed_bridge", max(1, args.query_count // 6))

    remaining = [query_id for query_id in sorted(available) if query_id not in used]
    rng.shuffle(remaining)
    for query_id in remaining:
        if len(selected) >= args.query_count:
            break
        selected.append((query_id, "random"))
        used.add(query_id)
    return selected[: args.query_count]


def anonymized_answers(query_id: str, by_variant: dict[str, Any], variants: list[str], seed: int) -> tuple[dict[str, str], dict[str, str]]:
    rng = random.Random(f"{seed}:{query_id}")
    shuffled = list(variants)
    rng.shuffle(shuffled)
    labels = [f"Answer {chr(ord('A') + index)}" for index in range(len(shuffled))]
    label_to_variant = dict(zip(labels, shuffled))
    label_to_answer = {
        label: by_variant[variant]["answer"]
        for label, variant in label_to_variant.items()
    }
    return label_to_variant, label_to_answer


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    answers = load_answers(Path(args.answers), args.variants)
    selected = choose_queries(args, answers)

    json_path = output_dir / "human_annotation_packet.jsonl"
    csv_path = output_dir / "human_annotation_packet.csv"
    csv_rows = []
    with json_path.open("w", encoding="utf-8") as json_handle:
        for packet_id, (query_id, bucket) in enumerate(selected, start=1):
            by_variant = answers[query_id]
            label_to_variant, label_to_answer = anonymized_answers(
                query_id, by_variant, args.variants, args.seed
            )
            query = next(iter(by_variant.values()))["query"]
            record = {
                "packet_id": packet_id,
                "query_id": query_id,
                "sampling_bucket": bucket,
                "query": query,
                "answers": label_to_answer,
                "annotation_fields": {
                    "best_answer_overall": "",
                    "factuality_score_1_to_5": "",
                    "completeness_score_1_to_5": "",
                    "relevance_score_1_to_5": "",
                    "notes": "",
                },
                "hidden_metadata": {
                    "label_to_variant": label_to_variant,
                },
            }
            json_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            row = {
                "packet_id": packet_id,
                "query": query,
                "sampling_bucket": bucket,
                "best_answer_overall": "",
                "factuality_score_1_to_5": "",
                "completeness_score_1_to_5": "",
                "relevance_score_1_to_5": "",
                "notes": "",
            }
            for label, answer in label_to_answer.items():
                row[label] = answer
            csv_rows.append(row)

    fieldnames = [
        "packet_id",
        "query",
        "sampling_bucket",
        *[f"Answer {chr(ord('A') + index)}" for index in range(len(args.variants))],
        "best_answer_overall",
        "factuality_score_1_to_5",
        "completeness_score_1_to_5",
        "relevance_score_1_to_5",
        "notes",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    readme = f"""# Human Annotation Packet

This packet contains {len(csv_rows)} anonymized question/answer comparisons.

For each row:

- read the query
- compare the anonymized answers
- choose the best answer overall
- score factuality, completeness, and relevance from 1 to 5
- add concise notes when useful

Variant labels are hidden in `human_annotation_packet.jsonl` under `hidden_metadata`.
"""
    (output_dir / "human_annotation_readme.md").write_text(readme, encoding="utf-8")
    print(f"Wrote human annotation packet to {output_dir}")


if __name__ == "__main__":
    main()
