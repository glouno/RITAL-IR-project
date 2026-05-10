#!/usr/bin/env python3
"""Pairwise LLM-as-judge evaluation for generated HiRAG answers."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import re
import statistics
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


CRITERIA = ["Comprehensiveness", "Diversity", "Empowerment", "Overall Winner"]


def parse_args() -> argparse.Namespace:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description="Judge generated answers pairwise")
    parser.add_argument("--answers", required=True)
    parser.add_argument("--baseline", default="hi")
    parser.add_argument("--variants", nargs="*", default=["hi_weighted", "hi_minmax", "hi_rerank_weighted"])
    parser.add_argument("--output-dir", default=f".runs/answer_eval/judge_{timestamp}")
    parser.add_argument("--judge-base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--judge-api-key", default="EMPTY")
    parser.add_argument("--judge-model", required=True)
    parser.add_argument("--max-concurrency", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_answers(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    records = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            data = json.loads(line)
            if data.get("error"):
                continue
            records[(str(data["query_id"]), data["variant"])] = data
    return records


def build_prompt(query: str, answer1: str, answer2: str) -> str:
    return f"""
You will evaluate two answers to the same question based on three criteria: Comprehensiveness, Diversity, and Empowerment.

- Comprehensiveness: How much detail does the answer provide to cover all aspects and details of the question?
- Diversity: How varied and rich is the answer in providing different perspectives and insights on the question?
- Empowerment: How well does the answer help the reader understand and make informed judgments about the topic?

For each criterion, choose the better answer, either "Answer 1", "Answer 2", or "Tie". Then select an overall winner.

Question:
{query}

Answer 1:
{answer1}

Answer 2:
{answer2}

Output only valid JSON in this format:
{{
  "Comprehensiveness": {{"Winner": "Answer 1|Answer 2|Tie", "Explanation": "..."}},
  "Diversity": {{"Winner": "Answer 1|Answer 2|Tie", "Explanation": "..."}},
  "Empowerment": {{"Winner": "Answer 1|Answer 2|Tie", "Explanation": "..."}},
  "Overall Winner": {{"Winner": "Answer 1|Answer 2|Tie", "Explanation": "..."}}
}}
""".strip()


def parse_judge_json(text: str) -> tuple[dict[str, Any] | None, str | None]:
    try:
        return json.loads(text), None
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0)), None
            except json.JSONDecodeError as exc:
                return None, str(exc)
        return None, "no JSON object found"


def map_winner(winner: str | None, order: list[str]) -> str:
    if winner == "Answer 1":
        return order[0]
    if winner == "Answer 2":
        return order[1]
    if winner == "Tie":
        return "Tie"
    return "Unknown"


async def judge_one(client: Any, args: argparse.Namespace, query: str, answer1: str, answer2: str) -> tuple[str, float]:
    prompt = build_prompt(query, answer1, answer2)
    start = time.perf_counter()
    response = await client.chat.completions.create(
        model=args.judge_model,
        messages=[
            {"role": "system", "content": "You are a careful answer evaluator. Return only JSON."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=args.max_tokens,
    )
    return response.choices[0].message.content or "", time.perf_counter() - start


async def run_judging(args: argparse.Namespace) -> None:
    from openai import AsyncOpenAI

    answers = load_answers(Path(args.answers))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "judge_results.jsonl"
    done = set()
    if result_path.exists() and not args.overwrite:
        with result_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    data = json.loads(line)
                    done.add((str(data["query_id"]), data["variant_a"], data["variant_b"], data["answer_order"]))

    client = AsyncOpenAI(base_url=args.judge_base_url, api_key=args.judge_api_key)
    semaphore = asyncio.Semaphore(args.max_concurrency)
    query_ids = sorted({query_id for query_id, _variant in answers.keys()}, key=lambda x: int(x) if x.isdigit() else x)

    async def evaluate_record(query_id: str, variant: str, order_name: str, order: list[str]) -> dict[str, Any] | None:
        baseline_record = answers.get((query_id, args.baseline))
        variant_record = answers.get((query_id, variant))
        if not baseline_record or not variant_record:
            return None
        key = (query_id, args.baseline, variant, order_name)
        if key in done:
            return None
        by_variant = {args.baseline: baseline_record, variant: variant_record}
        async with semaphore:
            raw, latency = await judge_one(
                client,
                args,
                baseline_record["query"],
                by_variant[order[0]]["answer"],
                by_variant[order[1]]["answer"],
            )
        parsed, parse_error = parse_judge_json(raw)
        criterion_winners = {}
        explanations = {}
        if parsed:
            for criterion in CRITERIA:
                item = parsed.get(criterion, {})
                criterion_winners[criterion] = map_winner(item.get("Winner"), order)
                explanations[criterion] = item.get("Explanation", "")
        return {
            "query_id": query_id,
            "query": baseline_record["query"],
            "variant_a": args.baseline,
            "variant_b": variant,
            "answer_order": order_name,
            "order": order,
            "criterion_winners": criterion_winners,
            "overall_winner": criterion_winners.get("Overall Winner", "Unknown"),
            "judge_explanation": explanations.get("Overall Winner", ""),
            "raw_judge_response": raw,
            "parse_error": parse_error,
            "latency_seconds": latency,
        }

    tasks = []
    for query_id in query_ids:
        for variant in args.variants:
            tasks.append(evaluate_record(query_id, variant, "baseline_first", [args.baseline, variant]))
            tasks.append(evaluate_record(query_id, variant, "variant_first", [variant, args.baseline]))

    mode = "w" if args.overwrite else "a"
    rows = []
    with result_path.open(mode, encoding="utf-8") as handle:
        for task in asyncio.as_completed(tasks):
            record = await task
            if record is None:
                continue
            rows.append(record)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
    write_summary(output_dir, list(load_result_rows(result_path)))
    print(f"Wrote judge results to {result_path}")


def load_result_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_summary(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["variant_b"]].append(row)
    csv_rows = []
    md = ["# Pairwise Answer Judge Summary", ""]
    for variant, items in sorted(grouped.items()):
        valid = [item for item in items if not item.get("parse_error")]
        wins = sum(1 for item in valid if item.get("overall_winner") == variant)
        losses = sum(1 for item in valid if item.get("overall_winner") == item.get("variant_a"))
        ties = sum(1 for item in valid if item.get("overall_winner") == "Tie")
        total = len(valid)
        latencies = [
            item["latency_seconds"]
            for item in valid
            if isinstance(item.get("latency_seconds"), (int, float))
        ]
        row = {
            "variant": variant,
            "valid_comparisons": total,
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "win_rate": wins / total if total else 0.0,
            "loss_rate": losses / total if total else 0.0,
            "tie_rate": ties / total if total else 0.0,
            "mean_latency_seconds": statistics.mean(latencies) if latencies else 0.0,
        }
        csv_rows.append(row)
        md.extend(
            [
                f"## {variant}",
                f"- valid comparisons: {total}",
                f"- wins/losses/ties: {wins}/{losses}/{ties}",
                f"- win rate: {row['win_rate']:.4f}",
                "",
            ]
        )
    with (output_dir / "judge_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0].keys()) if csv_rows else ["variant"])
        writer.writeheader()
        writer.writerows(csv_rows)
    (output_dir / "judge_summary.md").write_text("\n".join(md), encoding="utf-8")


def main() -> None:
    args = parse_args()
    asyncio.run(run_judging(args))


if __name__ == "__main__":
    main()
