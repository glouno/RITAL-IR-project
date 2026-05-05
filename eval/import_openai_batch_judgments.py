#!/usr/bin/env python3
"""Convert OpenAI Batch judge outputs into judge_results.jsonl."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.pairwise_answer_judge import CRITERIA, map_winner, parse_judge_json, write_summary


def parse_args() -> argparse.Namespace:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description="Import OpenAI Batch pairwise judge results")
    parser.add_argument("--batch-output", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--output-dir", default=f".runs/answer_eval/judge_import_{timestamp}")
    return parser.parse_args()


def load_metadata(path: Path) -> dict[str, dict[str, Any]]:
    metadata = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                data = json.loads(line)
                metadata[str(data["custom_id"])] = data
    return metadata


def extract_response_text(row: dict[str, Any]) -> tuple[str, str | None]:
    if row.get("error"):
        return "", json.dumps(row["error"], ensure_ascii=False)
    response = row.get("response") or {}
    body = response.get("body") or {}
    status_code = response.get("status_code")
    if status_code and int(status_code) >= 400:
        return "", json.dumps(body, ensure_ascii=False)
    try:
        return body["choices"][0]["message"]["content"] or "", None
    except (KeyError, IndexError, TypeError) as exc:
        return "", f"{type(exc).__name__}: unable to extract judge content"


def convert_rows(batch_output: Path, metadata: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    with batch_output.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            batch_row = json.loads(line)
            custom_id = str(batch_row.get("custom_id", ""))
            meta = metadata.get(custom_id)
            if not meta:
                rows.append(
                    {
                        "query_id": "",
                        "query": "",
                        "variant_a": "",
                        "variant_b": "",
                        "answer_order": "",
                        "order": [],
                        "criterion_winners": {},
                        "overall_winner": "Unknown",
                        "judge_explanation": "",
                        "raw_judge_response": "",
                        "parse_error": f"missing metadata for {custom_id}",
                        "latency_seconds": None,
                        "custom_id": custom_id,
                    }
                )
                continue
            raw, response_error = extract_response_text(batch_row)
            parsed, parse_error = (None, response_error)
            if response_error is None:
                parsed, parse_error = parse_judge_json(raw)
            criterion_winners = {}
            explanations = {}
            if parsed:
                for criterion in CRITERIA:
                    item = parsed.get(criterion, {})
                    criterion_winners[criterion] = map_winner(item.get("Winner"), meta["order"])
                    explanations[criterion] = item.get("Explanation", "")
            rows.append(
                {
                    "query_id": meta["query_id"],
                    "query": meta["query"],
                    "variant_a": meta["variant_a"],
                    "variant_b": meta["variant_b"],
                    "answer_order": meta["answer_order"],
                    "order": meta["order"],
                    "criterion_winners": criterion_winners,
                    "overall_winner": criterion_winners.get("Overall Winner", "Unknown"),
                    "judge_explanation": explanations.get("Overall Winner", ""),
                    "raw_judge_response": raw,
                    "parse_error": parse_error,
                    "latency_seconds": None,
                    "custom_id": custom_id,
                }
            )
    rows.sort(
        key=lambda item: (
            int(item["query_id"]) if str(item["query_id"]).isdigit() else str(item["query_id"]),
            item["variant_b"],
            item["answer_order"],
        )
    )
    return rows


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = convert_rows(Path(args.batch_output), load_metadata(Path(args.metadata)))
    result_path = output_dir / "judge_results.jsonl"
    with result_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_summary(output_dir, rows)
    print(f"Wrote imported judge results to {result_path}")


if __name__ == "__main__":
    main()
