#!/usr/bin/env python3
"""Export HiRAG community-report generation as OpenAI Batch JSONL."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import networkx as nx
import tiktoken

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hirag._op import _pack_single_community_describe
from hirag._storage import NetworkXStorage
from hirag.regimes import resolve_prompts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create OpenAI Batch requests for HiRAG community reports.")
    parser.add_argument("--graphml", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument("--prompt-regime", choices=["baseline", "lean", "ultra_lean"], default="baseline")
    parser.add_argument("--level", type=int, default=None, help="If omitted, export all levels without prior reports.")
    parser.add_argument("--existing-reports", default=None)
    parser.add_argument("--input-max-tokens", type=int, default=12000)
    parser.add_argument("--max-completion-tokens", type=int, default=2048)
    parser.add_argument("--completion-token-param", choices=["max_completion_tokens", "max_tokens", "none"], default="max_completion_tokens")
    parser.add_argument("--tiktoken-model-name", default="gpt-4o")
    parser.add_argument("--batch-file-name", default="community_report_requests.jsonl")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def request_body(args: argparse.Namespace, prompt: str) -> dict[str, Any]:
    body: dict[str, Any] = {"model": args.model, "messages": [{"role": "user", "content": prompt}]}
    if args.completion_token_param != "none":
        body[args.completion_token_param] = args.max_completion_tokens
    return body


def load_existing_reports(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object keyed by community id")
    return data


async def build_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    prompts = resolve_prompts(args.prompt_regime)
    graph = nx.read_graphml(args.graphml)
    storage = NetworkXStorage(namespace="chunk_entity_relation", global_config={"working_dir": str(Path(args.output_dir))})
    storage._graph = graph
    schema = await storage.community_schema()
    existing_reports = load_existing_reports(args.existing_reports)
    levels = sorted({int(c["level"]) for c in schema.values()}, reverse=True)
    target_levels = [args.level] if args.level is not None else levels
    encoder = tiktoken.encoding_for_model(args.tiktoken_model_name)
    batch_rows: list[dict[str, Any]] = []
    metadata_rows: list[dict[str, Any]] = []
    total_prompt_tokens = 0
    for community_id, community in sorted(schema.items(), key=lambda item: (int(item[1]["level"]), str(item[0]))):
        if int(community["level"]) not in target_levels:
            continue
        packed = await _pack_single_community_describe(
            storage,
            community,
            max_token_size=args.input_max_tokens,
            already_reports=existing_reports,
            global_config={"addon_params": {}},
        )
        prompt = str(prompts["community_report"]).format(input_text=packed)
        prompt_tokens = len(encoder.encode(prompt))
        total_prompt_tokens += prompt_tokens
        custom_id = f"community_report|{community_id}"
        batch_rows.append({"custom_id": custom_id, "method": "POST", "url": "/v1/chat/completions", "body": request_body(args, prompt)})
        metadata_rows.append(
            {
                "custom_id": custom_id,
                "community_id": community_id,
                "level": community["level"],
                "community": community,
                "prompt_tokens": prompt_tokens,
            }
        )
    manifest = {
        "graphml": args.graphml,
        "model": args.model,
        "prompt_regime": args.prompt_regime,
        "levels": levels,
        "exported_levels": target_levels,
        "request_count": len(batch_rows),
        "existing_reports": args.existing_reports,
        "input_max_tokens": args.input_max_tokens,
        "max_completion_tokens_per_request": args.max_completion_tokens,
        "total_prompt_tokens_estimate": total_prompt_tokens,
    }
    return batch_rows, metadata_rows, manifest


async def amain() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    batch_path = output_dir / args.batch_file_name
    metadata_path = output_dir / "community_report_metadata.jsonl"
    if batch_path.exists() and not args.overwrite:
        raise FileExistsError(f"{batch_path} exists; pass --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)
    batch_rows, metadata_rows, manifest = await build_rows(args)
    with batch_path.open("w", encoding="utf-8") as batch_handle:
        for row in batch_rows:
            batch_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with metadata_path.open("w", encoding="utf-8") as meta_handle:
        for row in metadata_rows:
            meta_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest.update({"batch_file": str(batch_path), "metadata_file": str(metadata_path)})
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {len(batch_rows)} community report requests to {batch_path}")


def main() -> None:
    asyncio.run(amain())


if __name__ == "__main__":
    main()
