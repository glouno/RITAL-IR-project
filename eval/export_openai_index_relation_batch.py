#!/usr/bin/env python3
"""Export second-stage HiRAG graph-indexing relation extraction as OpenAI Batch JSONL."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import tiktoken

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.export_openai_index_entity_batch import DEFAULT_CONTEXT_FILE, build_chunks, load_contexts
from hirag.regimes import resolve_prompts, resolve_stage_max_tokens


def parse_args() -> argparse.Namespace:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(
        description=(
            "Create OpenAI Batch JSONL for the second HiRAG indexing stage: "
            "relationship extraction over chunks using imported entity results."
        )
    )
    parser.add_argument("--context-file", default=DEFAULT_CONTEXT_FILE)
    parser.add_argument("--entity-results", required=True)
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument("--output-dir", default=f".runs/openai_index_batch/relation_extract_{timestamp}")
    parser.add_argument("--batch-file-name", default="relation_extract_requests.jsonl")
    parser.add_argument("--prompt-regime", choices=["baseline", "lean", "ultra_lean"], default="ultra_lean")
    parser.add_argument("--chunk-token-size", type=int, default=1200)
    parser.add_argument("--chunk-overlap-token-size", type=int, default=100)
    parser.add_argument("--tiktoken-model-name", default="gpt-4o")
    parser.add_argument("--entity-summary-to-max-tokens", type=int, default=500)
    parser.add_argument("--best-model-max-token-size", type=int, default=8192)
    parser.add_argument("--max-completion-tokens", type=int, default=None)
    parser.add_argument("--completion-token-param", choices=["max_completion_tokens", "max_tokens", "none"], default="max_completion_tokens")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_entity_results(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            chunk_id = str(row.get("chunk_id") or "").strip()
            if chunk_id:
                rows[chunk_id] = row
    return rows


def build_relation_prompt(chunk_content: str, entity_names: list[str], prompts: dict[str, Any]) -> str:
    context_base = {
        "tuple_delimiter": prompts["DEFAULT_TUPLE_DELIMITER"],
        "record_delimiter": prompts["DEFAULT_RECORD_DELIMITER"],
        "completion_delimiter": prompts["DEFAULT_COMPLETION_DELIMITER"],
        "entities": ",".join(entity_names),
    }
    return str(prompts["hi_relation_extraction"]).format(
        **context_base,
        input_text=chunk_content,
    )


def request_body(args: argparse.Namespace, prompt: str, max_completion_tokens: int | None) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
    }
    if args.completion_token_param != "none" and max_completion_tokens:
        body[args.completion_token_param] = max_completion_tokens
    if args.temperature is not None:
        body["temperature"] = args.temperature
    return body


def entity_names_for_chunk(entity_row: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for entity in entity_row.get("entities", []):
        name = str(entity.get("entity_name", "")).strip()
        if name and name not in names:
            names.append(name)
    return names


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    batch_path = output_dir / args.batch_file_name
    metadata_path = output_dir / "relation_extract_metadata.jsonl"
    manifest_path = output_dir / "manifest.json"
    if batch_path.exists() and not args.overwrite:
        raise FileExistsError(f"{batch_path} exists; pass --overwrite to replace it")

    prompts = resolve_prompts(args.prompt_regime)
    stage_max_tokens = resolve_stage_max_tokens(
        args.prompt_regime,
        best_model_max_token_size=args.best_model_max_token_size,
        entity_summary_to_max_tokens=args.entity_summary_to_max_tokens,
        overrides={},
    )
    max_completion_tokens = args.max_completion_tokens or stage_max_tokens.get("relation_extract")
    chunks = build_chunks(load_contexts(Path(args.context_file)), args)
    entity_results = load_entity_results(Path(args.entity_results))
    encoder = tiktoken.encoding_for_model(args.tiktoken_model_name)
    total_prompt_tokens = 0
    max_prompt_tokens = 0
    written = 0
    skipped_empty_entities = 0
    missing_chunks = 0

    with batch_path.open("w", encoding="utf-8") as batch_handle, metadata_path.open("w", encoding="utf-8") as meta_handle:
        for chunk_id, entity_row in entity_results.items():
            chunk = chunks.get(chunk_id)
            if chunk is None:
                missing_chunks += 1
                continue
            entity_names = entity_names_for_chunk(entity_row)
            if not entity_names:
                skipped_empty_entities += 1
                continue
            prompt = build_relation_prompt(str(chunk["content"]), entity_names, prompts)
            prompt_tokens = len(encoder.encode(prompt))
            total_prompt_tokens += prompt_tokens
            max_prompt_tokens = max(max_prompt_tokens, prompt_tokens)
            custom_id = f"index_relation|{chunk_id}"
            batch_handle.write(
                json.dumps(
                    {
                        "custom_id": custom_id,
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": request_body(args, prompt, max_completion_tokens),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            meta_handle.write(
                json.dumps(
                    {
                        "custom_id": custom_id,
                        "chunk_id": chunk_id,
                        "full_doc_id": chunk.get("full_doc_id"),
                        "chunk_order_index": chunk.get("chunk_order_index"),
                        "chunk_tokens": chunk.get("tokens"),
                        "entity_count": len(entity_names),
                        "entities": entity_names,
                        "prompt_tokens": prompt_tokens,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            written += 1

    manifest = {
        "batch_file": str(batch_path),
        "metadata_file": str(metadata_path),
        "context_file": args.context_file,
        "entity_results_file": args.entity_results,
        "model": args.model,
        "prompt_regime": args.prompt_regime,
        "request_count": written,
        "input_entity_chunk_count": len(entity_results),
        "skipped_empty_entities": skipped_empty_entities,
        "missing_chunks": missing_chunks,
        "completion_token_param": args.completion_token_param,
        "max_completion_tokens_per_request": max_completion_tokens,
        "total_prompt_tokens_estimate": total_prompt_tokens,
        "mean_prompt_tokens_estimate": total_prompt_tokens / written if written else 0,
        "max_prompt_tokens_estimate": max_prompt_tokens,
        "upload_hint": (
            "Upload with purpose='batch', then create a batch with "
            "endpoint='/v1/chat/completions' and completion_window='24h'."
        ),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {written} relation extraction requests to {batch_path}")
    print(f"Wrote metadata to {metadata_path}")
    print(f"Wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
