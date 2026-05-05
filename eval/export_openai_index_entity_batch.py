#!/usr/bin/env python3
"""Export first-stage HiRAG graph-indexing entity extraction as OpenAI Batch JSONL."""

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

from eval.eval_utils import REPO_MAIN
from hirag._op import get_chunks
from hirag._utils import compute_mdhash_id
from hirag.regimes import resolve_entity_extract_max_gleaning, resolve_prompts, resolve_stage_max_tokens


DEFAULT_CONTEXT_FILE = "eval/datasets/agriculture/agriculture_unique_contexts.json"


def parse_args() -> argparse.Namespace:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(
        description=(
            "Create OpenAI Batch JSONL for the first HiRAG indexing stage: "
            "hierarchical entity extraction over chunks."
        )
    )
    parser.add_argument("--context-file", default=DEFAULT_CONTEXT_FILE)
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument("--output-dir", default=f".runs/openai_index_batch/entity_extract_{timestamp}")
    parser.add_argument("--batch-file-name", default="entity_extract_requests.jsonl")
    parser.add_argument("--prompt-regime", choices=["baseline", "lean", "ultra_lean"], default="ultra_lean")
    parser.add_argument("--chunk-token-size", type=int, default=1200)
    parser.add_argument("--chunk-overlap-token-size", type=int, default=100)
    parser.add_argument("--tiktoken-model-name", default="gpt-4o")
    parser.add_argument("--entity-summary-to-max-tokens", type=int, default=500)
    parser.add_argument("--best-model-max-token-size", type=int, default=8192)
    parser.add_argument("--entity-extract-max-gleaning", type=int, default=None)
    parser.add_argument("--max-completion-tokens", type=int, default=None)
    parser.add_argument("--completion-token-param", choices=["max_completion_tokens", "max_tokens", "none"], default="max_completion_tokens")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_contexts(path: Path) -> list[str]:
    loaded = REPO_MAIN.load_context_input(path)
    contexts = [loaded] if isinstance(loaded, str) else loaded
    contexts = [context.strip() for context in contexts if isinstance(context, str) and context.strip()]
    if not contexts:
        raise ValueError(f"No non-empty contexts found in {path}")
    return contexts


def build_chunks(contexts: list[str], args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    docs = {
        compute_mdhash_id(context, prefix="doc-"): {"content": context}
        for context in contexts
    }
    return get_chunks(
        docs,
        overlap_token_size=args.chunk_overlap_token_size,
        max_token_size=args.chunk_token_size,
    )


def build_entity_prompt(chunk_content: str, prompts: dict[str, Any]) -> str:
    context_base = {
        "tuple_delimiter": prompts["DEFAULT_TUPLE_DELIMITER"],
        "record_delimiter": prompts["DEFAULT_RECORD_DELIMITER"],
        "completion_delimiter": prompts["DEFAULT_COMPLETION_DELIMITER"],
        "entity_types": ",".join(prompts["META_ENTITY_TYPES"]),
    }
    return str(prompts["hi_entity_extraction"]).format(
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


def stage_plan(chunks: dict[str, dict[str, Any]], entity_extract_max_gleaning: int) -> list[dict[str, Any]]:
    return [
        {
            "stage": "entity_extract",
            "batchable_now": True,
            "batch_count": 1,
            "request_count": len(chunks),
            "notes": "Independent one request per chunk. This script exports this stage.",
        },
        {
            "stage": "entity_glean_check_and_continue",
            "batchable_now": False,
            "batch_count": entity_extract_max_gleaning + 1 if entity_extract_max_gleaning > 0 else 0,
            "request_count": "up to chunks * (1 + entity_extract_max_gleaning)",
            "notes": "Depends on entity_extract responses. Recommended for Batch indexing: use ultra_lean/no gleaning first.",
        },
        {
            "stage": "relation_extract",
            "batchable_now": False,
            "batch_count": 1,
            "request_count": "one per chunk after entity output import",
            "notes": "Requires parsed entities from entity_extract for each chunk.",
        },
        {
            "stage": "cluster_summary",
            "batchable_now": False,
            "batch_count": "one per hierarchy layer",
            "request_count": "one per GMM cluster in that layer",
            "notes": "Layer N depends on imported summaries/embeddings from layer N-1.",
        },
        {
            "stage": "community_report",
            "batchable_now": False,
            "batch_count": "one per Leiden community level",
            "request_count": "one per community in that level",
            "notes": "Community levels are sequential because lower-level reports can depend on higher-level reports.",
        },
    ]


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    batch_path = output_dir / args.batch_file_name
    metadata_path = output_dir / "entity_extract_metadata.jsonl"
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
    max_completion_tokens = args.max_completion_tokens or stage_max_tokens.get("entity_extract")
    entity_extract_max_gleaning = resolve_entity_extract_max_gleaning(
        args.prompt_regime,
        args.entity_extract_max_gleaning,
    )

    contexts = load_contexts(Path(args.context_file))
    chunks = build_chunks(contexts, args)
    encoder = tiktoken.encoding_for_model(args.tiktoken_model_name)
    total_prompt_tokens = 0
    max_prompt_tokens = 0

    with batch_path.open("w", encoding="utf-8") as batch_handle, metadata_path.open("w", encoding="utf-8") as meta_handle:
        for chunk_id, chunk in chunks.items():
            prompt = build_entity_prompt(str(chunk["content"]), prompts)
            prompt_tokens = len(encoder.encode(prompt))
            total_prompt_tokens += prompt_tokens
            max_prompt_tokens = max(max_prompt_tokens, prompt_tokens)
            custom_id = f"index_entity|{chunk_id}"
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
                        "prompt_tokens": prompt_tokens,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    manifest = {
        "batch_file": str(batch_path),
        "metadata_file": str(metadata_path),
        "context_file": args.context_file,
        "model": args.model,
        "prompt_regime": args.prompt_regime,
        "chunk_count": len(chunks),
        "document_count": len(contexts),
        "chunk_token_size": args.chunk_token_size,
        "chunk_overlap_token_size": args.chunk_overlap_token_size,
        "completion_token_param": args.completion_token_param,
        "max_completion_tokens_per_request": max_completion_tokens,
        "total_prompt_tokens_estimate": total_prompt_tokens,
        "mean_prompt_tokens_estimate": total_prompt_tokens / len(chunks) if chunks else 0,
        "max_prompt_tokens_estimate": max_prompt_tokens,
        "stage_plan": stage_plan(chunks, int(entity_extract_max_gleaning or 0)),
        "upload_hint": (
            "Upload with purpose='batch', then create a batch with "
            "endpoint='/v1/chat/completions' and completion_window='24h'."
        ),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {len(chunks)} entity extraction requests to {batch_path}")
    print(f"Wrote metadata to {metadata_path}")
    print(f"Wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
