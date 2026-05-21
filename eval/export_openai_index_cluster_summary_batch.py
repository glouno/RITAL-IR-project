#!/usr/bin/env python3
"""Export cluster-summary prompts for clustered HiRAG graphs as OpenAI Batch JSONL."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import networkx as nx
import tiktoken

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hirag._utils import truncate_list_by_token_size
from hirag.regimes import resolve_prompts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create OpenAI Batch requests for HiRAG cluster summaries.")
    parser.add_argument("--graphml", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument("--prompt-regime", choices=["baseline", "lean", "ultra_lean"], default="baseline")
    parser.add_argument("--level", type=int, default=None)
    parser.add_argument("--input-max-tokens", type=int, default=12000)
    parser.add_argument("--max-completion-tokens", type=int, default=2048)
    parser.add_argument("--completion-token-param", choices=["max_completion_tokens", "max_tokens", "none"], default="max_completion_tokens")
    parser.add_argument("--tiktoken-model-name", default="gpt-4o")
    parser.add_argument("--batch-file-name", default="cluster_summary_requests.jsonl")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def graph_clusters(graph: nx.Graph) -> dict[tuple[int, str], list[str]]:
    clusters: dict[tuple[int, str], list[str]] = {}
    for node, data in graph.nodes(data=True):
        if not data.get("clusters"):
            continue
        for cluster in json.loads(data["clusters"]):
            key = (int(cluster["level"]), str(cluster["cluster"]))
            clusters.setdefault(key, []).append(str(node))
    return clusters


def request_body(args: argparse.Namespace, prompt: str) -> dict[str, Any]:
    body: dict[str, Any] = {"model": args.model, "messages": [{"role": "user", "content": prompt}]}
    if args.completion_token_param != "none":
        body[args.completion_token_param] = args.max_completion_tokens
    return body


def build_cluster_prompt(graph: nx.Graph, nodes: list[str], prompts: dict[str, Any], input_max_tokens: int) -> str:
    descriptions = [
        f"({node}, {graph.nodes[node].get('description', '')})"
        for node in sorted(nodes)
        if graph.has_node(node)
    ]
    descriptions = truncate_list_by_token_size(descriptions, key=lambda item: item, max_token_size=input_max_tokens)
    context = {
        "tuple_delimiter": prompts["DEFAULT_TUPLE_DELIMITER"],
        "record_delimiter": prompts["DEFAULT_RECORD_DELIMITER"],
        "completion_delimiter": prompts["DEFAULT_COMPLETION_DELIMITER"],
        "meta_attribute_list": prompts["META_ENTITY_TYPES"],
        "entity_description_list": ",".join(descriptions),
    }
    return str(prompts["summary_clusters"]).format(**context)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    batch_path = output_dir / args.batch_file_name
    metadata_path = output_dir / "cluster_summary_metadata.jsonl"
    if batch_path.exists() and not args.overwrite:
        raise FileExistsError(f"{batch_path} exists; pass --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)
    graph = nx.read_graphml(args.graphml)
    prompts = resolve_prompts(args.prompt_regime)
    clusters = graph_clusters(graph)
    levels = sorted({level for level, _cluster_id in clusters})
    target_levels = [args.level] if args.level is not None else levels
    encoder = tiktoken.encoding_for_model(args.tiktoken_model_name)
    total_prompt_tokens = 0
    rows = 0
    with batch_path.open("w", encoding="utf-8") as batch_handle, metadata_path.open("w", encoding="utf-8") as meta_handle:
        for (level, cluster_id), nodes in sorted(clusters.items()):
            if level not in target_levels:
                continue
            prompt = build_cluster_prompt(graph, nodes, prompts, args.input_max_tokens)
            prompt_tokens = len(encoder.encode(prompt))
            total_prompt_tokens += prompt_tokens
            custom_id = f"cluster_summary|{level}|{cluster_id}"
            batch_handle.write(json.dumps({"custom_id": custom_id, "method": "POST", "url": "/v1/chat/completions", "body": request_body(args, prompt)}, ensure_ascii=False) + "\n")
            meta_handle.write(json.dumps({"custom_id": custom_id, "level": level, "cluster_id": cluster_id, "nodes": nodes, "prompt_tokens": prompt_tokens}, ensure_ascii=False) + "\n")
            rows += 1
    manifest = {
        "graphml": args.graphml,
        "batch_file": str(batch_path),
        "metadata_file": str(metadata_path),
        "model": args.model,
        "prompt_regime": args.prompt_regime,
        "levels": levels,
        "exported_levels": target_levels,
        "request_count": rows,
        "input_max_tokens": args.input_max_tokens,
        "max_completion_tokens_per_request": args.max_completion_tokens,
        "total_prompt_tokens_estimate": total_prompt_tokens,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {rows} cluster summary requests to {batch_path}")


if __name__ == "__main__":
    main()
