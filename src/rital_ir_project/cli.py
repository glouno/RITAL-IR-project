"""Command-line interface for the HiRAG project scaffold."""

from __future__ import annotations

import argparse
from pathlib import Path

from .constants import ALL_DATASETS, MAIN_PAPER_DATASETS, OPTIONAL_DATASETS, PLAN_TEXT
from .datasets import DEFAULT_DATA_DIR, collect_dataset_stats, download_ultradomain_dataset, iter_jsonl
from .pipeline import HiRAGScaffold, load_ultradomain_documents


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="HiRAG paper helper CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    datasets_parser = subparsers.add_parser("datasets", help="List, download, or inspect datasets")
    datasets_subparsers = datasets_parser.add_subparsers(dest="datasets_command", required=True)

    datasets_subparsers.add_parser("list", help="List datasets used by the paper")

    download_parser = datasets_subparsers.add_parser("download", help="Download UltraDomain subsets")
    download_parser.add_argument(
        "names",
        nargs="*",
        default=[spec.name for spec in MAIN_PAPER_DATASETS],
        help="Dataset names to download.",
    )
    download_parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_DATA_DIR),
        help="Directory where JSONL files should be stored.",
    )

    stats_parser = datasets_subparsers.add_parser("stats", help="Show basic dataset stats")
    stats_parser.add_argument("path", help="Path to a JSONL dataset file.")
    stats_parser.add_argument("--limit", type=int, default=None, help="Optional row limit.")

    subparsers.add_parser("plan", help="Print the recommended reproduction plan")

    demo_parser = subparsers.add_parser("demo", help="Run the local HiRAG-style scaffold on a small slice")
    demo_parser.add_argument("--dataset", required=True, help="Path to an UltraDomain JSONL file.")
    demo_parser.add_argument("--limit", type=int, default=5, help="Number of contexts to index.")
    demo_parser.add_argument("--query", default=None, help="Optional query override.")
    demo_parser.add_argument("--top-k", type=int, default=10, help="Number of local entities to retrieve.")
    demo_parser.add_argument("--top-m", type=int, default=3, help="Number of key entities per community.")

    return parser


def handle_datasets_list() -> int:
    print("Main paper datasets:")
    for spec in MAIN_PAPER_DATASETS:
        print(f"- {spec.name}: {spec.url}")
        print(f"  {spec.note}")

    print("\nOptional appendix datasets:")
    for spec in OPTIONAL_DATASETS:
        print(f"- {spec.name}: {spec.url}")
        print(f"  {spec.note}")
    return 0


def handle_datasets_download(names: list[str], output_dir: str) -> int:
    for name in names:
        spec = ALL_DATASETS.get(name)
        if spec is None:
            raise SystemExit(f"Unknown dataset: {name}")
        if not spec.url.endswith(".jsonl"):
            raise SystemExit(f"{name} is not downloadable from this command. Use the source URL instead: {spec.url}")
        path = download_ultradomain_dataset(name, output_dir)
        print(path)
    return 0


def handle_datasets_stats(path: str, limit: int | None) -> int:
    stats = collect_dataset_stats(path, limit=limit)
    for key, value in stats.items():
        print(f"{key}: {value}")
    return 0


def handle_demo(dataset_path: str, limit: int, query: str | None, top_k: int, top_m: int) -> int:
    documents, default_query = load_ultradomain_documents(iter_jsonl(dataset_path), limit=limit)
    if not documents:
        raise SystemExit(f"No documents found in {dataset_path}")

    scaffold = HiRAGScaffold()
    artifacts = scaffold.index_documents(documents)
    effective_query = query or default_query or "Summarize the main topic of the indexed documents."

    print("Layer statistics:")
    for stat in artifacts.layer_statistics:
        print(
            f"- layer={stat.layer} input_entities={stat.input_entities} "
            f"clusters={stat.cluster_count} sparsity={stat.cluster_sparsity:.4f} "
            f"change_rate={stat.change_rate:.4f}"
        )

    print("\nCommunities:")
    for community in artifacts.communities[:10]:
        print(f"- {community.id} (level={community.level}) :: {community.summary}")

    context = scaffold.query(effective_query, top_k=top_k, top_m=top_m)
    print("\nRetrieved context:")
    print(context.render())
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "datasets":
        if args.datasets_command == "list":
            return handle_datasets_list()
        if args.datasets_command == "download":
            return handle_datasets_download(args.names, args.output_dir)
        if args.datasets_command == "stats":
            return handle_datasets_stats(args.path, args.limit)

    if args.command == "plan":
        print(PLAN_TEXT)
        return 0

    if args.command == "demo":
        return handle_demo(
            dataset_path=args.dataset,
            limit=args.limit,
            query=args.query,
            top_k=args.top_k,
            top_m=args.top_m,
        )

    parser.error(f"Unhandled command: {args.command}")
    return 2
