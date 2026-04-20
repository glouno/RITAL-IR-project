import argparse
import copy
import importlib.util
import json
import time
from pathlib import Path
from typing import Any

from hirag import QueryParam
from hirag.regimes import STAGE_NAMES, expand_benchmark_variants
from hirag.telemetry import new_telemetry_session, summarize_telemetry


REPO_ROOT = Path(__file__).resolve().parents[1]
MAIN_SPEC = importlib.util.spec_from_file_location("hirag_repo_main", REPO_ROOT / "main.py")
if MAIN_SPEC is None or MAIN_SPEC.loader is None:
    raise RuntimeError("Unable to load top-level main.py")
REPO_MAIN = importlib.util.module_from_spec(MAIN_SPEC)
MAIN_SPEC.loader.exec_module(REPO_MAIN)

build_graph_runtime = REPO_MAIN.build_graph_runtime
create_graph_working_dir = REPO_MAIN.create_graph_working_dir
load_context_input = REPO_MAIN.load_context_input
parse_int_or_none = REPO_MAIN.parse_int_or_none
resolve_context_file = REPO_MAIN.resolve_context_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark HiRAG prompt regimes")
    parser.add_argument("--context-file", required=True, help="Path to a context JSON file")
    parser.add_argument(
        "--variants",
        nargs="*",
        default=["baseline", "baseline_no_glean", "lean", "ultra_lean"],
        help="Named benchmark variants to run",
    )
    parser.add_argument(
        "--limit-contexts",
        type=int,
        default=None,
        help="Optional limit on number of contexts loaded from the JSON list",
    )
    parser.add_argument("--repeat", type=int, default=1, help="Repeat count per variant")
    parser.add_argument("--output-json", required=True, help="Path for benchmark JSON output")
    parser.add_argument("--knowledge-graph-path", required=True, help="Directory for graph outputs")
    parser.add_argument("--graph-vis-path", required=True, help="Directory for graph visualization outputs")
    parser.add_argument("--base-url", required=True, help="Local vLLM OpenAI-compatible URL")
    parser.add_argument("--api-key", required=True, help="API key for the local endpoint")
    parser.add_argument("--chat-model", default=None, help="Optional explicit chat model id")
    parser.add_argument(
        "--datasets-path",
        default=".",
        help="Root directory used when resolving relative dataset paths",
    )
    parser.add_argument("--graph-name", default="benchmark", help="Graph name prefix")
    parser.add_argument(
        "--embed-model",
        required=True,
        help="FastEmbed model name used for indexing",
    )
    parser.add_argument("--embed-dim", type=int, required=True, help="Embedding dimension")
    parser.add_argument("--max-token-size", type=int, required=True, help="Embedding metadata max token size")
    parser.add_argument("--embedding-batch-num", type=int, default=6, help="HiRAG embedding batch size")
    parser.add_argument(
        "--embedding-func-max-async",
        type=int,
        default=8,
        help="Maximum parallel embedding calls",
    )
    parser.add_argument("--fastembed-threads", type=int, default=None, help="FastEmbed threads")
    parser.add_argument("--fastembed-parallel", type=int, default=None, help="FastEmbed parallel workers")
    parser.add_argument(
        "--fastembed-cache-path",
        default=None,
        help="FastEmbed cache path",
    )
    parser.add_argument(
        "--disable-hierachical-mode",
        action="store_true",
        default=False,
        help="Disable hierarchical retrieval mode",
    )
    parser.add_argument(
        "--disable-naive-rag",
        action="store_true",
        default=False,
        help="Disable naive RAG mode",
    )
    parser.add_argument(
        "--disable-llm-cache",
        action="store_true",
        default=True,
        help="Disable HiRAG LLM cache; benchmark default is disabled",
    )
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    parser.add_argument(
        "--smoke-query-file",
        default=None,
        help="Optional query file used for post-index smoke queries",
    )
    parser.add_argument(
        "--smoke-query-limit",
        type=int,
        default=None,
        help="Optional maximum number of smoke queries to run",
    )
    parser.add_argument(
        "--stage-max-token",
        action="append",
        default=[],
        help="Optional stage override in stage=value form; may be repeated",
    )
    return parser.parse_args()


def parse_stage_max_token_overrides(values: list[str]) -> dict[str, int]:
    overrides: dict[str, int] = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"Invalid stage max token override: {item}")
        stage, raw_value = item.split("=", 1)
        stage = stage.strip()
        if stage not in STAGE_NAMES:
            raise ValueError(f"Unknown stage name: {stage}")
        value = parse_int_or_none(raw_value.strip())
        if value is None:
            raise ValueError(f"Invalid stage token value: {item}")
        overrides[stage] = value
    return overrides


def limit_context_input(context_input: str | list[str], limit_contexts: int | None) -> str | list[str]:
    if limit_contexts is None or not isinstance(context_input, list):
        return context_input
    return context_input[:limit_contexts]


def load_smoke_queries(smoke_query_file: str | None, limit: int | None) -> list[str]:
    if not smoke_query_file:
        return []

    path = Path(smoke_query_file)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Smoke query file not found: {path}")

    queries: list[str] = []
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                query = data.get("input") or data.get("query") or data.get("question")
                if isinstance(query, str) and query.strip():
                    queries.append(query.strip())
    elif path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            for item in data:
                if isinstance(item, str) and item.strip():
                    queries.append(item.strip())
                elif isinstance(item, dict):
                    query = item.get("input") or item.get("query") or item.get("question")
                    if isinstance(query, str) and query.strip():
                        queries.append(query.strip())
    else:
        queries = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    if limit is not None:
        queries = queries[:limit]
    return queries


def collect_graph_metrics(graph_func) -> dict[str, int | None]:
    graph = getattr(graph_func.chunk_entity_relation_graph, "_graph", None)
    node_count = graph.number_of_nodes() if graph is not None else None
    edge_count = graph.number_of_edges() if graph is not None else None
    community_store = getattr(graph_func.community_reports, "_data", None)
    community_count = len(community_store) if isinstance(community_store, dict) else None
    chunk_store = getattr(graph_func.text_chunks, "_data", None)
    chunk_count = len(chunk_store) if isinstance(chunk_store, dict) else None
    return {
        "chunk_count": chunk_count,
        "entity_count": node_count,
        "relation_count": edge_count,
        "community_count": community_count,
    }


def make_run_record(
    *,
    variant_name: str,
    prompt_regime: str,
    entity_extract_max_gleaning: int,
    context_count: int,
    graph_dir: Path,
    indexing_seconds: float,
    telemetry_payload: dict[str, Any],
    graph_metrics: dict[str, int | None],
    smoke_query_results: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    summary = telemetry_payload.get("summary", {})
    return {
        "variant": variant_name,
        "prompt_regime": prompt_regime,
        "entity_extract_max_gleaning": entity_extract_max_gleaning,
        "context_count": context_count,
        "graph_dir": str(graph_dir),
        "indexing_seconds": indexing_seconds,
        "chunk_count": graph_metrics.get("chunk_count"),
        "entity_count": graph_metrics.get("entity_count"),
        "relation_count": graph_metrics.get("relation_count"),
        "community_count": graph_metrics.get("community_count"),
        "cache_hit_count": summary.get("cache_hits", 0),
        "per_stage": summary.get("stages", {}),
        "telemetry": telemetry_payload,
        "smoke_queries": smoke_query_results or [],
    }


def run_variant(
    base_args: argparse.Namespace,
    *,
    variant: dict[str, Any],
    context_input: str | list[str],
    stage_max_token_overrides: dict[str, int],
    smoke_queries: list[str],
    run_index: int,
) -> dict[str, Any]:
    run_args = copy.deepcopy(base_args)
    run_args.prompt_regime = variant["prompt_regime"]
    run_args.entity_extract_max_gleaning = variant["entity_extract_max_gleaning"]
    run_args.disable_llm_cache = True
    run_args.query = None
    run_args.query_mode = "hi"
    run_args.benchmark_label = f"{variant['name']}-run{run_index + 1}"
    run_args.graph_name = f"{base_args.graph_name}_{variant['name']}_run{run_index + 1}"

    graph_working_dir = create_graph_working_dir(run_args)
    telemetry = new_telemetry_session(run_args.benchmark_label)
    graph_func, runtime = build_graph_runtime(
        run_args,
        graph_working_dir=graph_working_dir,
        telemetry=telemetry,
        stage_max_token_overrides=stage_max_token_overrides,
    )

    context_count = len(context_input) if isinstance(context_input, list) else 1
    print(
        f"[{variant['name']}] run={run_index + 1} "
        f"prompt_regime={run_args.prompt_regime} "
        f"gleaning={graph_func.entity_extract_max_gleaning} "
        f"chat_model={runtime['chat_model']}"
    )

    start = time.perf_counter()
    graph_func.insert(context_input)
    indexing_seconds = time.perf_counter() - start

    smoke_query_results: list[dict[str, Any]] = []
    for smoke_query in smoke_queries:
        query_start = time.perf_counter()
        graph_func.query(smoke_query, param=QueryParam(mode="hi"))
        smoke_query_results.append(
            {
                "query": smoke_query,
                "seconds": time.perf_counter() - query_start,
            }
        )

    telemetry_payload = summarize_telemetry(telemetry)
    graph_metrics = collect_graph_metrics(graph_func)
    run_record = make_run_record(
        variant_name=variant["name"],
        prompt_regime=run_args.prompt_regime,
        entity_extract_max_gleaning=graph_func.entity_extract_max_gleaning,
        context_count=context_count,
        graph_dir=graph_working_dir,
        indexing_seconds=indexing_seconds,
        telemetry_payload=telemetry_payload,
        graph_metrics=graph_metrics,
        smoke_query_results=smoke_query_results,
    )
    print(
        f"[{variant['name']}] run={run_index + 1} "
        f"indexing_seconds={indexing_seconds:.2f} "
        f"completion_tokens={telemetry_payload.get('summary', {}).get('completion_tokens', 0)}"
    )
    return run_record


def main() -> None:
    args = parse_args()
    context_file = resolve_context_file(args)
    context_input = limit_context_input(load_context_input(context_file), args.limit_contexts)
    variants = expand_benchmark_variants(args.variants)
    stage_max_token_overrides = parse_stage_max_token_overrides(args.stage_max_token)
    smoke_queries = load_smoke_queries(args.smoke_query_file, args.smoke_query_limit)

    runs = []
    for variant in variants:
        for run_index in range(args.repeat):
            runs.append(
                run_variant(
                    args,
                    variant=variant,
                    context_input=context_input,
                    stage_max_token_overrides=stage_max_token_overrides,
                    smoke_queries=smoke_queries,
                    run_index=run_index,
                )
            )

    report = {
        "dataset": str(context_file),
        "context_count": len(context_input) if isinstance(context_input, list) else 1,
        "variants": [variant["name"] for variant in variants],
        "repeat": args.repeat,
        "smoke_query_count": len(smoke_queries),
        "runs": runs,
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Benchmark results written to {output_path}")


if __name__ == "__main__":
    main()
