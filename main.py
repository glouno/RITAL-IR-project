import argparse
import asyncio
import importlib
import json
import logging
import os
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import tiktoken
from dotenv import load_dotenv

from hirag import HiRAG, QueryParam
from hirag._storage import NetworkXStorage
from hirag._utils import compute_args_hash, wrap_embedding_func_with_attrs
from hirag.base import BaseKVStorage
from hirag.regimes import PROMPT_REGIME_CHOICES, resolve_prompts
from hirag.telemetry import new_telemetry_session, record_llm_call, summarize_telemetry


LOGGER = logging.getLogger("HiRAG.main")


def env_str(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value is None:
        return default
    stripped = value.strip()
    if not stripped:
        return default
    return stripped


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return int(value.strip())


def env_int_or_none(name: str) -> int | None:
    value = os.getenv(name)
    if value is None or not value.strip():
        return None
    return int(value.strip())


def env_path(name: str) -> str | None:
    value = os.getenv(name)
    if value is None or not value.strip():
        return None
    return value.strip()


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default

    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def discover_local_chat_model(base_url: str, api_key: str) -> str | None:
    model_id, _ = discover_local_chat_model_info(base_url, api_key)
    return model_id


def discover_local_chat_model_info(base_url: str, api_key: str) -> tuple[str | None, int | None]:
    try:
        openai_module = importlib.import_module("openai")
        client = openai_module.OpenAI(base_url=base_url, api_key=api_key)
        models = client.models.list()
    except Exception as exc:
        LOGGER.warning("Unable to discover model from %s: %s", base_url, exc)
        return None, None

    for item in getattr(models, "data", []):
        model_id = getattr(item, "id", None)
        if isinstance(model_id, str) and model_id.strip():
            max_model_len = getattr(item, "max_model_len", None)
            return model_id.strip(), int(max_model_len) if max_model_len is not None else None
    return None, None


def parse_int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip():
        return int(value.strip())
    return None


def extract_context(item: Any) -> str | None:
    if isinstance(item, str):
        text = item.strip()
        return text if text else None

    if isinstance(item, dict):
        for key in ("context", "text", "content", "passage"):
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

    return None


def load_context_input(context_file: Path) -> str | list[str]:
    if context_file.suffix.lower() != ".json":
        context_text = context_file.read_text(encoding="utf-8").strip()
        if not context_text:
            raise ValueError(f"Context file is empty: {context_file}")
        return context_text

    with context_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(
            f"Expected a JSON list in {context_file}, got {type(data).__name__}"
        )

    contexts: list[str] = []
    for item in data:
        text = extract_context(item)
        if text:
            contexts.append(text)

    if not contexts:
        raise ValueError(f"No non-empty contexts found in {context_file}")

    return contexts


def create_fastembed_embedding(runtime: dict[str, Any]) -> Any:
    fastembed_module = importlib.import_module("fastembed")
    fastembed_model: Any | None = None
    model_init_lock = threading.Lock()

    def get_model() -> Any:
        nonlocal fastembed_model
        if fastembed_model is not None:
            return fastembed_model

        with model_init_lock:
            if fastembed_model is not None:
                return fastembed_model

            kwargs: dict[str, Any] = {"model_name": runtime["embed_model"]}
            threads = parse_int_or_none(runtime["fastembed_threads"])
            if threads is not None and threads > 0:
                kwargs["threads"] = threads
            if runtime["fastembed_cache_path"]:
                kwargs["cache_dir"] = runtime["fastembed_cache_path"]

            fastembed_model = fastembed_module.TextEmbedding(**kwargs)
        return fastembed_model

    def embed_sync(texts: list[str]) -> np.ndarray:
        model = get_model()
        parallel = parse_int_or_none(runtime["fastembed_parallel"])
        if parallel is not None and parallel > 0:
            try:
                vectors = list(model.embed(texts, parallel=parallel))
            except TypeError:
                vectors = list(model.embed(texts))
        else:
            vectors = list(model.embed(texts))
        return np.array(vectors)

    @wrap_embedding_func_with_attrs(
        embedding_dim=runtime["embed_dim"],
        max_token_size=runtime["max_token_size"],
    )
    async def fastembed_embedding(texts: list[str]) -> np.ndarray:
        return await asyncio.to_thread(embed_sync, texts)

    return fastembed_embedding


def create_vllm_chat_model(runtime: dict[str, Any]) -> Any:
    openai_module = importlib.import_module("openai")
    openai_async_client: Any | None = None
    token_encoder = tiktoken.get_encoding("cl100k_base")

    def get_async_client() -> Any:
        nonlocal openai_async_client
        if openai_async_client is not None:
            return openai_async_client

        openai_async_client = openai_module.AsyncOpenAI(
            base_url=runtime["base_url"],
            api_key=runtime["api_key"],
        )
        return openai_async_client

    async def vllm_chat_if_cache(
        prompt: str,
        system_prompt: str | None = None,
        history_messages: list[dict[str, str]] | None = None,
        **kwargs: Any,
    ) -> str:
        stage = kwargs.pop("stage", "unspecified")
        telemetry = kwargs.pop("telemetry", runtime.get("telemetry"))
        benchmark_label = kwargs.pop("benchmark_label", runtime.get("benchmark_label"))
        client = get_async_client()
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if history_messages:
            messages.extend(history_messages)
        messages.append({"role": "user", "content": prompt})
        requested_max_tokens = kwargs.get("max_tokens")
        model_max_context = runtime.get("model_max_context")
        if requested_max_tokens is not None and model_max_context:
            # Keep a conservative buffer because local token estimation can differ
            # from server-side counting by a few hundred tokens on long prompts.
            safety_buffer = 1024
            prompt_tokens_estimate = sum(
                len(token_encoder.encode(message["content"])) for message in messages
            )
            safe_max_tokens = max(
                1,
                min(
                    int(requested_max_tokens),
                    int(model_max_context) - prompt_tokens_estimate - safety_buffer,
                ),
            )
            kwargs["max_tokens"] = safe_max_tokens

        hashing_kv: BaseKVStorage | None = kwargs.pop("hashing_kv", None)
        args_hash: str | None = None
        if hashing_kv is not None:
            args_hash = compute_args_hash(runtime["chat_model"], messages)
            cached = await hashing_kv.get_by_id(args_hash)
            if cached is not None:
                cached_content = cached["return"]
                estimated_prompt_tokens = sum(
                    len(token_encoder.encode(message["content"])) for message in messages
                )
                estimated_completion_tokens = len(token_encoder.encode(cached_content))
                record_llm_call(
                    telemetry,
                    {
                        "stage": stage,
                        "benchmark_label": benchmark_label,
                        "cache_hit": True,
                        "prompt_tokens": estimated_prompt_tokens,
                        "completion_tokens": estimated_completion_tokens,
                        "total_tokens": estimated_prompt_tokens + estimated_completion_tokens,
                        "latency_seconds": 0.0,
                        "finish_reason": "cache",
                        "model": runtime["chat_model"],
                    },
                )
                return cached["return"]

        start = time.perf_counter()
        response = await client.chat.completions.create(
            model=runtime["chat_model"],
            messages=messages,
            **kwargs,
        )
        latency_seconds = time.perf_counter() - start
        content = response.choices[0].message.content or ""
        usage = getattr(response, "usage", None)
        if usage is None:
            prompt_tokens = sum(
                len(token_encoder.encode(message["content"])) for message in messages
            )
            completion_tokens = len(token_encoder.encode(content))
            total_tokens = prompt_tokens + completion_tokens
        else:
            prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
            completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
            total_tokens = int(getattr(usage, "total_tokens", 0) or 0)

        finish_reason = None
        if getattr(response, "choices", None):
            finish_reason = getattr(response.choices[0], "finish_reason", None)
        record_llm_call(
            telemetry,
            {
                "stage": stage,
                "benchmark_label": benchmark_label,
                "cache_hit": False,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "latency_seconds": latency_seconds,
                "finish_reason": finish_reason,
                "model": runtime["chat_model"],
            },
        )

        if hashing_kv is not None and args_hash is not None:
            await hashing_kv.upsert(
                {args_hash: {"return": content, "model": runtime["chat_model"]}}
            )

        return content

    return vllm_chat_if_cache


def parse_args() -> argparse.Namespace:
    load_dotenv()

    default_context_file = env_str("CONTEXT_FILE")
    default_working_dir = env_str("WORKING_DIR")
    default_base_url = env_str("VLLM_BASE_URL")
    default_api_key = env_str("VLLM_API_KEY")
    default_chat_model = env_str("VLLM_CHAT_MODEL")
    default_embed_model = env_str("EMBED_MODEL")
    default_embed_dim = env_int_or_none("EMBED_DIM")
    default_max_token_size = env_int_or_none("MAX_TOKEN_SIZE")
    default_embedding_batch_num = env_int(
        "EMBEDDING_BATCH_NUM", 6
    )
    default_embedding_func_max_async = env_int(
        "EMBEDDING_FUNC_MAX_ASYNC", 8
    )
    default_best_model_max_async = env_int("BEST_MODEL_MAX_ASYNC", 8)
    default_cheap_model_max_async = env_int("CHEAP_MODEL_MAX_ASYNC", 8)
    default_fastembed_threads = env_int_or_none("FASTEMBED_THREADS")
    default_fastembed_parallel = env_int_or_none("FASTEMBED_PARALLEL")
    default_fastembed_cache_path = env_str("FASTEMBED_CACHE_PATH")
    default_community_report_input_max_tokens = env_int(
        "COMMUNITY_REPORT_INPUT_MAX_TOKENS", 8192
    )
    default_cluster_summary_input_max_tokens = env_int(
        "CLUSTER_SUMMARY_INPUT_MAX_TOKENS", 6144
    )
    default_disable_llm_cache = env_bool("DISABLE_LLM_CACHE", False)
    default_disable_hierachical_mode = env_bool("DISABLE_HIERACHICAL_MODE", False)
    default_disable_naive_rag = env_bool("DISABLE_NAIVE_RAG", False)
    default_prompt_regime = env_str("PROMPT_REGIME", "baseline")
    default_entity_extract_max_gleaning = env_int_or_none("ENTITY_EXTRACT_MAX_GLEANING")
    default_query = env_str("QUERY")
    default_query_mode = env_str("QUERY_MODE", "hi")
    default_log_level = env_str("LOG_LEVEL", "INFO")
    default_graph_name = env_str("GRAPH_NAME", "graph")
    default_datasets_path = env_str("DATASETS_PATH")
    default_knowledge_graph_path = env_str("KNOWLEDGE_GRAPH_PATH")
    default_graph_vis_path = env_str("GRAPH_VIS_PATH")
    default_telemetry_output = env_path("TELEMETRY_OUTPUT")
    default_benchmark_label = env_str("BENCHMARK_LABEL")

    query_mode_choices = [
        "hi",
        "naive",
        "hi_nobridge",
        "hi_local",
        "hi_global",
        "hi_bridge",
    ]
    if default_query_mode not in query_mode_choices:
        LOGGER.warning(
            "Invalid QUERY_MODE=%s in .env. Falling back to '%s'.",
            default_query_mode,
            "hi",
        )
        default_query_mode = "hi"

    log_level_choices = ["DEBUG", "INFO", "WARNING", "ERROR"]
    if default_log_level is None:
        default_log_level = "INFO"
    default_log_level = default_log_level.upper()
    if default_log_level not in log_level_choices:
        LOGGER.warning(
            "Invalid LOG_LEVEL=%s in .env. Falling back to INFO.",
            default_log_level,
        )
        default_log_level = "INFO"

    parser = argparse.ArgumentParser(
        description="README-style HiRAG indexing script with local vLLM + FastEmbed"
    )
    parser.add_argument(
        "--context-file",
        type=str,
        default=default_context_file,
        help="Path to context file used for indexing (.txt or .json list)",
    )
    parser.add_argument(
        "--working-dir",
        type=str,
        default=default_working_dir,
        help="HiRAG working directory",
    )
    parser.add_argument(
        "--datasets-path",
        type=str,
        default=default_datasets_path,
        help="Root directory containing datasets",
    )
    parser.add_argument(
        "--knowledge-graph-path",
        type=str,
        default=default_knowledge_graph_path,
        help="Directory where built knowledge graphs are saved",
    )
    parser.add_argument(
        "--graph-name",
        type=str,
        default=default_graph_name,
        help="Graph folder prefix used when saving outputs",
    )
    parser.add_argument(
        "--graph-vis-path",
        type=str,
        default=default_graph_vis_path,
        help="Directory where graph visualizations are saved",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=default_base_url,
        help="Local vLLM OpenAI-compatible URL",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=default_api_key,
        help="API key for local vLLM endpoint",
    )
    parser.add_argument(
        "--chat-model",
        type=str,
        default=default_chat_model,
        help="vLLM chat model id; auto-discovered when omitted",
    )
    parser.add_argument(
        "--embed-model",
        type=str,
        default=default_embed_model,
        help="FastEmbed model name",
    )
    parser.add_argument(
        "--embed-dim",
        type=int,
        default=default_embed_dim,
        help="Embedding output dimension",
    )
    parser.add_argument(
        "--max-token-size",
        type=int,
        default=default_max_token_size,
        help="Max token size metadata used by HiRAG embedding wrapper",
    )
    parser.add_argument(
        "--embedding-batch-num",
        type=int,
        default=default_embedding_batch_num,
        help="HiRAG embedding batch size",
    )
    parser.add_argument(
        "--embedding-func-max-async",
        type=int,
        default=default_embedding_func_max_async,
        help="Maximum parallel embedding calls in HiRAG",
    )
    parser.add_argument(
        "--best-model-max-async",
        type=int,
        default=default_best_model_max_async,
        help="Maximum in-flight best-model LLM calls",
    )
    parser.add_argument(
        "--cheap-model-max-async",
        type=int,
        default=default_cheap_model_max_async,
        help="Maximum in-flight cheap-model LLM calls",
    )
    parser.add_argument(
        "--fastembed-threads",
        type=int,
        default=default_fastembed_threads,
        help="FastEmbed threads",
    )
    parser.add_argument(
        "--fastembed-parallel",
        type=int,
        default=default_fastembed_parallel,
        help="FastEmbed parallel workers",
    )
    parser.add_argument(
        "--fastembed-cache-path",
        type=str,
        default=default_fastembed_cache_path,
        help="FastEmbed cache path",
    )
    parser.add_argument(
        "--disable-llm-cache",
        action="store_true",
        default=default_disable_llm_cache,
        help="Disable HiRAG LLM response cache",
    )
    parser.add_argument(
        "--disable-hierachical-mode",
        action="store_true",
        default=default_disable_hierachical_mode,
        help="Disable hierarchical retrieval mode",
    )
    parser.add_argument(
        "--disable-naive-rag",
        action="store_true",
        default=default_disable_naive_rag,
        help="Disable naive RAG mode",
    )
    parser.add_argument(
        "--prompt-regime",
        type=str,
        default=default_prompt_regime,
        choices=PROMPT_REGIME_CHOICES,
        help="Prompt regime preset used for HiRAG generation stages",
    )
    parser.add_argument(
        "--entity-extract-max-gleaning",
        type=int,
        default=default_entity_extract_max_gleaning,
        help="Override entity/relation gleaning iterations; default depends on prompt regime",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=default_query,
        help="Optional question to run after indexing",
    )
    parser.add_argument(
        "--query-mode",
        type=str,
        default=default_query_mode,
        choices=query_mode_choices,
        help="Query mode passed to QueryParam",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=default_log_level,
        choices=log_level_choices,
        help="Logging level",
    )
    parser.add_argument(
        "--telemetry-output",
        type=str,
        default=default_telemetry_output,
        help="Optional path for machine-readable telemetry JSON",
    )
    parser.add_argument(
        "--benchmark-label",
        type=str,
        default=default_benchmark_label,
        help="Optional label stored in telemetry and benchmark outputs",
    )
    parser.add_argument(
        "--community-report-input-max-tokens",
        type=int,
        default=default_community_report_input_max_tokens,
        help="Input packing cap for community report generation",
    )
    parser.add_argument(
        "--cluster-summary-input-max-tokens",
        type=int,
        default=default_cluster_summary_input_max_tokens,
        help="Input packing cap for cluster summary generation",
    )
    return parser.parse_args()


def resolve_context_file(args: argparse.Namespace) -> Path:
    context_file = Path(args.context_file)
    if not context_file.exists() and not context_file.is_absolute():
        candidate = Path(args.datasets_path) / args.context_file
        if candidate.exists() and candidate.is_file():
            context_file = candidate
    if not context_file.exists() or not context_file.is_file():
        raise FileNotFoundError(f"Context file not found: {context_file}")
    return context_file


def create_graph_working_dir(args: argparse.Namespace) -> Path:
    Path(args.knowledge_graph_path).mkdir(parents=True, exist_ok=True)
    Path(args.graph_vis_path).mkdir(parents=True, exist_ok=True)

    current_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    graph_working_dir = (
        Path(args.knowledge_graph_path) / f"{args.graph_name}_{current_timestamp}"
    )
    graph_working_dir.mkdir(parents=True, exist_ok=True)
    return graph_working_dir


def build_graph_runtime(
    args: argparse.Namespace,
    *,
    graph_working_dir: Path,
    telemetry: dict[str, Any] | None = None,
    stage_max_token_overrides: dict[str, int] | None = None,
) -> tuple[HiRAG, dict[str, Any]]:
    discovered_model, discovered_context = discover_local_chat_model_info(
        args.base_url, args.api_key
    )
    chat_model = args.chat_model or discovered_model
    if not chat_model:
        raise RuntimeError(
            "No chat model available. Set --chat-model or VLLM_CHAT_MODEL, "
            "or ensure local vLLM exposes /models."
        )
    model_max_context = discovered_context

    runtime = {
        "base_url": args.base_url,
        "api_key": args.api_key,
        "chat_model": chat_model,
        "embed_model": args.embed_model,
        "embed_dim": args.embed_dim,
        "max_token_size": args.max_token_size,
        "fastembed_threads": args.fastembed_threads,
        "fastembed_parallel": args.fastembed_parallel,
        "fastembed_cache_path": args.fastembed_cache_path,
        "telemetry": telemetry,
        "benchmark_label": args.benchmark_label,
        "model_max_context": model_max_context,
    }

    embedding_func = create_fastembed_embedding(runtime)
    llm_func = create_vllm_chat_model(runtime)
    prompts = resolve_prompts(args.prompt_regime)

    graph_func = HiRAG(
        working_dir=str(graph_working_dir),
        enable_llm_cache=not args.disable_llm_cache,
        embedding_func=embedding_func,
        best_model_func=llm_func,
        cheap_model_func=llm_func,
        enable_hierachical_mode=not args.disable_hierachical_mode,
        embedding_batch_num=args.embedding_batch_num,
        embedding_func_max_async=args.embedding_func_max_async,
        best_model_max_async=args.best_model_max_async,
        cheap_model_max_async=args.cheap_model_max_async,
        enable_naive_rag=not args.disable_naive_rag,
        graph_storage_cls=NetworkXStorage,
        prompt_regime=args.prompt_regime,
        prompts=prompts,
        stage_max_tokens=stage_max_token_overrides or {},
        entity_extract_max_gleaning=args.entity_extract_max_gleaning,
        community_report_input_max_tokens=args.community_report_input_max_tokens,
        cluster_summary_input_max_tokens=args.cluster_summary_input_max_tokens,
    )
    return graph_func, runtime


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level))

    if not args.context_file:
        raise ValueError(
            "No context file configured. Set --context-file or CONTEXT_FILE in .env."
        )
    if not args.graph_name:
        raise ValueError("GRAPH_NAME cannot be empty.")

    graph_working_dir = create_graph_working_dir(args)
    context_file = resolve_context_file(args)
    context_input = load_context_input(context_file)
    telemetry = new_telemetry_session(args.benchmark_label)
    graph_func, runtime = build_graph_runtime(
        args, graph_working_dir=graph_working_dir, telemetry=telemetry
    )

    print(
        "Runtime: "
        f"chat_model={runtime['chat_model']} | "
        f"vllm_base_url={runtime['base_url']} | "
        f"embed_model={runtime['embed_model']} | "
        f"prompt_regime={args.prompt_regime} | "
        f"entity_extract_max_gleaning={graph_func.entity_extract_max_gleaning} | "
        f"best_model_max_async={graph_func.best_model_max_async} | "
        f"cheap_model_max_async={graph_func.cheap_model_max_async} | "
        f"community_report_input_max_tokens={graph_func.community_report_input_max_tokens} | "
        f"cluster_summary_input_max_tokens={graph_func.cluster_summary_input_max_tokens} | "
        f"graph_dir={graph_working_dir}"
    )

    start = time.perf_counter()
    graph_func.insert(context_input)
    elapsed = time.perf_counter() - start
    if isinstance(context_input, list):
        print(f"Indexed {len(context_input)} contexts from {context_file} in {elapsed:.2f}s")
    else:
        print(f"Indexed graph from {context_file} in {elapsed:.2f}s")

    if args.query:
        print("Perform hi search:")
        result = graph_func.query(args.query, param=QueryParam(mode=args.query_mode))
        print(result)

    if args.telemetry_output:
        telemetry_payload = {
            "benchmark_label": args.benchmark_label,
            "prompt_regime": args.prompt_regime,
            "entity_extract_max_gleaning": graph_func.entity_extract_max_gleaning,
            "best_model_max_async": graph_func.best_model_max_async,
            "cheap_model_max_async": graph_func.cheap_model_max_async,
            "community_report_input_max_tokens": graph_func.community_report_input_max_tokens,
            "cluster_summary_input_max_tokens": graph_func.cluster_summary_input_max_tokens,
            "graph_dir": str(graph_working_dir),
            "context_file": str(context_file),
            **summarize_telemetry(telemetry),
        }
        output_path = Path(args.telemetry_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(telemetry_payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
