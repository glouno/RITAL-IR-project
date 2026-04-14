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
from dotenv import load_dotenv

from hirag import HiRAG, QueryParam
from hirag._storage import NetworkXStorage
from hirag._utils import compute_args_hash, wrap_embedding_func_with_attrs
from hirag.base import BaseKVStorage


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
    try:
        openai_module = importlib.import_module("openai")
        client = openai_module.OpenAI(base_url=base_url, api_key=api_key)
        models = client.models.list()
    except Exception as exc:
        LOGGER.warning("Unable to discover model from %s: %s", base_url, exc)
        return None

    for item in getattr(models, "data", []):
        model_id = getattr(item, "id", None)
        if isinstance(model_id, str) and model_id.strip():
            return model_id.strip()
    return None


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
        client = get_async_client()
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if history_messages:
            messages.extend(history_messages)
        messages.append({"role": "user", "content": prompt})

        hashing_kv: BaseKVStorage | None = kwargs.pop("hashing_kv", None)
        args_hash: str | None = None
        if hashing_kv is not None:
            args_hash = compute_args_hash(runtime["chat_model"], messages)
            cached = await hashing_kv.get_by_id(args_hash)
            if cached is not None:
                return cached["return"]

        response = await client.chat.completions.create(
            model=runtime["chat_model"],
            messages=messages,
            **kwargs,
        )
        content = response.choices[0].message.content or ""

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
    default_fastembed_threads = env_int_or_none("FASTEMBED_THREADS")
    default_fastembed_parallel = env_int_or_none("FASTEMBED_PARALLEL")
    default_fastembed_cache_path = env_str("FASTEMBED_CACHE_PATH")
    default_disable_llm_cache = env_bool("DISABLE_LLM_CACHE", False)
    default_disable_hierachical_mode = env_bool("DISABLE_HIERACHICAL_MODE", False)
    default_disable_naive_rag = env_bool("DISABLE_NAIVE_RAG", False)
    default_query = env_str("QUERY")
    default_query_mode = env_str("QUERY_MODE", "hi")
    default_log_level = env_str("LOG_LEVEL", "INFO")
    default_graph_name = env_str("GRAPH_NAME", "graph")
    default_datasets_path = env_str("DATASETS_PATH")
    default_knowledge_graph_path = env_str("KNOWLEDGE_GRAPH_PATH")
    default_graph_vis_path = env_str("GRAPH_VIS_PATH")

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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level))

    if not args.context_file:
        raise ValueError(
            "No context file configured. Set --context-file or CONTEXT_FILE in .env."
        )
    if not args.graph_name:
        raise ValueError("GRAPH_NAME cannot be empty.")

    Path(args.knowledge_graph_path).mkdir(parents=True, exist_ok=True)
    Path(args.graph_vis_path).mkdir(parents=True, exist_ok=True)

    current_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    graph_working_dir = (
        Path(args.knowledge_graph_path) / f"{args.graph_name}_{current_timestamp}"
    )
    graph_working_dir.mkdir(parents=True, exist_ok=True)

    context_file = Path(args.context_file)
    if not context_file.exists() and not context_file.is_absolute():
        candidate = Path(args.datasets_path) / args.context_file
        if candidate.exists() and candidate.is_file():
            context_file = candidate
    if not context_file.exists() or not context_file.is_file():
        raise FileNotFoundError(f"Context file not found: {context_file}")

    context_input = load_context_input(context_file)

    chat_model = args.chat_model or discover_local_chat_model(args.base_url, args.api_key)
    if not chat_model:
        raise RuntimeError(
            "No chat model available. Set --chat-model or VLLM_CHAT_MODEL, "
            "or ensure local vLLM exposes /models."
        )

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
    }

    embedding_func = create_fastembed_embedding(runtime)
    llm_func = create_vllm_chat_model(runtime)

    graph_func = HiRAG(
        working_dir=str(graph_working_dir),
        enable_llm_cache=not args.disable_llm_cache,
        embedding_func=embedding_func,
        best_model_func=llm_func,
        cheap_model_func=llm_func,
        enable_hierachical_mode=not args.disable_hierachical_mode,
        embedding_batch_num=args.embedding_batch_num,
        embedding_func_max_async=args.embedding_func_max_async,
        enable_naive_rag=not args.disable_naive_rag,
        graph_storage_cls=NetworkXStorage,
    )

    print(
        "Runtime: "
        f"chat_model={runtime['chat_model']} | "
        f"vllm_base_url={runtime['base_url']} | "
        f"embed_model={runtime['embed_model']} | "
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


if __name__ == "__main__":
    main()
