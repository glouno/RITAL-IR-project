import asyncio
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from functools import partial
from typing import Callable, Dict, List, Optional, Type, Union, cast

import tiktoken


from ._llm import (
    gpt_4o_complete,
    gpt_4o_mini_complete,
    gpt_35_turbo_complete,
    openai_embedding,
    azure_gpt_4o_complete,
    azure_openai_embedding,
    azure_gpt_4o_mini_complete,
)
from ._op import (
    chunking_by_token_size,
    extract_entities,
    extract_hierarchical_entities,
    generate_community_report,
    get_chunks,
    hierarchical_query,
    hierarchical_bridge_query,
    hierarchical_local_query,
    hierarchical_global_query,
    hierarchical_nobridge_query,
    naive_query,
)
from ._storage import (
    JsonKVStorage,
    NanoVectorDBStorage,
    NetworkXStorage,
    Neo4jStorage,
)
from ._utils import (
    EmbeddingFunc,
    compute_mdhash_id,
    limit_async_func_call,
    convert_response_to_json,
    always_get_an_event_loop,
    logger,
)
from .base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    StorageNameSpace,
    QueryParam,
)
from .regimes import (
    resolve_entity_extract_max_gleaning,
    resolve_prompt_regime,
    resolve_prompts,
    resolve_stage_max_tokens,
)


@dataclass
class HiRAG:
    working_dir: str = field(
        default_factory=lambda: f"./hirag_cache_{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}"
    )
    # graph mode
    enable_local: bool = True
    enable_naive_rag: bool = False
    enable_hierachical_mode: bool = True

    # text chunking
    chunk_func: Callable[
        [
            list[list[int]],
            List[str],
            tiktoken.Encoding,
            Optional[int],
            Optional[int],
        ],
        List[Dict[str, Union[str, int]]],
    ] = chunking_by_token_size
    chunk_token_size: int = 1200
    chunk_overlap_token_size: int = 100
    tiktoken_model_name: str = "gpt-4o"

    # entity extraction
    entity_extract_max_gleaning: int | None = None
    entity_summary_to_max_tokens: int = 500

    # graph clustering
    graph_cluster_algorithm: str = "leiden"
    max_graph_cluster_size: int = 10
    graph_cluster_seed: int = 0xDEADBEEF

    # node embedding
    node_embedding_algorithm: str = "node2vec"
    node2vec_params: dict = field(
        default_factory=lambda: {
            "dimensions": 1536,
            "num_walks": 10,
            "walk_length": 40,
            "num_walks": 10,
            "window_size": 2,
            "iterations": 3,
            "random_seed": 3,
        }
    )

    # community reports
    special_community_report_llm_kwargs: dict = field(
        default_factory=lambda: {"response_format": {"type": "json_object"}}
    )

    # text embedding
    embedding_func: EmbeddingFunc = field(default_factory=lambda: openai_embedding)
    embedding_batch_num: int = 32
    embedding_func_max_async: int = 8
    query_better_than_threshold: float = 0.2

    # LLM
    using_azure_openai: bool = False
    # best_model_func: callable = gpt_35_turbo_complete
    best_model_func: callable = gpt_4o_mini_complete
    best_model_max_token_size: int = 32768
    best_model_max_async: int = 8
    cheap_model_func: callable = gpt_35_turbo_complete
    cheap_model_max_token_size: int = 32768
    cheap_model_max_async: int = 8
    community_report_input_max_tokens: int = 8192
    cluster_summary_input_max_tokens: int = 6144
    entity_extract_input_max_tokens: int | None = None
    reranker_model: str = "answerdotai/answerai-colbert-small-v1"
    reranker_cache_path: str | None = None
    local_rerank_top_n: int = 100
    edge_embedding_cache_path: str = ".runs/edge_embedding_cache"
    disable_edge_embedding_cache: bool = False

    # entity extraction
    entity_extraction_func: callable = extract_entities
    hierarchical_entity_extraction_func: callable = extract_hierarchical_entities

    # storage
    key_string_value_json_storage_cls: Type[BaseKVStorage] = JsonKVStorage
    vector_db_storage_cls: Type[BaseVectorStorage] = NanoVectorDBStorage
    vector_db_storage_cls_kwargs: dict = field(default_factory=dict)
    graph_storage_cls: Type[BaseGraphStorage] = NetworkXStorage
    enable_llm_cache: bool = True

    # extension
    always_create_working_dir: bool = True
    addon_params: dict = field(default_factory=dict)
    convert_response_to_json_func: callable = convert_response_to_json
    prompt_regime: str = "baseline"
    prompts: dict = field(default_factory=dict, repr=False)
    stage_max_tokens: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        self.prompt_regime = resolve_prompt_regime(self.prompt_regime)
        self.prompts = self.prompts or resolve_prompts(self.prompt_regime)
        self.stage_max_tokens = resolve_stage_max_tokens(
            self.prompt_regime,
            best_model_max_token_size=self.best_model_max_token_size,
            entity_summary_to_max_tokens=self.entity_summary_to_max_tokens,
            overrides=self.stage_max_tokens,
        )
        self.entity_extract_max_gleaning = resolve_entity_extract_max_gleaning(
            self.prompt_regime, self.entity_extract_max_gleaning
        )

        _print_config = ",\n  ".join(
            [f"{k} = {v}" for k, v in self._runtime_global_config().items()]
        )
        logger.debug(f"HiRAG init with param:\n\n  {_print_config}\n")

        if self.using_azure_openai:
            # If there's no OpenAI API key, use Azure OpenAI
            if self.best_model_func == gpt_4o_complete:
                self.best_model_func = azure_gpt_4o_complete
            if self.cheap_model_func == gpt_4o_mini_complete:
                self.cheap_model_func = azure_gpt_4o_mini_complete
            if self.embedding_func == openai_embedding:
                self.embedding_func = azure_openai_embedding
            logger.info(
                "Switched the default openai funcs to Azure OpenAI if you didn't set any of it"
            )

        if not os.path.exists(self.working_dir) and self.always_create_working_dir:
            logger.info(f"Creating working directory {self.working_dir}")
            os.makedirs(self.working_dir)

        storage_config = self._storage_global_config()
        self.full_docs = self.key_string_value_json_storage_cls(
            namespace="full_docs", global_config=storage_config
        )

        self.text_chunks = self.key_string_value_json_storage_cls(
            namespace="text_chunks", global_config=storage_config
        )

        self.llm_response_cache = (
            self.key_string_value_json_storage_cls(
                namespace="llm_response_cache", global_config=storage_config
            )
            if self.enable_llm_cache
            else None
        )

        self.community_reports = self.key_string_value_json_storage_cls(
            namespace="community_reports", global_config=storage_config
        )
        self.chunk_entity_relation_graph = self.graph_storage_cls(
            namespace="chunk_entity_relation", global_config=storage_config
        )

        self.embedding_func = limit_async_func_call(self.embedding_func_max_async)(
            self.embedding_func
        )
        self.entities_vdb = (
            self.vector_db_storage_cls(
                namespace="entities",
                global_config=storage_config,
                embedding_func=self.embedding_func,
                meta_fields={"entity_name"},
            )
            if self.enable_local
            else None
        )
        self.chunks_vdb = (
            self.vector_db_storage_cls(
                namespace="chunks",
                global_config=storage_config,
                embedding_func=self.embedding_func,
            )
            if self.enable_naive_rag
            else None
        )

        self.best_model_func = limit_async_func_call(self.best_model_max_async)(
            partial(self.best_model_func, hashing_kv=self.llm_response_cache)
        )
        self.cheap_model_func = limit_async_func_call(self.cheap_model_max_async)(
            partial(self.cheap_model_func, hashing_kv=self.llm_response_cache)
        )

    def _storage_global_config(self) -> dict:
        return asdict(self)

    def _runtime_global_config(self) -> dict:
        config = asdict(self)
        config["prompts"] = self.prompts
        config["stage_max_tokens"] = self.stage_max_tokens
        config["prompt_regime"] = self.prompt_regime
        config["entity_extract_max_gleaning"] = self.entity_extract_max_gleaning
        config["embedding_func"] = self.embedding_func
        config["reranker_model"] = self.reranker_model
        config["reranker_cache_path"] = self.reranker_cache_path
        config["edge_embedding_cache_path"] = self.edge_embedding_cache_path
        config["disable_edge_embedding_cache"] = self.disable_edge_embedding_cache
        return config

    def insert(self, string_or_strings):
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.ainsert(string_or_strings))

    def query(self, query: str, param: QueryParam = QueryParam()):
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.aquery(query, param))

    def _resolve_query_param(self, param: QueryParam) -> QueryParam:
        if param.mode == "hi_weighted":
            param.bridge_strategy = "query_weighted"
        elif param.mode == "hi_minmax":
            param.bridge_strategy = "minmax"
        elif param.mode == "hi_minmax_budgeted":
            param.bridge_strategy = "minmax_budgeted"
        elif param.mode == "hi_rerank":
            param.local_rerank_strategy = "fastembed_late_interaction"
        elif param.mode == "hi_rerank_weighted":
            param.bridge_strategy = "query_weighted"
            param.local_rerank_strategy = "fastembed_late_interaction"
        if param.local_rerank_strategy != "none":
            param.local_rerank_top_n = self.local_rerank_top_n
        return param

    async def aquery(self, query: str, param: QueryParam = QueryParam()):
        param = self._resolve_query_param(param)
        if param.mode == "naive" and not self.enable_naive_rag:
            raise ValueError("enable_naive_rag is False, cannot query in naive mode")
        if param.mode in {"hi", "hi_weighted", "hi_minmax", "hi_minmax_budgeted", "hi_rerank", "hi_rerank_weighted"} and not self.enable_hierachical_mode:
            raise ValueError(f"enable_hierachical_mode is False, cannot query in {param.mode} mode")
        if param.mode == "hi_nobridge" and not self.enable_hierachical_mode:
            raise ValueError("enable_hierachical_mode is False, cannot query in hierarchical_nobridge mode")
        if param.mode == "hi_bridge" and not self.enable_hierachical_mode:
            raise ValueError("enable_hierachical_mode is False, cannot query in hierarchical_bridge mode")
        if param.mode == "hi_local" and not self.enable_hierachical_mode:
            raise ValueError("enable_hierachical_mode is False, cannot query in hierarchical_local mode")
        if param.mode == "hi_global" and not self.enable_hierachical_mode:
            raise ValueError("enable_hierachical_mode is False, cannot query in hierarchical_global mode")

        if param.mode in {"hi", "hi_weighted", "hi_minmax", "hi_minmax_budgeted", "hi_rerank", "hi_rerank_weighted"}:  # retrieve with hierarchical knowledge
            response = await hierarchical_query(
                query,
                self.chunk_entity_relation_graph,
                self.entities_vdb,
                self.community_reports,
                self.text_chunks,
                param,
                self._runtime_global_config(),
            )
        elif param.mode == "hi_bridge":                 # retrieve with only bridge knowledge
            response = await hierarchical_bridge_query(
                query,
                self.chunk_entity_relation_graph,
                self.entities_vdb,
                self.community_reports,
                self.text_chunks,
                param,
                self._runtime_global_config(),
            )
        elif param.mode == "hi_local":                  # retrieve with only local knowledge
            response = await hierarchical_local_query(
                query,
                self.chunk_entity_relation_graph,
                self.entities_vdb,
                self.community_reports,
                self.text_chunks,
                param,
                self._runtime_global_config(),
            )
        elif param.mode == "hi_global":                 # retrieve with only global knowledge
            response = await hierarchical_global_query(
                query,
                self.chunk_entity_relation_graph,
                self.entities_vdb,
                self.community_reports,
                self.text_chunks,
                param,
                self._runtime_global_config(),
            )
        elif param.mode == "hi_nobridge":               # retrieve with no bridge knowledge
            response = await hierarchical_nobridge_query(
                query,
                self.chunk_entity_relation_graph,
                self.entities_vdb,
                self.community_reports,
                self.text_chunks,
                param,
                self._runtime_global_config(),
            )
        elif param.mode == "naive":                     # retrieve with only text units
            response = await naive_query(
                query,
                self.chunks_vdb,
                self.text_chunks,
                param,
                self._runtime_global_config(),
            )
        else:
            raise ValueError(f"Unknown mode {param.mode}")
        await self._query_done()
        return response

    async def ainsert(self, string_or_strings):
        await self._insert_start()
        try:
            if isinstance(string_or_strings, str):
                string_or_strings = [string_or_strings]
            # ---------- new docs
            new_docs = {    # dict: {hash: ori_content}
                compute_mdhash_id(c.strip(), prefix="doc-"): {"content": c.strip()}
                for c in string_or_strings
            }
            _add_doc_keys = await self.full_docs.filter_keys(list(new_docs.keys()))     # filter the docs that has already in the storage.
            new_docs = {k: v for k, v in new_docs.items() if k in _add_doc_keys}
            if not len(new_docs):
                logger.warning(f"All docs are already in the storage")
                return
            logger.info(f"[New Docs] inserting {len(new_docs)} docs")

            # ---------- chunking

            inserting_chunks = get_chunks(
                new_docs=new_docs,
                chunk_func=self.chunk_func,
                overlap_token_size=self.chunk_overlap_token_size,
                max_token_size=self.chunk_token_size,
            )

            _add_chunk_keys = await self.text_chunks.filter_keys(
                list(inserting_chunks.keys())
            )
            inserting_chunks = {
                k: v for k, v in inserting_chunks.items() if k in _add_chunk_keys
            }
            if not len(inserting_chunks):
                logger.warning(f"All chunks are already in the storage")
                return
            logger.info(f"[New Chunks] inserting {len(inserting_chunks)} chunks")
            if self.enable_naive_rag:
                logger.info("Insert chunks for naive RAG")
                await self.chunks_vdb.upsert(inserting_chunks)

            # TODO: no incremental update for communities now, so just drop all
            await self.community_reports.drop()                             # empty the data

            # ---------- extract/summary entity and upsert to graph
            if not self.enable_hierachical_mode:
                logger.info("\033[94m[[Entity Extraction]...\033[0m")
                maybe_new_kg = await self.entity_extraction_func(
                    inserting_chunks,
                    knwoledge_graph_inst=self.chunk_entity_relation_graph,
                    entity_vdb=self.entities_vdb,
                    global_config=self._runtime_global_config(),
                )
            else:
                logger.info("\033[94m[Hierachical Entity Extraction]...\033[0m")
                maybe_new_kg = await self.hierarchical_entity_extraction_func(
                    inserting_chunks,
                    knowledge_graph_inst=self.chunk_entity_relation_graph,
                    entity_vdb=self.entities_vdb,
                    global_config=self._runtime_global_config(),
                )
            if maybe_new_kg is None:
                logger.warning("No new entities found")
                return
            self.chunk_entity_relation_graph = maybe_new_kg
            # ---------- update clusterings of graph
            logger.info("\033[94m[Community Report]...\033[0m")
            await self.chunk_entity_relation_graph.clustering(
                self.graph_cluster_algorithm                    # use leiden
            )
            await generate_community_report(
                self.community_reports,
                self.chunk_entity_relation_graph,
                self._runtime_global_config(),
            )

            # ---------- commit upsertings and indexing
            await self.full_docs.upsert(new_docs)
            await self.text_chunks.upsert(inserting_chunks)
        finally:
            await self._insert_done()

    async def _insert_start(self):
        tasks = []
        for storage_inst in [
            self.chunk_entity_relation_graph,
        ]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_start_callback())
        await asyncio.gather(*tasks)

    async def _insert_done(self):
        tasks = []
        for storage_inst in [
            self.full_docs,
            self.text_chunks,
            self.llm_response_cache,
            self.community_reports,
            self.entities_vdb,
            self.chunks_vdb,
            self.chunk_entity_relation_graph,
        ]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        await asyncio.gather(*tasks)

    async def _query_done(self):
        tasks = []
        for storage_inst in [self.llm_response_cache]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        await asyncio.gather(*tasks)
