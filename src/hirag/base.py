from dataclasses import dataclass, field
from typing import TypedDict, Union, Literal, Generic, TypeVar
from ._utils import EmbeddingFunc
import numpy as np


@dataclass
class QueryParam:
    mode: Literal[
        "hi_global",
        "hi_local",
        "hi_bridge",
        "hi_nobridge",
        "naive",
        "hi",
        "hi_weighted",
        "hi_minmax",
        "hi_minmax_budgeted",
        "hi_mcts",
        "hi_rerank",
        "hi_rerank_weighted",
    ] = "hi"
    only_need_context: bool = False
    response_type: str = "Multiple Paragraphs"
    level: int = 2
    top_k: int = 20         # retrieve top-k entities
    top_m: int = 10         # retrieve top-m entities in each retrieved community
    bridge_strategy: Literal["unweighted", "query_weighted", "minmax", "minmax_budgeted", "mcts"] = "unweighted"
    local_rerank_strategy: Literal["none", "fastembed_late_interaction"] = "none"
    local_candidate_multiplier: int = 10
    local_rerank_top_n: int = 100
    bridge_alpha: float = 1.0
    bridge_beta: float = 0.15
    bridge_gamma: float = 0.05
    bridge_max_path_edges: int = 120
    bridge_max_total_edges: int = 220
    bridge_length_penalty: float = 0.02
    bridge_budget_fallback: Literal["weighted", "unweighted"] = "weighted"
    mcts_max_iterations: int = 96
    mcts_exploration_constant: float = 1.2
    mcts_candidate_hops: int = 2
    mcts_candidate_top_neighbors: int = 12
    mcts_progressive_widening_coefficient: float = 2.0
    mcts_progressive_widening_exponent: float = 0.5
    mcts_rollout_top_k: int = 3
    mcts_rollout_depth: int = 4
    mcts_rollout_epsilon: float = 0.1
    mcts_token_penalty: float = 0.15
    mcts_target_reward: float = 1.0
    # naive search
    naive_max_token_for_text_unit = 10000
    # hi search
    max_token_for_text_unit: int = 20000
    max_token_for_local_context: int = 20000
    max_token_for_bridge_knowledge: int = 12500
    max_token_for_community_report: int = 12500
    community_single_one: bool = False
    text_unit_snippet_chars: int | None = None
    text_unit_snippet_strategy: Literal["prefix", "query_overlap"] = "prefix"
    debug_info: dict = field(default_factory=dict)


TextChunkSchema = TypedDict(
    "TextChunkSchema",
    {"tokens": int, "content": str, "full_doc_id": str, "chunk_order_index": int},
)

SingleCommunitySchema = TypedDict(
    "SingleCommunitySchema",
    {
        "level": int,
        "title": str,
        "edges": list[list[str, str]],
        "nodes": list[str],
        "chunk_ids": list[str],
        "occurrence": float,
        "sub_communities": list[str],
    },
)


class CommunitySchema(SingleCommunitySchema):
    report_string: str
    report_json: dict


T = TypeVar("T")


@dataclass
class StorageNameSpace:
    namespace: str
    global_config: dict

    async def index_start_callback(self):
        """commit the storage operations after indexing"""
        pass

    async def index_done_callback(self):
        """commit the storage operations after indexing"""
        pass

    async def query_done_callback(self):
        """commit the storage operations after querying"""
        pass


@dataclass
class BaseVectorStorage(StorageNameSpace):
    embedding_func: EmbeddingFunc
    meta_fields: set = field(default_factory=set)

    async def query(self, query: str, top_k: int) -> list[dict]:
        raise NotImplementedError

    async def upsert(self, data: dict[str, dict]):
        """Use 'content' field from value for embedding, use key as id.
        If embedding_func is None, use 'embedding' field from value
        """
        raise NotImplementedError


@dataclass
class BaseKVStorage(Generic[T], StorageNameSpace):
    async def all_keys(self) -> list[str]:
        raise NotImplementedError

    async def get_by_id(self, id: str) -> Union[T, None]:
        raise NotImplementedError

    async def get_by_ids(
        self, ids: list[str], fields: Union[set[str], None] = None
    ) -> list[Union[T, None]]:
        raise NotImplementedError

    async def filter_keys(self, data: list[str]) -> set[str]:
        """return un-exist keys"""
        raise NotImplementedError

    async def upsert(self, data: dict[str, T]):
        raise NotImplementedError

    async def drop(self):
        raise NotImplementedError


@dataclass
class BaseGraphStorage(StorageNameSpace):
    async def has_node(self, node_id: str) -> bool:
        raise NotImplementedError

    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        raise NotImplementedError

    async def node_degree(self, node_id: str) -> int:
        raise NotImplementedError

    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        raise NotImplementedError

    async def get_node(self, node_id: str) -> Union[dict, None]:
        raise NotImplementedError

    async def get_edge(
        self, source_node_id: str, target_node_id: str
    ) -> Union[dict, None]:
        raise NotImplementedError

    async def get_node_edges(
        self, source_node_id: str
    ) -> Union[list[tuple[str, str]], None]:
        raise NotImplementedError

    async def upsert_node(self, node_id: str, node_data: dict[str, str]):
        raise NotImplementedError

    async def upsert_edge(
        self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]
    ):
        raise NotImplementedError

    async def clustering(self, algorithm: str):
        raise NotImplementedError

    async def community_schema(self) -> dict[str, SingleCommunitySchema]:
        """Return the community representation with report and nodes"""
        raise NotImplementedError

    async def embed_nodes(self, algorithm: str) -> tuple[np.ndarray, list[str]]:
        raise NotImplementedError("Node embedding is not used in HiRAG.")
