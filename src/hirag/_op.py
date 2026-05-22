import re
import json
import asyncio
import tiktoken
import networkx as nx
import time
import logging
import hashlib
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Union
from collections import Counter, defaultdict
from ._splitter import SeparatorSplitter
from ._utils import (
    logger,
    clean_str,
    compute_mdhash_id,
    decode_tokens_by_tiktoken,
    encode_string_by_tiktoken,
    is_float_regex,
    list_of_list_to_csv,
    pack_user_ass_to_openai_messages,
    split_string_by_multi_markers,
    truncate_list_by_token_size,
)
from .base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    SingleCommunitySchema,
    CommunitySchema,
    TextChunkSchema,
    QueryParam,
)
from .prompt import GRAPH_FIELD_SEP, PROMPTS
from ._cluster_utils import Hierarchical_Clustering
from .mcts import MCTSBridgeConfig, find_mcts_bridge_path


@contextmanager
def timer():
    start_time = time.perf_counter()
    try:
        yield
    finally:
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        logging.info(f"\033[94m[Retrieval Time: {elapsed_time:.6f} seconds]\033[0m")


def _get_prompts(global_config: dict | None = None) -> dict:
    if global_config is None:
        return PROMPTS
    return global_config.get("prompts", PROMPTS)


def _get_stage_max_tokens(
    global_config: dict, stage: str, fallback: int | None = None
) -> int | None:
    stage_max_tokens = global_config.get("stage_max_tokens", {})
    if stage in stage_max_tokens:
        return stage_max_tokens[stage]
    return fallback


def _ordered_unique(items: list[str]) -> list[str]:
    return list(dict.fromkeys(item for item in items if item))


def _safe_float(value: Any, default: float = 1.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _cosine_similarity(left: Any, right: Any) -> float:
    import numpy as np

    left_arr = np.asarray(left, dtype=np.float64)
    right_arr = np.asarray(right, dtype=np.float64)
    left_norm = np.linalg.norm(left_arr)
    right_norm = np.linalg.norm(right_arr)
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return float(np.dot(left_arr, right_arr) / (left_norm * right_norm))


def _normalize_token_embeddings(embeddings: Any):
    import numpy as np

    arr = np.asarray(embeddings, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


def _late_interaction_score(query_embedding: Any, document_embedding: Any) -> float:
    query_tokens = _normalize_token_embeddings(query_embedding)
    document_tokens = _normalize_token_embeddings(document_embedding)
    if query_tokens.size == 0 or document_tokens.size == 0:
        return 0.0
    scores = query_tokens @ document_tokens.T
    return float(scores.max(axis=1).sum())


_RERANKER_CACHE: dict[tuple[str, str | None], Any] = {}
_RERANKER_FAILURES: set[tuple[str, str | None]] = set()
_EDGE_EMBEDDING_CACHE: dict[tuple[Any, ...], dict[str, Any]] = {}


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _reranker_cache_key(global_config: dict) -> tuple[str, str | None]:
    model_name = global_config.get(
        "reranker_model", "answerdotai/answerai-colbert-small-v1"
    )
    cache_path = global_config.get("reranker_cache_path")
    return model_name, cache_path


def _get_late_interaction_reranker(global_config: dict):
    model_name, cache_path = _reranker_cache_key(global_config)
    key = (model_name, cache_path)
    if key in _RERANKER_FAILURES:
        raise RuntimeError(
            f"late-interaction reranker previously failed for {model_name}"
        )
    if key not in _RERANKER_CACHE:
        from fastembed import LateInteractionTextEmbedding

        _RERANKER_CACHE[key] = LateInteractionTextEmbedding(
            model_name=model_name,
            cache_dir=cache_path,
            lazy_load=True,
        )
    return _RERANKER_CACHE[key]


def _entity_text(node_data: dict) -> str:
    return "\n".join(
        [
            str(node_data.get("entity_name", "")),
            f"type: {node_data.get('entity_type', 'UNKNOWN')}",
            f"description: {node_data.get('description', '')}",
        ]
    )


async def _fetch_entity_node_datas(
    query: str,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    query_param: QueryParam,
    *,
    global_config: dict | None = None,
    candidate_multiplier: int | None = None,
) -> list[dict]:
    multiplier = candidate_multiplier or query_param.local_candidate_multiplier
    candidate_count = max(query_param.top_k, query_param.top_k * max(1, multiplier))
    results = await entities_vdb.query(query, top_k=candidate_count)
    if not len(results):
        return []

    node_datas_raw = await asyncio.gather(
        *[knowledge_graph_inst.get_node(r["entity_name"]) for r in results]
    )
    if not all([n is not None for n in node_datas_raw]):
        logger.warning("Some nodes are missing, maybe the storage is damaged")
    node_degrees = await asyncio.gather(
        *[knowledge_graph_inst.node_degree(r["entity_name"]) for r in results]
    )
    node_datas = [
        {
            **n,
            "entity_name": r["entity_name"],
            "rank": d,
            "dense_order": index,
            "dense_score": r.get("distance", r.get("__metrics__", 0.0)),
        }
        for index, (r, n, d) in enumerate(zip(results, node_datas_raw, node_degrees))
        if n is not None
    ]
    if (
        global_config is not None
        and query_param.local_rerank_strategy == "fastembed_late_interaction"
    ):
        node_datas = _rerank_entity_node_datas(query, node_datas, query_param, global_config)
    return node_datas


def _rerank_entity_node_datas(
    query: str,
    node_datas: list[dict],
    query_param: QueryParam,
    global_config: dict,
) -> list[dict]:
    if not node_datas:
        return node_datas
    rerank_limit = max(query_param.top_k, query_param.local_rerank_top_n)
    rerank_items = node_datas[:rerank_limit]
    untouched = node_datas[rerank_limit:]
    try:
        reranker = _get_late_interaction_reranker(global_config)
        texts = [query] + [_entity_text(node) for node in rerank_items]
        embeddings = list(reranker.embed(texts))
        query_embedding = embeddings[0]
        doc_embeddings = embeddings[1:]
        rescored = []
        for node, doc_embedding in zip(rerank_items, doc_embeddings):
            rescored.append(
                {
                    **node,
                    "rerank_score": _late_interaction_score(query_embedding, doc_embedding),
                }
            )
        rescored.sort(
            key=lambda item: (item.get("rerank_score", 0.0), item.get("dense_score", 0.0)),
            reverse=True,
        )
        return rescored + untouched
    except Exception as exc:
        _RERANKER_FAILURES.add(_reranker_cache_key(global_config))
        logger.warning("Late-interaction reranking failed; using dense order: %s", exc)
        return node_datas


def chunking_by_token_size(
    tokens_list: list[list[int]],
    doc_keys,
    tiktoken_model,
    overlap_token_size=128,
    max_token_size=1024,
):
  # tokenizer
    results = []
    for index, tokens in enumerate(tokens_list):
        chunk_token = []
        lengths = []
        for start in range(0, len(tokens), max_token_size - overlap_token_size):

            chunk_token.append(tokens[start : start + max_token_size])
            lengths.append(min(max_token_size, len(tokens) - start))

        # here somehow tricky, since the whole chunk tokens is list[list[list[int]]] for corpus(doc(chunk)),so it can't be decode entirely
        chunk_token = tiktoken_model.decode_batch(chunk_token)
        for i, chunk in enumerate(chunk_token):

            results.append(
                {
                    "tokens": lengths[i],
                    "content": chunk.strip(),
                    "chunk_order_index": i,
                    "full_doc_id": doc_keys[index],
                }
            )

    return results


def chunking_by_seperators(
    tokens_list: list[list[int]],
    doc_keys,
    tiktoken_model,
    overlap_token_size=128,
    max_token_size=1024,
):

    splitter = SeparatorSplitter(
        separators=[
            tiktoken_model.encode(s) for s in PROMPTS["default_text_separator"]
        ],
        chunk_size=max_token_size,
        chunk_overlap=overlap_token_size,
    )
    results = []
    for index, tokens in enumerate(tokens_list):
        chunk_token = splitter.split_tokens(tokens)
        lengths = [len(c) for c in chunk_token]

        # here somehow tricky, since the whole chunk tokens is list[list[list[int]]] for corpus(doc(chunk)),so it can't be decode entirely
        chunk_token = tiktoken_model.decode_batch(chunk_token)
        for i, chunk in enumerate(chunk_token):

            results.append(
                {
                    "tokens": lengths[i],
                    "content": chunk.strip(),
                    "chunk_order_index": i,
                    "full_doc_id": doc_keys[index],
                }
            )

    return results


def get_chunks(new_docs, chunk_func=chunking_by_token_size, **chunk_func_params):
    inserting_chunks = {}

    new_docs_list = list(new_docs.items())
    docs = [new_doc[1]["content"] for new_doc in new_docs_list]
    doc_keys = [new_doc[0] for new_doc in new_docs_list]

    ENCODER = tiktoken.encoding_for_model("gpt-4o")
    tokens = ENCODER.encode_batch(docs, num_threads=16)
    chunks = chunk_func(
        tokens, doc_keys=doc_keys, tiktoken_model=ENCODER, **chunk_func_params
    )

    for chunk in chunks:
        inserting_chunks.update(
            {compute_mdhash_id(chunk["content"], prefix="chunk-"): chunk}
        )

    return inserting_chunks


async def _handle_entity_relation_summary(
    entity_or_relation_name: str,
    description: str,
    global_config: dict,
) -> str:
    """Summarize the entity or relation description,is used during entity extraction and when merging nodes or edges in the knowledge graph

    Args:
        entity_or_relation_name: entity or relation name
        description: description
        global_config: global configuration
    """
    use_llm_func: callable = global_config["cheap_model_func"]
    llm_max_tokens = global_config["cheap_model_max_token_size"]
    tiktoken_model_name = global_config["tiktoken_model_name"]
    summary_max_tokens = global_config["entity_summary_to_max_tokens"]

    tokens = encode_string_by_tiktoken(description, model_name=tiktoken_model_name)
    if len(tokens) < summary_max_tokens:  # No need for summary
        return description
    prompts = _get_prompts(global_config)
    prompt_template = prompts["summarize_entity_descriptions"]
    use_description = decode_tokens_by_tiktoken(
        tokens[:llm_max_tokens], model_name=tiktoken_model_name
    )
    context_base = dict(
        entity_name=entity_or_relation_name,
        description_list=use_description.split(GRAPH_FIELD_SEP),
    )
    use_prompt = prompt_template.format(**context_base)
    logger.debug(f"Trigger summary: {entity_or_relation_name}")
    summary = await use_llm_func(
        use_prompt,
        stage="entity_merge_summary",
        max_tokens=_get_stage_max_tokens(
            global_config, "entity_merge_summary", summary_max_tokens
        ),
    )
    return summary


async def _handle_single_entity_extraction(
    record_attributes: list[str],
    chunk_key: str,
):
    if len(record_attributes) < 4 or record_attributes[0] != '"entity"':
        return None
    # add this record as a node in the G
    entity_name = clean_str(record_attributes[1].upper())
    if not entity_name.strip():
        return None
    entity_type = clean_str(record_attributes[2].upper())
    entity_description = clean_str(record_attributes[3])
    entity_source_id = chunk_key
    return dict(
        entity_name=entity_name,
        entity_type=entity_type,
        description=entity_description,
        source_id=entity_source_id,
    )


async def _handle_single_relationship_extraction(
    record_attributes: list[str],
    chunk_key: str,
):
    if len(record_attributes) < 5 or record_attributes[0] != '"relationship"':
        return None
    # add this record as edge
    source = clean_str(record_attributes[1].upper())
    target = clean_str(record_attributes[2].upper())
    edge_description = clean_str(record_attributes[3])
    edge_source_id = chunk_key
    weight = (
        float(record_attributes[-1]) if is_float_regex(record_attributes[-1]) else 1.0
    )
    return dict(
        src_id=source,
        tgt_id=target,
        weight=weight,
        description=edge_description,
        source_id=edge_source_id,
    )


async def _merge_nodes_then_upsert(
    entity_name: str,
    nodes_data: list[dict],
    knwoledge_graph_inst: BaseGraphStorage,
    global_config: dict,
):
    already_entitiy_types = []
    already_source_ids = []
    already_description = []

    already_node = await knwoledge_graph_inst.get_node(entity_name)
    if already_node is not None:                                            # already exist
        already_entitiy_types.append(already_node["entity_type"])
        already_source_ids.extend(
            split_string_by_multi_markers(already_node["source_id"], [GRAPH_FIELD_SEP])
        )
        already_description.append(already_node["description"])

    entity_type = sorted(
        Counter(
            [dp["entity_type"] for dp in nodes_data] + already_entitiy_types
        ).items(),
        key=lambda x: x[1],
        reverse=True,
    )[0][0]
    description = GRAPH_FIELD_SEP.join(
        sorted(set([dp["description"] for dp in nodes_data] + already_description))
    )
    source_id = GRAPH_FIELD_SEP.join(
        set([dp["source_id"] for dp in nodes_data] + already_source_ids)
    )
    description = await _handle_entity_relation_summary(
        entity_name, description, global_config
    )
    node_data = dict(
        entity_type=entity_type,
        description=description,
        source_id=source_id,
    )
    await knwoledge_graph_inst.upsert_node(
        entity_name,
        node_data=node_data,
    )
    node_data["entity_name"] = entity_name
    return node_data


async def _merge_edges_then_upsert(
    src_id: str,
    tgt_id: str,
    edges_data: list[dict],
    knwoledge_graph_inst: BaseGraphStorage,
    global_config: dict,
):
    already_weights = []
    already_source_ids = []
    already_description = []
    already_order = []
    if await knwoledge_graph_inst.has_edge(src_id, tgt_id):
        already_edge = await knwoledge_graph_inst.get_edge(src_id, tgt_id)
        already_weights.append(already_edge["weight"])
        already_source_ids.extend(
            split_string_by_multi_markers(already_edge["source_id"], [GRAPH_FIELD_SEP])
        )
        already_description.append(already_edge["description"])
        already_order.append(already_edge.get("order", 1))

    # [numberchiffre]: `Relationship.order` is only returned from DSPy's predictions
    order = min([dp.get("order", 1) for dp in edges_data] + already_order)
    weight = sum([dp["weight"] for dp in edges_data] + already_weights)
    description = GRAPH_FIELD_SEP.join(
        sorted(set([dp["description"] for dp in edges_data] + already_description))
    )
    source_id = GRAPH_FIELD_SEP.join(
        set([dp["source_id"] for dp in edges_data] + already_source_ids)
    )
    for need_insert_id in [src_id, tgt_id]:
        if not (await knwoledge_graph_inst.has_node(need_insert_id)):
            await knwoledge_graph_inst.upsert_node(
                need_insert_id,
                node_data={
                    "source_id": source_id,
                    "description": description,
                    "entity_type": '"UNKNOWN"',
                },
            )
    description = await _handle_entity_relation_summary(
        (src_id, tgt_id), description, global_config
    )
    await knwoledge_graph_inst.upsert_edge(
        src_id,
        tgt_id,
        edge_data=dict(
            weight=weight, description=description, source_id=source_id, order=order
        ),
    )

async def extract_hierarchical_entities(
    chunks: dict[str, TextChunkSchema],
    knowledge_graph_inst: BaseGraphStorage,
    entity_vdb: BaseVectorStorage,
    global_config: dict,
)-> Union[BaseGraphStorage, None]:
    """Extract entities and relations from text chunks

    Args:
        chunks: text chunks
        knowledge_graph_inst: knowledge graph instance
        entity_vdb: entity vector database
        global_config: global configuration

    Returns:
        Union[BaseGraphStorage, None]: knowledge graph instance
    """
    use_llm_func: callable = global_config["best_model_func"]
    entity_extract_max_gleaning = global_config["entity_extract_max_gleaning"]

    ordered_chunks = list(chunks.items())
    prompts = _get_prompts(global_config)
    entity_extract_prompt = prompts["hi_entity_extraction"]
    relation_extract_prompt = prompts["hi_relation_extraction"]

    context_base_entity = dict(
        tuple_delimiter=prompts["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=prompts["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=prompts["DEFAULT_COMPLETION_DELIMITER"],
        entity_types=",".join(prompts["META_ENTITY_TYPES"])
    )
    continue_prompt = prompts["entiti_continue_extraction"]
    if_loop_prompt = prompts["entiti_if_loop_extraction"]

    already_processed = 0
    already_entities = 0
    already_relations = 0

    async def _process_single_content_entity(chunk_key_dp: tuple[str, TextChunkSchema]):           # for each chunk, run the func
        nonlocal already_processed, already_entities, already_relations
        chunk_key = chunk_key_dp[0]
        chunk_dp = chunk_key_dp[1]
        content = chunk_dp["content"]
        hint_prompt = entity_extract_prompt.format(**context_base_entity, input_text=content)      # fill in the parameter
        final_result = await use_llm_func(
            hint_prompt,
            stage="entity_extract",
            max_tokens=_get_stage_max_tokens(global_config, "entity_extract"),
        )

        # check if need gleaning
        history = pack_user_ass_to_openai_messages(hint_prompt, final_result)               # set as history
        if_loop_result: str = await use_llm_func(
            if_loop_prompt,
            history_messages=history,
            stage="entity_glean_check",
            max_tokens=_get_stage_max_tokens(global_config, "entity_glean_check"),
        )
        if_loop_result = if_loop_result.strip().strip('"').strip("'").lower()
        if if_loop_result == "yes":
            logger.info(f"\033[91m[Found Missed Entities, Gleaning {entity_extract_max_gleaning} times]\033[0m")
            for now_glean_index in range(entity_extract_max_gleaning):
                glean_result = await use_llm_func(
                    continue_prompt,
                    history_messages=history,
                    stage="entity_glean_continue",
                    max_tokens=_get_stage_max_tokens(global_config, "entity_glean_continue"),
                )
                if not glean_result:
                    glean_result = ""
                history += pack_user_ass_to_openai_messages(continue_prompt, glean_result)      # add to history
                final_result += glean_result
                if now_glean_index == entity_extract_max_gleaning - 1:
                    break
                if_loop_result: str = await use_llm_func(                                       # judge if we still need the next iteration
                    if_loop_prompt,
                    history_messages=history,
                    stage="entity_glean_check",
                    max_tokens=_get_stage_max_tokens(global_config, "entity_glean_check"),
                )
                if_loop_result = if_loop_result.strip().strip('"').strip("'").lower()
                if if_loop_result != "yes":
                    break
        else:
            logger.info(f"\033[92m[No Missed Entities]\033[0m")

        records = split_string_by_multi_markers(                                            # split entities from result --> list of entities
            final_result,
            [context_base_entity["record_delimiter"], context_base_entity["completion_delimiter"]],
        )
        # resolve the entities
        maybe_nodes = defaultdict(list)
        maybe_edges = defaultdict(list)
        for record in records:
            record = re.search(r"\((.*)\)", record)
            if record is None:
                continue
            record = record.group(1)
            record_attributes = split_string_by_multi_markers(          # split entity
                record, [context_base_entity["tuple_delimiter"]]
            )
            if_entities = await _handle_single_entity_extraction(       # get the name, type, desc, source_id of entity--> dict
                record_attributes, chunk_key
            )
            if if_entities is not None:
                maybe_nodes[if_entities["entity_name"]].append(if_entities)
                continue

            if_relation = await _handle_single_relationship_extraction(
                record_attributes, chunk_key
            )
            if if_relation is not None:
                maybe_edges[(if_relation["src_id"], if_relation["tgt_id"])].append(
                    if_relation
                )
        already_processed += 1                                      # already processed chunks
        already_entities += len(maybe_nodes)
        already_relations += len(maybe_edges)
        now_ticks = prompts["process_tickers"][
            already_processed % len(prompts["process_tickers"])
        ]
        print(
            f"{now_ticks} Processed {already_processed}({already_processed*100//len(ordered_chunks)}%) chunks,  {already_entities} entities(duplicated), {already_relations} relations(duplicated)\r",
            end="",
            flush=True,
        )
        return dict(maybe_nodes), dict(maybe_edges)
    
    # extract entities
    # use_llm_func is wrapped in ascynio.Semaphore, limiting max_async callings
    entity_results = await asyncio.gather(
        *[_process_single_content_entity(c) for c in ordered_chunks]
    )
    print()  # clear the progress bar

    # fetch all entities from results
    all_entities = {}
    for item in entity_results:
        for k, v in item[0].items():
            value = v[0]
            all_entities[k] = v[0]
    context_entities = {key[0]: list(x[0].keys()) for key, x in zip(ordered_chunks, entity_results)}
    
    # fetch embeddings
    entity_discriptions = [v["description"] for k, v in all_entities.items()]
    entity_sequence_embeddings = []
    embeddings_batch_size = 64
    num_embeddings_batches = (len(entity_discriptions) + embeddings_batch_size - 1) // embeddings_batch_size
    for i in range(num_embeddings_batches):
        start_index = i * embeddings_batch_size
        end_index = min((i + 1) * embeddings_batch_size, len(entity_discriptions))
        batch = entity_discriptions[start_index:end_index]
        result = await entity_vdb.embedding_func(batch)
        entity_sequence_embeddings.extend(result)
    entity_embeddings = entity_sequence_embeddings
    for (k, v), x in zip(all_entities.items(), entity_embeddings):
        value = v
        value["embedding"] = x
        all_entities[k] = value
    
    already_processed = 0
    async def _process_single_content_relation(chunk_key_dp: tuple[str, TextChunkSchema]):           # for each chunk, run the func
        nonlocal already_processed, already_entities, already_relations
        chunk_key = chunk_key_dp[0]
        chunk_dp = chunk_key_dp[1]
        content = chunk_dp["content"]

        entities = context_entities[chunk_key]
        context_base_relation = dict(
            tuple_delimiter=prompts["DEFAULT_TUPLE_DELIMITER"],
            record_delimiter=prompts["DEFAULT_RECORD_DELIMITER"],
            completion_delimiter=prompts["DEFAULT_COMPLETION_DELIMITER"],
            entities=",".join(entities)
            )
        hint_prompt = relation_extract_prompt.format(**context_base_relation, input_text=content)
        final_result = await use_llm_func(
            hint_prompt,
            stage="relation_extract",
            max_tokens=_get_stage_max_tokens(global_config, "relation_extract"),
        )

        # check if need gleaning
        history = pack_user_ass_to_openai_messages(hint_prompt, final_result)               # set as history
        if_loop_result: str = await use_llm_func(                                       # judge if we still need the next iteration
            if_loop_prompt,
            history_messages=history,
            stage="relation_glean_check",
            max_tokens=_get_stage_max_tokens(global_config, "relation_glean_check"),
        )
        if_loop_result = if_loop_result.strip().strip('"').strip("'").lower()
        if if_loop_result == "yes":
            logger.info(f"\033[91m[Found Missed Relations, Gleaning {entity_extract_max_gleaning} times]\033[0m")
            for now_glean_index in range(entity_extract_max_gleaning):
                glean_result = await use_llm_func(
                    continue_prompt,
                    history_messages=history,
                    stage="relation_glean_continue",
                    max_tokens=_get_stage_max_tokens(global_config, "relation_glean_continue"),
                )

                history += pack_user_ass_to_openai_messages(continue_prompt, glean_result)      # add to history
                final_result += glean_result
                if now_glean_index == entity_extract_max_gleaning - 1:
                    break

                if_loop_result: str = await use_llm_func(                                       # judge if we still need the next iteration
                    if_loop_prompt,
                    history_messages=history,
                    stage="relation_glean_check",
                    max_tokens=_get_stage_max_tokens(global_config, "relation_glean_check"),
                )
                if_loop_result = if_loop_result.strip().strip('"').strip("'").lower()
                if if_loop_result != "yes":
                    break
        else:
            logger.info(f"\033[92m[No Missed Relations]\033[0m")

        records = split_string_by_multi_markers(                                            # split entities from result --> list of entities
            final_result,
            [context_base_relation["record_delimiter"], context_base_relation["completion_delimiter"]],
        )
        # resolve the entities
        maybe_nodes = defaultdict(list)
        maybe_edges = defaultdict(list)
        for record in records:
            record = re.search(r"\((.*)\)", record)
            if record is None:
                continue
            record = record.group(1)
            record_attributes = split_string_by_multi_markers(          # split entity
                record, [context_base_relation["tuple_delimiter"]]
            )
            if_entities = await _handle_single_entity_extraction(       # get the name, type, desc, source_id of entity--> dict
                record_attributes, chunk_key
            )
            if if_entities is not None:
                maybe_nodes[if_entities["entity_name"]].append(if_entities)
                continue

            if_relation = await _handle_single_relationship_extraction(
                record_attributes, chunk_key
            )
            if if_relation is not None:
                maybe_edges[(if_relation["src_id"], if_relation["tgt_id"])].append(
                    if_relation
                )
        already_processed += 1                                      # already processed chunks
        already_entities += len(maybe_nodes)
        already_relations += len(maybe_edges)
        now_ticks = prompts["process_tickers"][
            already_processed % len(prompts["process_tickers"])
        ]
        print(
            f"{now_ticks} Processed {already_processed}({already_processed*100//len(ordered_chunks)}%) chunks,  {already_entities} entities(duplicated), {already_relations} relations(duplicated)\r",
            end="",
            flush=True,
        )
        return dict(maybe_nodes), dict(maybe_edges)

    # extract entities
    # use_llm_func is wrapped in ascynio.Semaphore, limiting max_async callings
    relation_results = await asyncio.gather(
        *[_process_single_content_relation(c) for c in ordered_chunks]
    )
    print()

    # fetch all relations from results
    all_relations = {}
    for item in relation_results:
        for k, v in item[1].items():
            all_relations[k] = v
    
    # TODO: hierarchical clustering
    logger.info(f"\033[94m[Hierarchical Clustering]\033[0m")
    hierarchical_cluster = Hierarchical_Clustering()
    hierarchical_clustered_entities_relations = await hierarchical_cluster.perform_clustering(entity_vdb=entity_vdb, global_config=global_config, entities=all_entities)
    hierarchical_clustered_entities = [[x for x in y if "entity_name" in x.keys()] for y in hierarchical_clustered_entities_relations]
    hierarchical_clustered_relations = [[x for x in y if "src_id" in x.keys()] for y in hierarchical_clustered_entities_relations]
    
    maybe_nodes = defaultdict(list)                     # for all chunks
    maybe_edges = defaultdict(list)
    # extracted entities and relations
    for m_nodes, m_edges in zip(entity_results, relation_results):
        for k, v in m_nodes[0].items():
            maybe_nodes[k].extend(v)
        for k, v in m_edges[1].items():
            # it's undirected graph
            maybe_edges[tuple(sorted(k))].extend(v)
    # clustered entities
    for cluster_layer in hierarchical_clustered_entities:
        for item in cluster_layer:
            maybe_nodes[item['entity_name']].extend([item])
    # clustered relations
    for cluster_layer in hierarchical_clustered_relations:
        for item in cluster_layer:
            maybe_edges[tuple(sorted((item["src_id"], item["tgt_id"])))].extend([item])
    # store the nodes
    all_entities_data = await asyncio.gather(           
        *[
            _merge_nodes_then_upsert(k, v, knowledge_graph_inst, global_config)
            for k, v in maybe_nodes.items()
        ]
    )
    # store the edges
    await asyncio.gather(                               
        *[
            _merge_edges_then_upsert(k[0], k[1], v, knowledge_graph_inst, global_config)
            for k, v in maybe_edges.items()
        ]
    )
    if not len(all_entities_data):
        logger.warning("Didn't extract any entities, maybe your LLM is not working")
        return None
    if entity_vdb is not None:
        data_for_vdb = {                                # key is the md5 hash of the entity name string
            compute_mdhash_id(dp["entity_name"], prefix="ent-"): {
                "content": dp["entity_name"] + dp["description"],   # entity name and description construct the content
                "entity_name": dp["entity_name"],
            }
            for dp in all_entities_data
        }
        await entity_vdb.upsert(data_for_vdb)
    return knowledge_graph_inst

async def extract_entities(
    chunks: dict[str, TextChunkSchema],
    knwoledge_graph_inst: BaseGraphStorage,
    entity_vdb: BaseVectorStorage,
    global_config: dict,
) -> Union[BaseGraphStorage, None]:
    use_llm_func: callable = global_config["best_model_func"]
    entity_extract_max_gleaning = global_config["entity_extract_max_gleaning"]

    ordered_chunks = list(chunks.items())                       # chunks

    prompts = _get_prompts(global_config)
    entity_extract_prompt = prompts["entity_extraction"]
    context_base = dict(
        tuple_delimiter=prompts["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=prompts["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=prompts["DEFAULT_COMPLETION_DELIMITER"],
        entity_types=",".join(prompts["DEFAULT_ENTITY_TYPES"]),
    )
    continue_prompt = prompts["entiti_continue_extraction"]
    if_loop_prompt = prompts["entiti_if_loop_extraction"]

    already_processed = 0
    already_entities = 0
    already_relations = 0

    async def _process_single_content(chunk_key_dp: tuple[str, TextChunkSchema]):           # for each chunk, run the func
        nonlocal already_processed, already_entities, already_relations
        chunk_key = chunk_key_dp[0]
        chunk_dp = chunk_key_dp[1]
        content = chunk_dp["content"]
        hint_prompt = entity_extract_prompt.format(**context_base, input_text=content)      # fill in the parameter
        final_result = await use_llm_func(
            hint_prompt,
            stage="entity_extract",
            max_tokens=_get_stage_max_tokens(global_config, "entity_extract"),
        )

        # check if need gleaning
        history = pack_user_ass_to_openai_messages(hint_prompt, final_result)               # set as history
        if_loop_result: str = await use_llm_func(
            if_loop_prompt,
            history_messages=history,
            stage="entity_glean_check",
            max_tokens=_get_stage_max_tokens(global_config, "entity_glean_check"),
        )
        if_loop_result = if_loop_result.strip().strip('"').strip("'").lower()
        if if_loop_result == "yes":
            logger.info(f"\033[91m[Found Missed Entities, Gleaning {entity_extract_max_gleaning} times]\033[0m")
            for now_glean_index in range(entity_extract_max_gleaning):
                glean_result = await use_llm_func(
                    continue_prompt,
                    history_messages=history,
                    stage="entity_glean_continue",
                    max_tokens=_get_stage_max_tokens(global_config, "entity_glean_continue"),
                )
                history += pack_user_ass_to_openai_messages(continue_prompt, glean_result)      # add to history
                final_result += glean_result
                if now_glean_index == entity_extract_max_gleaning - 1:
                    break
                if_loop_result: str = await use_llm_func(                                       # judge if we still need the next iteration
                    if_loop_prompt,
                    history_messages=history,
                    stage="entity_glean_check",
                    max_tokens=_get_stage_max_tokens(global_config, "entity_glean_check"),
                )
                if_loop_result = if_loop_result.strip().strip('"').strip("'").lower()
                if if_loop_result != "yes":
                    break
        else:
            logger.info(f"\033[92m[No Missed Entities]\033[0m")

        records = split_string_by_multi_markers(                                            # split entities from result --> list of entities
            final_result,
            [context_base["record_delimiter"], context_base["completion_delimiter"]],
        )

        maybe_nodes = defaultdict(list)
        maybe_edges = defaultdict(list)
        for record in records:
            record = re.search(r"\((.*)\)", record)
            if record is None:
                continue
            record = record.group(1)
            record_attributes = split_string_by_multi_markers(          # split entity
                record, [context_base["tuple_delimiter"]]
            )
            if_entities = await _handle_single_entity_extraction(       # get the name, type, desc, source_id of entity--> dict
                record_attributes, chunk_key
            )
            if if_entities is not None:
                maybe_nodes[if_entities["entity_name"]].append(if_entities)
                continue

            if_relation = await _handle_single_relationship_extraction(
                record_attributes, chunk_key
            )
            if if_relation is not None:
                maybe_edges[(if_relation["src_id"], if_relation["tgt_id"])].append(
                    if_relation
                )
        already_processed += 1                                      # already processed chunks
        already_entities += len(maybe_nodes)
        already_relations += len(maybe_edges)
        now_ticks = prompts["process_tickers"][
            already_processed % len(prompts["process_tickers"])
        ]
        print(
            f"{now_ticks} Processed {already_processed}({already_processed*100//len(ordered_chunks)}%) chunks,  {already_entities} entities(duplicated), {already_relations} relations(duplicated)\r",
            end="",
            flush=True,
        )
        return dict(maybe_nodes), dict(maybe_edges)

    # use_llm_func is wrapped in ascynio.Semaphore, limiting max_async callings
    results = await asyncio.gather(
        *[_process_single_content(c) for c in ordered_chunks]
    )
    print()  # clear the progress bar
    maybe_nodes = defaultdict(list)                     # for all chunks
    maybe_edges = defaultdict(list)
    for m_nodes, m_edges in results:
        for k, v in m_nodes.items():
            maybe_nodes[k].extend(v)
        for k, v in m_edges.items():
            # it's undirected graph
            maybe_edges[tuple(sorted(k))].extend(v)
    all_entities_data = await asyncio.gather(           # store the nodes
        *[
            _merge_nodes_then_upsert(k, v, knwoledge_graph_inst, global_config)
            for k, v in maybe_nodes.items()
        ]
    )
    await asyncio.gather(                               # store the edges
        *[
            _merge_edges_then_upsert(k[0], k[1], v, knwoledge_graph_inst, global_config)
            for k, v in maybe_edges.items()
        ]
    )
    if not len(all_entities_data):
        logger.warning("Didn't extract any entities, maybe your LLM is not working")
        return None
    if entity_vdb is not None:
        data_for_vdb = {                                # key is the md5 hash of the entity name string
            compute_mdhash_id(dp["entity_name"], prefix="ent-"): {
                "content": dp["entity_name"] + dp["description"],   # entity name and description construct the content
                "entity_name": dp["entity_name"],
            }
            for dp in all_entities_data
        }
        await entity_vdb.upsert(data_for_vdb)
    return knwoledge_graph_inst


def _pack_single_community_by_sub_communities(
    community: SingleCommunitySchema,
    max_token_size: int,
    already_reports: dict[str, CommunitySchema],
) -> tuple[str, int]:
    # TODO
    all_sub_communities = [
        already_reports[k] for k in community["sub_communities"] if k in already_reports
    ]
    all_sub_communities = sorted(
        all_sub_communities, key=lambda x: x["occurrence"], reverse=True
    )
    may_trun_all_sub_communities = truncate_list_by_token_size(
        all_sub_communities,
        key=lambda x: x["report_string"],
        max_token_size=max_token_size,
    )
    sub_fields = ["id", "report", "rating", "importance"]
    sub_communities_describe = list_of_list_to_csv(
        [sub_fields]
        + [
            [
                i,
                c["report_string"],
                c["report_json"].get("rating", -1),
                c["occurrence"],
            ]
            for i, c in enumerate(may_trun_all_sub_communities)
        ]
    )
    already_nodes = []
    already_edges = []
    for c in may_trun_all_sub_communities:
        already_nodes.extend(c["nodes"])
        already_edges.extend([tuple(e) for e in c["edges"]])
    return (
        sub_communities_describe,
        len(encode_string_by_tiktoken(sub_communities_describe)),
        set(already_nodes),
        set(already_edges),
    )


async def _pack_single_community_describe(
    knwoledge_graph_inst: BaseGraphStorage,
    community: SingleCommunitySchema,
    max_token_size: int = 12000,
    already_reports: dict[str, CommunitySchema] = {},
    global_config: dict = {},
) -> str:
    nodes_in_order = sorted(community["nodes"])
    edges_in_order = sorted(community["edges"], key=lambda x: x[0] + x[1])

    nodes_data = await asyncio.gather(
        *[knwoledge_graph_inst.get_node(n) for n in nodes_in_order]
    )
    edges_data = await asyncio.gather(
        *[knwoledge_graph_inst.get_edge(src, tgt) for src, tgt in edges_in_order]
    )
    node_fields = ["id", "entity", "type", "description", "degree"]
    edge_fields = ["id", "source", "target", "description", "rank"]
    nodes_list_data = [
        [
            i,
            node_name,
            node_data.get("entity_type", "UNKNOWN"),
            node_data.get("description", "UNKNOWN"),
            await knwoledge_graph_inst.node_degree(node_name),
        ]
        for i, (node_name, node_data) in enumerate(zip(nodes_in_order, nodes_data))
    ]
    nodes_list_data = sorted(nodes_list_data, key=lambda x: x[-1], reverse=True)
    nodes_may_truncate_list_data = truncate_list_by_token_size(
        nodes_list_data, key=lambda x: x[3], max_token_size=max_token_size // 2
    )
    edges_list_data = [
        [
            i,
            edge_name[0],
            edge_name[1],
            edge_data.get("description", "UNKNOWN"),
            await knwoledge_graph_inst.edge_degree(*edge_name),
        ]
        for i, (edge_name, edge_data) in enumerate(zip(edges_in_order, edges_data))
    ]
    edges_list_data = sorted(edges_list_data, key=lambda x: x[-1], reverse=True)
    edges_may_truncate_list_data = truncate_list_by_token_size(
        edges_list_data, key=lambda x: x[3], max_token_size=max_token_size // 2
    )

    truncated = len(nodes_list_data) > len(nodes_may_truncate_list_data) or len(
        edges_list_data
    ) > len(edges_may_truncate_list_data)

    # If context is exceed the limit and have sub-communities:
    report_describe = ""
    need_to_use_sub_communities = (
        truncated and len(community["sub_communities"]) and len(already_reports)
    )
    force_to_use_sub_communities = global_config["addon_params"].get(
        "force_to_use_sub_communities", False
    )
    if need_to_use_sub_communities or force_to_use_sub_communities:
        logger.debug(
            f"Community {community['title']} exceeds the limit or you set force_to_use_sub_communities to True, using its sub-communities"
        )
        report_describe, report_size, contain_nodes, contain_edges = (
            _pack_single_community_by_sub_communities(
                community, max_token_size, already_reports
            )
        )
        report_exclude_nodes_list_data = [
            n for n in nodes_list_data if n[1] not in contain_nodes
        ]
        report_include_nodes_list_data = [
            n for n in nodes_list_data if n[1] in contain_nodes
        ]
        report_exclude_edges_list_data = [
            e for e in edges_list_data if (e[1], e[2]) not in contain_edges
        ]
        report_include_edges_list_data = [
            e for e in edges_list_data if (e[1], e[2]) in contain_edges
        ]
        # if report size is bigger than max_token_size, nodes and edges are []
        nodes_may_truncate_list_data = truncate_list_by_token_size(
            report_exclude_nodes_list_data + report_include_nodes_list_data,
            key=lambda x: x[3],
            max_token_size=(max_token_size - report_size) // 2,
        )
        edges_may_truncate_list_data = truncate_list_by_token_size(
            report_exclude_edges_list_data + report_include_edges_list_data,
            key=lambda x: x[3],
            max_token_size=(max_token_size - report_size) // 2,
        )
    nodes_describe = list_of_list_to_csv([node_fields] + nodes_may_truncate_list_data)
    edges_describe = list_of_list_to_csv([edge_fields] + edges_may_truncate_list_data)
    return f"""-----Reports-----
```csv
{report_describe}
```
-----Entities-----
```csv
{nodes_describe}
```
-----Relationships-----
```csv
{edges_describe}
```"""


def estimate_community_describe_tokens(
    prompt_template: str, packed_input_tokens: int
) -> int:
    overhead_prompt = prompt_template.replace("{input_text}", "")
    return len(encode_string_by_tiktoken(overhead_prompt)) + packed_input_tokens


def _community_report_json_to_str(parsed_output: dict) -> str:
    """refer official graphrag: index/graph/extractors/community_reports"""
    title = parsed_output.get("title", "Report")
    summary = parsed_output.get("summary", "")
    findings = parsed_output.get("findings", [])

    def finding_summary(finding: dict):
        if isinstance(finding, str):
            return finding
        return finding.get("summary")

    def finding_explanation(finding: dict):
        if isinstance(finding, str):
            return ""
        return finding.get("explanation")

    report_sections = "\n\n".join(
        f"## {finding_summary(f)}\n\n{finding_explanation(f)}" for f in findings
    )
    return f"# {title}\n\n{summary}\n\n{report_sections}"


async def generate_community_report(
    community_report_kv: BaseKVStorage[CommunitySchema],
    knwoledge_graph_inst: BaseGraphStorage,
    global_config: dict,
):
    llm_extra_kwargs = global_config["special_community_report_llm_kwargs"]
    use_llm_func: callable = global_config["best_model_func"]
    use_string_json_convert_func: callable = global_config[
        "convert_response_to_json_func"
    ]

    prompts = _get_prompts(global_config)
    community_report_prompt = prompts["community_report"]

    communities_schema = await knwoledge_graph_inst.community_schema()
    community_keys, community_values = list(communities_schema.keys()), list(
        communities_schema.values()
    )
    already_processed = 0

    async def _form_single_community_report(
        community: SingleCommunitySchema, already_reports: dict[str, CommunitySchema]
    ):
        nonlocal already_processed
        describe = await _pack_single_community_describe(
            knwoledge_graph_inst,
            community,
            max_token_size=global_config.get(
                "community_report_input_max_tokens",
                global_config["best_model_max_token_size"],
            ),
            already_reports=already_reports,
            global_config=global_config,
        )
        prompt = community_report_prompt.format(input_text=describe)
        response = await use_llm_func(
            prompt,
            stage="community_report",
            max_tokens=_get_stage_max_tokens(global_config, "community_report"),
            **llm_extra_kwargs,
        )
        data = use_string_json_convert_func(response)
        already_processed += 1
        now_ticks = prompts["process_tickers"][
            already_processed % len(prompts["process_tickers"])
        ]
        print(
            f"{now_ticks} Processed {already_processed} communities\r",
            end="",
            flush=True,
        )
        return data

    levels = sorted(set([c["level"] for c in community_values]), reverse=True)
    logger.info(f"Generating by levels: {levels}")
    community_datas = {}
    for level in levels:
        this_level_community_keys, this_level_community_values = zip(
            *[
                (k, v)
                for k, v in zip(community_keys, community_values)
                if v["level"] == level
            ]
        )
        this_level_communities_reports = await asyncio.gather(
            *[
                _form_single_community_report(c, community_datas)
                for c in this_level_community_values
            ]
        )
        community_datas.update(
            {
                k: {
                    "report_string": _community_report_json_to_str(r),
                    "report_json": r,
                    **v,
                }
                for k, r, v in zip(
                    this_level_community_keys,
                    this_level_communities_reports,
                    this_level_community_values,
                )
            }
        )
    print()  # clear the progress bar
    await community_report_kv.upsert(community_datas)


async def _find_most_related_community_from_entities(
    node_datas: list[dict],
    query_param: QueryParam,
    community_reports: BaseKVStorage[CommunitySchema],
):
    related_communities = []
    for node_d in node_datas:
        if "clusters" not in node_d:
            continue
        related_communities.extend(json.loads(node_d["clusters"]))
    related_community_dup_keys = [
        str(dp["cluster"])
        for dp in related_communities
        if dp["level"] <= query_param.level
    ]
    related_community_keys_counts = dict(Counter(related_community_dup_keys))
    _related_community_datas = await asyncio.gather(        # get community reports
        *[community_reports.get_by_id(k) for k in related_community_keys_counts.keys()]
    )
    related_community_datas = {
        k: v
        for k, v in zip(related_community_keys_counts.keys(), _related_community_datas)
        if v is not None
    }
    related_community_keys = sorted(            # sort by ratings
        related_community_keys_counts.keys(),
        key=lambda k: (
            related_community_keys_counts[k],
            related_community_datas[k]["report_json"].get("rating", -1),
        ),
        reverse=True,
    )
    sorted_community_datas = [                  # community reports sorted by ratings
        related_community_datas[k] for k in related_community_keys
    ]

    use_community_reports = truncate_list_by_token_size(    # in case community reprot is longer than token limitation
        sorted_community_datas,
        key=lambda x: x["report_string"],
        max_token_size=query_param.max_token_for_community_report,
    )
    if query_param.community_single_one:
        use_community_reports = use_community_reports[:1]
    return use_community_reports


async def _find_most_related_text_unit_from_entities(
    node_datas: list[dict],
    query_param: QueryParam,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    knowledge_graph_inst: BaseGraphStorage,
):
    text_units = [      # the entities related to the retrieved entities
        split_string_by_multi_markers(dp["source_id"], [GRAPH_FIELD_SEP])
        for dp in node_datas
    ]
    edges = await asyncio.gather(   # get relations related to the retrieved entities
        *[knowledge_graph_inst.get_node_edges(dp["entity_name"]) for dp in node_datas]
    )                               # where the source entities are the retrieved entities
    all_one_hop_nodes = set()       # find the one hop neighbors
    for this_edges in edges:
        if not this_edges:
            continue
        all_one_hop_nodes.update([e[1] for e in this_edges])
    all_one_hop_nodes = list(all_one_hop_nodes)
    all_one_hop_nodes_data = await asyncio.gather(  # get node information from storage
        *[knowledge_graph_inst.get_node(e) for e in all_one_hop_nodes]
    )
    all_one_hop_text_units_lookup = {               # find the text chunks of the 1-hop neighbors entities
        k: set(split_string_by_multi_markers(v["source_id"], [GRAPH_FIELD_SEP]))
        for k, v in zip(all_one_hop_nodes, all_one_hop_nodes_data)
        if v is not None
    }
    all_text_units_lookup = {}
    for index, (this_text_units, this_edges) in enumerate(zip(text_units, edges)):
        for c_id in this_text_units:
            if c_id in all_text_units_lookup:
                continue
            relation_counts = 0
            for e in this_edges:
                if (
                    e[1] in all_one_hop_text_units_lookup
                    and c_id in all_one_hop_text_units_lookup[e[1]]
                ):
                    relation_counts += 1
            all_text_units_lookup[c_id] = {
                "data": await text_chunks_db.get_by_id(c_id),
                "order": index,
                "relation_counts": relation_counts,     # count of relations related to the chunk
            }
    if any([v is None for v in all_text_units_lookup.values()]):
        logger.warning("Text chunks are missing, maybe the storage is damaged")
    all_text_units = [
        {"id": k, **v} for k, v in all_text_units_lookup.items() if v is not None
    ]
    all_text_units = sorted(        # sort by relation counts
        all_text_units, key=lambda x: (x["order"], -x["relation_counts"])
    )
    all_text_units = truncate_list_by_token_size(
        all_text_units,
        key=lambda x: x["data"]["content"],
        max_token_size=query_param.max_token_for_text_unit,
    )
    all_text_units: list[TextChunkSchema] = [t["data"] for t in all_text_units]
    return all_text_units


async def _find_most_related_edges_from_entities(
    node_datas: list[dict],
    query_param: QueryParam,
    knowledge_graph_inst: BaseGraphStorage,
):
    all_related_edges = await asyncio.gather(
        *[knowledge_graph_inst.get_node_edges(dp["entity_name"]) for dp in node_datas]
    )
    all_edges = set()
    for this_edges in all_related_edges:
        all_edges.update([tuple(sorted(e)) for e in this_edges])
    all_edges = list(all_edges)
    all_edges_pack = await asyncio.gather(
        *[knowledge_graph_inst.get_edge(e[0], e[1]) for e in all_edges]
    )
    all_edges_degree = await asyncio.gather(
        *[knowledge_graph_inst.edge_degree(e[0], e[1]) for e in all_edges]
    )
    all_edges_data = [
        {"src_tgt": k, "rank": d, **v}
        for k, v, d in zip(all_edges, all_edges_pack, all_edges_degree)
        if v is not None
    ]
    all_edges_data = sorted(
        all_edges_data, key=lambda x: (x["rank"], x["weight"]), reverse=True
    )
    all_edges_data = truncate_list_by_token_size(
        all_edges_data,
        key=lambda x: x["description"],
        max_token_size=query_param.max_token_for_local_context,
    )
    return all_edges_data


async def _find_most_related_edges_from_paths(
    path_datas: list[dict],
    path: list[str],
    query_param: QueryParam,
    knowledge_graph_inst: BaseGraphStorage,
):
    # all_related_edges = await asyncio.gather(
    #     *[knowledge_graph_inst.get_node_edges(dp["entity_name"]) for dp in node_datas]
    # )
    # all_reasoning_path = await asyncio.gather(
    #                         *[knowledge_graph_inst.get_edge(e[0], e[1]) for e in knowledge_graph_inst._graph.subgraph(path).edges()]
    #                     )
    all_reasoning_path = await knowledge_graph_inst.subgraph_edges(path)
    all_edges = set()
    all_edges.update([tuple(sorted(e[:2])) for e in all_reasoning_path])
    all_edges = list(all_edges)
    all_edges_pack = await asyncio.gather(
        *[knowledge_graph_inst.get_edge(e[0], e[1]) for e in all_edges]
    )
    all_edges_degree = await asyncio.gather(
        *[knowledge_graph_inst.edge_degree(e[0], e[1]) for e in all_edges]
    )
    all_edges_data = [
        {"src_tgt": k, "rank": d, **v}
        for k, v, d in zip(all_edges, all_edges_pack, all_edges_degree)
        if v is not None
    ]
    all_edges_data = sorted(
        all_edges_data, key=lambda x: (x["rank"], x["weight"]), reverse=True
    )
    all_edges_data = truncate_list_by_token_size(
        all_edges_data,
        key=lambda x: x["description"],
        max_token_size=query_param.max_token_for_bridge_knowledge,
    )
    return all_edges_data


def _networkx_graph_from_storage(knowledge_graph_inst: BaseGraphStorage) -> nx.Graph | None:
    return getattr(knowledge_graph_inst, "_graph", None)


def _edge_key(left: str, right: str) -> tuple[str, str]:
    return tuple(sorted((left, right)))


def _edge_description_text(graph: nx.Graph, source: str, target: str, edge_data: dict) -> str:
    source_data = graph.nodes.get(source, {})
    target_data = graph.nodes.get(target, {})
    return "\n".join(
        [
            f"source: {source}",
            str(source_data.get("description", "")),
            f"relation: {edge_data.get('description', '')}",
            f"target: {target}",
            str(target_data.get("description", "")),
        ]
    )


_SNIPPET_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "to",
    "what",
    "when",
    "where",
    "which",
    "with",
}


def _snippet_tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) > 1 and token not in _SNIPPET_STOPWORDS
    }


def _query_overlap_snippet(content: str, query: str, snippet_chars: int) -> str:
    query_tokens = _snippet_tokens(query)
    if not query_tokens:
        return content[:snippet_chars].rstrip()
    if len(content) <= snippet_chars:
        return content

    pieces = [
        piece.strip()
        for piece in re.split(r"(?<=[.!?])\s+|\n+", content)
        if piece.strip()
    ]
    if not pieces:
        pieces = [
            content[start : start + snippet_chars]
            for start in range(0, len(content), max(1, snippet_chars // 2))
        ]

    best_text = ""
    best_score: tuple[int, float, int] | None = None
    window: list[str] = []
    window_len = 0
    for piece in pieces:
        if len(piece) > snippet_chars:
            sub_windows = [
                piece[start : start + snippet_chars]
                for start in range(0, len(piece), max(1, snippet_chars // 2))
            ]
            candidates = sub_windows
        else:
            candidates = [piece]
        for candidate in candidates:
            if len(candidate) > snippet_chars:
                candidate = candidate[:snippet_chars]
            while window and window_len + len(candidate) + 1 > snippet_chars:
                removed = window.pop(0)
                window_len -= len(removed) + 1
            window.append(candidate)
            window_len += len(candidate) + 1
            current = " ".join(window).strip()
            current_tokens = _snippet_tokens(current)
            overlap = len(query_tokens & current_tokens)
            density = overlap / max(1, len(current_tokens))
            score = (overlap, density, -len(current))
            if best_score is None or score > best_score:
                best_score = score
                best_text = current
    if not best_text:
        best_text = content[:snippet_chars]
    return best_text[:snippet_chars].rstrip()


def _text_unit_context_content(
    text_unit: dict,
    query_param: QueryParam,
    query: str | None = None,
) -> str:
    content = str(text_unit.get("content", ""))
    snippet_chars = query_param.text_unit_snippet_chars
    if snippet_chars is None or snippet_chars <= 0 or len(content) <= snippet_chars:
        return content
    if query_param.text_unit_snippet_strategy == "query_overlap" and query:
        return _query_overlap_snippet(content, query, snippet_chars) + " ..."
    return content[:snippet_chars].rstrip() + " ..."


def _graph_cache_identity(graph: nx.Graph, knowledge_graph_inst: BaseGraphStorage) -> str:
    graph_file = getattr(knowledge_graph_inst, "_graphml_xml_file", None)
    if graph_file:
        graph_path = Path(graph_file)
        if graph_path.exists():
            stat = graph_path.stat()
            return _hash_text(
                f"{graph_path.resolve()}:{stat.st_size}:{int(stat.st_mtime)}"
            )
    sample_edges = [
        "->".join(_edge_key(str(source), str(target)))
        for source, target in list(graph.edges())[:1000]
    ]
    return _hash_text(
        json.dumps(
            {
                "nodes": graph.number_of_nodes(),
                "edges": graph.number_of_edges(),
                "sample": sorted(sample_edges),
            },
            sort_keys=True,
        )
    )


def _edge_embedding_cache_file(
    graph: nx.Graph,
    knowledge_graph_inst: BaseGraphStorage,
    global_config: dict,
) -> Path | None:
    if global_config.get("disable_edge_embedding_cache"):
        return None
    cache_root = global_config.get("edge_embedding_cache_path")
    if not cache_root:
        return None
    embedding_func = global_config.get("embedding_func")
    model = global_config.get("embed_model") or getattr(embedding_func, "__name__", "embedding")
    dim = getattr(embedding_func, "embedding_dim", "unknown")
    graph_id = _graph_cache_identity(graph, knowledge_graph_inst)
    return Path(cache_root) / f"{graph_id}_{_hash_text(str(model))[:12]}_{dim}.json"


def _load_edge_embedding_disk_cache(cache_file: Path | None) -> dict[str, Any]:
    if cache_file is None:
        return {"edge_embeddings": {}}
    cache_file = Path(cache_file)
    if not cache_file.exists():
        return {"edge_embeddings": {}}
    try:
        data = json.loads(cache_file.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Failed to load edge embedding cache %s: %s", cache_file, exc)
        return {"edge_embeddings": {}}
    if not isinstance(data, dict) or not isinstance(data.get("edge_embeddings"), dict):
        return {"edge_embeddings": {}}
    return data


def _write_edge_embedding_disk_cache(cache_file: Path | None, data: dict[str, Any]) -> None:
    if cache_file is None:
        return
    cache_file = Path(cache_file)
    try:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        tmp_file = cache_file.with_suffix(cache_file.suffix + ".tmp")
        tmp_file.write_text(json.dumps(data), encoding="utf-8")
        tmp_file.replace(cache_file)
    except Exception as exc:
        logger.warning("Failed to write edge embedding cache %s: %s", cache_file, exc)


async def _build_query_edge_costs(
    query: str,
    knowledge_graph_inst: BaseGraphStorage,
    query_param: QueryParam,
    embedding_func: Callable | None,
    global_config: dict | None = None,
) -> dict[tuple[str, str], dict[str, float]]:
    graph = _networkx_graph_from_storage(knowledge_graph_inst)
    if graph is None or embedding_func is None:
        return {}
    edges = list(graph.edges(data=True))
    if not edges:
        return {}
    try:
        query_embedding = (await embedding_func([query]))[0]
    except Exception as exc:
        logger.warning("Query-weighted bridge query embedding failed: %s", exc)
        return {}

    global_config = global_config or {}
    cache_embedding_func = global_config.get("embedding_func") or embedding_func
    cache_model = global_config.get("embed_model") or getattr(cache_embedding_func, "__name__", "embedding")
    cache_dim = getattr(cache_embedding_func, "embedding_dim", "unknown")
    cache_key = (
        id(graph),
        graph.number_of_nodes(),
        graph.number_of_edges(),
        str(cache_model),
        str(cache_dim),
    )
    if cache_key not in _EDGE_EMBEDDING_CACHE:
        edge_texts = {
            _edge_key(source, target): _edge_description_text(graph, source, target, data)
            for source, target, data in edges
        }
        disk_cache_file = _edge_embedding_cache_file(
            graph, knowledge_graph_inst, global_config
        )
        disk_cache = _load_edge_embedding_disk_cache(disk_cache_file)
        cached_embeddings = disk_cache.get("edge_embeddings", {})
        edge_embeddings_by_key: dict[tuple[str, str], Any] = {}
        missing_keys: list[tuple[str, str]] = []
        missing_texts: list[str] = []
        for edge_key, edge_text in edge_texts.items():
            cache_item = cached_embeddings.get("|".join(edge_key))
            text_hash = _hash_text(edge_text)
            if (
                isinstance(cache_item, dict)
                and cache_item.get("text_hash") == text_hash
                and "embedding" in cache_item
            ):
                edge_embeddings_by_key[edge_key] = cache_item["embedding"]
            else:
                missing_keys.append(edge_key)
                missing_texts.append(edge_text)
        if missing_texts:
            try:
                missing_embeddings = await embedding_func(missing_texts)
            except Exception as exc:
                logger.warning("Query-weighted bridge edge embedding failed: %s", exc)
                return {}
            for edge_key, edge_text, embedding in zip(missing_keys, missing_texts, missing_embeddings):
                embedding_list = embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)
                edge_embeddings_by_key[edge_key] = embedding_list
                cached_embeddings["|".join(edge_key)] = {
                    "text_hash": _hash_text(edge_text),
                    "embedding": embedding_list,
                }
            disk_cache["edge_embeddings"] = cached_embeddings
            disk_cache["graph_nodes"] = graph.number_of_nodes()
            disk_cache["graph_edges"] = graph.number_of_edges()
            _write_edge_embedding_disk_cache(disk_cache_file, disk_cache)
        edge_embeddings = [
            edge_embeddings_by_key[_edge_key(source, target)]
            for source, target, _data in edges
        ]
        _EDGE_EMBEDDING_CACHE[cache_key] = {
            "edges": edges,
            "edge_embeddings": edge_embeddings,
            "max_hub": max(
                (graph.degree(source) + graph.degree(target) for source, target, _ in edges),
                default=1,
            ),
        }
    cached = _EDGE_EMBEDDING_CACHE[cache_key]
    edges = cached["edges"]
    edge_embeddings = cached["edge_embeddings"]
    max_hub = max(
        int(cached["max_hub"]),
        1,
    )
    costs: dict[tuple[str, str], dict[str, float]] = {}
    for (source, target, data), edge_embedding in zip(edges, edge_embeddings):
        semantic_score = _cosine_similarity(query_embedding, edge_embedding)
        semantic_cost = max(0.0, 1.0 - semantic_score)
        hub_penalty = (graph.degree(source) + graph.degree(target)) / max_hub
        weight = max(_safe_float(data.get("weight"), 1.0), 0.0)
        inverse_weight_penalty = 1.0 / (1.0 + weight)
        cost = (
            query_param.bridge_alpha * semantic_cost
            + query_param.bridge_beta * hub_penalty
            + query_param.bridge_gamma * inverse_weight_penalty
        )
        costs[_edge_key(source, target)] = {
            "cost": float(cost),
            "semantic_score": float(semantic_score),
            "semantic_cost": float(semantic_cost),
            "hub_penalty": float(hub_penalty),
            "inverse_weight_penalty": float(inverse_weight_penalty),
        }
    return costs


def _weighted_edge_cost(
    costs: dict[tuple[str, str], dict[str, float]],
    source: str,
    target: str,
) -> float:
    return costs.get(_edge_key(source, target), {}).get("cost", 1.0)


def _weighted_dijkstra_path(
    graph: nx.Graph,
    source: str,
    target: str,
    edge_costs: dict[tuple[str, str], dict[str, float]],
) -> list[str]:
    return nx.dijkstra_path(
        graph,
        source=source,
        target=target,
        weight=lambda u, v, _data: _weighted_edge_cost(edge_costs, u, v),
    )


def _mcts_config_from_query(query_param: QueryParam) -> MCTSBridgeConfig:
    return MCTSBridgeConfig(
        max_iterations=query_param.mcts_max_iterations,
        exploration_constant=query_param.mcts_exploration_constant,
        candidate_hops=query_param.mcts_candidate_hops,
        candidate_max_hops=query_param.mcts_candidate_max_hops,
        candidate_top_neighbors=query_param.mcts_candidate_top_neighbors,
        progressive_widening_coefficient=query_param.mcts_progressive_widening_coefficient,
        progressive_widening_exponent=query_param.mcts_progressive_widening_exponent,
        rollout_top_k=query_param.mcts_rollout_top_k,
        rollout_depth=query_param.mcts_rollout_depth,
        rollout_epsilon=query_param.mcts_rollout_epsilon,
        max_path_edges=query_param.bridge_max_path_edges,
        max_token_budget=query_param.max_token_for_bridge_knowledge,
        length_penalty=query_param.bridge_length_penalty,
        token_penalty=query_param.mcts_token_penalty,
        target_reward=query_param.mcts_target_reward,
    )


def _minmax_path(
    graph: nx.Graph,
    source: str,
    target: str,
    edge_costs: dict[tuple[str, str], dict[str, float]],
) -> list[str]:
    thresholds = sorted({data["cost"] for data in edge_costs.values()})
    if not thresholds:
        return nx.shortest_path(graph, source=source, target=target)
    best_path: list[str] | None = None
    low = 0
    high = len(thresholds) - 1
    while low <= high:
        mid = (low + high) // 2
        threshold = thresholds[mid]
        allowed_edges = [
            (u, v)
            for u, v in graph.edges()
            if _weighted_edge_cost(edge_costs, u, v) <= threshold
        ]
        filtered = nx.Graph()
        filtered.add_nodes_from(graph.nodes())
        filtered.add_edges_from(allowed_edges)
        try:
            candidate = nx.shortest_path(filtered, source=source, target=target)
            best_path = candidate
            high = mid - 1
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            low = mid + 1
    if best_path is None:
        return _weighted_dijkstra_path(graph, source, target, edge_costs)
    return best_path


def _minmax_budgeted_path(
    graph: nx.Graph,
    source: str,
    target: str,
    edge_costs: dict[tuple[str, str], dict[str, float]],
    query_param: QueryParam,
) -> tuple[list[str], str]:
    thresholds = sorted({data["cost"] for data in edge_costs.values()})
    max_nodes = max(2, query_param.bridge_max_path_edges + 1)
    if thresholds:
        best_path: list[str] | None = None
        low = 0
        high = len(thresholds) - 1
        while low <= high:
            mid = (low + high) // 2
            threshold = thresholds[mid]
            allowed_edges = [
                (u, v)
                for u, v in graph.edges()
                if _weighted_edge_cost(edge_costs, u, v) <= threshold
            ]
            filtered = nx.Graph()
            filtered.add_nodes_from(graph.nodes())
            filtered.add_edges_from(allowed_edges)
            try:
                candidate = nx.shortest_path(filtered, source=source, target=target)
                if len(candidate) <= max_nodes:
                    best_path = candidate
                    high = mid - 1
                else:
                    low = mid + 1
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                low = mid + 1
        if best_path is not None:
            return best_path, "minmax_budgeted"

    if query_param.bridge_budget_fallback == "weighted":
        try:
            weighted = _weighted_dijkstra_path(graph, source, target, edge_costs)
            if len(weighted) <= max_nodes:
                return weighted, "weighted_fallback"
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            pass
    return nx.shortest_path(graph, source=source, target=target), "unweighted_fallback"


async def _path_between(
    knowledge_graph_inst: BaseGraphStorage,
    source: str,
    target: str,
    query_param: QueryParam,
    edge_costs: dict[tuple[str, str], dict[str, float]],
) -> tuple[list[str], str, dict[str, Any]]:
    from ._storage.gdb_neo4j import Neo4jStorage

    graph = _networkx_graph_from_storage(knowledge_graph_inst)
    try:
        if (
            graph is not None
            and query_param.bridge_strategy == "mcts"
            and edge_costs
        ):
            path, metadata = find_mcts_bridge_path(
                graph,
                source,
                target,
                edge_costs,
                _mcts_config_from_query(query_param),
            )
            return path, metadata.get("decision", "mcts"), metadata
        if (
            graph is not None
            and query_param.bridge_strategy == "query_weighted"
            and edge_costs
        ):
            return _weighted_dijkstra_path(graph, source, target, edge_costs), "query_weighted", {}
        if (
            graph is not None
            and query_param.bridge_strategy == "minmax"
            and edge_costs
        ):
            return _minmax_path(graph, source, target, edge_costs), "minmax", {}
        if (
            graph is not None
            and query_param.bridge_strategy == "minmax_budgeted"
            and edge_costs
        ):
            path, decision = _minmax_budgeted_path(graph, source, target, edge_costs, query_param)
            return path, decision, {}
        if isinstance(knowledge_graph_inst, Neo4jStorage):
            return await knowledge_graph_inst.shortest_path(source, target), "unweighted", {}
        if graph is not None:
            return nx.shortest_path(graph, source=source, target=target), "unweighted", {}
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return [], "no_path", {}
    except Exception as exc:
        logger.warning(
            "Bridge path strategy %s failed between %s and %s; falling back to unweighted path: %s",
            query_param.bridge_strategy,
            source,
            target,
            exc,
        )
        if graph is not None:
            try:
                return nx.shortest_path(graph, source=source, target=target), "unweighted_error_fallback", {}
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                return [], "no_path", {}
    return [], "no_path", {}


async def _find_path_with_required_nodes(
    knowledge_graph_inst: BaseGraphStorage,
    key_entities: list[str],
    query_param: QueryParam,
    edge_costs: dict[tuple[str, str], dict[str, float]],
) -> tuple[list[str], list[dict[str, Any]]]:
    key_entities = _ordered_unique(key_entities)
    if len(key_entities) < 2:
        return key_entities, []
    final_path: list[str] = []
    decisions: list[dict[str, Any]] = []
    total_edges = 0
    for source, target in zip(key_entities, key_entities[1:]):
        sub_path, decision, metadata = await _path_between(
            knowledge_graph_inst, source, target, query_param, edge_costs
        )
        sub_edges = max(0, len(sub_path) - 1)
        if (
            query_param.bridge_strategy in {"minmax_budgeted", "mcts"}
            and query_param.bridge_max_total_edges > 0
            and total_edges + sub_edges > query_param.bridge_max_total_edges
        ):
            decisions.append(
                {
                    "source": source,
                    "target": target,
                    "decision": "budget_stop",
                    "path_edges": sub_edges,
                    "total_edges_before": total_edges,
                    **metadata,
                }
            )
            break
        decisions.append(
            {
                "source": source,
                "target": target,
                "decision": decision,
                "path_edges": sub_edges,
                **metadata,
            }
        )
        if not sub_path:
            if not final_path or final_path[-1] != target:
                final_path.append(target)
            continue
        if final_path:
            final_path.extend(sub_path[1:])
        else:
            final_path.extend(sub_path)
        total_edges = max(0, len(final_path) - 1)
    return final_path, decisions


def _select_key_entities(
    overall_node_datas: list[dict],
    use_communities: list[CommunitySchema],
    query_param: QueryParam,
) -> list[str]:
    key_entities: list[str] = []
    max_entity_num = query_param.top_m
    if use_communities:
        for community in use_communities:
            community_entities = set(community["nodes"])
            key_entities.extend(
                [
                    entity["entity_name"]
                    for entity in overall_node_datas
                    if entity["entity_name"] in community_entities
                ][:max_entity_num]
            )
    else:
        key_entities.extend(
            [entity["entity_name"] for entity in overall_node_datas[:max_entity_num]]
        )
    return _ordered_unique(key_entities)


async def _build_bridge_path_data(
    query: str,
    knowledge_graph_inst: BaseGraphStorage,
    key_entities: list[str],
    query_param: QueryParam,
    global_config: dict,
) -> tuple[list[str], list[dict], dict[tuple[str, str], dict[str, float]]]:
    embedding_func = global_config.get("embedding_func")
    edge_costs: dict[tuple[str, str], dict[str, float]] = {}
    if query_param.bridge_strategy in {"query_weighted", "minmax", "minmax_budgeted", "mcts"}:
        edge_costs = await _build_query_edge_costs(
            query, knowledge_graph_inst, query_param, embedding_func, global_config
        )
        if not edge_costs:
            query_param.debug_info["bridge_fallback"] = "unweighted"
    path, decisions = await _find_path_with_required_nodes(
        knowledge_graph_inst, key_entities, query_param, edge_costs
    )
    if decisions:
        query_param.debug_info["bridge_path_decisions"] = decisions
        query_param.debug_info["bridge_budget_fallbacks"] = sum(
            1 for item in decisions if "fallback" in item.get("decision", "")
        )
        query_param.debug_info["bridge_budget_stopped"] = any(
            item.get("decision") == "budget_stop" for item in decisions
        )
    if not path:
        return [], [], edge_costs
    path_datas_raw = await asyncio.gather(
        *[knowledge_graph_inst.get_node(node_id) for node_id in path]
    )
    path_degrees = await asyncio.gather(
        *[knowledge_graph_inst.node_degree(node_id) for node_id in path]
    )
    path_datas = [
        {**node, "entity_name": node_id, "rank": degree}
        for node_id, node, degree in zip(path, path_datas_raw, path_degrees)
        if node is not None
    ]
    return path, path_datas, edge_costs


def _edge_metrics_for_path(
    path: list[str],
    edge_costs: dict[tuple[str, str], dict[str, float]],
) -> dict[str, Any]:
    path_edge_keys = [_edge_key(left, right) for left, right in zip(path, path[1:])]
    scored = [edge_costs[key] for key in path_edge_keys if key in edge_costs]
    costs = [item["cost"] for item in scored]
    scores = [item["semantic_score"] for item in scored]
    return {
        "bridge_path_nodes": len(path),
        "bridge_path_edges": max(0, len(path) - 1),
        "mean_edge_cost": sum(costs) / len(costs) if costs else None,
        "max_edge_cost": max(costs) if costs else None,
        "mean_edge_score": sum(scores) / len(scores) if scores else None,
    }


# context functions
async def _build_local_query_context(
    query,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    community_reports: BaseKVStorage[CommunitySchema],
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
):
    results = await entities_vdb.query(query, top_k=query_param.top_k)          # find the top-k(20) related entities
    if not len(results):
        return None
    node_datas = await asyncio.gather(
        *[knowledge_graph_inst.get_node(r["entity_name"]) for r in results]
    )
    if not all([n is not None for n in node_datas]):
        logger.warning("Some nodes are missing, maybe the storage is damaged")
    node_degrees = await asyncio.gather(
        *[knowledge_graph_inst.node_degree(r["entity_name"]) for r in results]
    )
    node_datas = [
        {**n, "entity_name": k["entity_name"], "rank": d}
        for k, n, d in zip(results, node_datas, node_degrees)
        if n is not None
    ]
    use_communities = await _find_most_related_community_from_entities(
        node_datas, query_param, community_reports
    )
    use_text_units = await _find_most_related_text_unit_from_entities(
        node_datas, query_param, text_chunks_db, knowledge_graph_inst
    )
    use_relations = await _find_most_related_edges_from_entities(
        node_datas, query_param, knowledge_graph_inst
    )
    logger.info(
        f"Using {len(node_datas)} entites, {len(use_communities)} communities, {len(use_relations)} relations, {len(use_text_units)} text units"
    )
    entites_section_list = [["id", "entity", "type", "description", "rank"]]
    for i, n in enumerate(node_datas):
        entites_section_list.append(
            [
                i,
                n["entity_name"],
                n.get("entity_type", "UNKNOWN"),
                n.get("description", "UNKNOWN"),
                n["rank"],
            ]
        )
    entities_context = list_of_list_to_csv(entites_section_list)

    relations_section_list = [
        ["id", "source", "target", "description", "weight", "rank"]
    ]
    for i, e in enumerate(use_relations):
        relations_section_list.append(
            [
                i,
                e["src_tgt"][0],
                e["src_tgt"][1],
                e["description"],
                e["weight"],
                e["rank"],
            ]
        )
    relations_context = list_of_list_to_csv(relations_section_list)

    communities_section_list = [["id", "content"]]
    for i, c in enumerate(use_communities):
        communities_section_list.append([i, c["report_string"]])
    communities_context = list_of_list_to_csv(communities_section_list)

    text_units_section_list = [["id", "content"]]
    for i, t in enumerate(use_text_units):
        text_units_section_list.append([i, _text_unit_context_content(t, query_param, query)])
    text_units_context = list_of_list_to_csv(text_units_section_list)
    return f"""
-----Reports-----
```csv
{communities_context}
```
-----Entities-----
```csv
{entities_context}
```
-----Relationships-----
```csv
{relations_context}
```
-----Sources-----
```csv
{text_units_context}
```
"""


async def _build_hierarchical_query_context(
    query,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    community_reports: BaseKVStorage[CommunitySchema],
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
):
    results = await entities_vdb.query(query, top_k=query_param.top_k * 10)          # find the top-k(20) related entities

    if not len(results):    # results just with entity name
        return None
    node_datas = await asyncio.gather(      # get full information of retrieved entities
        *[knowledge_graph_inst.get_node(r["entity_name"]) for r in results]
    )
    if not all([n is not None for n in node_datas]):    # for robustness
        logger.warning("Some nodes are missing, maybe the storage is damaged")
    node_degrees = await asyncio.gather(
        *[knowledge_graph_inst.node_degree(r["entity_name"]) for r in results]
    )
    node_datas = [                          # add rank, which is the degree
        {**n, "entity_name": k["entity_name"], "rank": d}
        for k, n, d in zip(results, node_datas, node_degrees)
        if n is not None
    ]
    overall_node_datas = node_datas
    node_datas = node_datas[:query_param.top_k]

    use_communities = await _find_most_related_community_from_entities(     # related communities
        node_datas, query_param, community_reports
    )
    use_text_units = await _find_most_related_text_unit_from_entities(
        node_datas, query_param, text_chunks_db, knowledge_graph_inst
    )
    # use_relations = await _find_most_related_edges_from_entities(
    #     node_datas, query_param, knowledge_graph_inst
    # )

    async def find_path_with_required_nodes(knowledge_graph_inst, source, target, required_nodes):
        # inital final path
        final_path = []
        # start node
        current_node = source

        # all gdb cls
        from ._storage.gdb_neo4j import Neo4jStorage
        from ._storage.gdb_networkx import NetworkXStorage
        
        # traverse the required nodes
        for next_node in required_nodes:
            # 找到从当前节点到下一个必经节点的最短路径
            try:
                if isinstance(knowledge_graph_inst, Neo4jStorage):
                    # use Neo4j's shortest_path method
                    sub_path = await knowledge_graph_inst.shortest_path(current_node, next_node)
                elif isinstance(knowledge_graph_inst, NetworkXStorage):
                    # use NetworkX's shortest_path method
                    sub_path = nx.shortest_path(knowledge_graph_inst._graph, source=current_node, target=next_node)
                else:
                    # use NetworkX by default
                    sub_path = nx.shortest_path(knowledge_graph_inst._graph, source=current_node, target=next_node)
            except nx.NetworkXNoPath:
                # raise ValueError(f"No path between {current_node} and {next_node}.")
                final_path.extend([next_node])
                current_node = next_node
                continue
            
            # merge paths (avoid adding the current node again)
            if final_path:
                final_path.extend(sub_path[1:])  # add from the second node, avoid adding the current node again
            else:
                final_path.extend(sub_path)
            
            # update the current node to the next required node
            current_node = next_node

        # finally, the path from the last required node to the target node
        try:
            if isinstance(knowledge_graph_inst, Neo4jStorage):
                # use Neo4j's shortest_path method
                sub_path = await knowledge_graph_inst.shortest_path(current_node, target)
            elif isinstance(knowledge_graph_inst, NetworkXStorage):
                # use NetworkX's shortest_path method
                sub_path = nx.shortest_path(knowledge_graph_inst._graph, source=current_node, target=target)
            else:
                # use NetworkX by default
                sub_path = nx.shortest_path(knowledge_graph_inst._graph, source=current_node, target=target)
            final_path.extend(sub_path[1:])  # add from the second node, avoid adding the current node again
        except nx.NetworkXNoPath:
            # raise ValueError(f"No path between {current_node} and {target}.")
            final_path.extend([target])

        return final_path

    # find some top-k entities in each communities in use_communities
    key_entities = []
    max_entity_num = query_param.top_m
    if use_communities:
        for c in use_communities:
            cur_community_key_entities = []
            community_entities = c['nodes']
            # find the top-k entities in this community
            cur_community_key_entities.extend(
                [e for e in overall_node_datas if e['entity_name'] in community_entities][:max_entity_num]
            )
            key_entities.append(cur_community_key_entities)
    else:
        key_entities = [overall_node_datas[:max_entity_num]]
    # unique key entities
    key_entities = [[e['entity_name'] for e in k] for k in key_entities]
    key_entities = list(set([k for kk in key_entities for k in kk]))
    # find the shortest path between the key entities
    try:
        path = await find_path_with_required_nodes(knowledge_graph_inst, key_entities[0], key_entities[-1], key_entities[1:-1])
        # path = list(set(path))
        path_datas = await asyncio.gather(      # get full information of retrieved entities
            *[knowledge_graph_inst.get_node(r) for r in path]
        )
        path_degrees = await asyncio.gather(
            *[knowledge_graph_inst.node_degree(r) for r in path]
        )
        path_datas = [                          # add rank, which is the degree
            {**n, "entity_name": k, "rank": d}
            for k, n, d in zip(path, path_datas, path_degrees)
            if n is not None
        ]
        # use_reasoning_path = await _find_most_related_edges_from_entities(
        #                     path_datas, query_param, knowledge_graph_inst
        #                 )
        use_reasoning_path = await _find_most_related_edges_from_paths(
                                path_datas, path, query_param, knowledge_graph_inst
                            )
    except ValueError as e:
        print(e)
    
    # # fetch the relations of the reasoning paths
    # reasoning_path = []
    # for i in range(len(path) - 1):
    #     src = path[i]
    #     tgt = path[i + 1]
    #     cur_relation = (await knowledge_graph_inst.get_edge(src, tgt))['description']
    #     reasoning_path.append(cur_relation)
    # reasoning_path = list(set(reasoning_path))

    logger.info(
        f"Using {len(node_datas)} entites, {len(use_communities)} communities, {len(use_reasoning_path)} reasoning path items, {len(use_text_units)} text units"
    )
    entites_section_list = [["id", "entity", "type", "description", "rank"]]
    for i, n in enumerate(node_datas):
        entites_section_list.append(
            [
                i,
                n["entity_name"],
                n.get("entity_type", "UNKNOWN"),
                n.get("description", "UNKNOWN"),
                n["rank"],
            ]
        )
    entities_context = list_of_list_to_csv(entites_section_list)
    
    reasoning_path_section_list = [
        ["id", "source", "target", "description", "weight", "rank"]
    ]
    for i, e in enumerate(use_reasoning_path):
        reasoning_path_section_list.append(
            [
                i,
                e["src_tgt"][0],
                e["src_tgt"][1],
                e["description"],
                e["weight"],
                e["rank"],
            ]
        )
    reasoning_path_context = list_of_list_to_csv(reasoning_path_section_list)
    
    # reasoning_path_context = list_of_list_to_csv([["id", "content"]] + [[i, p] for i, p in enumerate(reasoning_path)])
    
    communities_section_list = [["id", "content"]]
    for i, c in enumerate(use_communities):
        communities_section_list.append([i, c["report_string"].replace("\n", " ")])
    communities_context = list_of_list_to_csv(communities_section_list)

    text_units_section_list = [["id", "content"]]
    for i, t in enumerate(use_text_units):
        text_units_section_list.append([i, _text_unit_context_content(t, query_param, query)])
    text_units_context = list_of_list_to_csv(text_units_section_list)

    # display reference info
    entities = [n["entity_name"] for n in node_datas]
    communities = [(c["level"], c["title"]) for c in use_communities]
    chunks = [(t["full_doc_id"], t["chunk_order_index"]) for t in use_text_units]

    references_context = (
        f"Entities ({len(entities)}): {entities}\n\n"
        f"Communities (level, cluster_id) ({len(communities)}): {communities}\n\n"
        f"Chunks (doc_id, chunk_index) ({len(chunks)}): {chunks}\n"
    )

    logging.info(f"====== References ======:\n{references_context}")
    return f"""
-----Backgrounds-----
```csv
{communities_context}
```
-----Reasoning Path-----
```csv
{reasoning_path_context}
```
-----Detail Entity Information-----
```csv
{entities_context}
```
-----Source Documents-----
```csv
{text_units_context}
```
"""


async def _build_hibridge_query_context(
    query,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    community_reports: BaseKVStorage[CommunitySchema],
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
):
    results = await entities_vdb.query(query, top_k=query_param.top_k * 10)          # find the top-k(20) related entities

    if not len(results):    # results just with entity name
        return None
    node_datas = await asyncio.gather(      # get full information of retrieved entities
        *[knowledge_graph_inst.get_node(r["entity_name"]) for r in results]
    )
    if not all([n is not None for n in node_datas]):    # for robustness
        logger.warning("Some nodes are missing, maybe the storage is damaged")
    node_degrees = await asyncio.gather(
        *[knowledge_graph_inst.node_degree(r["entity_name"]) for r in results]
    )
    node_datas = [                          # add rank, which is the degree
        {**n, "entity_name": k["entity_name"], "rank": d}
        for k, n, d in zip(results, node_datas, node_degrees)
        if n is not None
    ]
    overall_node_datas = node_datas
    node_datas = node_datas[:query_param.top_k]

    use_communities = await _find_most_related_community_from_entities(     # related communities
        node_datas, query_param, community_reports
    )
    use_text_units = await _find_most_related_text_unit_from_entities(
        node_datas, query_param, text_chunks_db, knowledge_graph_inst
    )
    # use_relations = await _find_most_related_edges_from_entities(
    #     node_datas, query_param, knowledge_graph_inst
    # )

    async def find_path_with_required_nodes(knowledge_graph_inst, source, target, required_nodes):
        # inital final path
        final_path = []
        current_node = source

        from ._storage.gdb_neo4j import Neo4jStorage
        from ._storage.gdb_networkx import NetworkXStorage
        
        for next_node in required_nodes:
            try:
                if isinstance(knowledge_graph_inst, Neo4jStorage):
                    sub_path = await knowledge_graph_inst.shortest_path(current_node, next_node)
                elif isinstance(knowledge_graph_inst, NetworkXStorage):
                    sub_path = nx.shortest_path(knowledge_graph_inst._graph, source=current_node, target=next_node)
                else:
                    sub_path = nx.shortest_path(knowledge_graph_inst._graph, source=current_node, target=next_node)
            except nx.NetworkXNoPath:
                # raise ValueError(f"No path between {current_node} and {next_node}.")
                final_path.extend([next_node])
                current_node = next_node
                continue
            
            if final_path:
                final_path.extend(sub_path[1:])
            else:
                final_path.extend(sub_path)
            
            current_node = next_node

        try:
            if isinstance(knowledge_graph_inst, Neo4jStorage):
                sub_path = await knowledge_graph_inst.shortest_path(current_node, target)
            elif isinstance(knowledge_graph_inst, NetworkXStorage):
                sub_path = nx.shortest_path(knowledge_graph_inst._graph, source=current_node, target=target)
            else:
                sub_path = nx.shortest_path(knowledge_graph_inst._graph, source=current_node, target=target)
            final_path.extend(sub_path[1:])
        except nx.NetworkXNoPath:
            # raise ValueError(f"No path between {current_node} and {target}.")
            final_path.extend([target])

        return final_path

    # find some top-k entities in each communities in use_communities
    key_entities = []
    max_entity_num = query_param.top_m
    if use_communities:
        for c in use_communities:
            cur_community_key_entities = []
            community_entities = c['nodes']
            # find the top-k entities in this community
            cur_community_key_entities.extend(
                [e for e in overall_node_datas if e['entity_name'] in community_entities][:max_entity_num]
            )
            key_entities.append(cur_community_key_entities)
    else:
        key_entities = [overall_node_datas[:max_entity_num]]
    # unique key entities
    key_entities = [[e['entity_name'] for e in k] for k in key_entities]
    key_entities = list(set([k for kk in key_entities for k in kk]))
    # find the shortest path between the key entities
    try:
        path = await find_path_with_required_nodes(knowledge_graph_inst, key_entities[0], key_entities[-1], key_entities[1:-1])
        # path = list(set(path))
        path_datas = await asyncio.gather(      # get full information of retrieved entities
            *[knowledge_graph_inst.get_node(r) for r in path]
        )
        path_degrees = await asyncio.gather(
            *[knowledge_graph_inst.node_degree(r) for r in path]
        )
        path_datas = [                          # add rank, which is the degree
            {**n, "entity_name": k, "rank": d}
            for k, n, d in zip(path, path_datas, path_degrees)
            if n is not None
        ]
        use_reasoning_path = await _find_most_related_edges_from_paths(
                                path_datas, path, query_param, knowledge_graph_inst
                            )
    except ValueError as e:
        print(e)

    logger.info(
        f"Using {len(node_datas)} entites, {len(use_communities)} communities, {len(use_reasoning_path)} reasoning path items, {len(use_text_units)} text units"
    )
    entites_section_list = [["id", "entity", "type", "description", "rank"]]
    for i, n in enumerate(node_datas):
        entites_section_list.append(
            [
                i,
                n["entity_name"],
                n.get("entity_type", "UNKNOWN"),
                n.get("description", "UNKNOWN"),
                n["rank"],
            ]
        )
    entities_context = list_of_list_to_csv(entites_section_list)
    
    reasoning_path_section_list = [
        ["id", "source", "target", "description", "weight", "rank"]
    ]
    for i, e in enumerate(use_reasoning_path):
        reasoning_path_section_list.append(
            [
                i,
                e["src_tgt"][0],
                e["src_tgt"][1],
                e["description"],
                e["weight"],
                e["rank"],
            ]
        )
    reasoning_path_context = list_of_list_to_csv(reasoning_path_section_list)
    
    # reasoning_path_context = list_of_list_to_csv([["id", "content"]] + [[i, p] for i, p in enumerate(reasoning_path)])
    
    communities_section_list = [["id", "content"]]
    for i, c in enumerate(use_communities):
        communities_section_list.append([i, c["report_string"]])
    communities_context = list_of_list_to_csv(communities_section_list)

    text_units_section_list = [["id", "content"]]
    for i, t in enumerate(use_text_units):
        text_units_section_list.append([i, _text_unit_context_content(t, query_param, query)])
    text_units_context = list_of_list_to_csv(text_units_section_list)
    return f"""
-----Reasoning Path-----
```csv
{reasoning_path_context}
```
-----Source Documents-----
```csv
{text_units_context}
```
"""


async def _build_higlobal_query_context(
    query,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    community_reports: BaseKVStorage[CommunitySchema],
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
):
    results = await entities_vdb.query(query, top_k=query_param.top_k * 10)          # find the top-k(20) related entities

    if not len(results):    # results just with entity name
        return None
    node_datas = await asyncio.gather(      # get full information of retrieved entities
        *[knowledge_graph_inst.get_node(r["entity_name"]) for r in results]
    )
    if not all([n is not None for n in node_datas]):    # for robustness
        logger.warning("Some nodes are missing, maybe the storage is damaged")
    node_degrees = await asyncio.gather(
        *[knowledge_graph_inst.node_degree(r["entity_name"]) for r in results]
    )
    node_datas = [                          # add rank, which is the degree
        {**n, "entity_name": k["entity_name"], "rank": d}
        for k, n, d in zip(results, node_datas, node_degrees)
        if n is not None
    ]
    overall_node_datas = node_datas
    node_datas = node_datas[:query_param.top_k]

    use_communities = await _find_most_related_community_from_entities(     # related communities
        node_datas, query_param, community_reports
    )
    use_text_units = await _find_most_related_text_unit_from_entities(
        node_datas, query_param, text_chunks_db, knowledge_graph_inst
    )


    logger.info(
        f"Using {len(use_communities)} communities, {len(use_text_units)} text units"
    )
    
    communities_section_list = [["id", "content"]]
    for i, c in enumerate(use_communities):
        communities_section_list.append([i, c["report_string"]])
    communities_context = list_of_list_to_csv(communities_section_list)

    text_units_section_list = [["id", "content"]]
    for i, t in enumerate(use_text_units):
        text_units_section_list.append([i, _text_unit_context_content(t, query_param, query)])
    text_units_context = list_of_list_to_csv(text_units_section_list)
    return f"""
-----Backgrounds-----
```csv
{communities_context}
```
-----Source Documents-----
```csv
{text_units_context}
```
"""


async def _build_hilocal_query_context(
    query,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    community_reports: BaseKVStorage[CommunitySchema],
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
):
    results = await entities_vdb.query(query, top_k=query_param.top_k)          # find the top-k(20) related entities

    if not len(results):    # results just with entity name
        return None
    node_datas = await asyncio.gather(      # get full information of retrieved entities
        *[knowledge_graph_inst.get_node(r["entity_name"]) for r in results]
    )
    if not all([n is not None for n in node_datas]):    # for robustness
        logger.warning("Some nodes are missing, maybe the storage is damaged")
    node_degrees = await asyncio.gather(
        *[knowledge_graph_inst.node_degree(r["entity_name"]) for r in results]
    )
    node_datas = [                          # add rank, which is the degree
        {**n, "entity_name": k["entity_name"], "rank": d}
        for k, n, d in zip(results, node_datas, node_degrees)
        if n is not None
    ]

    use_text_units = await _find_most_related_text_unit_from_entities(
        node_datas, query_param, text_chunks_db, knowledge_graph_inst
    )
    use_relations = await _find_most_related_edges_from_entities(
        node_datas, query_param, knowledge_graph_inst
    )

    logger.info(
        f"Using {len(node_datas)} entites, {len(use_relations)} relations, {len(use_text_units)} text units"
    )
    entites_section_list = [["id", "entity", "type", "description", "rank"]]
    for i, n in enumerate(node_datas):
        entites_section_list.append(
            [
                i,
                n["entity_name"],
                n.get("entity_type", "UNKNOWN"),
                n.get("description", "UNKNOWN"),
                n["rank"],
            ]
        )
    entities_context = list_of_list_to_csv(entites_section_list)
    
    relation_section_list = [
        ["id", "source", "target", "description", "weight", "rank"]
    ]
    for i, e in enumerate(use_relations):
        relation_section_list.append(
            [
                i,
                e["src_tgt"][0],
                e["src_tgt"][1],
                e["description"],
                e["weight"],
                e["rank"],
            ]
        )
    relation_context = list_of_list_to_csv(relation_section_list)
    
    text_units_section_list = [["id", "content"]]
    for i, t in enumerate(use_text_units):
        text_units_section_list.append([i, _text_unit_context_content(t, query_param, query)])
    text_units_context = list_of_list_to_csv(text_units_section_list)
    return f"""
-----Entities-----
```csv
{entities_context}
```
-----Relations-----
```csv
{relation_context}
```
-----Sources-----
```csv
{text_units_context}
```
"""


async def _build_hierarchical_query_context_common(
    query,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    community_reports: BaseKVStorage[CommunitySchema],
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    global_config: dict,
    *,
    bridge_only: bool = False,
):
    overall_node_datas = await _fetch_entity_node_datas(
        query,
        knowledge_graph_inst,
        entities_vdb,
        query_param,
        global_config=global_config,
    )
    if not overall_node_datas:
        return None
    node_datas = overall_node_datas[:query_param.top_k]

    use_communities = await _find_most_related_community_from_entities(
        node_datas, query_param, community_reports
    )
    use_text_units = await _find_most_related_text_unit_from_entities(
        node_datas, query_param, text_chunks_db, knowledge_graph_inst
    )
    key_entities = _select_key_entities(overall_node_datas, use_communities, query_param)
    path, path_datas, edge_costs = await _build_bridge_path_data(
        query, knowledge_graph_inst, key_entities, query_param, global_config
    )
    use_reasoning_path = (
        await _find_most_related_edges_from_paths(
            path_datas, path, query_param, knowledge_graph_inst
        )
        if path
        else []
    )

    query_param.debug_info.update(
        {
            "mode": query_param.mode,
            "bridge_strategy": query_param.bridge_strategy,
            "local_rerank_strategy": query_param.local_rerank_strategy,
            "local_entities": [n["entity_name"] for n in node_datas],
            "communities": [(c["level"], c["title"]) for c in use_communities],
            "bridge_edges": [
                {
                    "source": e["src_tgt"][0],
                    "target": e["src_tgt"][1],
                    "description": e.get("description", ""),
                    "weight": e.get("weight"),
                    "rank": e.get("rank"),
                    **edge_costs.get(_edge_key(e["src_tgt"][0], e["src_tgt"][1]), {}),
                }
                for e in use_reasoning_path
            ],
            **_edge_metrics_for_path(path, edge_costs),
        }
    )

    logger.info(
        f"Using {len(node_datas)} entites, {len(use_communities)} communities, {len(use_reasoning_path)} reasoning path items, {len(use_text_units)} text units"
    )
    entities_section_list = [["id", "entity", "type", "description", "rank"]]
    for i, n in enumerate(node_datas):
        entities_section_list.append(
            [
                i,
                n["entity_name"],
                n.get("entity_type", "UNKNOWN"),
                n.get("description", "UNKNOWN"),
                n["rank"],
            ]
        )
    entities_context = list_of_list_to_csv(entities_section_list)

    reasoning_path_section_list = [
        ["id", "source", "target", "description", "weight", "rank"]
    ]
    for i, e in enumerate(use_reasoning_path):
        reasoning_path_section_list.append(
            [
                i,
                e["src_tgt"][0],
                e["src_tgt"][1],
                e["description"],
                e["weight"],
                e["rank"],
            ]
        )
    reasoning_path_context = list_of_list_to_csv(reasoning_path_section_list)

    communities_section_list = [["id", "content"]]
    for i, c in enumerate(use_communities):
        communities_section_list.append([i, c["report_string"].replace("\n", " ")])
    communities_context = list_of_list_to_csv(communities_section_list)

    text_units_section_list = [["id", "content"]]
    for i, t in enumerate(use_text_units):
        text_units_section_list.append([i, _text_unit_context_content(t, query_param, query)])
    text_units_context = list_of_list_to_csv(text_units_section_list)

    query_param.debug_info.update(
        {
            "local_context_text": entities_context,
            "global_context_text": communities_context,
            "bridge_context_text": reasoning_path_context,
        }
    )

    if bridge_only:
        return f"""
-----Reasoning Path-----
```csv
{reasoning_path_context}
```
-----Source Documents-----
```csv
{text_units_context}
```
"""

    entities = [n["entity_name"] for n in node_datas]
    communities = [(c["level"], c["title"]) for c in use_communities]
    chunks = [(t["full_doc_id"], t["chunk_order_index"]) for t in use_text_units]
    references_context = (
        f"Entities ({len(entities)}): {entities}\n\n"
        f"Communities (level, cluster_id) ({len(communities)}): {communities}\n\n"
        f"Chunks (doc_id, chunk_index) ({len(chunks)}): {chunks}\n"
    )
    logging.info(f"====== References ======:\n{references_context}")
    return f"""
-----Backgrounds-----
```csv
{communities_context}
```
-----Reasoning Path-----
```csv
{reasoning_path_context}
```
-----Detail Entity Information-----
```csv
{entities_context}
```
-----Source Documents-----
```csv
{text_units_context}
```
"""


# query functions
async def hierarchical_query(
    query,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    community_reports: BaseKVStorage[CommunitySchema],
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    global_config: dict,
) -> str:
    prompts = _get_prompts(global_config)
    use_model_func = global_config["best_model_func"]
    with timer():
        context = await _build_hierarchical_query_context_common(
            query,
            knowledge_graph_inst,
            entities_vdb,
            community_reports,
            text_chunks_db,
            query_param,
            global_config,
        )
    if query_param.only_need_context:
        return context
    if context is None:
        return prompts["fail_response"]
    sys_prompt_temp = prompts["local_rag_response"]
    sys_prompt = sys_prompt_temp.format(
        context_data=context, response_type=query_param.response_type
    )
    response = await use_model_func(
        query,
        system_prompt=sys_prompt,
        stage="query_answer",
        max_tokens=_get_stage_max_tokens(global_config, "query_answer"),
    )
    return response

async def hierarchical_bridge_query(
    query,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    community_reports: BaseKVStorage[CommunitySchema],
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    global_config: dict,
) -> str:
    prompts = _get_prompts(global_config)
    use_model_func = global_config["best_model_func"]
    with timer():
        context = await _build_hierarchical_query_context_common(
            query,
            knowledge_graph_inst,
            entities_vdb,
            community_reports,
            text_chunks_db,
            query_param,
            global_config,
            bridge_only=True,
        )
    if query_param.only_need_context:
        return context
    if context is None:
        return prompts["fail_response"]
    sys_prompt_temp = prompts["local_rag_response"]
    sys_prompt = sys_prompt_temp.format(
        context_data=context, response_type=query_param.response_type
    )
    response = await use_model_func(
        query,
        system_prompt=sys_prompt,
        stage="query_answer",
        max_tokens=_get_stage_max_tokens(global_config, "query_answer"),
    )
    return response

async def hierarchical_local_query(
    query,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    community_reports: BaseKVStorage[CommunitySchema],
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    global_config: dict,
) -> str:
    prompts = _get_prompts(global_config)
    use_model_func = global_config["best_model_func"]
    with timer():
        context = await _build_hilocal_query_context(
            query,
            knowledge_graph_inst,
            entities_vdb,
            community_reports,
            text_chunks_db,
            query_param,
        )
    if query_param.only_need_context:
        return context
    if context is None:
        return prompts["fail_response"]
    sys_prompt_temp = prompts["local_rag_response"]
    sys_prompt = sys_prompt_temp.format(
        context_data=context, response_type=query_param.response_type
    )
    response = await use_model_func(
        query,
        system_prompt=sys_prompt,
        stage="query_answer",
        max_tokens=_get_stage_max_tokens(global_config, "query_answer"),
    )
    return response

async def hierarchical_global_query(
    query,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    community_reports: BaseKVStorage[CommunitySchema],
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    global_config: dict,
) -> str:
    prompts = _get_prompts(global_config)
    use_model_func = global_config["best_model_func"]
    with timer():
        context = await _build_higlobal_query_context(
            query,
            knowledge_graph_inst,
            entities_vdb,
            community_reports,
            text_chunks_db,
            query_param,
        )
    if query_param.only_need_context:
        return context
    if context is None:
        return prompts["fail_response"]
    sys_prompt_temp = prompts["local_rag_response"]
    sys_prompt = sys_prompt_temp.format(
        context_data=context, response_type=query_param.response_type
    )
    response = await use_model_func(
        query,
        system_prompt=sys_prompt,
        stage="query_answer",
        max_tokens=_get_stage_max_tokens(global_config, "query_answer"),
    )
    return response

async def hierarchical_nobridge_query(
    query,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    community_reports: BaseKVStorage[CommunitySchema],
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    global_config: dict,
) -> str:
    prompts = _get_prompts(global_config)
    """
    retrieve with only related entities
    """
    use_model_func = global_config["best_model_func"]
    with timer():
        context = await _build_local_query_context(
            query,
            knowledge_graph_inst,
            entities_vdb,
            community_reports,
            text_chunks_db,
            query_param,
        )
    if query_param.only_need_context:
        return context
    if context is None:
        return prompts["fail_response"]
    sys_prompt_temp = prompts["local_rag_response"]
    sys_prompt = sys_prompt_temp.format(
        context_data=context, response_type=query_param.response_type
    )
    response = await use_model_func(
        query,
        system_prompt=sys_prompt,
        stage="query_answer",
        max_tokens=_get_stage_max_tokens(global_config, "query_answer"),
    )
    return response

async def naive_query(
    query,
    chunks_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    global_config: dict,
):
    prompts = _get_prompts(global_config)
    use_model_func = global_config["best_model_func"]
    with timer():
        results = await chunks_vdb.query(query, top_k=query_param.top_k)
        if not len(results):
            return prompts["fail_response"]
        chunks_ids = [r["id"] for r in results]
        chunks = await text_chunks_db.get_by_ids(chunks_ids)

        maybe_trun_chunks = truncate_list_by_token_size(
            chunks,
            key=lambda x: x["content"],
            max_token_size=query_param.naive_max_token_for_text_unit,
        )
        logger.info(f"Truncate {len(chunks)} to {len(maybe_trun_chunks)} chunks")
        section = "--New Chunk--\n".join([c["content"] for c in maybe_trun_chunks])
        if query_param.only_need_context:
            return section
    sys_prompt_temp = prompts["naive_rag_response"]
    sys_prompt = sys_prompt_temp.format(
        content_data=section, response_type=query_param.response_type
    )
    response = await use_model_func(
        query,
        system_prompt=sys_prompt,
        stage="query_answer",
        max_tokens=_get_stage_max_tokens(global_config, "query_answer"),
    )
    return response
