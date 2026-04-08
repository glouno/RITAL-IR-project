"""HiRetrieval-style community selection and bridge path construction."""

from __future__ import annotations

from collections import OrderedDict
from typing import Iterable

import networkx as nx
import numpy as np

from .models import CommunityReport, Entity, Relation, RetrievedContext, TextChunk


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left_norm = np.linalg.norm(left)
    right_norm = np.linalg.norm(right)
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return float(np.dot(left, right) / (left_norm * right_norm))


def detect_communities(graph: nx.Graph) -> list[set[str]]:
    if graph.number_of_nodes() == 0:
        return []

    try:
        communities = nx.community.louvain_communities(graph, seed=0)
    except AttributeError:
        communities = list(nx.community.greedy_modularity_communities(graph))
    return [set(community) for community in communities]


def build_community_reports(
    graph: nx.Graph,
    communities: Iterable[set[str]],
    reporter: callable,
) -> list[CommunityReport]:
    reports: list[CommunityReport] = []
    for index, nodes in enumerate(communities):
        entities = [
            Entity(
                name=node_name,
                entity_type=str(graph.nodes[node_name].get("entity_type", "normal_entity")),
                description=str(graph.nodes[node_name].get("description", "")),
                source_chunks=tuple(graph.nodes[node_name].get("source_chunks", ())),
                layer=int(graph.nodes[node_name].get("layer", 0)),
                members=tuple(graph.nodes[node_name].get("members", ())),
            )
            for node_name in sorted(nodes)
        ]
        reports.append(reporter(f"community-{index}", entities))
    return reports


def retrieve_context(
    graph: nx.Graph,
    community_reports: list[CommunityReport],
    chunks: dict[str, TextChunk],
    query: str,
    embed_query: callable,
    top_k: int = 20,
    top_m: int = 10,
) -> RetrievedContext:
    if graph.number_of_nodes() == 0:
        return RetrievedContext(query=query, local_entities=(), global_communities=(), bridge_relations=(), source_chunks=())

    query_vector = embed_query(query)
    scored_nodes: list[tuple[str, float]] = []
    for node_name, attributes in graph.nodes(data=True):
        score = cosine_similarity(np.asarray(attributes["embedding"]), query_vector)
        scored_nodes.append((node_name, score))
    scored_nodes.sort(key=lambda item: item[1], reverse=True)

    local_names = [name for name, _ in scored_nodes[:top_k]]
    local_entities = tuple(
        Entity(
            name=name,
            entity_type=str(graph.nodes[name].get("entity_type", "normal_entity")),
            description=str(graph.nodes[name].get("description", "")),
            source_chunks=tuple(graph.nodes[name].get("source_chunks", ())),
            layer=int(graph.nodes[name].get("layer", 0)),
            members=tuple(graph.nodes[name].get("members", ())),
        )
        for name in local_names
    )

    community_scores: list[tuple[CommunityReport, int]] = []
    for report in community_reports:
        overlap = len(set(report.nodes).intersection(local_names))
        if overlap:
            community_scores.append((report, overlap))
    community_scores.sort(key=lambda item: item[1], reverse=True)
    selected_communities = tuple(report for report, _ in community_scores)

    ordered_local_names = OrderedDict((name, None) for name in local_names)
    key_entities: list[str] = []
    for community in selected_communities:
        matches = [name for name in ordered_local_names if name in community.nodes][:top_m]
        key_entities.extend(matches)
    if not key_entities:
        key_entities = list(ordered_local_names)[:top_m]

    unique_key_entities = list(OrderedDict((name, None) for name in key_entities))
    bridge_pairs: list[tuple[str, str]] = []
    for current, following in zip(unique_key_entities, unique_key_entities[1:]):
        try:
            path = nx.shortest_path(graph, current, following)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue
        for left, right in zip(path, path[1:]):
            pair = tuple(sorted((left, right)))
            if pair not in bridge_pairs:
                bridge_pairs.append(pair)

    bridge_relations = tuple(
        Relation(
            source=left,
            target=right,
            description=str(graph.edges[left, right].get("description", "")),
            weight=float(graph.edges[left, right].get("weight", 1.0)),
            source_chunks=tuple(graph.edges[left, right].get("source_chunks", ())),
        )
        for left, right in bridge_pairs
    )

    chunk_ids: OrderedDict[str, None] = OrderedDict()
    for entity in local_entities:
        for chunk_id in entity.source_chunks:
            chunk_ids[chunk_id] = None
    for relation in bridge_relations:
        for chunk_id in relation.source_chunks:
            chunk_ids[chunk_id] = None
    source_chunks = tuple(chunks[chunk_id] for chunk_id in chunk_ids if chunk_id in chunks)

    return RetrievedContext(
        query=query,
        local_entities=local_entities,
        global_communities=selected_communities,
        bridge_relations=bridge_relations,
        source_chunks=source_chunks,
    )
