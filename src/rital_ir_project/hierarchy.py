"""Hierarchical clustering utilities mirroring the paper's HiIndex logic."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.mixture import GaussianMixture

from .models import Entity, LayerStatistics, Relation


@dataclass(slots=True)
class HierarchyConfig:
    max_layers: int = 6
    max_components: int = 8
    probability_threshold: float = 0.10
    sparsity_stop_threshold: float = 0.98
    change_rate_stop_threshold: float = 0.05
    random_state: int = 0


def choose_gmm_components(
    embeddings: np.ndarray,
    max_components: int,
    random_state: int,
) -> int:
    max_components = max(1, min(max_components, len(embeddings)))
    if max_components == 1:
        return 1

    best_components = 1
    best_bic = float("inf")

    for component_count in range(1, max_components + 1):
        model = GaussianMixture(
            n_components=component_count,
            random_state=random_state,
            covariance_type="full",
            n_init=3,
        )
        model.fit(embeddings)
        bic = model.bic(embeddings)
        if bic < best_bic:
            best_bic = bic
            best_components = component_count

    return best_components


def assign_overlapping_clusters(
    embeddings: np.ndarray,
    component_count: int,
    probability_threshold: float,
    random_state: int,
) -> list[list[int]]:
    if len(embeddings) == 1:
        return [[0]]

    model = GaussianMixture(
        n_components=component_count,
        random_state=random_state,
        covariance_type="full",
        n_init=3,
    )
    model.fit(embeddings)
    probabilities = model.predict_proba(embeddings)

    assignments: list[list[int]] = []
    for row in probabilities:
        labels = np.flatnonzero(row >= probability_threshold).tolist()
        if not labels:
            labels = [int(np.argmax(row))]
        assignments.append(labels)
    return assignments


def build_cluster_members(assignments: Sequence[Sequence[int]]) -> dict[int, list[int]]:
    clusters: dict[int, list[int]] = defaultdict(list)
    for entity_index, labels in enumerate(assignments):
        for label in labels:
            clusters[int(label)].append(entity_index)
    return dict(clusters)


def compute_cluster_sparsity(cluster_sizes: Sequence[int], entity_count: int) -> float:
    if entity_count <= 1:
        return 1.0
    numerator = sum(size * (size - 1) for size in cluster_sizes)
    denominator = entity_count * (entity_count - 1)
    return 1.0 - (numerator / denominator)


def build_hierarchy(
    base_entities: Sequence[Entity],
    embed_texts: callable,
    summarizer: callable,
    config: HierarchyConfig,
) -> tuple[list[Entity], list[Relation], list[LayerStatistics]]:
    current_layer_entities = list(base_entities)
    created_entities: list[Entity] = []
    created_relations: list[Relation] = []
    layer_statistics: list[LayerStatistics] = []
    previous_sparsity = 0.01

    for layer in range(1, config.max_layers + 1):
        if len(current_layer_entities) <= 2:
            break

        embeddings = embed_texts(
            [f"{entity.name} {entity.description}" for entity in current_layer_entities]
        )
        component_count = choose_gmm_components(
            embeddings,
            max_components=config.max_components,
            random_state=config.random_state,
        )
        assignments = assign_overlapping_clusters(
            embeddings,
            component_count=component_count,
            probability_threshold=config.probability_threshold,
            random_state=config.random_state,
        )
        clusters = build_cluster_members(assignments)
        cluster_sparsity = compute_cluster_sparsity(
            [len(cluster) for cluster in clusters.values()],
            len(current_layer_entities),
        )
        change_rate = abs(cluster_sparsity - previous_sparsity) / max(previous_sparsity, 1e-8)

        layer_statistics.append(
            LayerStatistics(
                layer=layer,
                input_entities=len(current_layer_entities),
                cluster_count=len(clusters),
                cluster_sparsity=cluster_sparsity,
                change_rate=change_rate,
            )
        )

        if cluster_sparsity >= config.sparsity_stop_threshold:
            break
        if change_rate <= config.change_rate_stop_threshold:
            break

        next_layer_entities: list[Entity] = []
        for cluster_index, member_indices in sorted(clusters.items()):
            cluster_entities = [current_layer_entities[index] for index in member_indices]
            if len(cluster_entities) <= 1:
                continue

            summary_entities, summary_relations = summarizer(
                layer,
                cluster_index,
                cluster_entities,
            )
            created_entities.extend(summary_entities)
            created_relations.extend(summary_relations)
            next_layer_entities.extend(summary_entities)

        unique_next_layer: dict[str, Entity] = {}
        for entity in next_layer_entities:
            unique_next_layer[entity.name] = entity

        current_layer_entities = list(unique_next_layer.values())
        previous_sparsity = cluster_sparsity

    return created_entities, created_relations, layer_statistics
