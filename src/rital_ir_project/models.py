"""Data structures and lightweight interfaces for the HiRAG scaffold."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence


@dataclass(slots=True, frozen=True)
class TextChunk:
    id: str
    document_id: str
    order: int
    text: str


@dataclass(slots=True, frozen=True)
class Entity:
    name: str
    entity_type: str
    description: str
    source_chunks: tuple[str, ...] = ()
    layer: int = 0
    members: tuple[str, ...] = ()


@dataclass(slots=True, frozen=True)
class Relation:
    source: str
    target: str
    description: str
    weight: float = 1.0
    source_chunks: tuple[str, ...] = ()


@dataclass(slots=True, frozen=True)
class LayerStatistics:
    layer: int
    input_entities: int
    cluster_count: int
    cluster_sparsity: float
    change_rate: float


@dataclass(slots=True, frozen=True)
class CommunityReport:
    id: str
    nodes: tuple[str, ...]
    level: int
    summary: str


@dataclass(slots=True, frozen=True)
class RetrievedContext:
    query: str
    local_entities: tuple[Entity, ...]
    global_communities: tuple[CommunityReport, ...]
    bridge_relations: tuple[Relation, ...]
    source_chunks: tuple[TextChunk, ...]

    def render(self) -> str:
        local_lines = [
            f"- {entity.name} [{entity.entity_type}] :: {entity.description}"
            for entity in self.local_entities
        ] or ["- none"]
        community_lines = [
            f"- {community.id} :: {community.summary}"
            for community in self.global_communities
        ] or ["- none"]
        bridge_lines = [
            f"- {relation.source} -> {relation.target} :: {relation.description}"
            for relation in self.bridge_relations
        ] or ["- none"]
        chunk_lines = [
            f"- {chunk.document_id}#{chunk.order} :: {chunk.text[:160].replace(chr(10), ' ')}"
            for chunk in self.source_chunks
        ] or ["- none"]

        return "\n".join(
            [
                f"Query: {self.query}",
                "",
                "Local entities:",
                *local_lines,
                "",
                "Global communities:",
                *community_lines,
                "",
                "Bridge relations:",
                *bridge_lines,
                "",
                "Supporting chunks:",
                *chunk_lines,
            ]
        )


@dataclass(slots=True)
class IndexArtifacts:
    layer_statistics: list[LayerStatistics] = field(default_factory=list)
    communities: list[CommunityReport] = field(default_factory=list)


class EntityExtractor(Protocol):
    def extract(self, chunk: TextChunk) -> tuple[list[Entity], list[Relation]]:
        """Extract base-layer entities and relations from a text chunk."""


class ClusterSummarizer(Protocol):
    def summarize(
        self,
        layer: int,
        cluster_index: int,
        cluster_entities: Sequence[Entity],
    ) -> tuple[list[Entity], list[Relation]]:
        """Produce summary entities and parent-child relations for one cluster."""


class CommunityReporter(Protocol):
    def summarize(self, community_id: str, entities: Sequence[Entity]) -> CommunityReport:
        """Produce a short report for a detected community."""
