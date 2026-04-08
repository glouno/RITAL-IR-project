"""A local HiRAG-style scaffold for indexing and retrieval experiments."""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import networkx as nx
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .hierarchy import HierarchyConfig, build_hierarchy
from .models import CommunityReport, Entity, IndexArtifacts, Relation, TextChunk
from .retrieval import build_community_reports, detect_communities, retrieve_context


ENTITY_PATTERN = re.compile(
    r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}|[A-Z]{2,}(?:\s+[A-Z]{2,}){0,2})\b"
)
ENTITY_STOPWORDS = {
    "The",
    "A",
    "An",
    "This",
    "That",
    "These",
    "Those",
    "Chapter",
    "Table",
    "Figure",
    "Section",
    "Appendix",
    "Introduction",
}


def stable_id(value: str) -> str:
    return hashlib.md5(value.encode("utf-8"), usedforsecurity=False).hexdigest()


def split_text_into_chunks(text: str, chunk_size: int, overlap: int) -> list[str]:
    words = text.split()
    if not words:
        return []
    step = max(1, chunk_size - overlap)
    chunks: list[str] = []
    for start in range(0, len(words), step):
        chunk_words = words[start : start + chunk_size]
        if not chunk_words:
            continue
        chunks.append(" ".join(chunk_words))
        if start + chunk_size >= len(words):
            break
    return chunks


class TfidfTextEmbedder:
    def __init__(self) -> None:
        self.vectorizer = TfidfVectorizer(max_features=2048, stop_words="english", ngram_range=(1, 2))
        self.is_fitted = False

    def fit(self, texts: Sequence[str]) -> None:
        corpus = list(texts) or ["empty"]
        self.vectorizer.fit(corpus)
        self.is_fitted = True

    def transform(self, texts: Sequence[str]) -> np.ndarray:
        if not self.is_fitted:
            self.fit(texts)
        return self.vectorizer.transform(list(texts)).toarray()

    def transform_one(self, text: str) -> np.ndarray:
        return self.transform([text])[0]


class SimpleRegexEntityExtractor:
    def extract(self, chunk: TextChunk) -> tuple[list[Entity], list[Relation]]:
        names: list[str] = []
        for match in ENTITY_PATTERN.findall(chunk.text):
            candidate = match.strip()
            if candidate in ENTITY_STOPWORDS:
                continue
            if candidate not in names:
                names.append(candidate)

        entities = [
            Entity(
                name=name,
                entity_type="normal_entity",
                description=f"Mentioned in chunk {chunk.document_id}#{chunk.order}.",
                source_chunks=(chunk.id,),
                layer=0,
            )
            for name in names
        ]

        relations: list[Relation] = []
        for index, source in enumerate(names):
            for target in names[index + 1 :]:
                relations.append(
                    Relation(
                        source=source,
                        target=target,
                        description="Co-occurs in the same text chunk.",
                        weight=1.0,
                        source_chunks=(chunk.id,),
                    )
                )

        return entities, relations


class SimpleClusterSummarizer:
    def summarize(
        self,
        layer: int,
        cluster_index: int,
        cluster_entities: Sequence[Entity],
    ) -> tuple[list[Entity], list[Relation]]:
        anchor_tokens = []
        for entity in cluster_entities[:3]:
            anchor_tokens.extend(re.findall(r"[A-Za-z0-9]+", entity.name))
        anchor = "_".join(anchor_tokens[:4]).upper() or f"LAYER_{layer}"
        summary_name = f"L{layer}_C{cluster_index}_{anchor}"
        member_names = tuple(entity.name for entity in cluster_entities)

        summary_entity = Entity(
            name=summary_name,
            entity_type="summary_entity",
            description=f"Summary node for: {', '.join(member_names[:6])}.",
            source_chunks=tuple(
                dict.fromkeys(
                    chunk_id
                    for entity in cluster_entities
                    for chunk_id in entity.source_chunks
                )
            ),
            layer=layer,
            members=member_names,
        )
        relations = [
            Relation(
                source=entity.name,
                target=summary_name,
                description=f"{entity.name} is summarized by {summary_name}.",
                weight=1.0,
                source_chunks=entity.source_chunks,
            )
            for entity in cluster_entities
        ]
        return [summary_entity], relations


class SimpleCommunityReporter:
    def summarize(self, community_id: str, entities: Sequence[Entity]) -> CommunityReport:
        top_names = [entity.name for entity in entities[:6]]
        summary = f"Community built around: {', '.join(top_names)}"
        return CommunityReport(
            id=community_id,
            nodes=tuple(entity.name for entity in entities),
            level=max((entity.layer for entity in entities), default=0),
            summary=summary,
        )


@dataclass(slots=True)
class HiRAGScaffold:
    chunk_size_words: int = 220
    chunk_overlap_words: int = 40
    hierarchy: HierarchyConfig = field(default_factory=HierarchyConfig)
    extractor: SimpleRegexEntityExtractor = field(default_factory=SimpleRegexEntityExtractor)
    summarizer: SimpleClusterSummarizer = field(default_factory=SimpleClusterSummarizer)
    community_reporter: SimpleCommunityReporter = field(default_factory=SimpleCommunityReporter)
    graph: nx.Graph = field(init=False, repr=False)
    chunks: dict[str, TextChunk] = field(init=False, repr=False)
    embedder: TfidfTextEmbedder = field(init=False, repr=False)
    artifacts: IndexArtifacts = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.graph = nx.Graph()
        self.chunks: dict[str, TextChunk] = {}
        self.embedder = TfidfTextEmbedder()
        self.artifacts = IndexArtifacts()

    def index_documents(self, documents: dict[str, str]) -> IndexArtifacts:
        self.graph.clear()
        self.chunks.clear()
        self.artifacts = IndexArtifacts()

        base_entities: dict[str, Entity] = {}
        base_relations: list[Relation] = []

        for document_id, text in documents.items():
            chunk_texts = split_text_into_chunks(text, self.chunk_size_words, self.chunk_overlap_words)
            for order, chunk_text in enumerate(chunk_texts):
                chunk_id = stable_id(f"{document_id}:{order}:{chunk_text[:200]}")
                chunk = TextChunk(id=chunk_id, document_id=document_id, order=order, text=chunk_text)
                self.chunks[chunk_id] = chunk
                entities, relations = self.extractor.extract(chunk)
                for entity in entities:
                    merged = base_entities.get(entity.name)
                    if merged is None:
                        base_entities[entity.name] = entity
                    else:
                        base_entities[entity.name] = Entity(
                            name=merged.name,
                            entity_type=merged.entity_type,
                            description=merged.description,
                            source_chunks=tuple(dict.fromkeys((*merged.source_chunks, *entity.source_chunks))),
                            layer=0,
                            members=merged.members,
                        )
                base_relations.extend(relations)

        for entity in base_entities.values():
            self._upsert_entity(entity)
        for relation in base_relations:
            self._upsert_relation(relation)

        base_entity_list = list(base_entities.values())
        hierarchy_entities, hierarchy_relations, layer_stats = build_hierarchy(
            base_entity_list,
            embed_texts=self._embed_texts_for_clustering,
            summarizer=self.summarizer.summarize,
            config=self.hierarchy,
        )

        for entity in hierarchy_entities:
            self._upsert_entity(entity)
        for relation in hierarchy_relations:
            self._upsert_relation(relation)

        self._refresh_node_embeddings()

        communities = detect_communities(self.graph)
        self.artifacts.layer_statistics.extend(layer_stats)
        self.artifacts.communities.extend(
            build_community_reports(self.graph, communities, self.community_reporter.summarize)
        )
        return self.artifacts

    def query(self, query: str, top_k: int = 20, top_m: int = 10):
        return retrieve_context(
            graph=self.graph,
            community_reports=self.artifacts.communities,
            chunks=self.chunks,
            query=query,
            embed_query=self.embedder.transform_one,
            top_k=top_k,
            top_m=top_m,
        )

    def _upsert_entity(self, entity: Entity) -> None:
        if self.graph.has_node(entity.name):
            node = self.graph.nodes[entity.name]
            node["description"] = str(node.get("description", entity.description))
            node["source_chunks"] = tuple(
                dict.fromkeys((*node.get("source_chunks", ()), *entity.source_chunks))
            )
            node["layer"] = max(int(node.get("layer", 0)), entity.layer)
            existing_members = tuple(node.get("members", ()))
            node["members"] = tuple(dict.fromkeys((*existing_members, *entity.members)))
            return

        self.graph.add_node(
            entity.name,
            entity_type=entity.entity_type,
            description=entity.description,
            source_chunks=entity.source_chunks,
            layer=entity.layer,
            members=entity.members,
        )

    def _upsert_relation(self, relation: Relation) -> None:
        left, right = sorted((relation.source, relation.target))
        if self.graph.has_edge(left, right):
            edge = self.graph.edges[left, right]
            edge["weight"] = float(edge.get("weight", 0.0)) + relation.weight
            edge["source_chunks"] = tuple(
                dict.fromkeys((*edge.get("source_chunks", ()), *relation.source_chunks))
            )
            return

        self.graph.add_edge(
            left,
            right,
            description=relation.description,
            weight=relation.weight,
            source_chunks=relation.source_chunks,
        )

    def _embed_texts_for_clustering(self, texts: Sequence[str]) -> np.ndarray:
        local_embedder = TfidfTextEmbedder()
        local_embedder.fit(texts)
        return local_embedder.transform(texts)

    def _refresh_node_embeddings(self) -> None:
        node_names = list(self.graph.nodes)
        node_texts = [
            f"{name} {self.graph.nodes[name].get('description', '')}"
            for name in node_names
        ]
        self.embedder.fit(node_texts)
        vectors = self.embedder.transform(node_texts)
        for node_name, vector in zip(node_names, vectors):
            self.graph.nodes[node_name]["embedding"] = vector


def load_ultradomain_documents(rows: Iterable[dict], limit: int | None = None) -> tuple[dict[str, str], str | None]:
    documents: dict[str, str] = {}
    first_query: str | None = None

    for index, row in enumerate(rows):
        if first_query is None:
            first_query = str(row.get("input", "")) or None
        document_id = str(row.get("_id") or row.get("context_id") or f"doc-{index}")
        documents[document_id] = str(row.get("context", ""))
        if limit is not None and len(documents) >= limit:
            break

    return documents, first_query
