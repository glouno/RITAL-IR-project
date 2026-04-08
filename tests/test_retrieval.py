import unittest

import networkx as nx
import numpy as np

from rital_ir_project.models import CommunityReport, TextChunk
from rital_ir_project.retrieval import retrieve_context


class RetrievalTests(unittest.TestCase):
    def test_bridge_relations_follow_shortest_path(self) -> None:
        graph = nx.Graph()
        graph.add_node("A", entity_type="entity", description="alpha", source_chunks=("c1",), layer=0, members=(), embedding=np.array([1.0, 0.0]))
        graph.add_node("B", entity_type="entity", description="beta", source_chunks=("c2",), layer=0, members=(), embedding=np.array([0.9, 0.1]))
        graph.add_node("C", entity_type="entity", description="gamma", source_chunks=("c3",), layer=0, members=(), embedding=np.array([0.0, 1.0]))
        graph.add_edge("A", "B", description="A to B", weight=1.0, source_chunks=("c1", "c2"))
        graph.add_edge("B", "C", description="B to C", weight=1.0, source_chunks=("c2", "c3"))

        communities = [CommunityReport(id="community-0", nodes=("A", "B", "C"), level=0, summary="demo community")]
        chunks = {
            "c1": TextChunk(id="c1", document_id="d1", order=0, text="alpha"),
            "c2": TextChunk(id="c2", document_id="d1", order=1, text="beta"),
            "c3": TextChunk(id="c3", document_id="d1", order=2, text="gamma"),
        }

        context = retrieve_context(
            graph=graph,
            community_reports=communities,
            chunks=chunks,
            query="alpha",
            embed_query=lambda _: np.array([1.0, 0.0]),
            top_k=3,
            top_m=3,
        )

        bridge_pairs = {(relation.source, relation.target) for relation in context.bridge_relations}
        self.assertIn(("A", "B"), bridge_pairs)


if __name__ == "__main__":
    unittest.main()
