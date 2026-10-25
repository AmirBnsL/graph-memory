import unittest
from graph_memory.pipeline.extract import MemoryRecord
from graph_memory.memory.graph.hipporag import HippoRAGMemory
from graph_memory.memory.graph.temporal_bteg import BiTemporalMemory


class MockEmbedder:
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        # Deterministic 4-d vector
        vecs = []
        for t in texts:
            tl = t.lower()
            vecs.append([
                float(len(t)),
                float("bmw" in tl or "car" in tl),
                float("berlin" in tl or "city" in tl),
                float("alice" in tl or "bob" in tl),
            ])
        return vecs

    def embed_query(self, query: str) -> list[float]:
        return self.embed_texts([query])[0]


class TestTicket2TripleAlignment(unittest.TestCase):
    def setUp(self):
        self.mock_emb = MockEmbedder()
        # Single turn with 2 distinct propositions and 2 distinct triples
        # In current extract.py, both MemoryRecords share the turn's full entity list & triples list
        shared_entities = ["Alice", "blue BMW", "Bob", "Berlin"]
        shared_triples = [("Alice", "bought", "blue BMW"), ("Bob", "moved to", "Berlin")]
        
        self.rec1 = MemoryRecord(
            id="rec_1",
            sample_id="conv_1",
            turn_id=0,
            speaker="User",
            timestamp="2023-01-01",
            fact_text="Alice bought a blue BMW.",
            entities=shared_entities,
            triples=shared_triples,
        )
        self.rec2 = MemoryRecord(
            id="rec_2",
            sample_id="conv_1",
            turn_id=0,
            speaker="User",
            timestamp="2023-01-01",
            fact_text="Bob moved to Berlin.",
            entities=shared_entities,
            triples=shared_triples,
        )

    def test_bteg_edge_binding_to_correct_proposition(self):
        bteg = BiTemporalMemory(embedder=self.mock_emb)
        bteg.add_records([self.rec1, self.rec2])
        g = bteg.export_graph()

        # The edge (bob, berlin) must be bound to rec2 ("Bob moved to Berlin.")
        # and NOT rec1 ("Alice bought a blue BMW.")
        bob_edges = [d for _, _, d in g.out_edges("bob", data=True)]
        self.assertEqual(len(bob_edges), 1)
        self.assertEqual(bob_edges[0]["record_id"], "rec_2")
        self.assertIn("Berlin", bob_edges[0]["fact_text"])

        alice_edges = [d for _, _, d in g.out_edges("alice", data=True)]
        self.assertEqual(len(alice_edges), 1)
        self.assertEqual(alice_edges[0]["record_id"], "rec_1")
        self.assertIn("BMW", alice_edges[0]["fact_text"])

    def test_hipporag_entity_matching_per_proposition(self):
        hippo = HippoRAGMemory(embedder=self.mock_emb)
        hippo.add_records([self.rec1, self.rec2])
        
        # When querying about Berlin/Bob, rec2 should be retrieved as top item, not contaminated by rec1
        res = hippo.retrieve(query="Where did Bob move?", k=1)
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].id, "rec_2")
        self.assertIn("Berlin", res[0].text)


if __name__ == "__main__":
    unittest.main()
