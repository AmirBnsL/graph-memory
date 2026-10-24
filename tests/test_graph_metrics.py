import sys
import unittest
from pathlib import Path
import networkx as nx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from graph_memory.analysis.graph_metrics import compute_graph_metrics


class TestGraphMetrics(unittest.TestCase):
    def test_empty_graph(self):
        g = nx.Graph()
        metrics = compute_graph_metrics(g)
        self.assertEqual(metrics["num_nodes"], 0)
        self.assertEqual(metrics["num_edges"], 0)
        self.assertEqual(metrics["largest_component_ratio"], 0.0)

    def test_star_graph(self):
        # Center node connected to 5 leaves -> high hub concentration
        g = nx.star_graph(5)
        metrics = compute_graph_metrics(g)
        self.assertEqual(metrics["num_nodes"], 6)
        self.assertEqual(metrics["num_edges"], 5)
        self.assertEqual(metrics["largest_component_ratio"], 1.0)
        self.assertEqual(metrics["num_connected_components"], 1)
        self.assertGreater(metrics["degree_gini"], 0.3)
        self.assertGreater(metrics["max_degree_ratio"], 2.0)

    def test_disconnected_graph(self):
        # Two disconnected triangles: 6 nodes, 2 components
        g = nx.Graph()
        g.add_edges_from([(1, 2), (2, 3), (3, 1)])
        g.add_edges_from([(4, 5), (5, 6), (6, 4)])
        metrics = compute_graph_metrics(g)
        self.assertEqual(metrics["num_nodes"], 6)
        self.assertEqual(metrics["num_edges"], 6)
        self.assertEqual(metrics["num_connected_components"], 2)
        self.assertEqual(metrics["largest_component_ratio"], 0.5)

    def test_temporal_multidigraph(self):
        g = nx.MultiDiGraph()
        g.add_node("Alice")
        g.add_node("Paris")
        g.add_node("Berlin")
        # 1 active edge, 1 superseded edge
        g.add_edge("Alice", "Paris", status="superseded", t_valid="2021", t_invalid="2023")
        g.add_edge("Alice", "Berlin", status="active", t_valid="2023", t_invalid="present")

        metrics = compute_graph_metrics(g)
        self.assertEqual(metrics["num_nodes"], 3)
        self.assertEqual(metrics["num_edges"], 2)
        self.assertEqual(metrics["temporal_edge_ratio"], 1.0)
        self.assertEqual(metrics["invalidated_edge_ratio"], 0.5)


if __name__ == "__main__":
    unittest.main()
