"""Topological Quality Metrics for Agent Memory Graphs (Hypothesis 3).

Computes cheap, non-parametric structural statistics on memory graphs:
  - Connectivity: Largest connected component ratio (LCC / |V|), number of components.
  - Hub Concentration: Degree variance, degree Gini coefficient, max degree / avg degree.
  - Density & Clustering: Graph density, average clustering coefficient.
  - Temporal Validity: Ratio of edges with temporal validity intervals, invalidated edge ratio.
"""
from __future__ import annotations

import math
from typing import Any
import networkx as nx


def _gini(values: list[float]) -> float:
    """Compute the Gini coefficient of a list of non-negative values."""
    if not values or all(v == 0 for v in values):
        return 0.0
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    cum_vals = [0.0]
    for v in sorted_vals:
        cum_vals.append(cum_vals[-1] + v)
    total = cum_vals[-1]
    if total == 0:
        return 0.0
    # Gini formula = 1 - 2 * sum(area under Lorenz curve)
    # Area under trapezoids: (cum_vals[i] + cum_vals[i+1]) / (2 * n * total)
    area = sum((cum_vals[i] + cum_vals[i + 1]) for i in range(n)) / (2 * n * total)
    return max(0.0, min(1.0, 1.0 - 2.0 * area))


def compute_graph_metrics(g: nx.Graph | nx.DiGraph | nx.MultiDiGraph | None) -> dict[str, Any]:
    """Compute structural graph statistics on a memory graph."""
    if g is None or g.number_of_nodes() == 0:
        return {
            "num_nodes": 0,
            "num_edges": 0,
            "largest_component_ratio": 0.0,
            "num_connected_components": 0,
            "degree_variance": 0.0,
            "degree_gini": 0.0,
            "max_degree_ratio": 0.0,
            "edge_density": 0.0,
            "clustering_coefficient": 0.0,
            "temporal_edge_ratio": 0.0,
            "invalidated_edge_ratio": 0.0,
        }

    n = g.number_of_nodes()
    m = g.number_of_edges()

    # Degree metrics
    degrees = [float(d) for _, d in g.degree()]
    avg_deg = sum(degrees) / n if n > 0 else 0.0
    deg_var = sum((d - avg_deg) ** 2 for d in degrees) / n if n > 0 else 0.0
    max_deg = max(degrees) if degrees else 0.0
    max_deg_ratio = max_deg / avg_deg if avg_deg > 0 else 0.0
    deg_gini = _gini(degrees)

    # Component metrics
    if g.is_directed():
        components = list(nx.weakly_connected_components(g))
    else:
        components = list(nx.connected_components(g))

    num_components = len(components)
    largest_cc_len = max(len(c) for c in components) if components else 0
    lcc_ratio = largest_cc_len / n if n > 0 else 0.0

    # Density & clustering (computed on undirected simple graph projection)
    if isinstance(g, (nx.MultiGraph, nx.MultiDiGraph)):
        undirected_g = nx.Graph(g)
    elif g.is_directed():
        undirected_g = g.to_undirected()
    else:
        undirected_g = g

    m_simple = undirected_g.number_of_edges()
    edge_density = nx.density(undirected_g)
    try:
        clustering_coeff = nx.average_clustering(undirected_g)
    except Exception:
        clustering_coeff = 0.0

    try:
        bridge_edges = len(list(nx.bridges(undirected_g)))
        bridge_edge_ratio = bridge_edges / m_simple if m_simple > 0 else 0.0
    except Exception:
        bridge_edge_ratio = 0.0

    # Temporal edge attributes (measuring non-default temporal validity coverage)
    temporal_edges = 0
    invalidated_edges = 0
    for _, _, data in g.edges(data=True):
        if data.get("t_valid") not in (None, "", "present"):
            temporal_edges += 1
        if data.get("status") == "superseded" or (data.get("t_invalid") and data.get("t_invalid") != "present"):
            invalidated_edges += 1

    temp_edge_ratio = temporal_edges / m if m > 0 else 0.0
    inval_edge_ratio = invalidated_edges / m if m > 0 else 0.0

    return {
        "num_nodes": n,
        "num_edges": m,
        "largest_component_ratio": round(lcc_ratio, 4),
        "num_connected_components": num_components,
        "degree_variance": round(deg_var, 4),
        "degree_gini": round(deg_gini, 4),
        "max_degree_ratio": round(max_deg_ratio, 4),
        "edge_density": round(edge_density, 4),
        "clustering_coefficient": round(clustering_coeff, 4),
        "bridge_edge_ratio": round(bridge_edge_ratio, 4),
        "temporal_edge_ratio": round(temp_edge_ratio, 4),
        "invalidated_edge_ratio": round(inval_edge_ratio, 4),
    }
