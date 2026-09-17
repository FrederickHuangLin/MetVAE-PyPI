"""Tests for the correlation reshaping helper and the GraphML export."""

import warnings

import numpy as np
import pandas as pd

from metvae.model import MetVAE
from metvae.utils import _corr_to_long


def _symmetric_corr(p=8, seed=13):
    """Return a symmetric correlation DataFrame with unit diagonal."""
    rng = np.random.default_rng(seed)
    a = rng.uniform(-1.0, 1.0, size=(p, p))
    mat = 0.5 * (a + a.T)
    np.fill_diagonal(mat, 1.0)
    names = [f"f{j}" for j in range(p)]
    return pd.DataFrame(mat, index=names, columns=names)


def test_corr_to_long():
    """The long form holds every upper-triangle pair, sorted by absolute correlation."""
    corr = _symmetric_corr()
    p = corr.shape[0]

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("error", FutureWarning)
        df_long = _corr_to_long(corr)

    assert list(df_long.columns) == ["node1", "node2", "correlation"]
    assert len(df_long) == p * (p - 1) // 2

    abs_corr = df_long["correlation"].abs().values
    assert np.all(np.diff(abs_corr) <= 1e-12)


def test_export_graphml_matches_manual_edges():
    """The exported graphs match an edge list built directly from the upper triangle."""
    import networkx as nx

    corr = _symmetric_corr()
    sparse_df = corr.where(corr.abs() >= 0.25, 0.0)
    np.fill_diagonal(sparse_df.values, 1.0)

    cutoffs = [0.3, 0.5]
    model = object.__new__(MetVAE)
    graphs = model.export_graphml(sparse_df=sparse_df, cutoffs=cutoffs, output_dir=None)

    names = list(sparse_df.index)
    expected = {}
    for cutoff in sorted({float(c) for c in cutoffs}, reverse=True):
        edge_type = f"Correlation_cutoff{cutoff:g}"
        graph = nx.Graph()
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                value = float(sparse_df.iloc[i, j])
                if abs(value) >= cutoff:
                    graph.add_edge(
                        names[i], names[j],
                        weight=value, correlation=value, EdgeScore=value,
                        EdgeType=edge_type, id=edge_type,
                    )
        if graph.number_of_edges() > 0:
            expected[edge_type] = graph

    assert set(graphs.keys()) == set(expected.keys())
    assert len(expected) == len(cutoffs)

    for key, graph in expected.items():
        obtained = graphs[key]
        assert set(obtained.nodes) == set(graph.nodes)
        assert {frozenset(e) for e in obtained.edges} == {frozenset(e) for e in graph.edges}
        for u, v, attrs in graph.edges(data=True):
            assert obtained.edges[u, v] == attrs
