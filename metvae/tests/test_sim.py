"""Tests for the data simulator."""

import numpy as np
import pandas as pd
import pytest

from metvae.sim import sim_data


def test_sim_data_shapes_and_seed():
    """Shapes, names, correlation structure, argument checks and seeding behave as specified."""
    n, d, cor_pairs = 40, 12, 4
    out = sim_data(n=n, d=d, cor_pairs=cor_pairs, mu=[2, 3, 4], seed=11)

    y = out['y']
    assert isinstance(y, pd.DataFrame)
    assert y.shape == (n, d)
    assert list(y.index) == ["s" + str(i) for i in range(n)]
    assert list(y.columns) == ["f" + str(j) for j in range(d)]

    cor = out['cor_matrix']
    assert cor.shape == (d, d)
    assert np.array_equal(cor, cor.T)
    assert np.all(np.diag(cor) == 1.0)

    off_diag_nonzero = int((cor != 0).sum() - d)
    assert off_diag_nonzero == 2 * cor_pairs

    with pytest.raises(ValueError):
        sim_data(n=n, d=d, cor_pairs=d // 2 + 1, mu=[2, 3, 4], seed=11)

    with pytest.raises(ValueError):
        sim_data(n=n, d=d, cor_pairs=cor_pairs, mu=None, seed=11)

    repeat = sim_data(n=n, d=d, cor_pairs=cor_pairs, mu=[2, 3, 4], seed=11)
    np.testing.assert_array_equal(repeat['y'].values, y.values)
    np.testing.assert_array_equal(repeat['cor_matrix'], cor)
