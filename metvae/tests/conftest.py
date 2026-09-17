"""Shared fixtures and command-line options for the MetVAE test suite."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from metvae.sim import sim_data


def pytest_addoption(parser):
    """Register the --runslow flag."""
    parser.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help="run tests marked slow",
    )


def pytest_configure(config):
    """Declare the slow marker so it is recognized without an ini file."""
    config.addinivalue_line("markers", "slow: long-running end-to-end tests")


def pytest_collection_modifyitems(config, items):
    """Skip tests marked slow unless --runslow is given."""
    if config.getoption("--runslow"):
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


def _simulate(n, d, cor_pairs, seed, zero_prop=0.3, da_prop=0.1):
    """
    Simulate an abundance matrix with sample and feature biases and censored zeros.

    Parameters
    ----------
    n : int
        Number of samples.
    d : int
        Number of features.
    cor_pairs : int
        Number of correlated feature pairs.
    seed : int
        Seed for the global numpy random state.
    zero_prop : float, default=0.3
        Per-feature proportion of observations set to zero.
    da_prop : float, default=0.1
        Proportion of uncorrelated features affected by the covariates.

    Returns
    -------
    dict
        Keys ``n_samples``, ``d_features``, ``abundance_data``, ``meta_data`` and
        ``true_cor``.
    """
    np.random.seed(seed)

    smd = pd.DataFrame({'x1': np.random.randn(n),
                        'x2': np.random.choice(['a', 'b'], size=n, replace=True)})
    smd.index = ["s" + str(i) for i in range(n)]

    sim = sim_data(n=n, d=d, cor_pairs=cor_pairs, mu=list(range(10, 15)),
                   x=smd, cont_list=['x1'], cat_list=['x2'], da_prop=da_prop)
    y = sim['y']
    true_cor = sim['cor_matrix']

    log_y = np.log(y)
    log_sample_bias = np.log(np.random.uniform(1e-3, 1e-1, size=n))
    log_feature_bias = np.log(np.random.uniform(1e-1, 1, size=d))
    log_data = log_y + log_sample_bias[:, np.newaxis]
    log_data = log_data + log_feature_bias.reshape(1, d)
    data = np.exp(log_data)

    # Per-feature quantile masking sets the lowest zero_prop of each column to zero
    thresholds = np.quantile(data, zero_prop, axis=0)
    data_miss = np.where(data < thresholds, 0, data)
    data_miss = pd.DataFrame(data_miss, index=y.index, columns=y.columns)

    return {
        'n_samples': n,
        'd_features': d,
        'abundance_data': data_miss,
        'meta_data': smd,
        'true_cor': true_cor,
    }


@pytest.fixture
def test_files():
    """Return the paths of the packaged abundance and metadata CSV files."""
    test_dir = Path(__file__).parent.absolute()
    data_path = test_dir / 'test_data.csv'
    meta_path = test_dir / 'test_smd.csv'

    assert data_path.exists(), f"Test data file not found at {data_path}"
    assert meta_path.exists(), f"Test metadata file not found at {meta_path}"

    return data_path, meta_path


@pytest.fixture
def small_simulation():
    """Return a 60 x 30 simulated dataset with 6 correlated pairs and 30 percent zeros."""
    return _simulate(n=60, d=30, cor_pairs=6, seed=123)


@pytest.fixture
def large_simulation():
    """Return a 100 x 50 simulated dataset with 10 correlated pairs and 30 percent zeros."""
    return _simulate(n=100, d=50, cor_pairs=10, seed=123)
