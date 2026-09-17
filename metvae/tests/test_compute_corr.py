"""Tests for the proportionality-based correlation estimator."""

import numpy as np
import pytest
import torch

from metvae.compute_corr import _compute_correlation


def _lognormal_matrix(n=40, d=12, seed=17):
    """Return a zero-free lognormal abundance matrix as a float64 tensor."""
    rng = np.random.default_rng(seed)
    return torch.tensor(np.exp(rng.standard_normal((n, d))), dtype=torch.float64)


def _reference_correlation(data, eps=1e-12):
    """
    Compute the correlation estimate with numpy for zero-free input.

    Parameters
    ----------
    data : numpy.ndarray
        Strictly positive abundance matrix of shape (n, d).
    eps : float, default=1e-12
        Floor applied to variances and to the product of standard deviations.

    Returns
    -------
    numpy.ndarray
        Correlation matrix of shape (d, d) with unit diagonal.
    """
    n, d = data.shape
    log_x = np.log(data)

    # Variance of the log ratio, using the column-centered log abundances
    clr_col = log_x - log_x.mean(axis=0, keepdims=True)
    diff = clr_col[:, :, None] - clr_col[:, None, :]
    vlr = (diff ** 2).mean(axis=0)
    vlr = 0.5 * (vlr + vlr.T)
    np.fill_diagonal(vlr, 0.0)

    # Per-feature variance of the row-centered log abundances
    clr_row = log_x - log_x.mean(axis=1, keepdims=True)
    clr_var = clr_row.var(axis=0, ddof=0)

    sum_log_var = clr_var.sum() * d / (d - 1)
    log_var = (clr_var - sum_log_var / d ** 2) * d / (d - 2)
    log_var = np.maximum(log_var, eps)

    std = np.sqrt(log_var)
    std_prod = np.maximum(np.outer(std, std), eps)

    rho = (vlr - log_var[None, :] - log_var[:, None]) / (-2.0 * std_prod)
    rho = 0.5 * (rho + rho.T)
    rho = np.clip(rho, -1.0, 1.0)
    np.fill_diagonal(rho, 1.0)
    return rho


def test_compute_correlation_matches_numpy_reference():
    """The torch estimator reproduces an independent numpy implementation."""
    data = _lognormal_matrix()
    expected = _reference_correlation(data.numpy())

    out = _compute_correlation(data, threshold=0.0)
    assert out.is_sparse
    obtained = out.to_dense().numpy()

    np.testing.assert_allclose(obtained, expected, rtol=1e-10, atol=1e-12)


def test_compute_correlation_invariants():
    """The estimate is symmetric, bounded, unit-diagonal and thresholded."""
    data = _lognormal_matrix()
    threshold = 0.2
    out = _compute_correlation(data, threshold=threshold)

    assert out.is_sparse
    assert out.layout == torch.sparse_coo
    dense = out.to_dense()

    assert torch.allclose(dense, dense.T, rtol=0.0, atol=1e-12)
    assert torch.all(dense.diagonal() == 1.0)
    assert torch.all(dense.abs() <= 1.0)

    unthresholded = _compute_correlation(data, threshold=0.0).to_dense()
    below = unthresholded.abs() < threshold
    below.fill_diagonal_(False)
    assert torch.all(dense[below] == 0.0)


def test_compute_correlation_rejects_small_d():
    """Fewer than three features raises ValueError."""
    data = _lognormal_matrix(n=20, d=2)
    with pytest.raises(ValueError):
        _compute_correlation(data, threshold=0.0)
