"""Tests for the SEC sparse correlation solver and its cross-validation wrapper."""

import numpy as np
import pytest
import torch

from metvae.compute_corr import _compute_correlation
from metvae.sparse import _SEC, _SEC_cv


def _sec_inputs(n=60, d=20, seed=23):
    """Return a lognormal abundance matrix and its dense correlation estimate."""
    rng = np.random.default_rng(seed)
    X = torch.tensor(np.exp(rng.standard_normal((n, d))), dtype=torch.float64)
    Rn = _compute_correlation(X, threshold=0.0).coalesce().to_dense()
    return X, Rn


def test_SEC_output_invariants():
    """The SEC estimate is a sparse, symmetric, unit-diagonal PSD matrix."""
    _, Rn = _sec_inputs()
    out = _SEC(Rn, rho=0.2, max_iter=100)

    assert out.is_sparse
    assert out.layout == torch.sparse_coo

    dense = out.to_dense()
    assert torch.allclose(dense, dense.T, atol=1e-10)
    assert torch.allclose(dense.diagonal(), torch.ones_like(dense.diagonal()), atol=1e-5)

    eig_min = torch.linalg.eigvalsh(0.5 * (dense + dense.T)).min().item()
    assert eig_min >= -1e-8


def test_SEC_dense_and_sparse_agree():
    """The dense solver and the sparse wrapper return the same values."""
    from metvae.sparse import _SEC_dense

    _, Rn = _sec_inputs()
    kwargs = dict(max_iter=100, tol=1e-3)
    sparse_out = _SEC(Rn, rho=0.2, **kwargs)
    dense_out = _SEC_dense(Rn, rho=0.2, **kwargs)

    assert torch.equal(sparse_out.to_dense(), dense_out)


def test_SEC_rejects_bad_max_iter():
    """A non-positive iteration budget raises ValueError."""
    _, Rn = _sec_inputs()
    with pytest.raises(ValueError):
        _SEC(Rn, rho=0.2, max_iter=0)


def test_SEC_cv_no_duplicate_rho():
    """The refinement pass never re-evaluates a penalty already scored."""
    X, Rn = _sec_inputs()
    _, scores_df, _ = _SEC_cv(
        X, Rn,
        c_grid=(1.0, 2.0, 3.0),
        n_splits=3,
        seed=0,
        workers=1,
        refine=True,
        refine_points=5,
        max_iter=50,
    )
    rhos = scores_df['rho'].tolist()
    assert len(rhos) == len(set(rhos))


def test_SEC_cv_workers_agree():
    """Cross-validation gives the same selection with one and with several workers."""
    X, Rn = _sec_inputs()
    kwargs = dict(c_grid=(1.0, 2.0, 3.0), n_splits=3, seed=0, refine=False, max_iter=50)

    rho_serial, _, est_serial = _SEC_cv(X, Rn, workers=1, **kwargs)
    rho_parallel, _, est_parallel = _SEC_cv(X, Rn, workers=4, **kwargs)

    assert rho_serial == rho_parallel
    assert torch.equal(est_serial, est_parallel)
