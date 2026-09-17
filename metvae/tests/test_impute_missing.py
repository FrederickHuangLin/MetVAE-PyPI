"""Tests for the ordinary least squares initializer and the censored normal fit."""

import numpy as np
import torch

from metvae.impute_missing import _fit_censored_normal, _ols_estimate


def _design(meta_np):
    """Return the design matrix with an intercept column prepended."""
    return np.column_stack([np.ones(meta_np.shape[0]), meta_np])


def _regression_data(n=50, d=6, seed=5):
    """Return a full-rank response matrix, a single covariate and the design matrix."""
    rng = np.random.default_rng(seed)
    meta_np = rng.standard_normal((n, 1))
    X = _design(meta_np)
    beta_true = rng.standard_normal((2, d))
    Y_np = X @ beta_true + 0.5 * rng.standard_normal((n, d))
    return Y_np, meta_np, X


def test_ols_estimate_matches_lstsq():
    """Coefficients match numpy least squares and the last row is the log standard deviation."""
    Y_np, meta_np, X = _regression_data()
    n, d = Y_np.shape
    p = X.shape[1]

    estimates = _ols_estimate(
        torch.tensor(Y_np, dtype=torch.float64),
        torch.tensor(meta_np, dtype=torch.float64),
    ).numpy()

    beta_ref, _, _, _ = np.linalg.lstsq(X, Y_np, rcond=None)
    np.testing.assert_allclose(estimates[:p, :], beta_ref, rtol=1e-8, atol=1e-10)

    resid = Y_np - X @ beta_ref
    sse = (resid ** 2).sum(axis=0)
    log_sigma_ref = 0.5 * np.log(sse / (n - p))
    np.testing.assert_allclose(estimates[p, :], log_sigma_ref, rtol=1e-8, atol=1e-10)


def test_ols_estimate_handles_nan_per_column():
    """Each column is fitted on its own observed rows."""
    Y_np, meta_np, X = _regression_data()
    n, d = Y_np.shape
    p = X.shape[1]

    rng = np.random.default_rng(99)
    Y_nan = Y_np.copy()
    masks = []
    for j in range(d):
        drop = rng.choice(n, size=5 + j, replace=False)
        Y_nan[drop, j] = np.nan
        masks.append(~np.isnan(Y_nan[:, j]))

    estimates = _ols_estimate(
        torch.tensor(Y_nan, dtype=torch.float64),
        torch.tensor(meta_np, dtype=torch.float64),
    ).numpy()

    for j in range(d):
        obs = masks[j]
        beta_ref, _, _, _ = np.linalg.lstsq(X[obs], Y_nan[obs, j], rcond=None)
        np.testing.assert_allclose(estimates[:p, j], beta_ref, rtol=1e-6, atol=1e-8)


def test_ols_estimate_zeroes_bad_columns():
    """A column with fewer observed rows than parameters is returned as zeros."""
    Y_np, meta_np, X = _regression_data()
    p = X.shape[1]

    Y_nan = Y_np.copy()
    Y_nan[1:, 2] = np.nan

    estimates = _ols_estimate(
        torch.tensor(Y_nan, dtype=torch.float64),
        torch.tensor(meta_np, dtype=torch.float64),
    ).numpy()

    assert np.all(estimates[:, 2] == 0.0)
    assert not np.all(estimates[:, 0] == 0.0)


def test_fit_censored_normal_recovers_parameters():
    """The Tobit fit recovers the location and scale of a left-censored normal sample."""
    n, d = 300, 5
    sigma_true = 1.0
    mu_true = np.array([0.0, 1.0, 2.0, 3.0, 4.0])

    rng = np.random.default_rng(31)
    y = mu_true[None, :] + sigma_true * rng.standard_normal((n, d))

    th_vec = np.quantile(y, 0.3, axis=0)
    censored = y <= th_vec[None, :]
    Y_np = np.where(censored, np.nan, y)
    assert abs(censored.mean() - 0.3) < 0.01

    Y = torch.tensor(Y_np, dtype=torch.float64)
    th = torch.tensor(np.broadcast_to(th_vec[None, :], (n, d)).copy(), dtype=torch.float64)

    init = _ols_estimate(Y, None)
    params = _fit_censored_normal(Y=Y, meta=None, th=th, init_estimates=init).numpy()

    mu_hat = params[0, :]
    sigma_hat = np.exp(params[1, :])

    np.testing.assert_allclose(mu_hat, mu_true, atol=0.15)
    assert np.all(np.abs(sigma_hat / sigma_true - 1.0) <= 0.2)
