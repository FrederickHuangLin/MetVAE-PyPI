"""Tests for the MetVAE estimator and its zero-initialization helper."""

import numpy as np
import pytest
import torch

from metvae.model import MetVAE, _random_initial


def _fit(simulation, **train_kwargs):
    """Build a MetVAE on a simulated dataset and train it."""
    model = MetVAE(
        data=simulation['abundance_data'],
        features_as_rows=False,
        meta=simulation['meta_data'],
        continuous_covariate_keys=['x1'],
        categorical_covariate_keys=['x2'],
        latent_dim=train_kwargs.pop('latent_dim', None),
        seed=train_kwargs.pop('seed', 0),
    )
    model.train(**train_kwargs)
    return model


def _tpr_fdr(true_cor, est_cor):
    """Return the true positive rate and false discovery rate of the edge set."""
    true_idx = true_cor[np.tril_indices_from(true_cor, k=-1)] != 0
    est_idx = est_cor[np.tril_indices_from(est_cor, k=-1)] != 0
    tpr = np.sum(est_idx & true_idx) / np.sum(true_idx)
    n_sel = np.sum(est_idx)
    fdr = np.sum(est_idx & ~true_idx) / n_sel if n_sel > 0 else 0.0
    return float(tpr), float(fdr)


def test_end_to_end_recovers_structure(small_simulation):
    """The full pipeline recovers a majority of the true edges and stays well formed."""
    model = _fit(small_simulation, batch_size=32, max_epochs=60, learning_rate=1e-2)
    model.get_corr(num_sim=20, workers=1, seed=0)
    results = model.sparse_by_p()

    est = results['estimate']
    sparse_est = results['sparse_estimate']

    tpr, fdr = _tpr_fdr(small_simulation['true_cor'], sparse_est.values)
    assert tpr >= 0.5
    assert fdr <= 0.5

    assert list(est.index) == list(model.feature_name)
    assert list(est.columns) == list(model.feature_name)
    assert list(sparse_est.index) == list(model.feature_name)
    assert list(sparse_est.columns) == list(model.feature_name)

    est_values = est.values
    sparse_values = sparse_est.values
    assert np.allclose(est_values, est_values.T)
    assert np.allclose(sparse_values, sparse_values.T)
    assert np.allclose(np.diag(est_values), 1.0)

    support_est = est_values != 0
    support_sparse = sparse_values != 0
    assert np.all(support_sparse <= support_est)


@pytest.mark.slow
def test_end_to_end_full_size(large_simulation):
    """The full-size configuration reaches the reference detection rates."""
    model = _fit(
        large_simulation,
        latent_dim=min(large_simulation['n_samples'], large_simulation['d_features']),
        batch_size=100,
        num_workers=0,
        max_epochs=1000,
        learning_rate=1e-2,
    )
    model.get_corr(num_sim=1000)
    results = model.sparse_by_p(p_adj_method='fdr_bh', cutoff=0.05)

    tpr, fdr = _tpr_fdr(large_simulation['true_cor'], results['sparse_estimate'].values)
    assert tpr >= 0.8
    assert fdr <= 0.25


def test_train_reports_unknown_kwargs(small_simulation):
    """Unrecognized keyword arguments to train raise a UserWarning."""
    model = MetVAE(
        data=small_simulation['abundance_data'],
        features_as_rows=False,
        latent_dim=8,
    )
    with pytest.warns(UserWarning):
        model.train(max_epochs=1, log_every_n_steps=1)


def test_train_sets_optimizer_attributes(small_simulation):
    """train exposes the optimizer and the epoch count, and does not accumulate losses."""
    model = MetVAE(
        data=small_simulation['abundance_data'],
        features_as_rows=False,
        latent_dim=8,
    )
    model.train(max_epochs=2)

    assert isinstance(model.optimizer, torch.optim.Optimizer)
    assert model.current_epoch == 2
    assert len(model.train_loss) == 2

    model.train(max_epochs=3)
    assert len(model.train_loss) == 3
    assert model.current_epoch == 3


def test_two_models_in_one_process_are_identical(small_simulation):
    """Two sequential fits in one process give bitwise identical parameters and estimates."""
    def run():
        model = MetVAE(
            data=small_simulation['abundance_data'],
            features_as_rows=False,
            meta=small_simulation['meta_data'],
            continuous_covariate_keys=['x1'],
            categorical_covariate_keys=['x2'],
            latent_dim=8,
            seed=0,
        )
        model.train(batch_size=32, max_epochs=30)
        threads_before = torch.get_num_threads()
        out = model.get_corr(num_sim=5, workers=1, seed=0)
        threads_after = torch.get_num_threads()
        return model, out['estimate'].to_dense(), threads_before, threads_after

    model_a, est_a, before_a, after_a = run()
    model_b, est_b, before_b, after_b = run()

    assert before_a == after_a
    assert before_b == after_b

    state_a = model_a.model.state_dict()
    state_b = model_b.model.state_dict()
    assert set(state_a.keys()) == set(state_b.keys())
    for key in state_a:
        assert torch.equal(state_a[key], state_b[key]), f"parameter {key} differs between runs"

    assert torch.equal(est_a, est_b)


def test_get_corr_independent_of_workers(small_simulation):
    """Sequential and parallel imputation give the same correlation estimate."""
    model = MetVAE(
        data=small_simulation['abundance_data'],
        features_as_rows=False,
        latent_dim=8,
        seed=0,
    )
    model.train(batch_size=32, max_epochs=20)

    est_serial = model.get_corr(num_sim=5, workers=1, seed=0)['estimate'].to_dense()
    est_parallel = model.get_corr(num_sim=5, workers=-1, seed=0)['estimate'].to_dense()

    assert torch.allclose(est_serial, est_parallel, rtol=0.0, atol=1e-12)
    assert torch.equal(est_serial != 0, est_parallel != 0)


def _random_initial_inputs(seed=7, n=25, d=6, n_zero=5):
    """Build the arguments of _random_initial together with a NaN mask."""
    torch.manual_seed(seed)
    y = torch.randn(n, d, dtype=torch.float64)
    nan_mask = torch.zeros(n, d, dtype=torch.bool)
    nan_mask[:n_zero, :] = True
    y[nan_mask] = float('nan')

    num_zero = torch.full((d,), float(n_zero))
    mean = torch.linspace(-1.0, 1.0, d, dtype=torch.float64)
    sd = torch.full((d,), 0.5, dtype=torch.float64)
    return y, nan_mask, num_zero, mean, sd


def test_random_initial_fills_only_nan():
    """Observed entries are untouched and every censored entry is filled."""
    y, nan_mask, num_zero, mean, sd = _random_initial_inputs()
    gen = torch.Generator().manual_seed(11)
    out = _random_initial(y=y, sample_size=y.shape[0], num_zero=num_zero,
                          mean=mean, sd=sd, generator=gen)

    assert out.shape == y.shape
    assert torch.equal(out[~nan_mask], y[~nan_mask])
    assert not torch.isnan(out).any()


def test_random_initial_values_from_lowest_k_pool():
    """Filled values come from the k smallest draws of the reference normal sample."""
    y, nan_mask, num_zero, mean, sd = _random_initial_inputs()
    n, d = y.shape

    gen = torch.Generator().manual_seed(11)
    out = _random_initial(y=y, sample_size=n, num_zero=num_zero,
                          mean=mean, sd=sd, generator=gen)

    ref_gen = torch.Generator().manual_seed(11)
    rand = torch.randn(n, d, dtype=torch.float64, generator=ref_gen)
    random_data = rand * sd + mean
    vals_sorted, _ = torch.sort(random_data, dim=0)

    k = torch.clamp(num_zero.to(torch.long), min=1, max=n)
    for j in range(d):
        kth = vals_sorted[k[j].item() - 1, j]
        filled = out[nan_mask[:, j], j]
        assert torch.all(filled <= kth + 1e-12)


def test_random_initial_generator_reproducible():
    """The same generator seed gives identical output and different seeds differ."""
    y, _, num_zero, mean, sd = _random_initial_inputs()
    n = y.shape[0]

    out1 = _random_initial(y=y, sample_size=n, num_zero=num_zero, mean=mean, sd=sd,
                           generator=torch.Generator().manual_seed(3))
    out2 = _random_initial(y=y, sample_size=n, num_zero=num_zero, mean=mean, sd=sd,
                           generator=torch.Generator().manual_seed(3))
    out3 = _random_initial(y=y, sample_size=n, num_zero=num_zero, mean=mean, sd=sd,
                           generator=torch.Generator().manual_seed(4))

    assert torch.equal(out1, out2)
    assert not torch.equal(out1, out3)


def test_random_initial_no_nan_returns_input():
    """An input without censored entries is returned unchanged."""
    torch.manual_seed(1)
    y = torch.randn(12, 4, dtype=torch.float64)
    num_zero = torch.zeros(4)
    mean = torch.zeros(4, dtype=torch.float64)
    sd = torch.ones(4, dtype=torch.float64)

    out = _random_initial(y=y, sample_size=12, num_zero=num_zero, mean=mean, sd=sd,
                          generator=torch.Generator().manual_seed(5))
    assert torch.equal(out, y)
