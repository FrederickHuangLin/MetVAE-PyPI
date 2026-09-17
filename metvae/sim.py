from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats


def _scale_continuous_covariates(x_cont):
    """
    Center and scale continuous covariates column-wise.

    Parameters
    ----------
    x_cont : pandas.DataFrame or None
        Continuous covariates.

    Returns
    -------
    pandas.DataFrame or None
        Centered and scaled covariates, or None when the input is None.

    Notes
    -----
    Constant columns are centered and assigned unit scale.
    """
    if x_cont is None:
        return None

    x_cont = x_cont.astype(float).copy()
    means = x_cont.mean(axis=0)
    stds = x_cont.std(axis=0, ddof=0).replace(0, 1.0)
    return (x_cont - means) / stds


def sim_data(n, d, cor_pairs=0, mu=None, sigma=1, x=None, cont_list=None,
             cat_list=None, da_prop=0.1, seed: Optional[int] = None):
    """
    Simulate abundance data with covariate effects and a sparse correlation structure.

    Parameters
    ----------
    n : int
        Number of samples.
    d : int
        Number of features.
    cor_pairs : int, default=0
        Number of correlated feature pairs. The pairs are disjoint and consecutive,
        so the maximum is ``d // 2``.
    mu : array-like or float
        Pool of mean values for log-abundance; feature means are drawn from it with
        replacement. Required.
    sigma : float, default=1
        Standard deviation shared by all features.
    x : pandas.DataFrame, optional
        Covariate matrix.
    cont_list : list of str, optional
        Column names of continuous covariates in ``x``.
    cat_list : list of str, optional
        Column names of categorical covariates in ``x``.
    da_prop : float, default=0.1
        Proportion of uncorrelated features affected by the covariates.
    seed : int, optional
        Seed passed to ``numpy.random.seed`` before any draw. None leaves the
        global random state untouched.

    Returns
    -------
    dict
        Keys ``'cor_matrix'`` (d x d correlation matrix), ``'y'`` (n x d abundance
        DataFrame), ``'x'`` (processed covariate matrix or None) and ``'beta'``
        (covariate effect sizes or None).

    Raises
    ------
    ValueError
        If ``mu`` is None, if ``cor_pairs`` exceeds ``d // 2``, or if ``x`` is given
        with neither ``cont_list`` nor ``cat_list``.

    Examples
    --------
    >>> result = sim_data(n=100, d=50, cor_pairs=10, mu=[2, 3, 4], seed=0)
    """
    if seed is not None:
        np.random.seed(seed)

    sample_name = ["s" + str(i) for i in range(n)]
    feature_name = ["f" + str(i) for i in range(d)]

    if mu is None:
        raise ValueError("mu parameter must be provided")
    mu_vector = np.random.choice(mu, size=d, replace=True)
    sd_vector = np.full(d, sigma)

    cor_matrix = np.eye(d)
    max_cor_pairs = d // 2
    if cor_pairs > max_cor_pairs:
        raise ValueError(f"The maximum number of correlated pairs is: {max_cor_pairs}. "
                         f"Please reduce the number of correlated pairs.")

    if cor_pairs != 0:
        # Disjoint consecutive pairs (0, 1), (2, 3), ...
        idx1 = np.arange(0, 2 * cor_pairs, 2)
        idx2 = np.arange(1, 2 * cor_pairs + 1, 2)

        cor_values = np.random.choice([-0.7, -0.6, -0.5, 0.5, 0.6, 0.7],
                                      size=cor_pairs, replace=True)

        cor_pairs_matrix = np.column_stack((idx1, idx2, cor_values))

        for i in range(cor_pairs):
            row_index = int(cor_pairs_matrix[i, 0])
            col_index = int(cor_pairs_matrix[i, 1])
            corr_value = cor_pairs_matrix[i, 2]
            cor_matrix[row_index, col_index] = corr_value
            cor_matrix[col_index, row_index] = corr_value

    cov_matrix = cor_matrix * np.outer(sd_vector, sd_vector)

    log_y = stats.multivariate_normal.rvs(mean=mu_vector, cov=cov_matrix, size=n)

    if x is not None:
        x_cont = x[cont_list] if cont_list is not None else None
        x_cont = _scale_continuous_covariates(x_cont)

        if cat_list is not None:
            x_cat = pd.get_dummies(x[cat_list], drop_first=True)
            x_cat = x_cat.astype(int)
        else:
            x_cat = None

        if x_cont is not None and x_cat is not None:
            x = pd.concat([x_cont, x_cat], axis=1)
        elif x_cont is not None:
            x = x_cont
        elif x_cat is not None:
            x = x_cat
        else:
            raise ValueError('At least one of `cont_list` and `cat_list` should be not None.')

    es = [0, -2, -1, 1, 2]
    if x is not None:
        p = x.shape[1]

        # Correlated features always carry a covariate effect
        beta1 = np.random.choice(es, size=2 * cor_pairs * p, replace=True, p=[0, 0.25, 0.25, 0.25, 0.25])
        beta1 = beta1.reshape((2 * cor_pairs, p))

        # Uncorrelated features carry an effect with probability da_prop
        beta2_prob = [1 - da_prop] + [da_prop / 4] * 4
        beta2 = np.random.choice(es, size=(d - 2 * cor_pairs) * p, replace=True, p=beta2_prob)
        beta2 = beta2.reshape((d - 2 * cor_pairs, p))

        beta = np.vstack([beta1, beta2])
        log_y = log_y + np.dot(x.values, beta.T)
    else:
        beta = None

    y = np.exp(log_y)
    y = pd.DataFrame(y, index=sample_name, columns=feature_name)

    return {'cor_matrix': cor_matrix, 'y': y, 'x': x, 'beta': beta}
