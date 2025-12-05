import pytest
import random
import numpy as np
import torch
from metvae.model import MetVAE

def test_vae(sample_metabolomics_data):
    """
    Test the MetVAE model using simulated metabolomics data.
    """
    simulation = sample_metabolomics_data
    
    n = simulation['n_samples']
    d = simulation['d_features']
    data = simulation['abundance_data']
    meta = simulation['meta_data']
    true_cor = simulation['true_cor']

    max_epochs=1000
    learning_rate=1e-2

    model = MetVAE(data=data,
                   features_as_rows=False,
                   meta=meta,
                   continuous_covariate_keys=['x1'],
                   categorical_covariate_keys=['x2'],
                   latent_dim=min(n, d))
    
    model.train(batch_size=100,
                num_workers=0,
                max_epochs=max_epochs,
                learning_rate=learning_rate,
                log_every_n_steps=1)
    
    model.get_corr(num_sim=1000)
    
    # Filtering
    results_metvae = model.sparse_by_p(p_adj_method='fdr_bh', cutoff=0.05)
    est_cor = results_metvae['sparse_estimate'].values

    # Calculate summary statistics
    true_idx = true_cor[np.tril_indices_from(true_cor, k=-1)] != 0
    est_idx = est_cor[np.tril_indices_from(est_cor, k=-1)] != 0
    tpr = np.sum(est_idx & true_idx) / np.sum(true_idx)
    fpr = np.sum(est_idx & ~true_idx) / np.sum(~true_idx)
    fdr = np.sum(est_idx & ~true_idx) / np.sum(est_idx)
    
    assert round(tpr, 4) == np.float64(0.9000)
    assert round(fpr, 4) == np.float64(0.0008)
    assert round(fdr, 4) == np.float64(0.1000)
    
    
    
    