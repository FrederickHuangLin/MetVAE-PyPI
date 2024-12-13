import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from metvae.sim import sim_data 

@pytest.fixture
def test_files():
    """Fixture to provide paths to test files and ensure they exist."""
    test_dir = Path(__file__).parent.absolute()
    data_path = test_dir / 'test_data.csv'
    meta_path = test_dir / 'test_smd.csv'
    
    # Verify files exist
    assert data_path.exists(), f"Test data file not found at {data_path}"
    assert meta_path.exists(), f"Test metadata file not found at {meta_path}"
    
    return data_path, meta_path

@pytest.fixture
def sample_metabolomics_data():
    """Create a small, synthetic metabolomics dataset for testing."""
    n = 100
    d = 50
    cor_pairs = int(d * 0.2)
    mu = list(range(10, 15))
    da_prop = 0.1
    zero_prop = 0.3
    
    np.random.seed(123)
    
    # Simulate metadata
    smd = pd.DataFrame({'x1': np.random.randn(n), 
                        'x2': np.random.choice(['a', 'b'], 
                                               size=n, 
                                               replace=True)})
    smd.index = ["s" + str(i) for i in range(n)]
    
    # Simulate absolute abundance
    sim = sim_data(n=n, 
                   d=d, 
                   cor_pairs=cor_pairs, 
                   mu=mu, 
                   x=smd, 
                   cont_list=['x1'], 
                   cat_list=['x2'], 
                   da_prop=da_prop)
    y = sim['y']
    true_cor = sim['cor_matrix']
    
    # Apply log transformation and add biases
    log_y = np.log(y)
    log_sample_bias = np.log(np.random.uniform(1e-3, 1e-1, size=n))
    log_feature_bias = np.log(np.random.uniform(1e-1, 1, size=d))
    log_data = log_y + log_sample_bias[:, np.newaxis]  # Adding sample bias
    log_data = log_data + log_feature_bias.reshape(1, d)  # Adding feature bias
    data = np.exp(log_data)
    
    # Calculate thresholds and apply zeros
    thresholds = np.quantile(data, zero_prop, axis=0)
    data_miss = np.where(data < thresholds, 0, data)
    data_miss = pd.DataFrame(
        data_miss,
        index=y.index,
        columns=y.columns
    )
    
    return {
        'n_samples': n,
        'd_features': d,
        'abundance_data': data_miss,
        'meta_data': smd,
        'true_cor': true_cor
        }
