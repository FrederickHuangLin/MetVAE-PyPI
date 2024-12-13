# MetVAE: A Variational Autoencoder for Metabolomics Correlation Analysis

MetVAE is a specialized Python package that implements a Variational Autoencoder (VAE) specifically designed for metabolomics correlation analysis. It addresses key challenges in metabolomics data, especially in untargeted metabolomics, including compositionality, missing values, and the influence of covariates or confounders.

## Key Features

- **Compositionality-Aware**: Automatically handles compositional data through centered log-ratio (CLR) transformation
- **Smart Zero Handling**: Uses censored estimation and multiple imputation for values below detection limit
- **Covariate/Confounder Adjustment**: Removes unwanted variation from experimental and biological covariates or confounders
- **Sparse Correlation Analysis**: Provides statistical and data-driven methods to construct a sparsified correlation matrix for metabolites
- **Latent Space Analysis**: Enables exploration of underlying metabolic patterns and modules

## Installation

```bash
pip install metvae
```

## Quick Start

Here's a simple example to get you started:

```python
from metvae import MetVAE

# Initialize and fit the model
model = MetVAE(
    data=abundance_data,
    meta=meta_data,
    continuous_covariate_keys=['age', 'bmi'],
    categorical_covariate_keys=['sex', 'batch'],
    latent_dim=10
)

# Train the model
model.train(max_epochs=1000)

# Analyze correlations
correlations = model.get_corr(num_sim=1000)

# Create sparse network
sparse_network = model.sparse_by_thresholding(
    th_len=30,
    n_cv=5,
    soft=False
)
```

## Detailed Usage

### Data Preprocessing

The package automatically handles several preprocessing steps:

1. **CLR Transformation**: Addresses the compositional nature of metabolomics data
2. **Zero Value Processing**: Uses sophisticated methods to handle measurements below detection limit
3. **Covariate/Confounder Adjustment**: Removes unwanted variation from known confounding factors

### Correlation Analysis

MetVAE provides multiple approaches for analyzing metabolite relationships:

```python
# Compute correlations with multiple imputation
model.get_corr(num_sim=1000)

# Create sparse network using statistical filtering
sparse_stats = model.sparse_by_filter(
    p_adj_method='fdr_bh',
    cutoff=0.05
)

# Create sparse network using threshold optimization
sparse_thresh = model.sparse_by_thresholding(
    th_len=30,
    n_cv=5
)
```

### Latent Space Analysis

Explore the patterns learned by the model:

```python
# Get feature loadings in CLR space
loadings = model.clr_loading()

# Compute metabolite co-occurrence patterns
cooccurrence = model.cooccurrence()
```

## Advanced Features

### Customizing Network Sparsification

The package offers two main approaches to network sparsification:

1. **Statistical Filtering**: Uses Fisher's z-test with multiple testing correction to control false discoveries
   
   ```python
   sparse_stats = model.sparse_by_filter(
       p_adj_method='fdr_bh',  # Benjamini-Hochberg FDR control
       cutoff=0.05
   )
   ```
   
2. **Threshold Optimization**: Uses cross-validation to find optimal sparsity levels
   ```python
   sparse_thresh = model.sparse_by_thresholding(
       th_len=30,  # Number of thresholds to test
       n_cv=5,     # Number of cross-validation folds
       soft=False  # Use hard thresholding
   )
   ```

### Handling Covariates

MetVAE can adjust for both continuous and categorical covariates:

```python
model = MetVAE(
    data=metabolite_data,
    meta=metadata,
    continuous_covariate_keys=['age', 'bmi'],
    categorical_covariate_keys=['sex', 'batch', 'treatment'],
    latent_dim=10
)
```

### Command Line Usage

The MetVAE package provides a command-line interface for easy model training and analysis. Here's an example showing how to train a model with both continuous and categorical covariates:

```bash
metvae-cli \
--data test_data.csv \
--meta test_smd.csv \
--continuous_covariate_keys x1 \
--categorical_covariate_keys x2 \
--latent_dim 100 \
--use_gpu \
--logging \
--batch_size 100 \
--learning_rate 0.01 \
--alpha_grid 0 0.0001
```

For a complete list of available parameters and their descriptions, run:

```bash
metvae-cli --help
```

## Technical Details

- **Data Format**: Expects pandas DataFrames with samples as either rows or columns
- **Missing Values**: Automatically handles zero values through sophisticated imputation
- **GPU Support**: Optional GPU acceleration for model training
- **Parallel Processing**: Supports multi-core processing for computational efficiency

# A Complete Simulation Example

This example demonstrates how to implement MetVAE utilizing a synthetic dataset.

## Setup and Data Generation

```python
import random
import numpy as np
import pandas as pd
import torch
from metvae.model import MetVAE
from metvae.sim import sim_data

# Initialize parameters
n, d, zero_prop, seed = 100, 50, 0.3, 123
# n: number of samples
# d: number of features
# zero_prop: proportion of zeros to introduce
# seed: random seed for reproducibility

# Set up correlation parameters
cor_pairs = int(0.2 * d)     # Number of correlated feature pairs (20% of features)
mu = list(range(10, 15))     # Mean values for data generation
da_prop = 0.1                # Differential abundance proportion

# Create sample metadata with continuous and categorical covariates
np.random.seed(seed)
smd = pd.DataFrame({
    'x1': np.random.randn(n),                           # Continuous covariate
    'x2': np.random.choice(['a', 'b'], size=n, replace=True)  # Categorical covariate
})
smd.index = ["s" + str(i) for i in range(n)]

# Generate correlated data using sim_data function
sim = sim_data(n=n, d=d, cor_pairs=cor_pairs, mu=mu, x=smd, 
               cont_list=['x1'], cat_list=['x2'], da_prop=da_prop)
y = sim['y']           # Original absolute abundance data
x = sim['x']           # Covariates
true_cor = sim['cor_matrix']  # True correlation matrix
beta = sim['beta']     # True regression coefficients

# Apply transformations and introduce biases
log_y = np.log(y)
log_sample_bias = np.log(np.random.uniform(1e-3, 1e-1, size=n))    # Sample-specific bias
log_feature_bias = np.log(np.random.uniform(1e-1, 1, size=d))      # Feature-specific bias
log_data = log_y + log_sample_bias[:, np.newaxis]  
log_data = log_data + log_feature_bias.reshape(1, d)  
data = np.exp(log_data)

# Introduce missing values (zeros) based on quantile thresholds
thresholds = np.quantile(data, zero_prop, axis=0)
data_miss = np.where(data < thresholds, 0, data)
data_miss = pd.DataFrame(data_miss, index=y.index, columns=y.columns)
```

## Model Training and Evaluation

```python
# Train MetVAE model
torch.manual_seed(123)
np.random.seed(123)
max_epochs = 1000
learning_rate = 1e-2

# Initialize model with metadata
model = MetVAE(data=data_miss,
               features_as_rows=False,
               meta=smd,
               continuous_covariate_keys=['x1'],
               categorical_covariate_keys=['x2'],
               latent_dim=min(n, d))

# Train the model
model.train(batch_size=100,
            num_workers=0,
            max_epochs=max_epochs,
            learning_rate=learning_rate,
            log_every_n_steps=1)

# Obtain the sparse correlation matrix by p-value filtering
model.get_corr(num_sim=1000)
results_metvae = model.sparse_by_filter(p_adj_method='fdr_bh', cutoff=0.05)
est_cor = results_metvae['sparse_estimate']

# Calculate performance metrics
true_idx = true_cor[np.tril_indices_from(true_cor, k=-1)] != 0
est_idx = est_cor[np.tril_indices_from(est_cor, k=-1)] != 0
tpr = np.sum(est_idx & true_idx) / np.sum(true_idx)      # True Positive Rate
fpr = np.sum(est_idx & ~true_idx) / np.sum(~true_idx)    # False Positive Rate
fdr = np.sum(est_idx & ~true_idx) / np.sum(est_idx)      # False Discovery Rate

# Obtain the sparse correlation matrix by thresholding
random.seed(123)
results_metvae = model.sparse_by_thresholding(th_len=100, n_cv=5, soft=False, n_jobs=1)
est_cor = results_metvae['sparse_estimate']

# Calculate performance metrics for thresholding approach
true_idx = true_cor[np.tril_indices_from(true_cor, k=-1)] != 0
est_idx = est_cor[np.tril_indices_from(est_cor, k=-1)] != 0
tpr = np.sum(est_idx & true_idx) / np.sum(true_idx)
fpr = np.sum(est_idx & ~true_idx) / np.sum(~true_idx)
fdr = np.sum(est_idx & ~true_idx) / np.sum(est_idx)
```

## Results

```python
# P-value filtering results
tpr =  0.9  
fpr =  0.0 
fdr =  0.0  

# Thresholding results
tpr =  0.9
fpr =  0.0
fdr =  0.0
```

## Citation

If you use MetVAE in your research, please cite:

[Paper in Preparation]

## Contributing

We welcome contributions! Please feel free to submit a Pull Request.

## License

This package is licensed under the MIT License - see the LICENSE file for details.

## Contact

Huang Lin

Email: hlin1239@umd.edu

GitHub: https://github.com/FrederickHuangLin/MetVAE-PyPI/issues