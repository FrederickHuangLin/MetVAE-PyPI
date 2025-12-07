# MetVAE: A Variational Autoencoder for Metabolomics Correlation Analysis

MetVAE is a variational autoencoder (VAE) framework for untargeted metabolomics data with:

- Compositional (CLR) preprocessing with zero/censoring handling
- Covariate/confounder adjustment
- Multiple-imputation–based correlation estimation
- Two sparsification strategies:
  - `sparse_by_p`: p-value based sparsification (Fisher’s z + multiple testing correction)
  - `sparse_by_sec`: Sparse Estimation of Correlation (SEC) with optional CV-based `rho` selection

The package can be used either as a Python API or via the command-line interface (`metvae-cli`).

------

## Installation

```bash
pip install metvae
```

(If you’re working from source, you can install with:)

```bash
pip install -e .
```

------

## Basic Python Usage

### 1. Prepare data

```python
import pandas as pd
from metvae.model import MetVAE

# data: samples x metabolites
data = pd.read_csv("data.csv", index_col=0)

# meta: optional sample metadata (rows = samples)
meta = pd.read_csv("meta.csv", index_col=0)
```

### 2. Initialize the model

```python
model = MetVAE(
    data=data,
    features_as_rows=False,          # set True if features are rows
    meta=meta,
    continuous_covariate_keys=["age", "bmi"],
    categorical_covariate_keys=["sex", "batch"],
    latent_dim=10,
    use_gpu=True,                    # use CUDA if available
    seed=0,
    feature_zero_threshold=0.3,      # drop features with zero proportion > 0.3
    sample_zero_threshold=None       # optionally drop samples with many zeros
)
```

`feature_zero_threshold` (float or `None`):

- If set (e.g. `0.3`), features with a proportion of zeros **> threshold** are removed during preprocessing.
- If `None`, no feature filtering by zero proportion is applied (all-zero features are still removed later).

`sample_zero_threshold` (float or `None`):

- If set (e.g. `0.8`), samples with a proportion of zeros **> threshold** are removed during preprocessing.
- If `None`, no sample filtering by zero proportion is applied (all-zero samples are still removed later).

### 3. Train

```python
model.train(
    batch_size=128,
    num_workers=0,
    max_epochs=1000,
    learning_rate=1e-3,
    log_every_n_steps=1
)
```

------

## Correlation Estimation

All sparsification methods rely on a **correlation estimate** produced via repeated imputations of censored zeros.

### `get_corr`

```python
corr_outputs = model.get_corr(
    num_sim=100,
    workers=-1,          # CPU workers (ignored on GPU)
    batch_size=100,      # batch size for GPU imputation
    threshold=0.2,       # |r| threshold inside _compute_correlation
    seed=0               # base seed; uses seed + sim_id internally
)
```

This computes:

- Multiple imputations of the CLR/log data using the trained VAE
- Back-transforms to the original scale
- Computes a **sparse** correlation matrix using `_compute_correlation` + a hard correlation threshold

The result is stored in:

```python
model.corr_outputs  # dict with:
# 'impute_log_data' : mean imputed log-data (dense tensor)
# 'estimate'        : sparse correlation matrix (COO tensor)
```

------

## Sparsification Method 1: `sparse_by_p` (p-value based)

`MetVAE.sparse_by_p` takes the correlation estimate in `self.corr_outputs['estimate']` and performs Fisher’s z-test with multiple testing correction, then zeroes out non-significant edges.

### API

```python
model.sparse_by_p(
    p_adj_method='fdr_bh',
    cutoff=0.05
)
```

### Parameters

- **`p_adj_method`**: multiple testing correction method passed to `_matrix_p_adjust`, options include
   `'bonferroni', 'sidak', 'holm-sidak', 'holm', 'simes-hochberg', 'hommel',  'fdr_bh', 'fdr_by', 'fdr_tsbh', 'fdr_tsbky'` (default `'fdr_bh'`).
- **`cutoff`**: threshold on *adjusted* p-values; correlations with `q_value > cutoff` are set to zero.

### Returns

```python
results_p = model.sparse_by_p()
```

`results_p` is a dictionary with:

- `'estimate'` – **dense** correlation matrix (DataFrame)
- `'p_value'` – unadjusted p-values (DataFrame)
- `'q_value'` – adjusted p-values (DataFrame)
- `'sparse_estimate'` – sparsified correlation matrix (DataFrame) with non-significant entries zeroed

Example:

```python
results_p = model.sparse_by_p(p_adj_method="fdr_bh", cutoff=0.05)
sparse_corr_p = results_p["sparse_estimate"].values
```

> ⚠️ `sparse_by_p` assumes `model.get_corr()` has already been called; otherwise it raises a `ValueError`.

------

## Sparsification Method 2: `sparse_by_sec` (SEC)

`sparse_by_sec` uses the **Sparse Estimation of Correlation (SEC)** algorithm to obtain a sparse correlation matrix. It can either:

1. **Fit once** with a fixed `rho` if you pass `rho`, or
2. **Select `rho` automatically** via K-fold cross-validation when `rho=None`.

The Python implementation in this package is adapted from the original [MATLAB reference code](https://warwick.ac.uk/fac/sci/statistics/staff/academic-research/leng/publications/sec.m) by Leng's lab.

### API

```python
results_sec = model.sparse_by_sec(
    rho=None,
    # SEC solver hyperparameters
    epsilon=1e-5,
    tol=1e-3,
    max_iter=1000,
    restart=50,
    line_search_apg=True,
    delta=None,
    n_samples=None,
    c_delta=0.1,
    threshold=0.1,
    # CV settings (used only when rho is None)
    c_grid=tuple(float(x) for x in range(1, 11)),  # 1.0, 2.0, ..., 10.0
    n_splits=5,
    seed=0,
    workers=-1,          # CPU: parallel across rho; GPU / workers<=1: sequential
    refine=True,         # single zoom after coarse pass
    refine_points=10
)
```

### Behavior

- If **`rho` is provided** (`rho=2.2`, say):

  - `sparse_by_sec` calls `_SEC` once on `model.corr_outputs['estimate']` and returns that fit.
  - No cross-validation is performed.
  - `scores_by_rho` is set to `None`.

- If **`rho` is `None`**:

  - Candidate penalties are given by 
    $$
    \rho = c \cdot \sqrt{\log(p)/n}
    $$
    


     for `c` in `c_grid`, where `p` = number of features and `n` = number of samples.

  - A K-fold (default 5-fold) CV loop:

    - For each `rho`, fits SEC on the training subset correlation
    - Computes the mean squared Frobenius error between the SEC estimate and the empirical correlation on validation subsets

  - After the coarse pass:

    - If `refine=True`, it **refines once** in a bracket between the best coarse `c` and its immediate left/right neighbors using `refine_points` equally spaced `c` values.

  - The final `best_rho` is the one with the smallest mean validation error (tie-broken by smaller `rho`).

### Outputs

```python
results_sec = model.sparse_by_sec(rho=None)
```

`results_sec` is a dictionary with:

- `'estimate'` – **dense** empirical correlation matrix before SEC (DataFrame)

- `'sparse_estimate'` – final **dense** SEC estimate after thresholding (DataFrame)

- `'best_rho'` – the selected `rho` (or your supplied `rho` if you passed one)

- `'scores_by_rho'` –

  - If you passed `rho` explicitly → `None`

  - If `rho=None` → a **pandas DataFrame** with one row per evaluated candidate:

    | c    | rho                | score                         |
    | ---- | ------------------ | ----------------------------- |
    | …    | c * sqrt(log(p)/n) | mean validation Frobenius err |

Values are sorted by `rho` (stable sort), which makes plotting easy:

```python
scores = results_sec["scores_by_rho"]
scores.plot(x="rho", y="score", marker="o")
```

Example usage with fixed `rho`:

```python
# After model.get_corr(...)
results_sec = model.sparse_by_sec(rho=2.2)
sparse_corr_sec = results_sec["sparse_estimate"].values
```

Example with automatic `rho` selection:

```python
results_sec = model.sparse_by_sec(
    rho=None,
    c_grid=tuple(float(x) for x in range(1, 11)),
    n_splits=5,
    seed=0,
    workers=-1,
    refine=True,
    refine_points=10
)

best_rho = results_sec["best_rho"]
scores_df = results_sec["scores_by_rho"]
sparse_corr_sec = results_sec["sparse_estimate"].values
```

------

## GraphML Export of Correlation Networks

After you obtain a sparsified correlation matrix (either from `sparse_by_p` or `sparse_by_sec`), you can export it as one or more **GraphML** files for downstream network analysis (e.g. in Cytoscape, Gephi, igraph).

### Python API

```
from metvae.model import MetVAE

# After training and correlation estimation
results_p = model.sparse_by_p(p_adj_method="fdr_bh", cutoff=0.01)
sparse_df = results_p["sparse_estimate"]

# Export one GraphML per cutoff
G = model.export_graphml(
    sparse_df=sparse_df,
    cutoffs=[0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3],
    output_dir="./results/graphs",
    file_prefix="correlation_graph_cutoff"
)
```

This will create files like:

- `correlation_graph_cutoff0.9.graphml`
- `correlation_graph_cutoff0.8.graphml`
- …
- `correlation_graph_cutoff0.3.graphml`

Each file contains an undirected graph where:

- Nodes = metabolites (feature names)
- Edges = pairs with `|correlation| >= cutoff`
- Edge attributes include `weight`, `correlation`, `EdgeScore`, and `EdgeType` (e.g. `"Correlation_cutoff0.7"`)

> Note: GraphML export requires `networkx` (`pip install networkx`).

## Command Line Interface (`metvae-cli`)

The CLI mirrors the Python API:

```bash
metvae-cli \
  --data data_miss.csv \
  --meta meta.csv \
  --continuous_covariate_keys age bmi \
  --categorical_covariate_keys sex batch \
  --feature_zero_threshold 0.3 \
  --latent_dim 10 \
  --batch_size 128 \
  --max_epochs 1000 \
  --learning_rate 0.001 \
  --num_sim 100 \
  --corr_threshold 0.2 \
  --sparse_method pval \
  --p_adj_method fdr_bh \
  --cutoff 0.05 \
  --export_graphml \
  --graphml_cutoffs 0.9 0.8 0.7 0.6 0.5 0.4 0.3 \
  --graphml_prefix hcc_correlation_graph_cutoff
```

To use SEC instead of p-value filtering:

```bash
metvae-cli \
  --data data_miss.csv \
  --meta meta.csv \
  --continuous_covariate_keys age bmi \
  --categorical_covariate_keys sex batch \
  --feature_zero_threshold 0.3 \
  --latent_dim 10 \
  --batch_size 128 \
  --max_epochs 1000 \
  --learning_rate 0.001 \
  --num_sim 100 \
  --corr_threshold 0.2 \
  --sparse_method sec \
  --rho 2.2 \
  --export_graphml \
  --graphml_cutoffs 0.9 0.8 0.7 0.6 0.5 0.4 0.3 \
  --graphml_prefix hcc_correlation_graph_cutoff
```

If you want automatic `rho` selection, use the CLI flags that correspond to the SEC hyperparameters (e.g., `--sec-epsilon`, `--sec-tol`, `--sec-c-grid`, `--sec-n-splits`, etc.), which map directly onto the `sparse_by_sec` parameters described above.

------

## Zero Imputation / Reconstruction

If you just want to impute censored zeros using the trained VAE:

```python
imputed_clr = model.impute_zeros()
```

This:

1. Initializes censored values by sampling from a censored Gaussian (per feature),
2. Refines them using the VAE decoder,
3. Returns a fully-imputed CLR matrix.

------

## Coefficients and Confounding Effects

```python
coef_df = model.confound_coef()  # covariate x metabolite effects (if meta was provided)
es_df   = model.confound_es()    # sample x metabolite confounding effects
```

------

## Latent Loadings and Co-occurrence

```python
loading_df = model.clr_loading()   # feature loadings on latent dims
cooccur_df = model.cooccurrence()  # co-occurrence matrix between features
```

