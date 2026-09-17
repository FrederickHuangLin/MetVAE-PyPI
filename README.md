# MetVAE

MetVAE builds a correlation network from untargeted metabolomics data. The input is a table of metabolite abundances, with one row per sample and one column per metabolite, and an optional table of sample metadata such as batch, age, or diet. Zeros in the abundance table are treated as measurements that fell below the detection limit rather than as true absences, and they are filled in by a variational autoencoder. Covariate effects listed in the metadata are removed before the correlations are estimated. The output is a sparse correlation matrix, in which most metabolite pairs are set to exactly zero and the remaining pairs form the edges of a network. Two methods are available for deciding which pairs to keep: a p-value method with multiple-testing correction, and the SEC penalized estimator, whose tuning parameter is chosen by cross-validation.

MetVAE can be used from Python or from the command line through `metvae-cli`.

## Table of contents

- [Installation](#installation)
- [Preparing your data](#preparing-your-data)
- [A complete example](#a-complete-example)
- [Example notebooks](#example-notebooks)
- [Arguments that matter](#arguments-that-matter)
- [Reading the results](#reading-the-results)
- [Reproducibility](#reproducibility)
- [Run time and memory](#run-time-and-memory)
- [Command line](#command-line)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)
- [License](#license)

## Installation

MetVAE requires Python 3.9 or later. A separate conda environment keeps its dependencies away from your other projects.

```bash
conda create -n metvae python=3.11
conda activate metvae
pip install metvae
```

TensorBoard is optional and is used only when you pass `logging=True`.

```bash
pip install "metvae[logging]"
```

To install from a copy of the source tree, run the following from the directory that contains `setup.py`.

```bash
pip install .
```

Check that the installation worked.

```bash
python -c "import metvae; print(metvae.__version__)"
metvae-cli --help
```

The install downloads PyTorch, which is a large package, so the first install can take several minutes. The default PyTorch wheel runs on the CPU. To use an NVIDIA GPU, install a CUDA build of PyTorch first, then pass `use_gpu=True` to `MetVAE`. Apple GPUs are not used; on a Mac, MetVAE runs on the CPU.

MetVAE installs `numpy`, `pandas`, `scipy`, `statsmodels`, `torch`, `networkx`, and `tqdm`.

## Preparing your data

### File layout

The abundance file has samples as rows and metabolites as columns. The first column holds the sample identifiers.

```csv
sample_id,M1,M2,M3,M4,M5
sample_01,120.5,0,88.1,45.9,17.2
sample_02,98.3,12.7,0,51.4,20.8
sample_03,143.9,9.4,102.6,0,15.5
sample_04,110.2,11.1,95.0,48.7,19.3
```

If your file is transposed, with metabolites as rows, pass `features_as_rows=True` to `MetVAE` instead of rearranging the file.

The metadata file has the same sample identifiers in its first column, in any order, and one column per covariate.

```csv
sample_id,age,batch,diet
sample_01,54,A,control
sample_02,61,A,treatment
sample_03,47,B,control
sample_04,58,B,treatment
```

### Rules for the abundance table

| Rule | Reason |
| --- | --- |
| Write a zero where a metabolite was not detected, and leave it as `0`. | MetVAE treats zeros as values below the detection limit and imputes them. |
| Do not impute, replace, or add a pseudo-count to the zeros yourself. | Any value you substitute is treated as a real measurement. |
| Do not include negative values. | Negative values are converted to zeros and a warning is issued. |
| Do not log-transform, scale, or normalize the abundances. | MetVAE applies the centered log-ratio transform internally. |
| Every sample in the abundance table must appear in the metadata index. | Samples with no metadata row raise an error. |

Empty cells are read by pandas as `NaN` and are counted as zeros.

### Loading the tables

```python
import pandas as pd

data = pd.read_csv("data.csv", index_col=0)
meta = pd.read_csv("meta.csv", index_col=0)
```

Check that the sample identifiers line up.

```python
pd.testing.assert_index_equal(data.index, meta.index)
```

If the check fails because the metadata rows are in a different order, or because the metadata has extra samples, reorder and subset it.

```python
meta = meta.loc[data.index]
```

## A complete example

This script simulates a small data set, fits the model, and runs both sparsification methods. It runs in about 3 seconds on a laptop.

```python
import numpy as np
import pandas as pd

from metvae.model import MetVAE
from metvae.sim import sim_data

np.random.seed(7)

n_samples = 100
n_metabolites = 50

sample_ids = [f"sample_{i:02d}" for i in range(1, n_samples + 1)]
meta = pd.DataFrame(
    {
        "age": np.random.normal(loc=50.0, scale=8.0, size=n_samples),
        "batch": np.random.choice(["A", "B"], size=n_samples),
    },
    index=sample_ids,
)

sim = sim_data(
    n=n_samples,
    d=n_metabolites,
    cor_pairs=10,
    mu=[10, 11, 12, 13, 14],
    x=meta,
    cont_list=["age"],
    cat_list=["batch"],
)

raw_data = sim["y"].copy()
true_cor = sim["cor_matrix"]

# Set the lowest 30 percent of the values of each metabolite to zero, to mimic
# the left-censoring seen in metabolomics.
thresholds = raw_data.quantile(0.30, axis=0)
data = raw_data.mask(raw_data.lt(thresholds), 0.0)

data.index = sample_ids
meta.index = sample_ids

pd.testing.assert_index_equal(data.index, meta.index)

model = MetVAE(
    data=data,
    meta=meta,
    continuous_covariate_keys=["age"],
    categorical_covariate_keys=["batch"],
)

model.train(learning_rate=1e-2)

corr_outputs = model.get_corr(workers=1)

results_p = model.sparse_by_p()
results_sec = model.sparse_by_sec(workers=1)

print("Objects returned by get_corr():", list(corr_outputs.keys()))
print("Objects returned by sparse_by_p():", list(results_p.keys()))
print("Best rho:", results_sec["best_rho"])
print(results_p["sparse_estimate"].iloc[:5, :5])
```

Constructing `MetVAE` prints a summary of the preprocessing.

```text
Start: samples=100, features=50
Filtered features: removed 0 (threshold 0.30)
Filtered samples: none (no sample_zero_threshold provided)
After zero filtering: samples=100, features=50
Post-cleaning (convert negative values to zeros and drop all-zero samples): samples=100, features=50
```

`train` then prints a progress bar over the 1000 epochs, and the final lines are

```text
Objects returned by get_corr(): ['impute_log_data', 'estimate']
Objects returned by sparse_by_p(): ['estimate', 'p_value', 'q_value', 'sparse_estimate']
Best rho: 1.2746360114795632
```

followed by the first five rows and columns of the sparsified correlation matrix.

In this example no metabolite is dropped, because each one has exactly 30 percent zeros and the default filter removes metabolites with a zero proportion strictly greater than 0.30.

## Example notebooks

The `notebooks/` folder holds three executed Jupyter notebooks: `01_quickstart.ipynb` (the example above with figures and a timing table), `02_sim_study.ipynb` (four simulated settings with and without confounders and zero inflation), and `03_hcc.ipynb` (a real 411 sample by 7217 metabolite data set analyzed from a saved checkpoint). The data and checkpoint they read are not part of this repository; the notebooks document the file paths and the exact calls.

## Arguments that matter

Every argument below has a default that works for a first run. The tables list the ones that change the results.

### `MetVAE(...)`

| Argument | What it controls | Default | When to change it |
| --- | --- | --- | --- |
| `data` | Abundance table, samples by metabolites. | required | Always supplied. |
| `features_as_rows` | Whether the table is transposed. | `False` | Set to `True` when metabolites are rows. |
| `meta` | Sample metadata table. | `None` | Supply it whenever you want covariates removed. |
| `continuous_covariate_keys` | Numeric metadata columns to adjust for. | `None` | For example `['age', 'bmi']`. |
| `categorical_covariate_keys` | Grouping metadata columns to adjust for. | `None` | For example `['batch', 'diet']`. |
| `feature_zero_threshold` | Metabolites with a zero proportion above this value are dropped. | `0.3` | Raise it to keep sparser metabolites, lower it to keep only well-measured ones, set `None` to keep all. |
| `sample_zero_threshold` | Samples with a zero proportion above this value are dropped. | `None` | Set a value such as `0.8` to drop samples with few detected metabolites. |
| `latent_dim` | Size of the latent space. | `None`, which is `min(n_samples, n_metabolites)` after filtering | Set a smaller value such as 10 to impose a lower-dimensional structure. |
| `hidden_dims` | Encoder hidden layer sizes. | `None`, a linear encoder | Set for example `[256, 128]` for a nonlinear encoder. |
| `activation` | Nonlinearity in the hidden layers. | `'relu'` | Options are `'relu'`, `'tanh'`, `'gelu'`, `'silu'`, or `None` for no nonlinearity. Has no effect when `hidden_dims` is `None`. |
| `use_gpu` | Whether to run on CUDA. | `False` | Set to `True` with a CUDA build of PyTorch. |
| `logging` | Whether to write TensorBoard logs to `runs/`. | `False` | Set to `True` after installing TensorBoard. |
| `seed` | Seed for preprocessing, initialization, and training. | `0` | Change it to check the stability of the results across runs. |

### `model.train(...)`

| Argument | What it controls | Default | When to change it |
| --- | --- | --- | --- |
| `batch_size` | Samples per gradient step. | `128` | Lower it when a single batch does not fit in memory. |
| `max_epochs` | Number of passes over the data. | `1000` | Raise it when the training loss is still falling at the end. |
| `learning_rate` | Initial step size of the optimizer. | `1e-3` | `1e-2` is used in the examples here and converges faster on small data. Lower it if the loss oscillates. |
| `max_grad_norm` | Gradient clipping threshold. | `1.0` | Set `None` to disable clipping. |
| `shuffle` | Whether the sample order is permuted each epoch. | `True` | Leave as is. |
| `num_workers` | Data loading subprocesses. | `0` | Leave at `0`; the data are already in memory. |

Inspect `model.train_loss`, a list of one mean loss per epoch, to judge whether training has converged.

### `model.get_corr(...)`

| Argument | What it controls | Default | When to change it |
| --- | --- | --- | --- |
| `num_sim` | Number of imputations averaged. | `100` | Lower it to 20 for a fast trial run, raise it for a more stable estimate. |
| `workers` | Number of PyTorch threads used. `-1` uses all CPU cores. | `-1` | Lower it when sharing a machine. The results do not depend on it. |
| `threshold` | Correlations with absolute value below this are set to zero within each imputation. | `0.2` | Lower it to keep weaker associations in the estimate. |
| `batch_size` | Imputations per round of CUDA streams. | `100` | Used only on a GPU. |
| `seed` | Base seed; imputation `i` uses `seed + i`. | `None`, which reuses the seed given to `MetVAE` | Change it to check stability. |

### `model.sparse_by_p(...)`

| Argument | What it controls | Default | When to change it |
| --- | --- | --- | --- |
| `p_adj_method` | Multiple-testing correction. | `'fdr_bh'` | `'bonferroni'`, `'sidak'`, `'holm-sidak'`, `'holm'`, `'simes-hochberg'`, `'hommel'`, `'fdr_bh'`, `'fdr_by'`, `'fdr_tsbh'`, `'fdr_tsbky'`. |
| `cutoff` | Adjusted p-value threshold. | `0.05` | Lower it to `0.01` for a smaller, more selective network. |

### `model.sparse_by_sec(...)`

| Argument | What it controls | Default | When to change it |
| --- | --- | --- | --- |
| `rho` | Penalty strength. | `None`, chosen by cross-validation | Supply a number to skip cross-validation and refit quickly. |
| `c_grid` | Grid of multipliers searched by cross-validation. | `1.0, 2.0, ..., 10.0` | Extend it when the selected value sits at an end of the grid. |
| `n_splits` | Number of cross-validation folds. | `5` | Lower it to `3` for a faster search on large data. |
| `refine` | Whether a second, finer pass runs around the best coarse value. | `True` | Set `False` for a faster search. |
| `refine_points` | Number of values in the refinement pass. | `10` | Raise it for a finer search. |
| `threshold` | Entries with absolute value below this are set to zero in the final estimate. | `0.1` | Raise it for a sparser network. |
| `workers` | Number of PyTorch threads used. `-1` uses all CPU cores. | `-1` | Lower it when sharing a machine. |
| `seed` | Seed for the fold assignment. | `0` | Change it to check the stability of the selected penalty. |

### How to choose

`feature_zero_threshold` sets how much missingness you accept in a metabolite. The default of `0.3` keeps metabolites detected in at least 70 percent of samples. A metabolite detected in very few samples contributes little information about correlation, and its imputed values dominate its column, so a low threshold gives a more reliable network and a high threshold gives a larger one.

`num_sim` sets how many imputed data sets are averaged. Averaging reduces the variability that the imputation itself introduces. Run a first analysis with `num_sim=20` to check that the pipeline works, then use the default of 100 for the reported results.

`cutoff` in `sparse_by_p` is the false discovery rate you are willing to accept when `p_adj_method` is `'fdr_bh'`. A cutoff of `0.05` means that about 5 percent of the reported edges are expected to be false.

`rho` in `sparse_by_sec` controls how strongly small correlations are pushed to zero: larger values give fewer edges. The candidate values are

```python
rho = c * sqrt(log(p) / n)
```

where `p` is the number of metabolites and `n` the number of samples. Cross-validation evaluates each `c` in `c_grid`, then a refinement pass evaluates `refine_points` values around the best one. If you see a warning that the best `c` sits at the lower or upper edge of `c_grid`, the search ran out of room; extend the grid in that direction and rerun, for example `c_grid=[0.1, 0.25, 0.5, 1.0, 2.0]` for the lower edge or `c_grid=list(range(1, 21))` for the upper edge.

## Reading the results

### What `sparse_by_p` returns

`sparse_by_p` returns a dictionary of four pandas DataFrames. Each is square, with the metabolite names as both the row index and the column names.

| Key | Contents |
| --- | --- |
| `estimate` | The correlation estimate before sparsification. |
| `p_value` | The unadjusted p-value of each pair. |
| `q_value` | The p-value after multiple-testing correction. |
| `sparse_estimate` | The correlation estimate with every pair whose `q_value` exceeds `cutoff` set to zero. |

### What `sparse_by_sec` returns

| Key | Contents |
| --- | --- |
| `estimate` | The correlation estimate before sparsification. |
| `sparse_estimate` | The SEC estimate after penalization and thresholding. |
| `best_rho` | The penalty used, either the one you supplied or the one chosen by cross-validation. |
| `scores_by_rho` | A DataFrame with columns `c`, `rho`, and `score`, one row per candidate evaluated, or `None` when you supplied `rho`. |

A zero in `sparse_estimate` means there is no evidence of an association between that pair of metabolites at the settings you chose. It is not evidence that the two metabolites are independent.

### Saving and reshaping

```python
results_p["sparse_estimate"].to_csv("sparse_correlations.csv")
results_p["q_value"].to_csv("q_values.csv")
```

The square matrix is convenient for storage, and a list of edges is convenient for reading. The following converts the upper triangle into one row per metabolite pair, dropping the zeros.

```python
import numpy as np
import pandas as pd

sparse_df = results_p["sparse_estimate"]
values = sparse_df.to_numpy()
i, j = np.triu_indices(values.shape[0], k=1)
edges = pd.DataFrame(
    {
        "metabolite_1": sparse_df.index[i],
        "metabolite_2": sparse_df.columns[j],
        "correlation": values[i, j],
    }
)
edges = edges.loc[edges["correlation"] != 0]
edges = edges.reindex(edges["correlation"].abs().sort_values(ascending=False).index)
edges.to_csv("edges.csv", index=False)
```

To plot the cross-validation curve from `sparse_by_sec`:

```python
scores = results_sec["scores_by_rho"]
scores.plot(x="rho", y="score", marker="o")
```

### Exporting a network file

`export_graphml` writes one GraphML file per absolute correlation cutoff, keeping the pairs whose absolute correlation is at least that cutoff.

```python
graphs = model.export_graphml(
    sparse_df=results_p["sparse_estimate"],
    cutoffs=[0.7],
    output_dir="results",
)
```

This writes `results/correlation_graph_cutoff0.7.graphml`. Pass several cutoffs, for example `cutoffs=[0.9, 0.8, 0.7]`, to write one file each. The nodes are metabolites and the edges carry the attributes `weight`, `correlation`, `EdgeScore`, and `EdgeType`. The function returns a dictionary of `networkx` graphs, and skips any cutoff that yields no edges.

To view the network in Cytoscape, choose File > Import > Network from File and select the `.graphml` file. In the Style panel, set the edge width or the edge color to be mapped from the `EdgeScore` attribute, which holds the signed correlation.

### Other methods

`impute_zeros` returns the centered log-ratio data with the censored zeros replaced by the reconstruction of the trained model. Observed entries are left unchanged. The result is a PyTorch tensor with samples as rows and metabolites as columns, in the order given by `model.sample_name` and `model.feature_name`.

```python
imputed = model.impute_zeros()
imputed_df = pd.DataFrame(imputed.cpu().numpy(), index=model.sample_name, columns=model.feature_name)
```

`confound_coef` returns the estimated covariate effects on the centered log-ratio scale, with metabolites as rows and covariates as columns. Categorical covariates appear as one indicator column per level after the first. It returns `None` when no metadata was supplied.

```python
coef = model.confound_coef()
```

`confound_es` returns the fitted covariate effect for each sample and metabolite, with samples as rows and metabolites as columns. It is the product of the covariate matrix and the coefficients, and it is the quantity subtracted from the data before the correlations are estimated. It returns `None` when no metadata was supplied.

```python
effects = model.confound_es()
```

`clr_loading` returns the decoder weights as metabolite loadings on the latent dimensions, with metabolites as rows and the columns named `latent_0`, `latent_1`, and so on. Metabolites with similar loadings move together in the fitted model.

```python
loadings = model.clr_loading()
```

`cooccurrence` returns a square matrix indexed by metabolite, holding the model-implied variance of the log ratio of each pair. A small value indicates two metabolites that move together, and a large value indicates two that vary independently.

```python
cooccur = model.cooccurrence()
```

## Reproducibility

Three seeds control the random draws.

| Seed | What it controls |
| --- | --- |
| `MetVAE(seed=...)` | Model initialization and the batch order during training. |
| `get_corr(seed=...)` | The multiple imputations. Defaults to the seed given to `MetVAE`. |
| `sim_data(seed=...)` | The simulated data set. |

On one machine, with one version of MetVAE and one version of PyTorch, a run with the same seeds gives identical results. This holds regardless of the value of `workers`, which changes only how many threads are used.

Across machines, results can differ in the last few digits of each correlation. Different CPUs, different BLAS libraries, and different PyTorch versions sum floating-point numbers in different orders. The number of threads PyTorch uses, reported by `torch.get_num_threads()`, has the same effect; MetVAE leaves that setting unchanged.

These differences are far smaller than the statistical uncertainty of a correlation. They matter only when a correlation sits almost exactly on a threshold, such as the `threshold` of `get_corr` or the `cutoff` of `sparse_by_p`, where a difference in the last digits can move a single pair from one side to the other. Record the versions of MetVAE and PyTorch alongside your results.

## Run time and memory

The following measurements were taken on an Apple M2 Max laptop with 12 cores and 64 GB of memory, running on the CPU. Memory is the peak resident set size of the whole process tree during the step.

| Workload | Step | Seconds | Peak memory (GB) |
| --- | --- | --- | --- |
| Quickstart, 100 samples x 50 metabolites | `train`, 1000 epochs | 1.2 | 0.35 |
| | `get_corr(num_sim=100)` | 0.07 | 0.35 |
| | `sparse_by_p` | 0.005 | 0.35 |
| | `sparse_by_sec`, cross-validation | 0.20 | 0.35 |
| Simulation example 1, 100 samples x 500 metabolites, 30 percent zeros | `train`, 1000 epochs | 6.4 | 0.43 |
| | `get_corr(num_sim=100)` | 0.5 | 0.43 |
| | `sparse_by_sec`, grid search | 2.6 | 1.19 |
| HCC, 411 samples x 7217 metabolites, model loaded from a checkpoint | `get_corr(num_sim=100)` | 33.5 | 2.65 |
| | `sparse_by_p` | 2.8 | 5.99 |

Training time grows with the number of epochs, the number of samples, and the number of metabolites. Correlation estimation grows with `num_sim` and with the square of the number of metabolites.

For memory, write `d` for the number of metabolites after filtering. `get_corr` holds three dense `d` by `d` float64 matrices, which is `24 * d^2` bytes, about 1.3 GB at `d = 7217`. `sparse_by_p` holds four such matrices, which is `32 * d^2` bytes, because it returns the estimate, the p-values, the adjusted p-values, and the sparsified estimate at the same time. Allow room for both the PyTorch tensors and the pandas copies that are returned. If you run out of memory on a large data set, raise `feature_zero_threshold` to reduce `d`, or use `sparse_by_sec`, which holds fewer matrices.

## Command line

`metvae-cli` runs the whole pipeline and writes the results to `--save_path`. Every flag has a default, and only `--data` is required.

### Input and output

| Flag | Meaning | Default |
| --- | --- | --- |
| `--data` | Path to the abundance CSV, samples as rows. | required |
| `--features_as_rows` | Set when metabolites are rows and samples are columns. | off |
| `--meta` | Path to the sample metadata CSV. | none |
| `--save_path` | Output directory. | `./` |
| `--seed` | Random seed. | `0` |

### Preprocessing

| Flag | Meaning | Default |
| --- | --- | --- |
| `--continuous_covariate_keys` | Names of numeric metadata columns, separated by spaces. Requires `--meta`. | none |
| `--categorical_covariate_keys` | Names of grouping metadata columns, separated by spaces. Requires `--meta`. | none |
| `--feature_zero_threshold` | Drop metabolites whose zero proportion exceeds this value. Pass `none` to keep them all. | `0.3` |
| `--no_feature_filter` | Same as `--feature_zero_threshold none`. | off |
| `--sample_zero_threshold` | Drop samples whose zero proportion exceeds this value. | `none` |

### Model

| Flag | Meaning | Default |
| --- | --- | --- |
| `--latent_dim` | Size of the latent space. | `10` |
| `--hidden_dims` | Encoder hidden layer sizes, separated by spaces. Omit for a linear encoder. | none |
| `--activation` | One of `relu`, `tanh`, `gelu`, `silu`, `linear`, `none`. | `relu` |
| `--use_gpu` | Use CUDA if available. | off |
| `--logging` | Write TensorBoard logs to `runs/`. | off |

The default `--latent_dim` of the command line differs from the Python default, which is `min(n_samples, n_metabolites)`.

### Training

| Flag | Meaning | Default |
| --- | --- | --- |
| `--batch_size` | Samples per gradient step. | `128` |
| `--num_workers` | Data loading subprocesses. | `0` |
| `--max_epochs` | Number of passes over the data. | `1000` |
| `--learning_rate` | Initial step size. | `0.001` |
| `--max_grad_norm` | Gradient clipping threshold; `-1` disables clipping. | `1.0` |
| `--deterministic` | Enable deterministic CUDA algorithms. | off |

### Correlation estimation

| Flag | Meaning | Default |
| --- | --- | --- |
| `--num_sim` | Number of imputations averaged. | `100` |
| `--workers` | PyTorch threads; `-1` uses all cores. | `-1` |
| `--threshold` | Absolute correlation cutoff applied within each imputation. | `0.2` |
| `--impute_batch_size` | Imputations per round of CUDA streams; used only on a GPU. | `100` |

### Sparsification

| Flag | Meaning | Default |
| --- | --- | --- |
| `--sparse_method` | `pval` or `sec`. | `sec` |

For `--sparse_method pval`:

| Flag | Meaning | Default |
| --- | --- | --- |
| `--p_adj_method` | Multiple-testing correction. | `fdr_bh` |
| `--cutoff` | Adjusted p-value cutoff. | `0.05` |

For `--sparse_method sec`:

| Flag | Meaning | Default |
| --- | --- | --- |
| `--rho` | Fixed penalty. A negative value selects the penalty by cross-validation. | `-1.0` |
| `--c_grid` | Grid of multipliers for cross-validation. | `1.0 2.0 ... 10.0` |
| `--n_splits` | Number of cross-validation folds. | `5` |
| `--no_refine` | Skip the refinement pass after the coarse search. | off |
| `--refine_points` | Number of values in the refinement pass. | `10` |
| `--sec_threshold` | Final hard threshold on the SEC estimate. | `0.1` |
| `--sec_workers` | PyTorch threads for cross-validation; `-1` uses all cores. | `-1` |
| `--sec_epsilon` | Eigenvalue floor used in the projection step. | `1e-05` |
| `--sec_tol` | Convergence tolerance of the solver. | `0.001` |
| `--sec_max_iter` | Maximum number of solver iterations. | `1000` |
| `--sec_restart` | Restart period of the solver; a negative value disables restarts. | `50` |
| `--no_line_search_apg` | Disable the solver line search. | off |
| `--sec_delta` | Small-correlation cutoff inside the solver. | computed from `--sec_c_delta` |
| `--sec_c_delta` | Scale used when `--sec_delta` is not given. | `0.1` |

### GraphML export

| Flag | Meaning | Default |
| --- | --- | --- |
| `--export_graphml` | Write GraphML files from the final sparse matrix. | off |
| `--graphml_cutoffs` | Absolute correlation cutoffs, separated by spaces. | `0.7` |
| `--graphml_prefix` | Filename prefix; the cutoff and `.graphml` are appended. | `correlation_graph_cutoff` |

### Worked example: p-value sparsification

```bash
metvae-cli \
  --data data.csv \
  --meta meta.csv \
  --save_path results_pval \
  --continuous_covariate_keys age \
  --categorical_covariate_keys batch diet \
  --feature_zero_threshold 0.3 \
  --latent_dim 10 \
  --batch_size 128 \
  --max_epochs 1000 \
  --learning_rate 0.01 \
  --num_sim 100 \
  --threshold 0.2 \
  --sparse_method pval \
  --p_adj_method fdr_bh \
  --cutoff 0.05 \
  --export_graphml \
  --graphml_cutoffs 0.9 0.8 0.7 \
  --graphml_prefix correlation_graph_cutoff
```

This writes into `results_pval`:

| File | Contents |
| --- | --- |
| `model_state.pth` | Model weights, optimizer state, and the per-epoch training losses. |
| `df_corr.csv` | The correlation estimate before sparsification. |
| `p_values.csv` | Unadjusted p-values. |
| `q_values.csv` | Adjusted p-values. |
| `df_sparse_pval.csv` | The sparsified correlation matrix. |
| `correlation_graph_cutoff0.9.graphml`, and one file per cutoff | Network files. |

### Worked example: SEC sparsification

```bash
metvae-cli \
  --data data.csv \
  --meta meta.csv \
  --save_path results_sec \
  --continuous_covariate_keys age \
  --categorical_covariate_keys batch diet \
  --feature_zero_threshold 0.3 \
  --latent_dim 10 \
  --max_epochs 1000 \
  --learning_rate 0.01 \
  --num_sim 100 \
  --threshold 0.2 \
  --sparse_method sec \
  --c_grid 1 2 3 4 5 6 7 8 9 10 \
  --n_splits 5 \
  --refine_points 10 \
  --sec_threshold 0.1 \
  --export_graphml \
  --graphml_cutoffs 0.7
```

This writes into `results_sec`:

| File | Contents |
| --- | --- |
| `model_state.pth` | Model weights, optimizer state, and the per-epoch training losses. |
| `df_corr.csv` | The correlation estimate before sparsification. |
| `df_sparse_sec.csv` | The SEC estimate after penalization and thresholding. |
| `sec_selected.txt` | One line of the form `best_rho=...`. |
| `sec_scores.csv` | The cross-validation score of each candidate penalty. Written only when the penalty was selected by cross-validation. |
| `correlation_graph_cutoff0.7.graphml` | Network file. |

To use a fixed penalty instead of cross-validation, add `--rho 2.2`. In that case `sec_scores.csv` is not written.

## Troubleshooting

| Message | Cause | Fix |
| --- | --- | --- |
| `ValueError: The following sample names are missing in the sample meta data: {...}` | Some samples in the abundance table have no row in the metadata. | Add the missing rows, or drop those samples with `data = data.loc[data.index.intersection(meta.index)]`. |
| `UserWarning: The dataset contains N negative values. They have been converted to zeros, but please double-check that this preprocessing step is appropriate for your data.` | The abundance table contains negative numbers, often from a background subtraction or a log transform applied earlier. | Supply raw, non-negative abundances. If the values are already on a log scale, exponentiate them before passing them to MetVAE. |
| `Removed N all-zero samples after cleaning.` | Some samples had no detected metabolite left after filtering. | No action is required. Check the sample identifiers if the count is larger than you expect. |
| `ValueError: Need at least d >= 3 features to compute stable correlations.` | Fewer than three metabolites remain after filtering. | Raise `feature_zero_threshold`, or set it to `None`, to keep more metabolites. |
| `ValueError: Sample size must be > 3 for Fisher's z-test.` | Fewer than four samples remain. | `sparse_by_p` cannot be used at this sample size. Use `sparse_by_sec`, or raise `sample_zero_threshold` if samples were dropped by filtering. |
| `ValueError: No correlation estimates. Please compute correlations the first using get_corr method.` | `sparse_by_p` was called before `get_corr`. | Call `model.get_corr()` first. |
| ``ValueError: No correlation estimates. Please compute correlations first using `get_corr`.`` | `sparse_by_sec` was called before `get_corr`. | Call `model.get_corr()` first. |
| `UserWarning: Best c = ... occurs at the LOWER edge of c_grid [...]. Consider expanding c_grid to include smaller c values.` | Cross-validation chose the smallest candidate, so a better penalty may lie below the grid. | Rerun with a grid that extends lower, for example `c_grid=[0.1, 0.25, 0.5, 1.0, 2.0]`. |
| `UserWarning: Best c = ... occurs at the UPPER edge of c_grid [...]. Consider expanding c_grid to include larger c values.` | Cross-validation chose the largest candidate. | Rerun with a grid that extends higher, for example `c_grid=list(range(1, 21))`. |
| The process is killed, or the machine swaps, during `get_corr` or `sparse_by_p`. | The dense matrices do not fit in memory. See the memory section for the sizes. | Raise `feature_zero_threshold` to reduce the number of metabolites, or use `sparse_by_sec` in place of `sparse_by_p`. |
| `CUDA not available. Falling back to CPU.` | `use_gpu=True` was passed but PyTorch cannot see a CUDA device. | Install a CUDA build of PyTorch, or leave `use_gpu=False`. This message also appears on a Mac, where Apple GPUs are not used. |
| ``RuntimeError: logging=True requires tensorboard. Install it with `pip install tensorboard`.`` | TensorBoard is not installed. | Run `pip install "metvae[logging]"`, or set `logging=False`. |
| ``RuntimeError: networkx is required for GraphML export. Install it with `pip install networkx`.`` | `networkx` is missing from the environment. | Run `pip install networkx`. |

## Citation

If you use MetVAE, please cite the protocol preprint and the SEC method. The protocol manuscript is under review at STAR Protocols.

```bibtex
@article{lin2026metvae,
  title   = {Protocol for constructing correlation-based molecular networks from large-scale untargeted metabolomics data},
  author  = {Lin, Huang and Zhang, Lijun and Lotfi, Ali and Jarmusch, Alan and Lee, Iris and Kim, Adam and Morton, James T. and Aksenov, Alexander},
  journal = {bioRxiv},
  year    = {2026},
  doi     = {10.1101/2025.04.26.649581},
  note    = {PMCID: PMC13131467. Under review at STAR Protocols.}
}

@article{cui2016sec,
  title   = {Sparse estimation of high-dimensional correlation matrices},
  author  = {Cui, Ying and Leng, Chenlei and Sun, Defeng},
  journal = {Computational Statistics and Data Analysis},
  volume  = {93},
  pages   = {390--403},
  year    = {2016}
}
```

The preprint is available at https://pmc.ncbi.nlm.nih.gov/articles/PMC13131467/. The SEC implementation in this package is adapted from the MATLAB reference code released by Leng's group at the University of Warwick.

## License

MetVAE is released under the MIT License. See the `LICENSE` file for the full text.
