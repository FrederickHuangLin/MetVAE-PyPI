# Changelog

All notable changes to `metvae` are recorded here.

## 1.1.0

### Performance

- Random initialization of the censored values is vectorized over features, replacing the per-feature loop.
- Training draws its batches from `torch.randperm` index batches instead of a `DataLoader`, with a one-time check that the draws match `DataLoader` exactly.
- Training runs about 1.5 to 2.5 times faster, with numerics identical to the previous release.
- `_compute_correlation_dense` computes the correlation matrix with a single matrix multiplication and writes into caller-supplied buffers.
- The multiple imputations of `get_corr` run in process, accumulating into a single dense matrix, and `get_corr` is several times faster.
- The cross-validation in `sparse_by_sec` computes the training-fold correlations once per fold instead of once per candidate `rho`.

### Memory

- `get_corr` holds three dense `d` by `d` float64 matrices in one process, in place of one copy per worker process.
- Peak memory for the 411 by 7217 workload falls from about 44 GB across worker processes to about 2.5 GB.
- `sparse_by_p` releases the intermediate `z` and `z_score` matrices before the p-values are adjusted.

### Reproducibility

- `get_corr` no longer changes the process-wide torch thread count; the previous value is restored on return.
- The results of `get_corr` no longer depend on `workers` or on the number of CPU cores.
- The imputations are reduced in simulation order, so the sum is performed in a fixed order.
- `sim_data` accepts a `seed` argument.

### Bug fixes

- `model.clr_sd` is a standard deviation on every code path. On the zero-free path it previously held a variance. Imputation and correlation estimates are unaffected, because that path does not use it.
- `train()` resets `train_loss` at the start of each call, so repeated calls no longer append to the losses of the previous run.
- `train()` warns when it receives unknown keyword arguments, which were previously accepted and ignored silently.
- `train()` stores `optimizer`, `scheduler`, and `current_epoch` on the model.
- The TensorBoard writer is flushed and closed when training ends or raises.
- The upper bound on `cor_pairs` in `sim_data` is `d // 2`.
- CLI: `--continuous_covariate_keys` and `--categorical_covariate_keys` default to no covariates, and passing either without `--meta` is an error.
- CLI: `--no_feature_filter` keeps every feature except the all-zero ones.
- CLI: `--feature_zero_threshold none` disables the feature filter through the same flag that sets it.
- CLI: the saved checkpoint holds the optimizer state, the epoch count, the per-epoch losses, and the learning rate alongside the model weights.

### Packaging

- `pytorch-lightning` and `joblib` are removed from the dependencies.
- `tensorboard` moves to the optional extra `logging`, installed with `pip install "metvae[logging]"`.
- Minimum versions raised to `torch >= 1.12` and `pandas >= 1.5`.
- The test package is excluded from the built wheel.
- `pyproject.toml` added, declaring the setuptools build backend and the pytest configuration.
- The test suite is rewritten with tolerance-based numerical checks in place of exact equality.

### Documentation

- README rewritten for readers with little coding experience, with a complete worked example, argument tables, expected run times and memory, the command-line flags as they are defined, and a troubleshooting table.
- The command-line flag names in the README match `build_parser`.
- Docstrings revised across `model.py`, `sparse.py`, `compute_corr.py`, `impute_missing.py`, and `sim.py`.

## 1.0.0

Previous release.
