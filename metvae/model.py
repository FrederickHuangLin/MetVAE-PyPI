import os
import warnings
from typing import Optional, Iterable, Dict, List, Literal, Sequence
import random
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from .vae import VAE
from .utils import _make_valid_column_name, _torch_to_df, _corr_to_long, _thread_limit
from .impute_missing import _ols_estimate, _fit_censored_normal
from .compute_corr import _compute_correlation, _compute_correlation_dense
from .sparse import _matrix_p_adjust, _p_filter, _SEC_dense, _SEC_cv

def _data_pre_process(
        data: pd.DataFrame,
        features_as_rows: bool = False,
        meta: Optional[pd.DataFrame] = None,
        continuous_covariate_keys: Optional[List[str]] = None,
        categorical_covariate_keys: Optional[List[str]] = None,
        device: Optional[str] = None,
        dtype: torch.dtype = torch.float64,
        feature_zero_threshold: float = 0.3,
        sample_zero_threshold: Optional[float] = None
    ):
    """
    Internal function for preprocessing compositional data with metadata covariates.
    Performs CLR transformation, handles zero values, and adjusts for confounding effects.

    Parameters
    ----------
    data : pandas.DataFrame
        Input abundance matrix. Can be organized with either features as columns (default)
        or features as rows (set features_as_rows=True)
    features_as_rows : bool, default=False
        If True, transposes the input data to ensure features are columns
    meta : pandas.DataFrame, optional
        Sample metadata containing covariates/confounders. Must have samples as index
        matching the abundance data
    continuous_covariate_keys : List[str], optional
        Column names in meta for continuous covariates to adjust for
    categorical_covariate_keys : List[str], optional
        Column names in meta for categorical covariates to adjust for
    device : str or torch.device, optional
        Torch device to place tensors on (e.g., 'cpu', 'cuda'). If None, uses default device.
    dtype : torch.dtype, optional
        Data type for returned torch tensors (default torch.float64).
    feature_zero_threshold: float
        Drop features with proportion of zeros > threshold (default 0.3)
    sample_zero_threshold: float, optional
        drop samples with proportion of zeros > threshold (default None: keep all)

    Returns
    -------
    dict
        A dictionary containing processed data and parameters:
        - clr_data: CLR-transformed and deconfounded data (torch.tensor)
        - meta: Processed metadata matrix (torch.tensor)
        - num_zero: Count of zeros per feature (torch.tensor)
        - shift: Sample-wise geometric mean for CLR transform (numpy.array)
        - clr_mean: Estimated means in CLR space (torch.tensor)
        - clr_sd: Estimated standard deviations in CLR space (torch.tensor)
        - clr_coef: Estimated covariate coefficients (torch.tensor)
        - sample_name: List of sample identifiers
        - feature_name: List of feature identifiers
        - confound_name: List of confounder names

    Notes
    -----
    Zeros are represented as NaN on the CLR scale. Feature means, standard deviations and
    covariate coefficients are estimated by a censored normal model when zeros are present
    and by ordinary least squares otherwise.
    """
    dev = torch.device(device) if device is not None else torch.device("cpu")

    if not isinstance(data, pd.DataFrame):
        raise TypeError('The input data must be a pandas.DataFrame')

    if features_as_rows:
        data = data.T

    if meta is not None:
        if not isinstance(meta, pd.DataFrame):
            raise TypeError('The meta data must be a pandas.DataFrame or None')

        missing = set(data.index) - set(meta.index)
        if missing:
            raise ValueError(f"The following sample names are missing in the sample meta data: {missing}")
        meta = meta.loc[data.index]

    # Drop features and samples whose proportion of zeros exceeds the thresholds.
    n0, d0 = data.shape
    print(f"Start: samples={n0}, features={d0}")

    # NaNs count as zeros for the sparsity calculation.
    data_zeros_view = data.fillna(0)

    if feature_zero_threshold is not None:
        feat_zero_prop = (data_zeros_view.eq(0)).mean(axis=0)  # per feature
        feat_drop_mask = feat_zero_prop > feature_zero_threshold
        n_feat_drop = int(feat_drop_mask.sum())
        if n_feat_drop > 0:
            data = data.loc[:, ~feat_drop_mask]
            print(f"Filtered features: removed {n_feat_drop} with zero proportion > {feature_zero_threshold:.2f}")
        else:
            print(f"Filtered features: removed 0 (threshold {feature_zero_threshold:.2f})")

    if sample_zero_threshold is not None:
        data_zeros_view = data.fillna(0)
        samp_zero_prop = (data_zeros_view.eq(0)).mean(axis=1)  # per sample
        samp_drop_mask = samp_zero_prop > sample_zero_threshold
        n_samp_drop = int(samp_drop_mask.sum())
        if n_samp_drop > 0:
            data = data.loc[~samp_drop_mask, :]
            if meta is not None:
                meta = meta.loc[data.index]
            print(f"Filtered samples: removed {n_samp_drop} with zero proportion > {sample_zero_threshold:.2f}")
        else:
            print(f"Filtered samples: removed 0 (threshold {sample_zero_threshold:.2f})")
    else:
        print("Filtered samples: none (no sample_zero_threshold provided)")

    n1, d1 = data.shape
    print(f"After zero filtering: samples={n1}, features={d1}")

    sample_name = data.index.tolist()
    feature_name = data.columns.tolist()

    tdata = torch.tensor(data.values, dtype=dtype, device=dev)
    tdata = torch.nan_to_num(tdata, nan=0.0)

    neg_mask = tdata < 0
    num_neg = int(neg_mask.sum().item())
    if num_neg > 0:
        warnings.warn(
            f"The dataset contains {num_neg} negative values. "
            "They have been converted to zeros, but please double-check "
            "that this preprocessing step is appropriate for your data."
        )
        tdata = torch.where(neg_mask, torch.zeros_like(tdata), tdata)

    row_all_zero = (tdata == 0).all(dim=1)
    if row_all_zero.any():
        keep_idx = ~row_all_zero
        dropped = int(row_all_zero.sum().item())
        tdata = tdata[keep_idx]
        sample_name = [s for i, s in enumerate(sample_name) if keep_idx[i].item()]
        if meta is not None:
            meta = meta.loc[sample_name]
        print(f"Removed {dropped} all-zero samples after cleaning.")
    n, d = tdata.shape
    print(f"Post-cleaning (convert negative values to zeros and drop all-zero samples): samples={n}, features={d}")

    num_zero = (tdata == 0).sum(dim=0).float()

    # Assemble the covariate matrix, one-hot encoding the categorical covariates.
    if meta is not None:
        smd_cont = meta.loc[:, continuous_covariate_keys] if continuous_covariate_keys is not None else None

        smd_cat = None
        if categorical_covariate_keys is not None:
            smd_cat = meta.loc[:, categorical_covariate_keys].apply(lambda x: x.astype('category'))
            smd_cat = pd.get_dummies(smd_cat, drop_first=True, dtype=float)
            smd_cat.columns = [_make_valid_column_name(c) for c in smd_cat.columns]

        if smd_cont is not None and smd_cat is not None:
            smd_df = pd.concat([smd_cont, smd_cat], axis=1)
            confound_name = smd_df.columns.tolist()
        elif smd_cont is not None:
            smd_df = smd_cont
            confound_name = smd_cont.columns.tolist()
        elif smd_cat is not None:
            smd_df = smd_cat
            confound_name = smd_cat.columns.tolist()
        else:
            smd_df = None
            confound_name = None
    else:
        smd_df = None
        confound_name = None

    if smd_df is not None:
        smd = torch.tensor(smd_df.values, dtype=dtype, device=dev)
        p = smd.shape[1] + 1
    else:
        smd = None
        p = 1

    # Center the log data by sample; zeros become NaN.
    log_data = torch.where(
        tdata > 0, torch.log(tdata),
        torch.tensor(float('nan'), dtype=tdata.dtype, device=dev)
    )
    shift = torch.nanmean(log_data, dim=1, keepdim=True)
    clr_data = log_data - shift

    # Per-feature smallest positive value, used as the censoring threshold.
    th_raw = torch.where(tdata > 0, tdata, torch.tensor(float('inf'), dtype=tdata.dtype, device=dev)).min(dim=0).values
    th_raw = torch.where(th_raw == float('inf'), torch.tensor(1e-5, dtype=tdata.dtype, device=dev), th_raw)
    clr_th = torch.log(th_raw) - shift

    init_params = _ols_estimate(clr_data, smd)  # (p+1, d)

    if torch.any(num_zero != 0):
        clr_params = _fit_censored_normal(
            clr_data, smd, clr_th,
            init_estimates=init_params,
            max_iter=100,
        )
    else:
        clr_params = init_params

    clr_log_sd = clr_params[p, :]
    clr_sd = torch.exp(clr_log_sd)
    clr_mean = clr_params[0, :]

    if smd is not None:
        clr_coef = clr_params[1:p, :]
        clr_data = clr_data - smd @ clr_coef
    else:
        clr_coef = None

    outputs = {
        'clr_data': clr_data,
        'meta': smd,
        'num_zero': num_zero,
        'shift': shift,
        'clr_mean': clr_mean,
        'clr_sd': clr_sd,
        'clr_coef': clr_coef,
        'sample_name': sample_name,
        'feature_name': feature_name,
        'confound_name': confound_name
    }
    return outputs

def _scale_continuous_metadata(
        meta: Optional[pd.DataFrame],
        continuous_covariate_keys: Optional[List[str]]
    ) -> Optional[pd.DataFrame]:
    """
    Center and scale continuous covariates while leaving all other columns unchanged.
    Constant columns are centered and left with unit scale to avoid division by zero.
    """
    if meta is None or continuous_covariate_keys is None:
        return meta

    scaled = meta.copy()
    cont = scaled.loc[:, continuous_covariate_keys].astype(float)
    means = cont.mean(axis=0)
    stds = cont.std(axis=0, ddof=0).replace(0, 1.0)
    scaled.loc[:, continuous_covariate_keys] = (cont - means) / stds
    return scaled

def _random_initial(
        y: torch.Tensor, 
        sample_size: int, 
        num_zero: torch.Tensor,
        mean: torch.Tensor, 
        sd: torch.Tensor,
        *,
        generator: Optional[torch.Generator] = None
    ) -> torch.Tensor:
    """
    Replace NaN entries of CLR-transformed data by draws from the lower tail of a normal.

    For each feature, sample_size normal variates are drawn with the feature mean and
    standard deviation, the k smallest are retained as candidates, and one candidate is
    drawn uniformly with replacement for each NaN of that feature. Here k is the number of
    zeros of the feature, clamped to [1, sample_size].

    Parameters
    ----------
    y : torch.Tensor
        CLR-transformed data of shape (batch_size, feature_size). NaN entries mark the
        censored zeros of the original data.
    sample_size : int
        Number of normal variates drawn per feature.
    num_zero : torch.Tensor
        Number of zeros per feature, of shape (feature_size,).
    mean : torch.Tensor
        Per-feature mean on the CLR scale, of shape (feature_size,).
    sd : torch.Tensor
        Per-feature standard deviation on the CLR scale, of shape (feature_size,).
    generator : torch.Generator, optional
        Generator used for both random draws. If None, the global torch RNG is used.

    Returns
    -------
    torch.Tensor
        Tensor of the same shape as y with every NaN replaced.

    Notes
    -----
    Within a feature, the t-th NaN in ascending row order receives the t-th draw.
    """
    device, dtype = y.device, y.dtype
    feature_size = y.shape[1]

    nan_mask = torch.isnan(y)
    num_nan = nan_mask.sum(dim=0)

    if (num_nan == 0).all():
        return y.clone()

    rand = torch.randn(sample_size, feature_size, device=device, dtype=dtype, generator=generator)
    random_data = rand.mul_(sd).add_(mean)

    # Number of candidates kept per feature.
    k0 = num_zero.to(dtype=torch.long)
    k = torch.clamp(torch.where(k0 > 0, k0, torch.ones_like(k0)), min=1, max=sample_size)
    k_max = int(k.max().item())
    candidates = torch.topk(random_data, k_max, dim=0, largest=False, sorted=True).values
    del random_data, rand

    # One uniform index in [0, k_j) per required draw of feature j.
    m_max = int(num_nan.max().item())
    r = (torch.rand(m_max, feature_size, device=device, generator=generator) * k.view(1, -1)).floor().to(torch.long)
    fills_full = torch.gather(candidates, dim=0, index=r)

    # Position of each NaN within its column, counted from the top.
    rank = (nan_mask.cumsum(0) - 1).clamp_(0, m_max - 1)
    picked = fills_full.gather(0, rank)

    return torch.where(nan_mask, picked, y)

def _epoch_index_batches(
        n: int,
        batch_size: int,
        shuffle: bool,
        generator: torch.Generator
    ) -> List[tuple]:
    """
    Build one epoch of index batches, drawing from the generator as a DataLoader does.

    Parameters
    ----------
    n : int
        Number of samples.
    batch_size : int
        Number of indices per batch. The final batch may be shorter.
    shuffle : bool
        Whether the sample order is permuted.
    generator : torch.Generator
        Generator consumed by the draws.

    Returns
    -------
    list of tuple
        One ``(indices, None)`` pair per batch, with indices in permutation order.

    Notes
    -----
    Each epoch consumes one int64 draw for the iterator base seed and, when shuffle is
    True, two permutations of length n, matching torch 2.5 DataLoader with a RandomSampler.
    """
    torch.empty((), dtype=torch.int64).random_(generator=generator)
    if shuffle:
        perm = torch.randperm(n, generator=generator)
        torch.randperm(n, generator=generator)
    else:
        perm = torch.arange(n)
    return [(perm[s:s + batch_size], None) for s in range(0, n, batch_size)]

_RANDPERM_BATCHING_OK: Optional[bool] = None

def _randperm_batching_matches_dataloader() -> bool:
    """
    Check once per process that index batching reproduces the DataLoader.

    Returns
    -------
    bool
        True when the batch contents and the generator state after one epoch agree with
        ``torch.utils.data.DataLoader`` on small reference cases.
    """
    global _RANDPERM_BATCHING_OK
    if _RANDPERM_BATCHING_OK is not None:
        return _RANDPERM_BATCHING_OK

    ok = True
    try:
        for (n, bs, shuffle) in ((7, 3, True), (7, 3, False), (4, 4, True)):
            ds = TensorDataset(torch.arange(n, dtype=torch.float64).view(n, 1))
            g_ref = torch.Generator(device="cpu")
            g_ref.manual_seed(1234)
            dl = DataLoader(ds, batch_size=bs, shuffle=shuffle, num_workers=0,
                            drop_last=False, generator=g_ref)
            ref = [[tuple(b.view(-1).to(torch.long).tolist()) for (b,) in dl] for _ in range(2)]
            ref_tail = torch.randn(2, generator=g_ref).tolist()

            g_new = torch.Generator(device="cpu")
            g_new.manual_seed(1234)
            got = [[tuple(idx.tolist()) for (idx, _) in _epoch_index_batches(n, bs, shuffle, g_new)]
                   for _ in range(2)]
            got_tail = torch.randn(2, generator=g_new).tolist()

            ok = ok and (ref == got) and (ref_tail == got_tail)
    except Exception:
        ok = False

    _RANDPERM_BATCHING_OK = bool(ok)
    return _RANDPERM_BATCHING_OK

class MetVAE():
    """
    Variational autoencoder for untargeted metabolomics data with covariate adjustment.

    On construction the input is centered-log-ratio transformed, zeros are recorded as
    censored observations, and the measured covariates are regressed out on the CLR scale.

    Parameters
    ----------
    data : pd.DataFrame
        Input metabolomics data matrix. Should contain abundances of metabolites across samples.
        Can be organized with either samples or features as rows (see ``features_as_rows``).
    
    features_as_rows : bool, default=False
        Data orientation flag. Set to True if features (metabolites) are rows and samples are columns.
        The model transposes the data internally to a samples by features layout.

    meta : pd.DataFrame, optional
        Sample metadata containing covariate/confounder information. Must have the same sample index as ``data``.
        Used to adjust for experimental and biological confounding factors.
    
    continuous_covariate_keys : list[str], optional
        Names of continuous covariates in ``meta`` to adjust for (e.g., ``['age', 'bmi']``).
        These variables are included directly in the adjustment.
    
    categorical_covariate_keys : list[str], optional
        Names of categorical covariates in ``meta`` to adjust for (e.g., ``['sex', 'treatment']``).
        These are one-hot encoded automatically before adjustment.
    
    latent_dim : int, optional
        Dimension of the latent space. If None, defaults to ``min(n_samples, n_features)``
        after preprocessing. Larger values allow more complex structure but require more data.
    
    hidden_dims : list[int] or None, default=None
        Hidden layer sizes of the encoder networks, e.g. ``[256, 128]``.
        If ``None`` or an empty list, the encoders are linear. The decoder is always linear.

    activation : str | callable | None, default="relu"
        Nonlinearity used in the MLP(s). One of ``{"relu", "tanh", "gelu", "silu"}``, ``None`` for identity,
        or a zero-argument callable returning an ``nn.Module`` (e.g., ``lambda: nn.LeakyReLU(0.1)``).
    
    use_gpu : bool, default=False
        Whether to use GPU acceleration for model training and inference.
        Automatically falls back to CPU if CUDA is unavailable.
    
    logging : bool, default=False
        If True, logs training progress/metrics (e.g., to TensorBoard).
    
    dtype : torch.dtype, default=torch.float64
        Numeric dtype used for tensors in the model and preprocessed data.
        
    feature_zero_threshold: float
        Drop features with proportion of zeros > threshold (default 0.3)
        
    sample_zero_threshold: float, optional
        drop samples with proportion of zeros > threshold (default None: keep all)
    
    seed : int, default=0
        Random seed used during preprocessing/model initialization/training for reproducibility.
        Applied via ``torch.manual_seed(seed)`` (and ``torch.cuda.manual_seed_all(seed)`` when on CUDA).
    
    Attributes
    ----------
    model : VAE
        The underlying VAE model architecture.
    
    device : torch.device
        The device (CPU/GPU) where the model and data reside.
    
    sample_dim : int
        Number of samples after preprocessing.
    
    feature_dim : int
        Number of features (metabolites) after preprocessing.
    
    latent_dim : int
        Dimension of the VAE latent space. Defaults to ``min(sample_dim, feature_dim)``
        when not specified.
    
    clr_data : torch.Tensor
        CLR-transformed and covariate-adjusted data of shape (n_samples, n_features).
    
    num_zero : torch.Tensor
        Per-feature count of zeros in the original data (shape: (n_features,)).
    
    shift : torch.Tensor
        Per-feature log-scale offset/bias vector added when mapping back to log/original scales.
    
    clr_mean : torch.Tensor
        Estimated per-feature mean on the CLR scale (shape: (n_features,)).
    
    clr_sd : torch.Tensor
        Estimated per-feature standard deviation on the CLR scale (shape: (n_features,)).
    
    clr_coef : torch.Tensor
        Regression coefficients used for covariate/confounder adjustment on the CLR scale.
    
    sample_name : list[str]
        Names/IDs of samples after preprocessing.
    
    feature_name : list[str]
        Names/IDs of features (metabolites) after preprocessing.
    
    confound_name : list[str]
        Names of covariates/confounders used in the adjustment.
    
    corr_outputs : dict | None
        Storage for correlation analysis results (populated after calling correlation methods).
        Typically contains keys like ``'impute_log_data'`` and ``'estimate'`` (sparse correlation).
    
    train_loss : list[float]
        Per-epoch training losses recorded during ``train()``.

    optimizer : torch.optim.Optimizer or None
        Optimizer created by the most recent call to ``train()``.

    scheduler : torch.optim.lr_scheduler.LRScheduler or None
        Learning rate scheduler created by the most recent call to ``train()``.

    current_epoch : int
        Index of the epoch most recently completed by ``train()``; 0 before training.

    Examples
    --------
    >>> # Basic usage without covariates
    >>> model = MetVAE(data=metabolite_data, latent_dim=8)
    >>>
    >>> # With covariate adjustment
    >>> model = MetVAE(
    ...     data=metabolite_data,
    ...     meta=metadata,
    ...     continuous_covariate_keys=['age', 'bmi'],
    ...     categorical_covariate_keys=['sex', 'batch']
    ... )
    """
    
    def __init__(
            self,
            data: pd.DataFrame,
            features_as_rows: bool = False,
            meta: Optional[pd.DataFrame] = None,
            continuous_covariate_keys: Optional[List[str]] = None,
            categorical_covariate_keys: Optional[List[str]] = None,
            latent_dim: Optional[int] = None,
            hidden_dims: Optional[List[int]] = None,
            activation: Optional[str] = "relu",
            use_gpu: bool = False,
            logging: bool = False,
            dtype: torch.dtype = torch.float64,
            feature_zero_threshold: float = 0.3,
            sample_zero_threshold: Optional[float] = None,
            seed: int = 0
    ):
        """
        Preprocess the data, select the device, and build the VAE.
        """
        self.device = torch.device("cuda") if (use_gpu and torch.cuda.is_available()) else torch.device("cpu")
        if use_gpu and not torch.cuda.is_available():
            print("CUDA not available. Falling back to CPU.")
        self.logging = logging
        self.dtype = dtype
        
        self.base_seed = int(seed)
        torch.manual_seed(self.base_seed)
        if self.device.type == 'cuda':
            torch.cuda.manual_seed_all(self.base_seed)

        scaled_meta = _scale_continuous_metadata(meta, continuous_covariate_keys)
            
        # This handles CLR transformation, zero value processing, and covariate/confounder adjustment
        pp = _data_pre_process(
            data=data,
            features_as_rows=features_as_rows,
            meta=scaled_meta,
            continuous_covariate_keys=continuous_covariate_keys,
            categorical_covariate_keys=categorical_covariate_keys,
            device=self.device,
            dtype=dtype,
            feature_zero_threshold=feature_zero_threshold,
            sample_zero_threshold=sample_zero_threshold
            )
        
        # Unpack preprocessed data components
        self.meta = pp['meta']
        self.num_zero = pp['num_zero']
        self.shift = pp['shift']
        self.clr_data = pp['clr_data']
        self.clr_mean = pp['clr_mean']
        self.clr_sd = pp['clr_sd']
        self.clr_coef = pp['clr_coef']
        self.sample_name = pp['sample_name']
        self.feature_name = pp['feature_name']
        self.confound_name = pp['confound_name']
        
        # Shapes
        self.sample_dim = self.clr_data.shape[0]
        self.feature_dim = self.clr_data.shape[1]
        self.latent_dim = min(self.sample_dim, self.feature_dim) if latent_dim is None else latent_dim

        # Initialize the VAE model architecture
        self.model = VAE(
            input_dim=self.feature_dim,
            latent_dim=self.latent_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            dtype=dtype
        ).to(self.device)
        
        # Placeholders filled by get_corr and train.
        self.corr_outputs = None
        self.train_loss = []
        self.optimizer = None
        self.scheduler = None
        self.current_epoch = 0

    def train(
            self,
            batch_size: int = 128,
            num_workers: int = 0,
            max_epochs: int = 1000,
            learning_rate: float = 1e-3,
            max_grad_norm: float = 1.0,
            shuffle: bool = True,
            deterministic: bool = False,
            **trainer_kwargs
    ):
        """
        Train the VAE by mini-batch stochastic gradient descent.

        The optimizer is AdamW with zero weight decay and the learning rate follows a
        cosine annealing schedule with warm restarts (T_0 = 20, T_mult = 2,
        eta_min = learning_rate / 2). Batches containing censored zeros are completed by
        ``_random_initial`` before the loss is evaluated.

        Parameters
        ----------
        batch_size : int, default=128
            Number of samples per mini-batch.
        num_workers : int, default=0
            Number of data loading subprocesses. Zero loads batches in the main process.
        max_epochs : int, default=1000
            Number of passes through the training data.
        learning_rate : float, default=1e-3
            Initial learning rate of the optimizer.
        max_grad_norm : float, default=1.0
            Maximum gradient norm used for clipping. None disables clipping.
        shuffle : bool, default=True
            Whether the sample order is permuted at each epoch.
        deterministic : bool, default=False
            If True and running on CUDA, sets ``CUBLAS_WORKSPACE_CONFIG``, disables cuDNN
            benchmarking, and enables deterministic algorithms.
        **trainer_kwargs : dict
            Ignored. Passing any value raises a UserWarning.

        Returns
        -------
        None
            Per-epoch mean losses are appended to ``self.train_loss``.

        Raises
        ------
        RuntimeError
            If ``self.logging`` is True and tensorboard is not installed.
        """
        if trainer_kwargs:
            warnings.warn(
                "train() ignores the extra keyword arguments "
                f"{sorted(trainer_kwargs)}.",
                UserWarning,
                stacklevel=2,
            )

        run_seed = int(self.base_seed)

        torch.manual_seed(run_seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(run_seed)

        if deterministic and self.device.type == "cuda":
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True)

        dl_gen = torch.Generator(device="cpu")
        dl_gen.manual_seed(run_seed)

        def _worker_init_fn(worker_id: int):
            wseed = run_seed + worker_id
            random.seed(wseed)
            np.random.seed(wseed)
            torch.manual_seed(wseed)

        y_data = self.clr_data
        n = self.sample_dim

        has_zero = bool(torch.any(self.num_zero != 0))
        rows_with_nan = torch.isnan(y_data).any(dim=1)

        # Index batching reproduces the DataLoader draws; fall back if it does not.
        use_index_batching = (num_workers == 0) and _randperm_batching_matches_dataloader()
        dl = None
        if not use_index_batching:
            ds = TensorDataset(y_data)
            pin_memory_flag = (y_data.device.type == "cpu" and self.device.type == "cuda")
            dl = DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=pin_memory_flag,
                drop_last=False,
                generator=dl_gen,
                worker_init_fn=_worker_init_fn if num_workers > 0 else None,
                persistent_workers=(num_workers > 0)
            )

        optim = torch.optim.AdamW(self.model.parameters(),
                                  lr=learning_rate,
                                  weight_decay=0.0)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optim,
            T_0=20,
            T_mult=2,
            eta_min=learning_rate/2
            )

        self.train_loss = []
        self.optimizer = optim
        self.scheduler = scheduler

        writer = None
        if self.logging:
            try:
                from torch.utils.tensorboard import SummaryWriter
            except ImportError as exc:
                raise RuntimeError(
                    "logging=True requires tensorboard. Install it with `pip install tensorboard`."
                ) from exc
            os.makedirs("runs", exist_ok=True)
            existing = [d for d in os.listdir("runs") if d.startswith("run") and d[len("run"):].isdigit()]
            next_id = max([int(d[len("run"):]) for d in existing], default=-1) + 1
            writer = SummaryWriter(os.path.join("runs", f"run{next_id}"))

        try:
            self.model.train()
            for epoch in tqdm(range(1, max_epochs + 1)):
                running = torch.zeros((), dtype=torch.float64, device=self.device)
                num_batches = 0

                if use_index_batching:
                    batches = _epoch_index_batches(n, batch_size, shuffle, dl_gen)
                else:
                    batches = ((None, y_batch) for (y_batch,) in dl)

                for idx, y_batch in batches:
                    if idx is not None:
                        y_batch = y_data[idx]
                        batch_has_nan = bool(rows_with_nan[idx].any())
                    else:
                        batch_has_nan = bool(torch.isnan(y_batch).any())

                    if has_zero and batch_has_nan:
                        complete_y = _random_initial(
                            y=y_batch,
                            sample_size=n,
                            num_zero=self.num_zero,
                            mean=self.clr_mean,
                            sd=self.clr_sd,
                        )
                    else:
                        complete_y = y_batch

                    loss = self.model.training_step(complete_y)

                    optim.zero_grad(set_to_none=True)
                    loss.backward()
                    if max_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=max_grad_norm)
                    optim.step()

                    running += loss.detach()
                    num_batches += 1

                if num_batches > 0:
                    avg = running.item() / num_batches
                    self.train_loss.append(avg)
                    if writer is not None:
                        writer.add_scalar("Loss/train", avg, epoch)

                scheduler.step()
                self.current_epoch = epoch
        finally:
            if writer is not None:
                writer.flush()
                writer.close()

    def confound_coef(self):
        """
        Return the estimated covariate coefficients on the CLR scale.

        Returns
        -------
        pandas.DataFrame or None
            Coefficients with features as rows and covariates as columns, or None when no
            metadata was supplied.
        """
        if self.meta is not None:
            clr_coef = self.clr_coef.clone().detach().cpu()
            clr_coef = pd.DataFrame(
                clr_coef.numpy().T,
                index=self.feature_name,
                columns=self.confound_name
            )
        else:
            clr_coef = None
        return clr_coef

    def confound_es(self):
        """
        Return the fitted covariate effect for each sample and feature on the CLR scale.

        Returns
        -------
        pandas.DataFrame or None
            Product of the covariate matrix and the coefficients, with samples as rows and
            features as columns, or None when no metadata was supplied.
        """
        if self.meta is not None:
            clr_coef = self.confound_coef().values
            X = self.meta.clone().detach().cpu().numpy()
            clr_es = X @ clr_coef.T
            clr_es = pd.DataFrame(
                clr_es,
                index=self.sample_name,
                columns=self.feature_name
            )
        else:
            clr_es = None
        return clr_es

    def impute_zeros(
            self,
            *, 
            generator: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        """
        Impute the censored zeros of the preprocessed CLR data.

        Missing entries of ``self.clr_data`` are first drawn from the lower tail of a
        per-feature normal by ``_random_initial``, then replaced by the reconstruction of
        the VAE evaluated on the completed matrix. Observed entries are left unchanged.
        If no zeros were detected during preprocessing, ``self.clr_data`` is returned.

        Parameters
        ----------
        generator : torch.Generator, optional
            Generator used by the initialization and by the latent sampling. If None, the
            global torch RNG is used.

        Returns
        -------
        torch.Tensor
            Dense tensor of shape (n_samples, n_features) on the CLR scale, with the same
            device and dtype as ``self.clr_data``.
        """
        y = self.clr_data
        num_zero = self.num_zero
        clr_mean = self.clr_mean
        clr_sd = self.clr_sd
        n, d = y.shape

        was_training = self.model.training
        self.model.eval()
        try:
            if torch.any(num_zero != 0):
                nan_mask = torch.isnan(y)
                complete_y = _random_initial(
                    y=y,
                    sample_size=n,
                    num_zero=num_zero,
                    mean=clr_mean,
                    sd=clr_sd,
                    generator=generator)

                with torch.no_grad():
                    _, _, _, recon_y = self.model(
                        complete_y,
                        generator=generator
                    )

                impute_y = torch.where(nan_mask, recon_y, y)
            else:
                impute_y = y
        finally:
            self.model.train(was_training)

        return impute_y
    
    @torch.no_grad()
    def _single_imputation(
        self,
        seed: int,
        shift: torch.Tensor,
        threshold: float,
        device: torch.device
    ) -> tuple:
        """
        Run one imputation and its correlation estimate with a per-call generator.

        Parameters
        ----------
        seed : int
            Seed of the generator used by this simulation.
        shift : torch.Tensor
            Per-sample offset added to the CLR data to return to the log scale.
        threshold : float
            Absolute correlation cutoff.
        device : torch.device
            Device of the generator and of the computation.

        Returns
        -------
        impute_log_data : torch.Tensor
            Imputed data on the log scale, of shape (n, d).
        corr : torch.Tensor
            Dense correlation matrix of shape (d, d).
        """
        gen = torch.Generator(device=device)
        gen.manual_seed(int(seed))

        if device.type == "cuda":
            with torch.cuda.device(device):
                impute_clr_data = self.impute_zeros(generator=gen)
        else:
            impute_clr_data = self.impute_zeros(generator=gen)

        impute_log_data = impute_clr_data + shift
        corr = _compute_correlation_dense(torch.exp(impute_log_data), threshold=threshold)
        return impute_log_data, corr

    def _gpu_parallel_imputation(
        self,
        num_sim: int,
        shift: torch.Tensor,
        threshold: float,
        batch_size: int,
        device: torch.device,
        base_seed: int = 0
    ) -> tuple:
        """
        Average multiple imputations and their correlation estimates on CUDA.

        Parameters
        ----------
        num_sim : int
            Number of simulations.
        shift : torch.Tensor
            Per-sample offset added to the CLR data to return to the log scale.
        threshold : float
            Absolute correlation cutoff applied within each simulation.
        batch_size : int
            Number of simulations issued per round of CUDA streams.
        device : torch.device
            CUDA device used for the computation.
        base_seed : int, default=0
            Seed of simulation ``sim_id`` is ``base_seed + sim_id``.

        Returns
        -------
        impute_log_data_mean : torch.Tensor
            Mean imputed data on the log scale, of shape (n, d).
        Rn_mean_sparse : torch.Tensor
            Sparse COO mean correlation matrix of shape (d, d).

        Raises
        ------
        ValueError
            If num_sim < 1.
        """
        if num_sim < 1:
            raise ValueError("num_sim must be >= 1")
        num_batches = (num_sim + batch_size - 1) // batch_size

        acc = None
        impute_log_sum = None

        was_training = self.model.training
        try:
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, num_sim)
                current_batch_size = end_idx - start_idx

                batch_corrs, batch_impute_log = [], []
                streams = [torch.cuda.Stream(device=device) for _ in range(min(4, current_batch_size))]

                for i in range(current_batch_size):
                    sim_id = start_idx + i
                    stream = streams[i % len(streams)]

                    with torch.cuda.stream(stream):
                        impute_log, corr = self._single_imputation(
                            base_seed + sim_id, shift, threshold, device
                        )
                        batch_impute_log.append(impute_log)
                        batch_corrs.append(corr)

                for s in streams:
                    s.synchronize()

                for impute_log in batch_impute_log:
                    impute_log_sum = impute_log.clone() if impute_log_sum is None else (impute_log_sum + impute_log)

                for corr in batch_corrs:
                    acc = corr.clone() if acc is None else acc.add_(corr)

                del batch_corrs, batch_impute_log
        finally:
            self.model.train(was_training)

        impute_log_data_mean = impute_log_sum / float(num_sim)
        Rn_mean_sparse = (acc / float(num_sim)).to_sparse_coo()

        return impute_log_data_mean, Rn_mean_sparse

    def _cpu_multiple_imputation(
            self,
            num_sim: int,
            shift: torch.Tensor,
            threshold: float,
            workers: Optional[int],
            base_seed: int = 0
    ) -> tuple:
        """
        Average multiple imputations and their correlation estimates on CPU.

        Simulations run sequentially inside a block that fixes the number of intra-op
        torch threads, so the reduction order is the simulation order and the result does
        not depend on ``workers``.

        Parameters
        ----------
        num_sim : int
            Number of simulations.
        shift : torch.Tensor
            Per-sample offset added to the CLR data to return to the log scale.
        threshold : float
            Absolute correlation cutoff applied within each simulation.
        workers : int or None
            Number of intra-op torch threads. None or a negative value uses all cores.
        base_seed : int, default=0
            Seed of simulation ``sim_id`` is ``base_seed + sim_id``.

        Returns
        -------
        impute_log_data_mean : torch.Tensor
            Mean imputed data on the log scale, of shape (n, d).
        Rn_mean_sparse : torch.Tensor
            Sparse COO mean correlation matrix of shape (d, d).

        Raises
        ------
        ValueError
            If num_sim < 1.
        """
        if num_sim < 1:
            raise ValueError("num_sim must be >= 1")

        if workers is None or int(workers) < 0:
            nthreads = os.cpu_count() or 1
        else:
            nthreads = max(1, int(workers))

        d = self.feature_dim
        dtype, device = self.clr_data.dtype, self.clr_data.device

        was_training = self.model.training
        with _thread_limit(nthreads), torch.no_grad():
            try:
                acc = torch.zeros((d, d), dtype=dtype, device=device)
                G = torch.empty((d, d), dtype=dtype, device=device)
                T = torch.empty((d, d), dtype=dtype, device=device)
                ilog = None

                for sim_id in range(num_sim):
                    gen = torch.Generator(device="cpu")
                    gen.manual_seed(base_seed + sim_id)
                    il = self.impute_zeros(generator=gen) + shift
                    _compute_correlation_dense(torch.exp(il), threshold=threshold, out=G, scratch=T)
                    acc.add_(G)
                    if ilog is None:
                        ilog = il
                    else:
                        ilog.add_(il)
            finally:
                self.model.train(was_training)

        del G, T
        return ilog / num_sim, (acc / num_sim).to_sparse_coo()

    def get_corr(
        self,
        num_sim: int = 100,
        workers: int = -1,
        batch_size: int = 100,   # For GPU batching
        threshold: float = 0.2,  # Correlation threshold for sparsity
        seed: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Estimate the correlation matrix from multiple imputations and cache the result.

        Each simulation imputes the censored zeros, returns to the log scale, computes the
        correlation matrix, and hard-thresholds it. The reported estimate is the mean of
        the thresholded matrices, and ``impute_log_data`` is the mean of the imputed log
        data.

        Parameters
        ----------
        num_sim : int, default=100
            Number of imputation simulations to run and average.

        workers : int, default=-1
            Number of intra-op torch threads used by the CPU path; ``-1`` uses all cores.
            The thread count is restored on return and the result does not depend on it.
            Ignored on CUDA.

        batch_size : int, default=100
            Number of simulations issued per round of CUDA streams on the GPU path.

        threshold : float, default=0.2
            Absolute correlation cutoff applied within each simulation. Entries with
            ``|r| < threshold`` are set to zero before averaging.

        seed : int or None, default=None
            Base seed. Simulation ``sim_id`` uses ``seed + sim_id``. If None, the seed
            given to the constructor is used.

        Returns
        -------
        dict
            Dictionary with ``'impute_log_data'``, a dense tensor of shape (n, d), and
            ``'estimate'``, a sparse COO tensor of shape (d, d).
        """
        shift = self.shift
        num_zero = self.num_zero
        device = self.device

        base_seed = int(seed) if seed is not None else self.base_seed

        if torch.any(num_zero != 0):
            if device.type == 'cuda':
                impute_log_data_mean, Rn_mean = self._gpu_parallel_imputation(
                    num_sim, shift, threshold, batch_size, device, base_seed=base_seed
                )
            else:
                impute_log_data_mean, Rn_mean = self._cpu_multiple_imputation(
                    num_sim, shift, threshold, workers, base_seed=base_seed
                )
        else:
            impute_clr_data = self.impute_zeros()
            impute_log_data_mean = impute_clr_data + shift
            impute_data = torch.exp(impute_log_data_mean)
            Rn_mean = _compute_correlation(data=impute_data,
                                           threshold=threshold)

        outputs = {
            'impute_log_data': impute_log_data_mean,
            'estimate': Rn_mean
        }
        self.corr_outputs = outputs
        return outputs

    def sparse_by_p(
            self,
            p_adj_method: Literal['bonferroni', 'sidak', 
                                  'holm-sidak', 'holm', 
                                  'simes-hochberg', 'hommel', 
                                  'fdr_bh', 'fdr_by', 
                                  'fdr_tsbh', 'fdr_tsbky'] = 'fdr_bh',
            cutoff: float = 0.05
    ):
        """
        Sparsify the correlation estimate by a Fisher z-test with multiplicity control.

        The correlations are clamped to (-1, 1), transformed by
        z = 0.5 * (log1p(r) - log1p(-r)), divided by the standard error 1 / sqrt(n - 3),
        converted to two-sided normal p-values, and adjusted for multiple testing.
        Correlations whose adjusted p-value exceeds the cutoff are set to zero.

        Parameters
        ----------
        p_adj_method : str, default='fdr_bh'
            Multiple testing correction passed to ``_matrix_p_adjust``.
        cutoff : float, default=0.05
            Adjusted p-value threshold.

        Returns
        -------
        dict
            Dictionary with ``'estimate'``, ``'p_value'``, ``'q_value'`` and
            ``'sparse_estimate'``, each a dense (d, d) float64 pandas.DataFrame indexed by
            feature name. All four matrices are held in memory simultaneously.

        Raises
        ------
        ValueError
            If ``get_corr`` has not been called, or if the sample size is not greater
            than 3.
        """
        if self.corr_outputs is None:
            raise ValueError("No correlation estimates. Please compute correlations the first using get_corr method.")
        if getattr(self, "sample_dim", None) is None or self.sample_dim <= 3:
            raise ValueError("Sample size must be > 3 for Fisher's z-test.")

        Rn = self.corr_outputs['estimate']
        n = self.sample_dim
        feature_names = self.feature_name

        Rn = Rn.to_dense() if Rn.is_sparse else Rn.clone()
        Rn.clamp_(min=-1.0 + 1e-7, max=1.0 - 1e-7)
        Rn.fill_diagonal_(1.0)
        device, dtype = Rn.device, Rn.dtype

        z = 0.5 * (torch.log1p(Rn) - torch.log1p(-Rn))
        se = 1.0 / torch.sqrt(torch.tensor(float(n-3), dtype=dtype, device=device))
        z_score = z / se
        del z
        z_score.fill_diagonal_(0.0)

        p_val = 2.0 * (1.0 - torch.special.ndtr(torch.abs(z_score)))
        del z_score
        p_val.fill_diagonal_(0.0)

        q_val = _matrix_p_adjust(p_val, method=p_adj_method)
        Rn_hat = _p_filter(Rn, q_val, max_p = cutoff, impute_value = 0)

        outputs = {
            'estimate' : _torch_to_df(Rn, names=feature_names),
            'p_value' : _torch_to_df(p_val, names=feature_names),
            'q_value' : _torch_to_df(q_val, names=feature_names),
            'sparse_estimate' : _torch_to_df(Rn_hat, names=feature_names),
            }
        
        return outputs
    
    def sparse_by_sec(
        self,
        rho: Optional[float] = None,
        *,
        # SEC solver hyperparameters
        epsilon: float = 1e-5,
        tol: float = 1e-3,
        max_iter: int = 1000,
        restart: Optional[int] = 50,
        line_search_apg: bool = True,
        delta: Optional[float] = None,
        n_samples: Optional[int] = None,
        c_delta: float = 0.1,
        threshold: float = 0.1,
        # CV settings (automatic rho selection)
        c_grid: Sequence[float] = tuple(float(x) for x in range(1, 11)),  # 1.0, 2.0, ..., 10.0 (coarse)
        n_splits: int = 5,
        seed: int = 0,
        workers: int = -1,          # CPU: parallel across rho; GPU/single worker: sequential
        refine: bool = True,        # single zoom after coarse pass
        refine_points: int = 10     # number of points in the refined bracket (inclusive)
    ):
        """
        Sparsify the correlation estimate with the sparse estimation of correlation algorithm.

        A fixed `rho` runs a single fit. When `rho` is None it is selected by K-fold
        cross-validation over the candidates `rho = c * sqrt(log(p)/n)` for `c` in `c_grid`,
        scored by the mean validation squared Frobenius error, with ties resolved in favor of
        the smaller `rho`. With `refine=True` a single further pass evaluates
        `refine_points` evenly spaced values between the neighbors of the best coarse `c`.

        Parameters
        ----------
        rho : float, optional
            Fixed L1 penalty. If None, selected by cross-validation.

        epsilon : float, default=1e-5
            Eigenvalue floor for PSD projection during calibration.
    
        tol : float, default=1e-3
            Convergence tolerance for APG iterations.
    
        max_iter : int, default=1000
            Maximum APG iterations.
    
        restart : int or None, default=50
            Nesterov restart period (iterations). Set None to disable.
    
        line_search_apg : bool, default=True
            Enable backtracking line search for adaptive step sizes.
    
        delta : float, optional
            Small-correlation threshold inside the optimization. If None, set to
            `c_delta * sqrt(log(p)/n_samples)`.
    
        n_samples : int, optional
            Required only if `delta` is None to compute it automatically.
    
        c_delta : float, default=0.1
            Scaling constant when auto-setting `delta`.
    
        threshold : float, default=0.1
            Hard threshold on the final SEC estimate; entries with |value| < threshold
            are set to zero (diagonal preserved).
    
        c_grid : sequence of float, default=(1.0, 2.0, ..., 10.0)
            Coarse grid of `c` values used to generate candidate penalties via
            `rho = c * sqrt(log(p)/n)`.
    
        n_splits : int, default=5
            Number of folds for K-fold CV.
    
        seed : int, default=0
            Random seed for deterministic fold assignment.
    
        workers : int, default=-1
            CPU thread parallelism for CV. `-1` uses all cores. On GPU or when
            `workers <= 1`, runs sequentially.
    
        refine : bool, default=True
            If True, perform one zoom-in refinement between the best coarse `c` and its
            immediate neighbors after the coarse pass.
    
        refine_points : int, default=10
            Number of evenly spaced points (inclusive) to evaluate within the refinement
            bracket when `refine=True`.
    
        Returns
        -------
        outputs : dict
            - 'estimate' : pandas.DataFrame
                Dense empirical correlation matrix (pre-SEC).
            - 'sparse_estimate' : pandas.DataFrame
                Final dense SEC estimate after sparsification/thresholding.
            - 'best_rho' : float or None
                Selected penalty parameter; equals provided `rho` if given, else the CV choice.
            - 'scores_by_rho' : dict or None
                `{rho: mean_validation_frobenius_error}` for all evaluated candidates;
                None if a fixed `rho` was supplied.
    
        Raises
        ------
        ValueError
            If `self.corr_outputs` is missing. Call `get_corr()` first.
    
        Notes
        -----
        Cross-validation scores the mean squared Frobenius error between the fit on the
        training folds and the empirical correlation on the validation fold. The refinement
        reuses the same folds and the cached validation correlations. The returned
        DataFrames hold two dense d by d float64 matrices.
        """
        if self.corr_outputs is None:
            raise ValueError("No correlation estimates. Please compute correlations first using `get_corr`.")

        impute_log_data = self.corr_outputs['impute_log_data']
        Rn = self.corr_outputs['estimate']
        feature_names = self.feature_name
        impute_data = torch.exp(impute_log_data)
        n = self.sample_dim

        Rn_dense = Rn.to_dense() if Rn.is_sparse else Rn

        best_rho = None
        scores_by_rho = None
        R_hat_dense = None

        if rho is not None:
            R_hat_dense = _SEC_dense(
                Rn=Rn_dense, rho=rho,
                epsilon=epsilon, tol=tol, max_iter=max_iter, restart=restart,
                line_search_apg=line_search_apg, delta=delta, n_samples=n,
                c_delta=c_delta, threshold=threshold
            )
            best_rho = rho
            scores_by_rho = None
        else:
            best_rho, scores_by_rho, R_hat_dense = _SEC_cv(
                X=impute_data,
                Rn=Rn_dense,
                c_grid=c_grid,
                n_splits=n_splits,
                seed=seed if seed is not None else self.base_seed,
                workers=workers,
                refine=refine,
                refine_points=refine_points,
                epsilon=epsilon, tol=tol, max_iter=max_iter, restart=restart,
                line_search_apg=line_search_apg, delta=delta, n_samples=n,
                c_delta=c_delta, threshold=threshold
            )

        outputs = {
            'estimate': _torch_to_df(Rn_dense, names=feature_names),
            'sparse_estimate': _torch_to_df(R_hat_dense, names=feature_names),
            'best_rho': best_rho,
            'scores_by_rho': scores_by_rho,
        }
        return outputs
    
    def export_graphml(
        self,
        sparse_df: pd.DataFrame,
        cutoffs: Iterable[float],
        output_dir: Optional[str] = None,
        file_prefix: str = "correlation_graph_cutoff",
    ):
        """
        Build one undirected graph per absolute correlation cutoff, optionally as GraphML.

        Parameters
        ----------
        sparse_df : pandas.DataFrame
            Square correlation matrix, typically ``sparse_by_p()["sparse_estimate"]``.
        cutoffs : iterable of float
            Absolute correlation cutoffs. Non-positive values are dropped.
        output_dir : str, optional
            Directory receiving the ``.graphml`` files. If None, nothing is written.
        file_prefix : str, default="correlation_graph_cutoff"
            Filename prefix; the cutoff and the ``.graphml`` extension are appended.

        Returns
        -------
        dict
            Mapping from ``f"Correlation_cutoff{cutoff:g}"`` to ``networkx.Graph``,
            containing only the cutoffs that yielded at least one edge.

        Raises
        ------
        RuntimeError
            If networkx is not installed.
        ValueError
            If sparse_df is not a pandas.DataFrame.
        """
        try:
            import networkx as nx
        except ImportError as exc:
            raise RuntimeError(
                "networkx is required for GraphML export. Install it with `pip install networkx`."
            ) from exc

        if sparse_df is None or not isinstance(sparse_df, pd.DataFrame):
            raise ValueError("sparse_df must be a pandas DataFrame with the sparse correlation matrix.")

        cutoffs = sorted({float(c) for c in cutoffs if float(c) > 0.0}, reverse=True)

        graphs: Dict[str, nx.Graph] = {}
        if not cutoffs:
            return graphs

        # Strict upper triangle of the matrix as an edge list, keeping only usable edges.
        names = [str(x) for x in sparse_df.index]
        values = sparse_df.to_numpy()
        iu, ju = np.triu_indices(values.shape[0], k=1)
        corr = values[iu, ju].astype(float, copy=False)
        keep = ~np.isnan(corr)
        keep &= np.abs(corr) >= min(cutoffs)
        kept = [float(v) for v in corr[keep]]
        df_long = pd.DataFrame({
            "node1": [names[i] for i in iu[keep]],
            "node2": [names[j] for j in ju[keep]],
            "correlation": pd.Series(kept, dtype=object),
            "abs_corr": np.abs(corr[keep]),
        })

        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)

        for cutoff in cutoffs:
            edge_type = f"Correlation_cutoff{cutoff:g}"
            sub = df_long.loc[df_long["abs_corr"] >= cutoff].copy()

            if sub.empty:
                continue

            sub["weight"] = sub["correlation"]
            sub["EdgeScore"] = sub["correlation"]
            sub["EdgeType"] = edge_type
            sub["id"] = edge_type
            G = nx.from_pandas_edgelist(
                sub,
                source="node1",
                target="node2",
                edge_attr=["weight", "correlation", "EdgeScore", "EdgeType", "id"],
                create_using=nx.Graph,
            )

            graphs[edge_type] = G

            if output_dir is not None:
                filename = f"{file_prefix}{cutoff:g}.graphml"
                path = os.path.join(output_dir, filename)
                nx.write_graphml(G, path)

        return graphs
    
    def clr_loading(self):
        """
        Return the decoder weight matrix as feature loadings on the CLR scale.

        Returns
        -------
        pandas.DataFrame
            Loadings with features as rows and latent dimensions as columns, the latter
            named ``latent_0``, ``latent_1``, and so on.
        """
        clr_loading = self.model.decode_mu.weight.clone().detach().cpu().numpy()
        clr_loading = pd.DataFrame(
            clr_loading,
            index=self.feature_name,
            columns=["latent_{}".format(i) for i in range(self.latent_dim)]
        )
        return clr_loading

    def cooccurrence(self):
        """
        Return the model-implied variance of pairwise log ratios between features.

        Returns
        -------
        pandas.DataFrame
            Symmetric matrix indexed by feature name, equal to the squared Euclidean
            distances between the feature loadings scaled by (n - 1) / n. Larger values
            indicate features that vary more independently.
        """
        clr_loading = self.clr_loading().values
        cooccur = squareform(pdist(clr_loading)) ** 2 * (self.sample_dim - 1) / self.sample_dim
        cooccur = pd.DataFrame(
            cooccur,
            index=self.feature_name,
            columns=self.feature_name
        )
        return cooccur

# Benchmarking function to compute correlations without VAE-based zero imputation
def _simple_inference(
        data: pd.DataFrame,
        features_as_rows: bool = False,
        meta: Optional[pd.DataFrame] = None,
        continuous_covariate_keys: Optional[List[str]] = None,
        categorical_covariate_keys: Optional[List[str]] = None,
        num_sim: int = 100,
        threshold: float = 0.2,   # kept for p_value branch only
        sparse_method: Literal['pval', 'sec'] = 'pval',
        p_adj_method: Literal['bonferroni','sidak','holm-sidak','holm','simes-hochberg','hommel','fdr_bh','fdr_by','fdr_tsbh','fdr_tsbky'] = 'fdr_bh',
        cutoff: float = 0.05,
        rho: float = 1,
        seed: Optional[int] = 0):
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    pp = _data_pre_process(
        data=data,
        features_as_rows=features_as_rows,
        meta=meta,
        continuous_covariate_keys=continuous_covariate_keys,
        categorical_covariate_keys=categorical_covariate_keys,
        device=device
        )
    
    meta = pp['meta']
    num_zero = pp['num_zero']
    shift = pp['shift']
    clr_data = pp['clr_data']
    clr_mean = pp['clr_mean']
    clr_sd = pp['clr_sd']
    feature_names = pp['feature_name']
    n, d = clr_data.shape
    dtype = clr_data.dtype

    # Impute missing values and compute correlations
    impute_log_sum = torch.zeros_like(clr_data, dtype=dtype)
    Rn_sum = torch.zeros((d, d), device=device, dtype=dtype)
    
    for x in range(num_sim):
        if seed is not None:
            torch.manual_seed(seed + x)
            np.random.seed(seed + x)
        
        if torch.any(num_zero != 0):
            impute_clr_data = _random_initial(
                y=clr_data, 
                sample_size=n, 
                num_zero=num_zero, 
                mean=clr_mean, 
                sd=clr_sd)
        else:
            impute_clr_data = clr_data
        
        impute_log_data = impute_clr_data + shift
        impute_data = torch.exp(impute_log_data)
        rho_k = _compute_correlation_dense(data=impute_data, threshold=threshold)
        
        impute_log_sum += torch.nan_to_num(impute_log_data, nan=0.0).to(dtype)
        Rn_sum += torch.nan_to_num(rho_k, nan=0.0).to(dtype)

    # Averages
    impute_log_mean = (impute_log_sum / float(num_sim)).to(dtype)
    impute_mean = torch.exp(impute_log_mean)
    Rn = (Rn_sum / float(num_sim)).to(dtype)

    if sparse_method == 'pval':
        # Fisher z-transformation of the correlations.
        Rn_clamped = Rn.clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        z = 0.5 * (torch.log1p(Rn_clamped) - torch.log1p(-Rn_clamped))
        
        se = 1.0 / torch.sqrt(torch.tensor(float(n - 3), device=device, dtype=Rn.dtype))
        z_score = z / se
        z_score.fill_diagonal_(0.0)
        
        p_value = 2.0 * (1.0 - torch.special.ndtr(torch.abs(z_score)))
        p_value.fill_diagonal_(0.0)  
        
        q_value = _matrix_p_adjust(p_value, method=p_adj_method)
        
        Rn_hat = _p_filter(Rn, q_value, max_p=cutoff, impute_value=0)
    else:
        Rn_hat = _SEC_dense(Rn=Rn, rho=rho)

    Rn_hat_dense = Rn_hat

    # Outputs
    outputs = {
        'estimate': _torch_to_df(Rn, names=feature_names), 
        'sparse_estimate': _torch_to_df(Rn_hat_dense, names=feature_names)
        }

    return outputs
