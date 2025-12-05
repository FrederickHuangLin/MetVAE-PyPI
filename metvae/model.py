import os
import warnings
from typing import Optional, Dict, List, Literal, Sequence
import random
import math
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from joblib import Parallel, delayed, parallel_backend
from .vae import VAE
from .utils import _make_valid_column_name, _torch_to_df
from .impute_missing import _ols_estimate, _fit_censored_normal
from .compute_corr import _compute_correlation
from .sparse import _matrix_p_adjust, _p_filter, _SEC, _SEC_cv

def _data_pre_process(
        data: pd.DataFrame,
        features_as_rows: bool = False,
        meta: Optional[pd.DataFrame] = None,
        continuous_covariate_keys: Optional[List[str]] = None,
        categorical_covariate_keys: Optional[List[str]] = None,
        device: Optional[str] = None,
        dtype: torch.dtype = torch.float64
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
    The function performs several key steps:
    1. Data validation and organization
    2. Zero handling and CLR transformation
    3. Parameter estimation with censoring for zero values
    4. Covariate adjustment if metadata is provided
    5. Conversion to PyTorch tensors for downstream analysis
    """
    # --- Input Validation and Data Organization --- 
    device = torch.device(device)
    
    if not isinstance(data, pd.DataFrame):
        raise TypeError('The input data must be a pandas.DataFrame')
     
    # Ensure features are columns for consistent processing
    if features_as_rows:
        data = data.T
    sample_name = data.index.tolist()
    feature_name = data.columns.tolist()
    data = torch.tensor(data.values, dtype=dtype, device=device)
    data = torch.nan_to_num(data, nan=0.0) # treats NaNs as zeros
    
    # Check for and fix negative values
    neg_mask = data < 0
    num_neg = int(neg_mask.sum().item())
    
    if num_neg > 0:
        warnings.warn(
            f"The dataset contains {num_neg} negative values. "
            "They have been converted to zeros, but please double-check "
            "that this preprocessing step is appropriate for your data."
        )
        data = torch.where(neg_mask, torch.zeros_like(data), data)
    
    # Discard rows that are all-zero-or-NaN
    row_all_zero = (data == 0).all(dim=1)             # (n,)
    if row_all_zero.any():
        # filter data and metadata/samplenames consistently
        keep_idx = ~row_all_zero
        # keep sample_name list in sync
        sample_name = [s for i, s in enumerate(sample_name) if keep_idx[i].item()]
        data = data[keep_idx]
        if meta is not None:
            # if smd still a pandas df at this point, do the same selection there
            # otherwise if it's torch already:
            meta = meta.loc[keep_idx.cpu().numpy(), :]
    n, d = data.shape

    # Count zeros in each feature for handling censored values
    num_zero = (data == 0).sum(dim=0).float()
    
    # --- Metadata Processing and Validation --- 
    if meta is not None and not isinstance(meta, pd.DataFrame):
        raise TypeError('The meta data must be a pandas.DataFrame or None')

    # Ensure all samples in abundance data have corresponding metadata
    if isinstance(meta, pd.DataFrame):
        if not set(sample_name).issubset(meta.index):
            missing = set(sample_name) - set(meta.index)
            raise ValueError(f"The following sample names are missing in the sample meta data: {missing}")

    # Process continuous and categorical covariates from metadata
    if isinstance(meta, pd.DataFrame):
        # Handle continuous covariates
        smd_cont = meta.loc[:, continuous_covariate_keys] if continuous_covariate_keys is not None else None
        
        # Process categorical covariates with dummy encoding
        smd_cat = None
        if categorical_covariate_keys is not None:
            smd_cat = meta.loc[:, categorical_covariate_keys].apply(lambda x: x.astype('category'))
            smd_cat = pd.get_dummies(smd_cat, drop_first=True, dtype=float) # One-hot encoding
            smd_cat.columns = [_make_valid_column_name(c) for c in smd_cat.columns] # Ensure valid column names
    
        # Combine processed covariates
        if smd_cont is not None and smd_cat is not None:
            smd = pd.concat([smd_cont, smd_cat], axis=1)
            confound_name = smd.columns.tolist()
        elif smd_cont is not None:
            smd = smd_cont
            confound_name = smd_cont.columns.tolist()
        elif smd_cat is not None:
            smd = smd_cat
            confound_name = smd_cat.columns.tolist()
        else:
            smd = None
            confound_name = None
    else:
        smd = None
        confound_name = None
    
    if smd is not None:
        smd = torch.tensor(smd.values, dtype=dtype, device=device)
        p = smd.shape[1] + 1
    else:
        confound_name = None
        p = 1

    # --- CLR Transformation --- 
    # Handle zeros by converting to NaN after log transform
    log_data = torch.where(
        data > 0, torch.log(data), 
        torch.tensor(float('nan'), dtype=data.dtype, device=device)
        )
    
    # Calculate geometric mean (shift) for CLR transformation
    shift = torch.nanmean(log_data, dim=1, keepdim=True)
    clr_data = log_data - shift

    # Calculate threshold values for censored observations (raw -> per-feature min positive)
    th_raw = torch.where(data > 0, data, torch.tensor(float('inf'), dtype=data.dtype, device=device)).min(dim=0).values
    th_raw = torch.where(th_raw == float('inf'), torch.tensor(1e-5, dtype=data.dtype, device=device), th_raw)
    
    # Convert thresholds to CLR space
    clr_th = torch.log(th_raw) - shift

    # --- Parameter Estimation --- 
    init_params = _ols_estimate(clr_data, smd)  # (p+1, d)

    # --- Handle Zero Values and Estimate Final Parameters --- 
    if torch.any(num_zero != 0):
        # Use censored normal MLE for features with zeros
        clr_params = _fit_censored_normal(
            clr_data, smd, clr_th,
            init_estimates=init_params,
            max_iter=100,
        )
        clr_log_sd = clr_params[p, :]
        clr_sd = torch.exp(clr_log_sd)
        clr_mean = clr_params[0, :]
    else:
        # Use initial estimates if no zeros present
        clr_params = init_params
        clr_log_sd = clr_params[p, :]
        clr_sd = torch.exp(clr_log_sd)
        clr_mean = clr_params[0, :]

    # Part 6: Deconfound Data if Metadata Present
    if smd is not None:
        clr_coef = clr_params[1:p, :]
        clr_data -= smd @ clr_coef
    else:
        clr_coef = None
    
    # Part 7: Prepare Outputs
    outputs = {'clr_data': clr_data, 'meta': smd, 'num_zero': num_zero, 'shift': shift,
               'clr_mean': clr_mean, 'clr_sd': clr_sd, 'clr_coef': clr_coef, 
               'sample_name': sample_name, 'feature_name': feature_name, 'confound_name': confound_name}
    
    return outputs

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
    Initialize missing values (NaN) in CLR-transformed compositional data using random sampling.
    This internal function handles the initialization of censored zero values by generating 
    random values from a normal distribution and strategically assigning them to NaN positions.
    
    Parameters
    ----------
    y : torch.Tensor
        Input tensor with shape (batch_size, feature_size) containing CLR-transformed data.
        NaN values in this tensor represent censored zeros from the original data.
    sample_size : int
        Number of random samples to generate per feature for selecting initialization values.
        A larger sample size provides more candidates for initialization.
    num_zero : torch.Tensor
        Number of zeros per feature in the original data. Shape: (feature_size,)
    mean : torch.Tensor
        Estimated means for each feature in CLR space. Shape: (feature_size,)
    sd : torch.Tensor
        Estimated standard deviations for each feature in CLR space. Shape: (feature_size,)
    generator : Optional[torch.Generator], default=None
        Pseudorandom number generator used by this function’s sampling steps.
        If None, PyTorch’s global RNG state is used.
    
    Returns
    -------
    torch.Tensor
        A complete tensor of same shape as input 'y' where all NaN values have been 
        replaced with appropriate random initializations.
    
    Notes
    -----
    The function works in three main steps:
    1. Generates random values from a normal distribution for each feature
    2. Selects the smallest values as candidates for zero replacement
    3. Randomly assigns these candidates to NaN positions in the data
    """
    device, dtype = y.device, y.dtype
    batch_size, feature_size = y.shape
    
    # Count number of NaN values (censored zeros) for each feature
    nan_mask = torch.isnan(y)                                 # (n, d)
    num_nan  = nan_mask.sum(dim=0)                            # (d,)
    
    # Nothing to fill
    if (num_nan == 0).all():
        return y.clone()
            
    # Generate random samples from normal distribution using feature-specific parameters
    # Shape: (sample_size, feature_size)
    rand = torch.randn(sample_size, feature_size, device=device, dtype=dtype, generator=generator)
    random_data = rand * sd + mean

    # Sort ascending once per feature to get the "smallest" values quickly: (sample_size, feature_size)
    vals_sorted, _ = torch.sort(random_data, dim=0)
    
    # How many smallest candidates to consider per feature:
    # clamp to [1, sample_size]; if num_zero[j] == 0, still take at least 1 candidate
    k0 = num_zero.to(dtype=torch.long)
    k = torch.clamp(torch.where(k0 > 0, k0, torch.ones_like(k0)), min=1, max=sample_size)  # (feature_size,)
    k_max = int(k.max().item())
    
    # Take the first k_max candidates per feature (others unused for columns with k_j < k_max)
    candidates = vals_sorted[:k_max, :]                       # (k_max, feature_size)
    
    # We need m_j = num_nan[j] draws per feature, with replacement from [0 .. k_j-1].
    m = num_nan.to(dtype=torch.long)                          # (feature_size,)
    m_max = int(m.max().item())
    
    # Build a (m_max, d) matrix of indices ~ Uniform{0, …, k_j-1} using vector bounds:
    # Use rand in [0,1), scale by k, floor, cast to long.
    if m_max > 0:
        r = (torch.rand(m_max, feature_size, device=device, generator=generator) * k.view(1, -1)).floor().to(torch.long)  # (m_max, feature_size)
        # Gather chosen fills: (m_max, feature_size)
        fills_full = torch.gather(candidates, dim=0, index=r)
    else:
        fills_full = torch.empty(0, feature_size, device=device, dtype=dtype)

    complete = y.clone()
    
    # Cheap per-column scatter into NaN positions (variable counts per column)
    for j in range(feature_size):
        mj = int(m[j].item())
        if mj == 0:
            continue
        idx_nan_j = nan_mask[:, j].nonzero(as_tuple=False).squeeze(1)  # (mj,)
        # take the first mj draws in column j
        complete[idx_nan_j, j] = fills_full[:mj, j]

    return complete

class MetVAE():
    """
    Variational Autoencoder (VAE) specifically designed for untargeted metabolomics data analysis with covariate/confounder handling.
    
    This class implements a specialized VAE that accounts for the unique characteristics of metabolomics data,
    including compositionality, zero values, and the influence of covariates/confounders. The model performs
    several key preprocessing steps before training:
    1. Centered log-ratio (CLR) transformation to handle compositional data
    2. Careful handling of zero values through censored estimation and multiple imputation
    3. Covariate/confounder adjustment to remove unwanted variation
    
    Parameters
    ----------
    data : pd.DataFrame
        Input metabolomics data matrix. Should contain abundances of metabolites across samples.
        Can be organized with either samples or features as rows (see ``features_as_rows``).
    
    features_as_rows : bool, default=False
        Data orientation flag. Set to True if features (metabolites) are rows and samples are columns.
        The model will transpose the data internally to maintain a consistent samples × features format.
    
    meta : pd.DataFrame, optional
        Sample metadata containing covariate/confounder information. Must have the same sample index as ``data``.
        Used to adjust for experimental and biological confounding factors.
    
    continuous_covariate_keys : list[str], optional
        Names of continuous covariates in ``meta`` to adjust for (e.g., ``['age', 'bmi']``).
        These variables are included directly in the adjustment.
    
    categorical_covariate_keys : list[str], optional
        Names of categorical covariates in ``meta`` to adjust for (e.g., ``['sex', 'treatment']``).
        These are one-hot encoded automatically before adjustment.
    
    latent_dim : int, default=10
        Dimension of the latent space. Larger values allow more complex structure but require more data.
    
    hidden_dims : list[int] or None, default=None
        Hidden layer sizes for the encoder (and, if applicable, decoder) MLP(s), e.g. ``[256, 128]``.
        If ``None`` or an empty list, the encoder/decoder are linear (no hidden layers).
    
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
        Dimension of the VAE latent space (echoes the constructor argument).
    
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
    
    Notes
    -----
    The model performs several important preprocessing steps automatically:
    - CLR transformation to handle the compositional nature of metabolomics data.
    - Zero-value handling via a censored-normal strategy with deterministic or stochastic imputation.
    - Covariate/confounder adjustment to remove unwanted technical/biological variation.
    - Optional logging for monitoring training progress and convergence.
    
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
            latent_dim: int = 10,
            hidden_dims: Optional[List[int]] = None,
            activation: Optional[str] = "relu",
            use_gpu: bool = False,
            logging: bool = False,
            dtype: torch.dtype = torch.float64,
            seed: int = 0
    ):
        """
        Initialize the MetVAE model with data preprocessing and model setup.
        
        This initialization process includes:
        1. Data preprocessing (CLR transformation, zero handling)
        2. Covariate/confounder processing and adjustment
        3. GPU/CPU device selection
        4. Model architecture setup
        """
        # Preprocess input data using internal utility function
        
        # Device
        self.device = torch.device("cuda") if (use_gpu and torch.cuda.is_available()) else torch.device("cpu")
        if use_gpu and not torch.cuda.is_available():
            print("CUDA not available. Falling back to CPU.")
        self.logging = logging
        self.dtype = dtype
        
        self.base_seed = int(seed)
        torch.manual_seed(self.base_seed)
        if self.device.type == 'cuda':
            torch.cuda.manual_seed_all(self.base_seed)
            
        # This handles CLR transformation, zero value processing, and covariate/confounder adjustment
        pp = _data_pre_process(
            data=data,
            features_as_rows=features_as_rows,
            meta=meta,
            continuous_covariate_keys=continuous_covariate_keys,
            categorical_covariate_keys=categorical_covariate_keys,
            device=self.device,
            dtype=dtype)
        
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
        self.latent_dim = latent_dim

        # Initialize the VAE model architecture
        self.model = VAE(
            input_dim=self.feature_dim,
            latent_dim=self.latent_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            dtype=dtype
        ).to(self.device)
        
        # Initialize placeholder for results
        self.corr_outputs = None
        self.train_loss = []

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
        Train the VAE model using mini-batch optimization.
        
        This method implements the full training loop for the VAE, including handling of zero values,
        gradient updates, and learning rate scheduling. The training process uses mini-batch
        stochastic gradient descent with the AdamW optimizer and cosine annealing learning rate
        scheduling for improved convergence.
        
        Parameters
        ----------
        batch_size : int, default=32
            Number of samples per mini-batch. Larger batches provide more stable gradients
            but require more memory. Recommended range: 16-128 depending on available memory.
            
        num_workers : int, default=0
            Number of subprocesses to use for data loading. Set to 0 for the main process.
            
        max_epochs : int, default=1000
            Maximum number of complete passes through the training data. The actual training
            might converge earlier depending on loss progression.
            
        learning_rate : float, default=1e-3
            Initial learning rate for the AdamW optimizer. The rate will be modulated by
            the cosine annealing scheduler during training.
            
        max_grad_norm : float, default=1.0
            Maximum norm for gradient clipping. Helps prevent exploding gradients and
            stabilizes training. Set to None to disable gradient clipping.
            
        shuffle : bool, default=True
            Whether to randomize the order in which data samples are loaded in each epoch.
            
        deterministic : bool, default=False
            If True and running on CUDA, enables deterministic kernels:
            sets ``CUBLAS_WORKSPACE_CONFIG``, disables cuDNN benchmarking, enables
            deterministic algorithms. This may reduce speed and can raise if a
            non-deterministic op is encountered.
            
        **trainer_kwargs : dict
            Additional keyword arguments for customizing the training process.
        
        Notes
        -----
        The training process includes several key components:
        1. Mini-batch data loading with optional parallel processing
        2. Zero-value handling through random initialization
        3. Gradient-based optimization with AdamW
        4. Learning rate scheduling with cosine annealing
        5. Optional TensorBoard logging for monitoring training progress
        
        The method stores training losses in self.train_loss for later analysis.
        """
        # Reproducibility setup
        run_seed = int(self.base_seed)
        
        # Global seeds
        torch.manual_seed(run_seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(run_seed)
           
        # Optional deterministic kernels (CUDA)
        if deterministic and self.device.type == "cuda":
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True)
            
        # DataLoader shuffle generator
        dl_gen = torch.Generator(device="cpu")
        dl_gen.manual_seed(run_seed)
        
        # Seed each worker for NumPy/Python as well
        def _worker_init_fn(worker_id: int):
            wseed = run_seed + worker_id
            random.seed(wseed)
            np.random.seed(wseed)
            torch.manual_seed(wseed)
        
        # Extract required data components from the class instance
        y_data = self.clr_data
        n = self.sample_dim

        # Set up data loading with mini-batches
        ds = TensorDataset(y_data)
        dl = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=(self.device.type == "cuda"),
            drop_last=False,
            generator=dl_gen,                  
            worker_init_fn=_worker_init_fn if num_workers > 0 else None,  
            persistent_workers=(num_workers > 0)
        )

        # Initialize optimizer and learning rate scheduler
        # AdamW combines Adam optimizer with decoupled weight decay
        optim = torch.optim.AdamW(self.model.parameters(),
                                  lr=learning_rate, 
                                  weight_decay=0.0)
        
        # Configure cosine annealing scheduler with warm restarts
        # This helps escape local minima and find better solutions
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optim, 
            T_0=20, # Initial restart period
            T_mult=2, # Period multiplier after each restart
            eta_min=learning_rate/2 # Minimum learning rate
            )

        # Set up TensorBoard logging if enabled
        writer = None
        if self.logging:
            os.makedirs("runs", exist_ok=True)
            existing = [d for d in os.listdir("runs") if d.startswith("run") and d[len("run"):].isdigit()]
            next_id = max([int(d[len("run"):]) for d in existing], default=-1) + 1
            writer = SummaryWriter(os.path.join("runs", f"run{next_id}"))
        
        # Begin training loop
        self.model.train()
        for epoch in tqdm(range(1, max_epochs + 1)):
            running = 0.0
            num_batches = 0

            for (y_batch,) in dl:
                # y_batch: (batch_size, d)
                # Impute NaNs if any (and only if original data had zeros)
                if torch.any(self.num_zero != 0) and torch.isnan(y_batch).any():
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

                running += float(loss.item())
                num_batches += 1

            # epoch end
            if num_batches > 0:
                avg = running / num_batches
                self.train_loss.append(avg)
                if writer is not None:
                    writer.add_scalar("Loss/train", avg, epoch)

            scheduler.step()
        
        # Ensure all logging data is written
        if self.logging:
            writer.flush()

    def confound_coef(self):
        """
        Extract and format the learned covariate/confounder coefficients from the model.
        
        This method retrieves the coefficients that describe how each covariate/confounder affects
        the metabolite abundances in the CLR-transformed space. These coefficients help
        us understand the strength and direction of confounding effects on each metabolite.
        
        Returns
        -------
        pd.DataFrame or None
            If metadata was provided during training:
                Returns a DataFrame where rows are metabolites, columns are covariates,
                and values represent the effect size of each covariate on each metabolite.
            If no metadata was provided:
                Returns None since no confounding effects were modeled.
        
        Notes
        -----
        Positive coefficients indicate that increasing the covariate/confounder value leads to
        higher metabolite abundance, while negative coefficients indicate the opposite.
        The coefficients are in the CLR space, so interpretations should consider
        the compositional nature of the data.
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
        Calculate the total confounding effect size for each sample and metabolite.
        
        This method computes the combined effect of all covariates on each metabolite
        for each sample by multiplying the covariate values with their corresponding
        coefficients. This shows us how much of each metabolite's variation can be
        attributed to the measured confounding factors.
        
        Returns
        -------
        pd.DataFrame or None
            If metadata was provided during training:
                Returns a DataFrame where rows are samples, columns are metabolites,
                and values represent the total confounding effect on each metabolite
                in each sample.
            If no metadata was provided:
                Returns None since no confounding effects were modeled.
        
        Notes
        -----
        The effect sizes are in the CLR space and represent how much each metabolite's
        abundance would be expected to change based solely on the confounding factors.
        This can be useful for:
        - Identifying samples with strong confounding effects
        - Understanding which metabolites are most affected by confounders
        - Validating the effectiveness of confounder adjustment
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
        Impute censored zeros (NaNs on the CLR/log scale) in the preprocessed data.
    
        This routine fills missing entries in ``self.clr_data`` using a two-stage approach:
        1) **Censored-normal initialization:** draw low-tail values per feature from a
           Gaussian parameterized by ``clr_mean`` and ``clr_sd``, respecting each
           feature's zero frequency (via ``_random_initial``).
        2) **VAE refinement:** run the trained VAE in eval mode
           on the initialized matrix and use the reconstruction to replace only the
           originally missing entries.
    
        If no zeros were detected during preprocessing (i.e., no NaNs in ``self.clr_data``),
        the input is returned unchanged.
    
        Returns
        -------
        torch.Tensor
            Dense tensor of shape ``(n_samples, n_features)`` on the CLR/log scale,
            same device/dtype as the inputs. Observed entries are preserved; NaNs are
            replaced by VAE-refined values.
        """
        # Extract required components
        y = self.clr_data
        num_zero = self.num_zero
        clr_mean = self.clr_mean
        clr_sd = self.clr_sd
        n, d = y.shape
    
        self.model.eval()
        if torch.any(num_zero != 0):
            # Step 1: Initialize missing values with random samples
            complete_y = _random_initial(
                y=y, 
                sample_size=n, 
                num_zero=num_zero, 
                mean=clr_mean, 
                sd=clr_sd,
                generator=generator)
            
            # Step 2: Use the VAE to refine the initial estimates
            with torch.no_grad():
                _, _, _, recon_y = self.model(
                    complete_y,
                    generator=generator
                )
            
            # Step 3: Combine original and imputed values
            impute_y = y.clone()
            impute_y[torch.isnan(y)] = recon_y[torch.isnan(y)]
        else:
            impute_y = y
    
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
        Single imputation round using a per-call RNG (works on CPU/GPU, stream-safe).
        """
        # Per-call generator; no global reseeding
        gen = torch.Generator(device=device)
        gen.manual_seed(int(seed))
    
        # If on CUDA, ensure we run on the intended device (esp. multi-GPU)
        if device.type == "cuda":
            with torch.cuda.device(device):
                impute_clr_data = self.impute_zeros(generator=gen)
        else:
            impute_clr_data = self.impute_zeros(generator=gen)
    
        # Back to log/original scales
        impute_log_data = impute_clr_data + shift
        impute_data = torch.exp(impute_log_data)
    
        # Sparse correlation
        sparse_corr = _compute_correlation(data=impute_data, threshold=threshold)
        return impute_log_data, sparse_corr
    
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
        GPU-optimized parallel imputation using batching and streams.
        """
        n_features = self.feature_dim
        accumulated_indices, accumulated_values = [], []
        impute_log_sum = None
    
        if num_sim < 1:
            raise ValueError("num_sim must be >= 1")
        num_batches = (num_sim + batch_size - 1) // batch_size
    
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_sim)
            current_batch_size = end_idx - start_idx
    
            batch_sparse_corrs, batch_impute_log = [], []
            streams = [torch.cuda.Stream(device=device) for _ in range(min(4, current_batch_size))]
    
            for i in range(current_batch_size):
                sim_id = start_idx + i
                sim_seed = base_seed + sim_id
                stream = streams[i % len(streams)]
    
                with torch.cuda.stream(stream):
                    # NOTE: we don't reseed globals; _single_imputation builds a local Generator
                    impute_log, sparse_corr = self._single_imputation(
                        sim_seed, shift, threshold, device
                    )
                    batch_impute_log.append(impute_log)
                    batch_sparse_corrs.append(sparse_corr)
    
            # Ensure kernels in this batch are complete
            for s in streams:
                s.synchronize()
    
            # Sum logs without keeping the whole stack (saves memory)
            for impute_log in batch_impute_log:
                impute_log_sum = impute_log.clone() if impute_log_sum is None else (impute_log_sum + impute_log)
    
            # Accumulate sparse pieces
            for sc in batch_sparse_corrs:
                idx = sc.indices()
                val = sc.values() / float(num_sim)  # pre-divide for mean
                accumulated_indices.append(idx)
                accumulated_values.append(val)
    
            # Explicit cleanup
            del batch_sparse_corrs, batch_impute_log
            torch.cuda.empty_cache()
    
        # Mean of log data
        impute_log_data_mean = impute_log_sum / float(num_sim)
    
        # Build averaged sparse correlation
        if accumulated_indices:
            all_indices = torch.cat(accumulated_indices, dim=1)
            all_values  = torch.cat(accumulated_values)
    
            Rn_mean_sparse = torch.sparse_coo_tensor(
                indices=all_indices,
                values=all_values,
                size=(n_features, n_features),
                device=device
            ).coalesce()
        else:
            Rn_mean_sparse = torch.sparse_coo_tensor(
                indices=torch.empty(2, 0, dtype=torch.long, device=device),
                values=torch.empty(0, dtype=impute_log_data_mean.dtype, device=device),
                size=(n_features, n_features),
                device=device
            )
    
        return impute_log_data_mean, Rn_mean_sparse
    
    def _cpu_parallel_imputation(
            self,
            num_sim: int,
            shift: torch.Tensor,
            threshold: float,
            workers: int,
            device: torch.device,
            base_seed: int = 0
    ) -> tuple:
        """
        Process-based parallel imputation with joblib (loky).
        - Chunks simulations inside each worker to reduce IPC/memory.
        - Returns (impute_log_data_mean, Rn_mean_sparse).
        """
        if workers is None:
            workers = 1
        elif workers < 0:
            workers = os.cpu_count() or 1
        else:
            workers = int(workers)
    
        if num_sim < 1:
            raise ValueError("num_sim must be >= 1")
    
        # Limit BLAS oversubscription in each process (optional but healthy)
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        try:
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
        except Exception:
            pass
    
        # Decide chunking: ~equal-size chunks across workers
        chunk_size = max(1, math.ceil(num_sim / max(1, workers)))
            
    
        # Build seed ranges for each chunk
        ranges = []
        for start in range(0, num_sim, chunk_size):
            end = min(start + chunk_size, num_sim)
            ranges.append((start, end))
    
        # Worker function: run a chunk of simulations and aggregate locally
        def _run_chunk(start: int, end: int):
            impute_sum = None
            idx_list, val_list = [], []
            # Use CPU in worker processes
            dev_cpu = torch.device("cpu")
    
            for sim_id in range(start, end):
                seed = base_seed + sim_id
                impute_log, sparse_corr = self._single_imputation(
                    seed, shift, threshold, dev_cpu
                )
                impute_sum = impute_log.clone() if impute_sum is None else impute_sum + impute_log
                idx_list.append(sparse_corr.indices())
                val_list.append(sparse_corr.values())
    
            # Concatenate once per chunk (smaller memory than per-sim)
            if idx_list:
                chunk_indices = torch.cat(idx_list, dim=1)
                chunk_values  = torch.cat(val_list)
            else:
                # Empty chunk (shouldn't happen if num_sim>0)
                chunk_indices = torch.empty(2, 0, dtype=torch.long, device="cpu")
                chunk_values  = torch.empty(0, device="cpu", dtype=impute_sum.dtype if impute_sum is not None else torch.float32)
    
            return impute_sum, chunk_indices, chunk_values
    
        # Parallel execution with processes; results are in input order
        if workers == 1:
            results = [_run_chunk(s, e) for (s, e) in ranges]
        else:
            with parallel_backend("loky"):  # explicit processes
                # pre_dispatch limits how many tasks are queued simultaneously (reduces memory)
                results = Parallel(
                    n_jobs=workers,
                    prefer="processes",
                    pre_dispatch=workers,   # queue ~workers chunks at a time
                    batch_size=1,           # 1 chunk per dispatched task
                    verbose=0,
                )(delayed(_run_chunk)(s, e) for (s, e) in ranges)
    
        # Reduce across chunks on the driver
        impute_log_sum = None
        all_indices, all_values = [], []
        for chunk_sum, chunk_idx, chunk_val in results:
            impute_log_sum = chunk_sum if impute_log_sum is None else impute_log_sum + chunk_sum
            if chunk_idx.numel() > 0:
                all_indices.append(chunk_idx)
                all_values.append(chunk_val)
    
        # Mean imputation
        impute_log_data_mean = impute_log_sum / float(num_sim)
    
        # Average sparse correlations
        n_features = impute_log_data_mean.shape[-1]
        if all_indices:
            combined_indices = torch.cat(all_indices, dim=1)                 # (2, nnz_total)
            combined_values  = torch.cat(all_values) / float(num_sim)        # (nnz_total,)
    
            # Canonical lexicographic order -> bit-stable coalesce
            lin = combined_indices[0] * n_features + combined_indices[1]
            perm = torch.argsort(lin)
            combined_indices = combined_indices[:, perm]
            combined_values  = combined_values[perm]
    
            Rn_mean_sparse = torch.sparse_coo_tensor(
                indices=combined_indices,
                values=combined_values,
                size=(n_features, n_features),
                device="cpu",
            ).coalesce()
        else:
            Rn_mean_sparse = torch.sparse_coo_tensor(
                indices=torch.empty(2, 0, dtype=torch.long, device="cpu"),
                values=torch.empty(0, device="cpu"),
                size=(n_features, n_features),
                device="cpu",
            )
    
        return impute_log_data_mean, Rn_mean_sparse
    
    def get_corr(
        self,
        num_sim: int = 100,
        workers: int = -1,
        batch_size: int = 100,   # For GPU batching
        threshold: float = 0.2,  # Correlation threshold for sparsity
        seed: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute a correlation estimate from multiple imputations, optionally in parallel,
        and cache the result in ``self.corr_outputs``.
    
        The method performs multiple stochastic imputations of the (log) data, averages
        the imputed matrices, transforms back to the original scale, and computes a
        **sparse** correlation matrix by hard-thresholding small correlations.
    
        Parameters
        ----------
        num_sim : int, default=100
            Number of imputation simulations to run and average.
            Larger values reduce Monte Carlo noise but increase time/memory.
    
        workers : int, default=-1
            CPU thread count for the **CPU** path (ignored on GPU).
            - ``-1`` uses all available CPU cores.
            - ``1`` runs sequentially (deterministic order).
            - ``>1`` uses a thread pool (shared memory; faster but see Notes on
              reproducibility).
    
        batch_size : int, default=100
            Batch size for the **GPU** path. Controls how many imputations are
            executed per batch/round of CUDA streams. Increase to use more GPU,
            decrease to reduce peak memory.
    
        threshold : float, default=0.2
            Absolute correlation cutoff used inside ``_compute_correlation``.
            Only entries with ``|r| >= threshold`` are kept; the result is returned
            as a sparse COO tensor (symmetric; diagonal kept).
    
        seed : int or None, default=None
            Base seed for reproducibility. When provided, each simulation uses
            ``seed + sim_id`` as its per-simulation seed. See Notes for details.
    
        Returns
        -------
        outputs : Dict[str, torch.Tensor]
            A dictionary with:
            - ``'impute_log_data'`` : ``torch.Tensor`` (dense, shape ~ ``(n, p)``)
                The mean of imputed log-scale data over ``num_sim`` simulations.
            - ``'estimate'`` : ``torch.Tensor`` (sparse COO, shape ``(p, p)``)
                Sparse correlation estimate obtained after thresholding.
                Convert with ``.to_dense()`` if a dense matrix is needed.
        """
        
        # Get required components from the model
        shift = self.shift
        num_zero = self.num_zero
        device = self.device
        
        base_seed = int(seed) if seed is not None else self.base_seed
        
        # Check if we need to handle zeros
        if torch.any(num_zero != 0):
            if device.type == 'cuda':
                # GPU implementation
                impute_log_data_mean, Rn_mean = self._gpu_parallel_imputation(
                    num_sim, shift, threshold, batch_size, device, base_seed=base_seed
                )
            else:
                # CPU implementation
                impute_log_data_mean, Rn_mean = self._cpu_parallel_imputation(
                    num_sim, shift, threshold, workers, device, base_seed=base_seed
                )
        else:
            # Direct computation without imputation
            impute_clr_data = self.impute_zeros()
            impute_log_data_mean = impute_clr_data + shift
            impute_data = torch.exp(impute_log_data_mean)
            Rn_mean = _compute_correlation(data=impute_data,
                                           threshold=threshold)
        
        # Store outputs
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
        Create a sparse correlation matrix by filtering based on statistical significance.
        
        This method implements Fisher's z-test to identify significant 
        correlations between metabolites while controlling for multiple testing. It follows 
        a three-step process:
        
        1. Transform correlation coefficients using Fisher's z-transformation to obtain
           normally distributed values that can be used for statistical testing.
           
        2. Calculate p-values using the transformed correlations and sample size, which
           tells us the probability of observing such correlations by chance.
           
        3. Adjust these p-values for multiple testing to control the false discovery rate
           or family-wise error rate, depending on the chosen method.
        
        Parameters
        ----------
        p_adj_method : str, default='fdr_bh'
            Method for multiple testing correction. Options include:
            - 'fdr_bh': Benjamini-Hochberg FDR control (recommended for most cases)
            - 'bonferroni': Most conservative, controls family-wise error rate
            - 'holm': Less conservative than Bonferroni but still controls FWER
            - Other methods provide different tradeoffs between power and error control
        
        cutoff : float, default=0.05
            Significance threshold for adjusted p-values. Correlations with adjusted
            p-values above this threshold will be set to zero in the sparse network.
        
        Returns
        -------
        dict
            A dictionary containing:
            - 'estimate': Original correlation matrix
            - 'p_value': Unadjusted p-values for each correlation
            - 'q_value': Adjusted p-values after multiple testing correction
            - 'sparse_estimate': Sparsified correlation matrix where non-significant
               correlations are set to zero
        """
        # Check if correlations have been computed
        if self.corr_outputs is None:
            raise ValueError("No correlation estimates. Please compute correlations the first using get_corr method.")
        if getattr(self, "sample_dim", None) is None or self.sample_dim <= 3:
            raise ValueError("Sample size must be > 3 for Fisher's z-test.")
        
        # Extract preprocessed data and correlation estimate
        Rn = self.corr_outputs['estimate']
        n = self.sample_dim
        feature_names = self.feature_name
        
        # Convert to regular matrix
        Rn = Rn.to_dense()
        Rn = Rn.clamp(min=-1.0 + 1e-7, max=1.0 - 1e-7)
        Rn.fill_diagonal_(1.0)
        device, dtype = Rn.device, Rn.dtype

        # Step 1: Fisher's z-transformation of correlations
        # This transforms correlation coefficients to approximate normal distribution
        # z = 0.5 * log((1+Rn)/(1-Rn)) = 0.5 * (log1p(Rn) - log1p(-Rn))
        z = 0.5 * (torch.log1p(Rn) - torch.log1p(-Rn))
        
        # Calculate standard error of the z-transformed correlations
        se = 1.0 / torch.sqrt(torch.tensor(float(n-3), dtype=dtype, device=device))
        
        # Calculate z-scores for hypothesis testing
        z_score = z / se
        z_score.fill_diagonal_(0.0)
        
        # Step 2: Calculate two-tailed p-values
        p_val = 2.0 * (1.0 - torch.special.ndtr(torch.abs(z_score)))
        p_val.fill_diagonal_(0.0)
        
        # Step 3: Apply multiple testing correction
        q_val = _matrix_p_adjust(p_val, method=p_adj_method)

        # Create sparse correlation matrix by filtering based on adjusted p-values
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
        Create a sparse correlation matrix using the Sparse Estimation of Correlation (SEC) algorithm.
    
        This method can:
        - run a **single SEC fit** when a fixed `rho` (penalty) is supplied, or
        - **automatically select `rho` via K-fold cross-validation (CV)** when `rho` is None.
    
        Automatic selection evaluates candidates `rho = c * sqrt(log(p)/n)` for `c` in `c_grid`.
        After a coarse pass over `c_grid`, if `refine=True` a **single refinement** zooms into
        the interval between the best coarse `c` and its immediate neighbors, evaluating an
        evenly spaced finer grid of size `refine_points`. The final `best_rho` minimizes the
        mean validation Frobenius error across folds (ties favor smaller `rho`).
    
        Parallelism:
        - **CPU** with `workers=-1` (default) or `workers>1`: evaluate candidate `rho`s in parallel.
        - **GPU** or `workers<=1`: evaluate sequentially.
    
        Parameters
        ----------
        rho : float, optional
            Fixed ℓ₁ regularization parameter for SEC. If provided, runs one SEC fit.
            If None (default), `rho` is selected by K-fold CV with an optional single refinement.
    
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
        - CV scoring uses mean squared Frobenius error between the SEC fit (train folds)
          and the empirical correlation on validation folds.
        - The refinement reuses the same folds and cached validation correlations.
        - Sparse (COO) tensors are densified internally where required.
        """
        # Prechecks
        if self.corr_outputs is None:
            raise ValueError("No correlation estimates. Please compute correlations first using `get_corr`.")
    
        # Extract preprocessed data and correlation estimate
        impute_log_data = self.corr_outputs['impute_log_data']
        Rn = self.corr_outputs['estimate']
        feature_names = self.feature_name
        impute_data = torch.exp(impute_log_data)  # stays on same device/dtype
        n = self.sample_dim
    
        best_rho = None
        scores_by_rho = None
        R_hat_dense = None
    
        if rho is not None:
            # --- fixed-ρ single fit ---
            R_hat_sparse = _SEC(
                Rn=Rn, rho=rho,
                epsilon=epsilon, tol=tol, max_iter=max_iter, restart=restart,
                line_search_apg=line_search_apg, delta=delta, n_samples=n,
                c_delta=c_delta, threshold=threshold
            )
            R_hat_dense = R_hat_sparse.to_dense()
            best_rho = rho
            scores_by_rho = None
        else:
            best_rho, scores_by_rho, R_hat_dense = _SEC_cv(
                X=impute_data,
                Rn=Rn,                         
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
    
        # Prepare outputs (dense DataFrames)
        Rn_dense = Rn.to_dense()
        outputs = {
            'estimate': _torch_to_df(Rn_dense, names=feature_names),
            'sparse_estimate': _torch_to_df(R_hat_dense, names=feature_names),
            'best_rho': best_rho,
            'scores_by_rho': scores_by_rho,
        }
        return outputs
    
    def clr_loading(self):
        """
        Extract and format the VAE's learned feature loadings in CLR space.
        
        Returns
        -------
        pandas.DataFrame
            A DataFrame where:
            - Rows represent metabolites (features)
            - Columns represent latent dimensions
            - Values indicate how strongly each metabolite contributes to each dimension
            - Column names are formatted as "latent_0", "latent_1", etc.
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
        Calculate the co-occurrence strength between metabolites based on their latent representations.
        
        This method transforms the VAE's learned feature loadings into co-occrrence measures, represented as the variances of log-ratios.
        
        Returns
        -------
        pandas.DataFrame
            A symmetric DataFrame where both rows and columns are metabolites.
            Each value represents the co-occurrence strength between two metabolites:
            - Higher values indicate metabolites that vary more independently
            - Lower values suggest metabolites that tend to change together
            - The diagonal represents self-co-occurrence (usually not meaningful)
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
        num_sim: int = 1000,
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
        rho_k = _compute_correlation(data=impute_data, threshold=threshold)
        rho_k = rho_k.to_dense()
        
        impute_log_sum += torch.nan_to_num(impute_log_data, nan=0.0).to(dtype)
        Rn_sum += torch.nan_to_num(rho_k, nan=0.0).to(dtype)

    # Averages
    impute_log_mean = (impute_log_sum / float(num_sim)).to(dtype)
    impute_mean = torch.exp(impute_log_mean)
    Rn = (Rn_sum / float(num_sim)).to(dtype)

    if sparse_method == 'pval':
        # Obtain p-values by Fisher z-transformation
        Rn_clamped = Rn.clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        z = 0.5 * (torch.log1p(Rn_clamped) - torch.log1p(-Rn_clamped))  # 0.5*log((1+ρ)/(1-ρ))
        
        se = 1.0 / torch.sqrt(torch.tensor(float(n - 3), device=device, dtype=Rn.dtype))
        z_score = z / se
        z_score.fill_diagonal_(0.0)
        
        p_value = 2.0 * (1.0 - torch.special.ndtr(torch.abs(z_score)))
        p_value.fill_diagonal_(0.0)  
        
        q_value = _matrix_p_adjust(p_value, method=p_adj_method)
        
        Rn_hat = _p_filter(Rn, q_value, max_p=cutoff, impute_value=0)
    else:
        Rn_hat = _SEC(
            Rn=Rn,
            rho=rho
        )
        
    Rn_hat_dense = Rn_hat.to_dense()

    # Outputs
    outputs = {
        'estimate': _torch_to_df(Rn, names=feature_names), 
        'sparse_estimate': _torch_to_df(Rn_hat_dense, names=feature_names)
        }

    return outputs