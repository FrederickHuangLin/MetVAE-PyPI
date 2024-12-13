import os
import warnings
from typing import Optional, List, Literal
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import norm
import statsmodels.formula.api as smf
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from .vae import VAE, IWAE
from .utils import _CustomDataModule, _make_valid_column_name
from .impute_missing import _censor_normal_mle
from .compute_corr import _compute_correlation
from .sparse import _matrix_p_adjust, _p_filter, _corr_thresholding

def _data_pre_process(data: pd.DataFrame,
                      features_as_rows: bool = False,
                      meta: Optional[pd.DataFrame] = None,
                      continuous_covariate_keys: Optional[List[str]] = None,
                      categorical_covariate_keys: Optional[List[str]] = None):
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
    # Part 1: Input Validation and Data Organization
    if not isinstance(data, pd.DataFrame):
        raise TypeError('The input data must be a pandas.DataFrame')
     
    # Ensure features are columns for consistent processing
    if features_as_rows:
        data = data.T
    sample_name = data.index.tolist()
    feature_name = data.columns.tolist()

    # Count zeros in each feature for handling censored values
    num_zero = np.apply_along_axis(lambda x: np.sum(x == 0), axis=0, arr=data.values)
    
    # Part 2: Metadata Processing and Validation
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
            smd_cat = smd_cat.map(_make_valid_column_name) # Ensure valid column names
            smd_cat = pd.get_dummies(smd_cat, drop_first=True, dtype=float) # One-hot encoding
    
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
     
    # Get dimensions for later use
    n, d = data.shape
    p = smd.shape[1] + 1 if meta is not None else 1 # Add 1 for intercept

    # Part 3: CLR Transformation
    # Handle zeros by converting to NaN after log transform
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        log_data = np.log(data)
        log_data[np.isinf(log_data)] = np.nan
    
    # Calculate geometric mean (shift) for CLR transformation
    shift = np.nanmean(log_data, axis=1, keepdims=True)
    clr_data = log_data - shift

    ## Calculate threshold values for censored observations
    th = np.apply_along_axis(lambda x: np.min(x[x != 0]), axis=0, arr=data)
    th = np.tile(th, (n, 1))
    clr_th = np.log(th) - shift

    # Part 4: Parameter Estimation
    # Function to estimate parameters for each feature
    columns_with_warnings = []
    def estimate_params(y, meta):
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter('always') 
    
                if meta is None:
                    df = pd.DataFrame({'y': y})
                    model = smf.ols('y ~ 1', data=df).fit()
                else:
                    df = pd.DataFrame({'y': y}).join(meta)
                    formula = 'y ~ ' + ' + '.join(meta.columns)
                    model = smf.ols(formula, data=df).fit()
    
                estimates = np.append(model.params.values, np.log(model.scale))
                if w:
                    columns_with_warnings.append(y.name)
    
                return estimates
    
        except Exception as e:
            print(f"Error in column {y.name}: {e}")
            return None
    
    # Initial parameter estimation
    init_params = clr_data.apply(lambda col: estimate_params(col, smd), axis=0)
    init_params[np.isnan(init_params)] = 1

    # Report any warning-generating features
    if columns_with_warnings:
        print("Warning: The following features triggered warnings and may need to be reviewed or removed:")
        print(", ".join(columns_with_warnings))

    # Part 5: Handle Zero Values and Estimate Final Parameters
    if np.any(num_zero != 0):
        # Use censored normal MLE for features with zeros
        clr_params = [_censor_normal_mle(init_params=init_params.values[:, i], meta=smd,
                                         data=clr_data.values[:, i], th=clr_th[:, i]) for i in range(d)]
        clr_params = np.column_stack(clr_params)
        clr_log_sd = clr_params[p, :]
        clr_sd = np.exp(clr_log_sd)
        clr_mean = clr_params[0, :]
    else:
        # Use initial estimates if no zeros present
        clr_params = init_params.values
        clr_log_sd = clr_params[p, :]
        clr_sd = np.exp(clr_log_sd)
        clr_mean = clr_params[0, :]

    # Part 6: Deconfound Data if Metadata Present
    if meta is not None:
        clr_coef = clr_params[1:p, :]
        X = smd.values
        clr_data -= X @ clr_coef
    
    # Part 7: Convert to PyTorch Tensors for Output
    clr_data = torch.tensor(clr_data.values, dtype=torch.float32)
    smd = torch.tensor(smd.values, dtype=torch.float32) if smd is not None else None
    num_zero = torch.tensor(num_zero, dtype=torch.float32)
    shift = shift.astype('float32')
    if torch.any(num_zero != 0):
        clr_mean = torch.tensor(clr_mean, dtype=torch.float32)
        clr_sd = torch.tensor(clr_sd, dtype=torch.float32)
        clr_coef = torch.tensor(clr_params[1:p, :], dtype=torch.float32) if smd is not None else None
    else:
        clr_mean = None
        clr_sd = None
        clr_coef = None
    
    outputs = {'clr_data': clr_data, 'meta': smd, 'num_zero': num_zero, 'shift': shift,
               'clr_mean': clr_mean, 'clr_sd': clr_sd, 'clr_coef': clr_coef, 
               'sample_name': sample_name, 'feature_name': feature_name, 'confound_name': confound_name}
    
    return outputs

def _random_initial(y: torch.Tensor, sample_size: int, num_zero: torch.Tensor, 
                    mean: torch.Tensor, sd: torch.Tensor) -> torch.Tensor:
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
    # Get dimensions of input tensor
    batch_size, feature_size = y.shape
    
    # Count number of NaN values (censored zeros) for each feature
    num_nan = torch.sum(torch.isnan(y), dim=0)
            
    # Generate random samples from normal distribution using feature-specific parameters
    # Shape: (sample_size, feature_size)
    random_data = torch.randn(sample_size, feature_size, device=y.device) * sd.repeat(sample_size, 1) + mean.repeat(sample_size, 1)

    # Process each feature separately to fill NaN values
    fill_values_list = []
    for j in range(feature_size):
        # Extract random values for current feature
        random_values = random_data[:, j]
        
        # Select the smallest values as candidates for zero initialization
        # num_zero[j] determines how many candidates we need for this feature
        _, indices = torch.topk(random_values, int(num_zero[j].item()), largest=False)
        fill_values_candidates = random_values[indices]
        
        # Randomly select required number of values from candidates
        # num_nan[j] determines how many NaN values need to be filled
        fill_values = fill_values_candidates[torch.randperm(int(num_zero[j].item()))[:int(num_nan[j].item())]]
        fill_values_list.append(fill_values)

    # Create a copy of input tensor for filling NaN values
    complete_data = y.clone() 
    
    # Fill NaN values feature by feature
    for j in range(feature_size):
        orig_values = y[:, j]
        fill_values = fill_values_list[j]
        
        # Find positions of NaN values in current feature
        nan_indices = torch.isnan(orig_values)
        
        # Replace NaN values with randomly selected fill values if any exist
        if nan_indices.sum() > 0:
            complete_data[nan_indices, j] = fill_values

    return complete_data

class MetVAE():
    """
    Variational Autoencoder (VAE) specifically designed for untargeted metabolomics data analysis with covariate/confounder handling.
    
    This class implements a specialized VAE that accounts for the unique characteristics of metabolomics data,
    including compositionality, zero values, and the influence of covariates/confounders. The model performs
    several key preprocessing steps before training:
    1. Centered log-ratio (CLR) transformation to handle compositional data
    2. Careful handling of zero values through censored estimation
    3. Covariate/confounder adjustment to remove unwanted variation
    
    Parameters
    ----------
    data : pd.DataFrame
        Input metabolomics data matrix. Should contain abundances of metabolites across samples.
        Can be organized with either samples or features as rows (see features_as_rows parameter).
    
    features_as_rows : bool, default=False
        Data orientation flag. Set to True if features (metabolites) are rows and samples are columns.
        The model will transpose the data internally to maintain a consistent samples × features format.
    
    meta : pd.DataFrame, optional
        Sample metadata containing covariate/confounder information. Must have the same sample index as data.
        Used to adjust for experimental and biological confounding factors.
    
    continuous_covariate_keys : List[str], optional
        Names of continuous covariates in meta to adjust for (e.g., ['age', 'bmi']).
        These variables will be used directly in the adjustment process.
    
    categorical_covariate_keys : List[str], optional
        Names of categorical covariates in meta to adjust for (e.g., ['sex', 'treatment']).
        These will be one-hot encoded automatically before adjustment.
    
    latent_dim : int, default=10
        Dimension of the latent space where data will be embedded.
        A larger dimension allows for more complex patterns but requires more data to train effectively.
    
    use_gpu : bool, default=False
        Whether to use GPU acceleration for model training.
        Will automatically fall back to CPU if CUDA is not available.
    
    logging : bool, default=False
        Whether to log training progress and model metrics.
    
    Attributes
    ----------
    model : VAE
        The underlying VAE model architecture.
    
    device : torch.device
        The device (CPU/GPU) where the model and data are stored.
    
    clr_data : torch.Tensor
        CLR-transformed and covariate-adjusted data.
    
    meta : torch.Tensor
        Processed metadata matrix used for covariate/confounder adjustment.
    
    corr_outputs : dict
        Storage for correlation analysis results (populated after training).
    
    Notes
    -----
    The model performs several important preprocessing steps automatically:
    - CLR transformation to handle the compositional nature of metabolomics data
    - Zero value handling through a censored normal estimation approach
    - Covariate/confounder adjustment to remove unwanted technical and biological variation
    - Data scaling and normalization for optimal model training
    
    Examples
    --------
    >>> # Basic usage without covariates
    >>> model = MetVAE(data=metabolite_data, latent_dim=8)
    
    >>> # Using with covariate adjustment
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
            use_gpu: bool = False,
            logging: bool = False
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
        # This handles CLR transformation, zero value processing, and covariate/confounder adjustment
        pre_process_data = _data_pre_process(data=data,
                                             features_as_rows=features_as_rows,
                                             meta=meta,
                                             continuous_covariate_keys=continuous_covariate_keys,
                                             categorical_covariate_keys=categorical_covariate_keys)
        
        # Unpack preprocessed data components
        meta = pre_process_data['meta']
        num_zero = pre_process_data['num_zero']
        shift = pre_process_data['shift']
        clr_data = pre_process_data['clr_data']
        clr_mean = pre_process_data['clr_mean']
        clr_sd = pre_process_data['clr_sd']
        clr_coef = pre_process_data['clr_coef']
        sample_name = pre_process_data['sample_name']
        feature_name = pre_process_data['feature_name']
        confound_name = pre_process_data['confound_name']

        # Set up GPU usage if requested and available
        self.use_gpu = use_gpu
        if self.use_gpu:
            if not torch.cuda.is_available():
                self.device = torch.device("cpu")
                print("CUDA not available. Check if PyTorch is installed with CUDA support.")
            else:
                self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        
        # Store training process
        self.logging = logging

        # Move all tensors to appropriate device (GPU/CPU)
        self.meta = meta.to(self.device) if meta is not None else None
        self.num_zero = num_zero.to(self.device)
        self.clr_data = clr_data.to(self.device)
        self.clr_mean = clr_mean.to(self.device) if clr_mean is not None else None
        self.clr_sd = clr_sd.to(self.device) if clr_sd is not None else None
        self.clr_coef = clr_coef.to(self.device) if clr_coef is not None else None
        
        # Store data characteristics
        self.shift = shift
        self.sample_dim = clr_data.shape[0]
        self.sample_name = sample_name
        self.feature_dim = clr_data.shape[1]
        self.feature_name = feature_name
        self.latent_dim = latent_dim
        
        # Calculate covariate/confounder dimension
        if meta is None:
            self.confound_dim = 0
        else:
            self.confound_dim = meta.shape[1]
        self.confound_name = confound_name

        # Initialize the VAE model architecture
        self.model = VAE(
            input_dim=self.feature_dim,
            latent_dim=self.latent_dim,
            confound_dim=self.confound_dim
        ).to(self.device)
        
        # Initialize placeholder for correlation analysis results
        self.corr_outputs = None

    def train(
            self,
            batch_size: int = 32,
            num_workers: int = 0,
            max_epochs: int = 1000,
            learning_rate: float = 1e-3,
            max_grad_norm: float = 1.0,
            **trainer_kwargs
    ):
        """
        Train the VAE model on metabolomics data using mini-batch optimization.
        
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
        # Extract required data components from the class instance
        x_data = self.meta if self.meta is not None else None
        y_data = self.clr_data
        num_zero = self.num_zero
        clr_mean = self.clr_mean
        clr_sd = self.clr_sd
        n = self.sample_dim

        # Set up data loading with mini-batches
        self.data_module = _CustomDataModule(x_data, y_data, batch_size, num_workers)
        self.data_module.setup()
        train_dataloader = self.data_module.train_dataloader()

        # Initialize optimizer and learning rate scheduler
        # AdamW combines Adam optimizer with decoupled weight decay
        optimizer = torch.optim.AdamW(self.model.parameters(), 
                                      lr=learning_rate, 
                                      weight_decay=0)
        
        # Configure cosine annealing scheduler with warm restarts
        # This helps escape local minima and find better solutions
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, 
            T_0=20, # Initial restart period
            T_mult=2, # Period multiplier after each restart
            eta_min=learning_rate/2 # Minimum learning rate
            )

        # Set up TensorBoard logging if enabled
        if self.logging:
            # Create unique log directory for this training run
            logdir = "runs"
            if not os.path.exists(logdir):
                os.makedirs(logdir)
            
            # Generate unique run identifier
            base_dirname = "run"
            existing_dirs = [d for d in os.listdir(logdir) if os.path.isdir(os.path.join(logdir, d))]
            run_numbers = [int(d[len(base_dirname):]) for d in existing_dirs if d.startswith(base_dirname) and d.replace(base_dirname, '').isdigit()]
            next_run_number = max(run_numbers) + 1 if run_numbers else 0
            next_dirname = f"{base_dirname}{next_run_number}"
            full_log_dir_path = os.path.join(logdir, next_dirname)
            
            # Initialize TensorBoard writer
            writer = SummaryWriter(full_log_dir_path)
        
        # Begin training loop
        train_losses = []
        for epoch in tqdm(range(1, max_epochs+1)):
            running_loss = 0.0
            self.model.train()
            torch.set_grad_enabled(True)
            
            # Process mini-batches
            for i, batch in enumerate(train_dataloader):
                x, y = batch
                
                # Handle zero values through random initialization if needed
                if torch.any(num_zero != 0):
                    complete_y = _random_initial(y, n, num_zero, clr_mean, clr_sd)
                else:
                    complete_y = y
                
                # Compute loss for current batch
                loss = self.model.training_step((x, complete_y))
                
                # Gradient descent step
                optimizer.zero_grad()
                loss.backward()
                
                # Apply gradient clipping to prevent explosive gradients
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    max_norm=max_grad_norm, 
                    norm_type=2
                    )
                
                # Update model parameters
                optimizer.step()
                
                # Accumulate batch loss
                running_loss += loss.item()
            
            # Update learning rate schedule
            scheduler.step(epoch - 1)
            
            # Calculate average loss for this epoch
            avg_loss = running_loss / (i + 1)
            train_losses.append(avg_loss)
            self.train_loss = train_losses
            
            # Log training progress if enabled
            if self.logging:
                writer.add_scalar("Loss/train", avg_loss, epoch)
        
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
            X = self.meta.detach().cpu().numpy()
            clr_es = X @ clr_coef.T
            clr_es = pd.DataFrame(
                clr_es,
                index=self.sample_name,
                columns=self.feature_name
            )
        else:
            clr_es = None
        return clr_es

    def impute_zeros(self):
        """
        Impute missing values in the original data.
        
        This method uses the trained VAE model to provide intelligent estimates for
        metabolite abundances that were originally zero or below detection limit.
        The process involves:
        1. Initial random value assignment based on the learned distribution
        2. Refinement of these values using the VAE's learned patterns
        3. Combination of original and imputed values
        
        Returns
        -------
        numpy.ndarray
            The complete dataset with imputed values replacing NaN entries.
            The array maintains the original data where values were observed
            and provides model-based estimates where values were missing.
        
        Notes
        -----
        The imputation process respects the compositional nature of the data
        and leverages both:
        - The overall distribution of each metabolite (through clr_mean and clr_sd)
        - The relationships between metabolites (through the VAE's learned representation)
        
        This can provide more reliable estimates than simple approaches like
        mean imputation or zero replacement.
        """
        # Extract required components
        x = self.meta
        y = self.clr_data
        num_zero = self.num_zero
        clr_mean = self.clr_mean
        clr_sd = self.clr_sd
        n, d = y.shape
    
        if torch.any(num_zero != 0):
            # Step 1: Initialize missing values with random samples
            complete_y = _random_initial(y, n, num_zero, clr_mean, clr_sd)
            
            # Step 2: Use the VAE to refine the initial estimates
            with torch.no_grad():
                _, _, _, recon_y = self.model(x, complete_y)
            
            # Step 3: Combine original and imputed values
            impute_y = y.clone()
            impute_y[torch.isnan(y)] = recon_y[torch.isnan(y)]
        else:
            impute_y = y
    
        impute_y = impute_y.detach().cpu().numpy()
        return impute_y

    def get_corr(
            self,
            num_sim: int = 1000):
        """
        Estimate metabolite correlations while accounting for zero values through multiple imputations.
        
        This method computes correlations in compositional
        metabolomics data that contains zeros or values below detection limit. Instead of computing
        correlations directly, which could be biased by missing values, it uses the trained VAE model
        to perform multiple rounds of imputation and then averages the results.
        
        The process involves several key steps:
        1. For each simulation round:
           - Impute missing values using the VAE's learned patterns
           - Transform the data back to the original scale
           - Compute correlations on the complete dataset
        2. Average the results across all simulations to get stable estimates
        
        This approach is particularly valuable because it:
        - Accounts for uncertainty in the missing value estimates
        - Preserves the compositional nature of the data
        - Leverages the VAE's learned relationships between metabolites
        
        Parameters
        ----------
        num_sim : int, default=1000
            Number of simulation rounds to perform when imputing missing values.
            More simulations provide more stable estimates but take longer to compute.
            For data with few zeros, smaller values (e.g., 100) might be sufficient.
        
        Notes
        -----
        The correlation estimates are stored in self.corr_outputs as a dictionary with:
        - 'impute_log_data': The average log-transformed imputed data
        - 'estimate': The average correlation matrix across all simulations
        
        The method handles two cases differently:
        1. Data with zeros: Performs multiple rounds of imputation and averaging
        2. Data without zeros: Computes correlations directly without simulation
        """
        # Prepare lists to store results from multiple simulations
        impute_log_data_list = [] # Will store log-transformed imputed data
        rho_list = [] # Will store correlation matrices
        
        # Get required components from the model
        shift = self.shift 
        num_zero = self.num_zero

        # Handle data with zeros through multiple imputation
        if torch.any(num_zero != 0):
            # Perform multiple rounds of imputation and correlation computation
            for x in range(num_sim):
                # Set random seeds for reproducibility
                torch.manual_seed(x)
                np.random.seed(x)
                
                # Step 1: Impute missing values using the VAE
                impute_clr_data = self.impute_zeros()
                
                # Step 2: Transform back to log scale by adding the CLR shift
                impute_log_data = impute_clr_data + shift
                
                # Step 3: Transform to original scale for correlation computation
                impute_data = np.exp(impute_log_data)
                
                # Step 4: Compute correlations for this imputation
                rho = _compute_correlation(data=impute_data)
    
                # Store results from this simulation
                impute_log_data_list.append(impute_log_data)
                rho_list.append(rho)
            
            # Calculate average results across all simulations
            # Stack arrays along a new axis (simulation axis)       
            impute_log_data_array = np.stack(impute_log_data_list)
            rho_array = np.stack(rho_list)
            
            # Compute means, handling any NaN values that might have occurred
            impute_log_data_mean = np.nan_to_num(impute_log_data_array).sum(axis=0) / num_sim    
            rho_mean = np.nan_to_num(rho_array).sum(axis=0) / num_sim
        else:
            # For data without zeros, we can compute correlations directly
            impute_clr_data = self.impute_zeros()
            impute_log_data_mean = impute_clr_data + shift
            impute_data = np.exp(impute_log_data_mean)
            rho_mean = _compute_correlation(data=impute_data)

        outputs = {
            'impute_log_data': impute_log_data_mean, 
            'estimate': rho_mean
            }
        self.corr_outputs = outputs

    def sparse_by_filter(self,
                         p_adj_method: Literal['bonferroni', 'sidak', 
                                               'holm-sidak', 'holm', 
                                               'simes-hochberg', 'hommel', 
                                               'fdr_bh', 'fdr_by', 
                                               'fdr_tsbh', 'fdr_tsbky'] = 'fdr_bh',
                         cutoff: float = 0.05):
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
        
        # Get correlation matrix and sample size
        rho = self.corr_outputs['estimate']
        n = self.sample_dim

        # Step 1: Fisher's z-transformation of correlations
        # This transforms correlation coefficients to approximate normal distribution
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            z = 0.5 * np.log((1 + rho) / (1 - rho))
        
        # Calculate standard error of the z-transformed correlations
        se = 1 / np.sqrt(n - 3)
        
        # Calculate z-scores for hypothesis testing
        z_score = z / se
        np.fill_diagonal(z_score, 0)  
        
        # Step 2: Calculate two-tailed p-values
        p_value = 2 * np.minimum(1 - norm.cdf(np.abs(z_score)), norm.cdf(np.abs(z_score)))
        np.fill_diagonal(p_value, 0) # Set diagonal to zero  
        
        # Step 3: Apply multiple testing correction
        q_value = _matrix_p_adjust(p_value, method = p_adj_method)

        # Create sparse correlation matrix by filtering based on adjusted p-values
        sparse_rho = _p_filter(rho, q_value, max_p = cutoff, impute_value = 0)
        
        outputs = {
            'estimate': rho, 
            'p_value': p_value, 
            'q_value': q_value, 
            'sparse_estimate': sparse_rho
            }
        
        return outputs

    def sparse_by_thresholding(self,
                               th_len: int = 30, 
                               n_cv: int = 5, 
                               soft: bool = False, 
                               max_th: Optional[float] = None, 
                               alpha_grid: Optional[np.ndarray] = None,
                               n_jobs: int = -1):
        """
        Create a sparse correlation network using data-driven thresholding.
        
        This method implements an adaptive approach to network sparsification that uses
        cross-validation to determine optimal correlation thresholds. Unlike simple
        statistical filtering, this approach learns from the data structure itself to
        decide which correlations to keep or remove.
        
        The method works by examining how well different threshold values can help
        reconstruct the original data patterns. It uses cross-validation to prevent
        overfitting and can apply either hard or soft thresholding to the correlations.
        
        Parameters
        ----------
        th_len : int, default=30
            Number of threshold values to test. A larger number provides finer
            granularity in finding the optimal threshold but increases computation time.
            Think of this as how many "candidate thresholds" we want to evaluate.
        
        n_cv : int, default=5
            Number of cross-validation folds. Higher values give more reliable estimates
            but take longer to compute. For example, with n_cv=5, we split the data
            into 5 parts and use each part as a validation set once.
        
        soft : bool, default=False
            Whether to use soft or hard thresholding:
            - False (hard): Sets correlations below threshold to exactly zero
            - True (soft): Shrinks correlations towards zero continuously
            Hard thresholding creates sparser networks but soft thresholding may
            better preserve subtle relationship patterns.
        
        max_th : float, optional
            Maximum correlation threshold to consider. If None, will be set based
            on the data. This caps how aggressive the sparsification can be.
        
        alpha_grid : numpy.ndarray, optional
            Custom grid of sparsity penalty parameters to test. Higher values encourage
            more sparsity, particularly for correlations that are less reliable in the
            validation data. If None, it will be set to 0s.
        
        n_jobs : int, default=-1
            Number of parallel jobs for cross-validation:
            - -1: Use all available CPU cores
            - 1: No parallelization
            - n: Use n cores
            Parallelization can significantly speed up computation for large datasets.
        
        Returns
        -------
        dict
            A dictionary containing:
            - 'estimate': Original correlation matrix
            - 'sparse_estimate': Optimally sparsified correlation matrix
            - 'threshold': Optimal threshold value found
            - 'cv_scores': Cross-validation scores for different thresholds
            - Additional diagnostic information from the thresholding process
        """
        # Check if correlations have been computed
        if self.corr_outputs is None:
            raise ValueError("No correlation estimates. Please compute correlations the first using get_corr method.")
        
        # Extract preprocessed data and correlation estimates
        log_data, rho = self.corr_outputs['impute_log_data'], self.corr_outputs['estimate']

        # Apply correlation thresholding using cross-validation
        results = _corr_thresholding(
            np.exp(log_data), 
            th_len=th_len, 
            n_cv=n_cv, 
            soft=soft, 
            max_th=max_th, 
            alpha_grid=alpha_grid, 
            n_jobs=n_jobs)
        
        outputs = {'estimate': rho}
        outputs.update(results)
        
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
        
class MetIWAE(MetVAE):
    """
    Importance-Weighted AutoEncoder (IWAE) specifically designed for untargeted metabolomics data analysis with covariate/confounder handling.
    
    This class implements a specialized VAE that accounts for the unique characteristics of metabolomics data,
    including compositionality, zero values, and the influence of covariates/confounders. The model performs
    several key preprocessing steps before training:
    1. Centered log-ratio (CLR) transformation to handle compositional data
    2. Careful handling of zero values through censored estimation
    3. Covariate/confounder adjustment to remove unwanted variation
    
    Parameters
    ----------
    data : pd.DataFrame
        Input metabolomics data matrix. Should contain abundances of metabolites across samples.
        Can be organized with either samples or features as rows (see features_as_rows parameter).
    
    features_as_rows : bool, default=False
        Data orientation flag. Set to True if features (metabolites) are rows and samples are columns.
        The model will transpose the data internally to maintain a consistent samples × features format.
    
    meta : pd.DataFrame, optional
        Sample metadata containing covariate/confounder information. Must have the same sample index as data.
        Used to adjust for experimental and biological confounding factors.
    
    continuous_covariate_keys : List[str], optional
        Names of continuous covariates in meta to adjust for (e.g., ['age', 'bmi']).
        These variables will be used directly in the adjustment process.
    
    categorical_covariate_keys : List[str], optional
        Names of categorical covariates in meta to adjust for (e.g., ['sex', 'treatment']).
        These will be one-hot encoded automatically before adjustment.
    
    latent_dim : int, default=10
        Dimension of the latent space where data will be embedded.
        A larger dimension allows for more complex patterns but requires more data to train effectively.
        
    num_importance_samples : int, default = 1
        Number of importance samples to generate.
    
    use_gpu : bool, default=False
        Whether to use GPU acceleration for model training.
        Will automatically fall back to CPU if CUDA is not available.
    
    logging : bool, default=False
        Whether to log training progress and model metrics.
    
    Attributes
    ----------
    model : VAE
        The underlying VAE model architecture.
    
    device : torch.device
        The device (CPU/GPU) where the model and data are stored.
    
    clr_data : torch.Tensor
        CLR-transformed and covariate-adjusted data.
    
    meta : torch.Tensor
        Processed metadata matrix used for covariate/confounder adjustment.
    
    corr_outputs : dict
        Storage for correlation analysis results (populated after training).
    
    Notes
    -----
    The model performs several important preprocessing steps automatically:
    - CLR transformation to handle the compositional nature of metabolomics data
    - Zero value handling through a censored normal estimation approach
    - Covariate/confounder adjustment to remove unwanted technical and biological variation
    - Data scaling and normalization for optimal model training
    
    Examples
    --------
    >>> # Basic usage without covariates
    >>> model = MetIWAE(data=metabolite_data, latent_dim=8, num_importance_samples=10)
    
    >>> # Using with covariate adjustment
    >>> model = MetIWAE(
    ...     data=metabolite_data,
    ...     meta=metadata,
    ...     continuous_covariate_keys=['age', 'bmi'],
    ...     categorical_covariate_keys=['sex', 'batch'],
    ...     num_importance_samples=10
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
            num_importance_samples: int = 1,
            use_gpu: bool = False,
            logging: bool = False,
    ):
        super().__init__(data=data,
                         features_as_rows=features_as_rows,
                         meta=meta,
                         continuous_covariate_keys=continuous_covariate_keys,
                         categorical_covariate_keys=categorical_covariate_keys,
                         latent_dim=latent_dim,
                         use_gpu=use_gpu,
                         logging=logging)

        self.model = IWAE(
            input_dim=self.feature_dim,
            latent_dim=self.latent_dim,
            confound_dim=self.confound_dim,
            num_importance_samples=num_importance_samples
        ).to(self.device)

# Function to compute correlations without zero imputation via VAE/IWAE

def _simple_inference(data: pd.DataFrame,
                      features_as_rows: bool = False,
                      meta: Optional[pd.DataFrame] = None,
                      continuous_covariate_keys: Optional[List[str]] = None,
                      categorical_covariate_keys: Optional[List[str]] = None,
                      num_sim: int = 1000,
                      sparse_method: Literal['filtering', 'thresholding'] = 'filtering',
                      p_adj_method: Literal['bonferroni', 'sidak', 'holm-sidak', 'holm', 'simes-hochberg', 'hommel', 'fdr_bh', 'fdr_by', 'fdr_tsbh', 'fdr_tsbky'] = 'fdr_bh',
                      cutoff: float = 0.05,
                      th_len: int = 100, 
                      n_cv: int = 5, 
                      soft: bool = False, 
                      n_jobs: int = -1):
    
    pre_process_data = _data_pre_process(data=data,
                                        features_as_rows=features_as_rows,
                                        meta=meta,
                                        continuous_covariate_keys=continuous_covariate_keys,
                                        categorical_covariate_keys=categorical_covariate_keys)
    meta = pre_process_data['meta']
    num_zero = pre_process_data['num_zero']
    shift = pre_process_data['shift']
    clr_data = pre_process_data['clr_data']
    clr_mean = pre_process_data['clr_mean']
    clr_sd = pre_process_data['clr_sd']
    n, d = clr_data.shape

    # Impute missing values and compute correlations
    impute_log_data_list = []
    rho_list = []
    
    for x in range(num_sim):
        torch.manual_seed(x)
        np.random.seed(x)
        
        if torch.any(num_zero != 0):
            impute_clr_data = _random_initial(y=clr_data, sample_size=n, num_zero=num_zero, mean=clr_mean, sd=clr_sd)
        else:
            impute_clr_data = clr_data
        impute_clr_data = impute_clr_data.numpy()
        impute_log_data = impute_clr_data + shift
        impute_data = np.exp(impute_log_data)
        rho = _compute_correlation(data=impute_data)
        
        impute_log_data = impute_log_data.astype('float32')
        rho = rho.astype('float32')
        impute_log_data_list.append(impute_log_data)
        rho_list.append(rho)

    impute_log_data_array = np.stack(impute_log_data_list)
    impute_log_data_mean = np.nan_to_num(impute_log_data_array).sum(axis=0) / num_sim
    rho_array = np.stack(rho_list)
    rho_mean = np.nan_to_num(rho_array).sum(axis=0) / num_sim

    if sparse_method == 'filtering':
        # Obtain p-values by Fisher z-transformation
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            z = 0.5 * np.log((1 + rho_mean) / (1 - rho_mean))
        se = 1 / np.sqrt(n - 3)
        z_score = z / se
        np.fill_diagonal(z_score, 0)  
        p_value = 2 * np.minimum(1 - norm.cdf(np.abs(z_score)), norm.cdf(np.abs(z_score)))
        np.fill_diagonal(p_value, 0)  
        q_value = _matrix_p_adjust(p_value, method = p_adj_method)
        sparse_rho = _p_filter(rho_mean, q_value, max_p = cutoff, impute_value = 0)
    else:
        sparse_rho = _corr_thresholding(np.exp(impute_log_data_mean), th_len=th_len, n_cv=n_cv, soft=soft, n_jobs=n_jobs)['sparse_estimate']

    # Outputs
    outputs = {'estimate': rho_mean, 'sparse_estimate': sparse_rho}

    return outputs