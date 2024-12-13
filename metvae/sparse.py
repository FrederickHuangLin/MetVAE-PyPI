import random
import numpy as np
from statsmodels.stats.multitest import multipletests
from joblib import Parallel, delayed
from .compute_corr import _compute_correlation

# Helper functions for p-values filtering

def _p_filter(mat, mat_p, max_p, impute_value=0):
    mat_filter = mat.copy()
    mat_filter[mat_p > max_p] = impute_value
    return mat_filter

def _matrix_p_adjust(p_matrix, method='fdr_bh'):
    n = p_matrix.shape[0]

    # Extract lower triangular part of the matrix into a vector
    p_vector = p_matrix[np.tril_indices(n, k=-1)]

    # Adjust the p-values
    _, q_vector, _, _ = multipletests(p_vector, method=method)

    # Create a datarix with adjusted q-values in the lower triangle
    q_datarix = np.zeros((n, n))
    q_datarix[np.tril_indices(n, k=-1)] = q_vector

    # Make the datarix symmetric
    q_datarix += q_datarix.T

    return q_datarix

# Helper functions for thresholding

def _soft_thresh(data, th):
    data_sign = np.sign(data)
    data_th = data_sign * np.maximum(np.abs(data)-th, 0) 
    return data_th

def _hard_thresh(data, th):
    data_th = np.copy(data)
    data_th[np.abs(data_th)<th] = 0
    return data_th

def _compute_loss(data1, data2, th, soft=False, alpha=0):
    corr1 = _compute_correlation(data=data1)
    corr2 = _compute_correlation(data=data2)

    # Apply thresholding
    corr1_th = _soft_thresh(corr1, th) if soft else _hard_thresh(corr1, th)
    corr_diff = corr1_th - corr2
    corr_diff[np.isnan(corr_diff)] = 0

    # Frobenius norm
    fro_norm = np.linalg.norm(corr_diff, 'fro')

    # Sparsity penalty
    if alpha > 0:
        epsilon = 1e-10  # Small constant to prevent division by zero
        weight = 1/np.abs(corr2 + epsilon)
        weight[corr2 == 0] = 0
        np.fill_diagonal(weight, 0)
        sparsity_penalty = alpha * np.sum(np.abs(weight * corr1_th))
    else:
        sparsity_penalty = 0
    return 0.5 * fro_norm + sparsity_penalty

def _cv_iteration(args):
    """
    Compute cross-validation losses for a single fold and alpha value.
    
    Parameters:
    -----------
    args : tuple
        Contains (data, train_indices, test_indices, threshold_grid, soft, alpha)
    
    Returns:
    --------
    numpy.ndarray
        Array of losses for each threshold value
    """
    data, idx_train, idx_test, threshold_grid, soft, alpha = args
    train_data = data[idx_train]
    test_data = data[idx_test]
    losses = np.array([
        _compute_loss(train_data, test_data, threshold, soft, alpha)
        for threshold in threshold_grid
    ])
    
    return losses
    
def _corr_thresholding(data, th_len=10, n_cv=5, soft=False, max_th=None, alpha_grid=None, n_jobs=-1):
    """
    Obtain sparse correlation matrix with cross-validation.
    
    Parameters:
    -----------
    data : numpy.ndarray
        Input matrix of shape (n_samples, n_features)
    th_len : int
        Number of threshold values to try
    n_cv : int
        Number of cross-validation folds
    soft : bool
        Whether to use soft thresholding
    max_th : float, optional
        Maximum threshold value
    alpha_grid : numpy.ndarray, optional
        Grid of alpha values for sparsity penalty
    n_jobs : int
        Number of parallel jobs (-1 for all cores)
    
    Returns:
    --------
    dict
        Contains CV errors, optimal parameters, and sparse correlation matrix
    """
    if alpha_grid is None:
        alpha_grid = np.array([0.0])
    
    n, d = data.shape
    n1 = n - int(n/np.log(n))  # Training set size
    
    # Calculate the correlation matrix
    corr = _compute_correlation(data=data)
    
    # Set up threshold grid
    if max_th is None:
        max_th = np.max(np.abs(corr[corr != 1]))
    threshold_grid = np.linspace(0, max_th, th_len)
    
    # Initialize arrays for storing results
    loss_array = np.zeros((n_cv, th_len, len(alpha_grid)))
    
    # Prepare cross-validation tasks
    cv_tasks = []
    for i in range(n_cv):
        idx_train = random.sample(range(n), n1)
        idx_test = list(set(range(n)) - set(idx_train))
        
        for j, alpha in enumerate(alpha_grid):
            cv_tasks.append((
                data, idx_train, idx_test, 
                threshold_grid, soft, alpha
            ))
    
    # Parallel processing for cross-validation
    cv_results = Parallel(n_jobs=n_jobs)(
        delayed(_cv_iteration)(args) 
        for args in cv_tasks
    )
    
    # Reshape results
    cv_results = np.array(cv_results)
    loss_array = cv_results.reshape(n_cv, len(alpha_grid), th_len).transpose(0, 2, 1)
    
    # Find optimal parameters
    mean_loss = np.mean(loss_array, axis=0)
    opt_idx = np.unravel_index(np.argmin(mean_loss), mean_loss.shape)
    th_opt = threshold_grid[opt_idx[0]]
    alpha_opt = alpha_grid[opt_idx[1]]

    # Apply optimal thresholding
    corr_th = _soft_thresh(corr, th_opt) if soft else _hard_thresh(corr, th_opt)
    
    return {
        'cv_error': loss_array,
        'thresh_grid': threshold_grid,
        'optimal_th': th_opt, 
        'optimal_alpha': alpha_opt,
        'sparse_estimate': corr_th
    }