import warnings
import numpy as np
from sklearn.metrics.pairwise import nan_euclidean_distances

# Helper functions for correlation computation

def _compute_vlr(data):
    n = data.shape[0]
    log_data = np.log(data)
    log_data[np.isinf(log_data)] = np.nan

    # CLR transformation by features
    shift = np.nanmean(log_data, axis=0)
    clr_data = log_data - shift

    # VLR is proportional to the squared Euclidean distance in CLR coordinates.
    clr_dist = nan_euclidean_distances(clr_data.T, clr_data.T)
    vlr = clr_dist**2 / n

    return vlr

def _compute_correlation(data):
    n, d = data.shape
    
    # Log transform the data, replacing -inf with NaN
    log_data = np.log(data)
    log_data[np.isinf(log_data)] = np.nan
    
    # CLR transformation by samples
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        shift = np.nanmean(log_data, axis=1)
    clr_data = log_data - shift.reshape(-1, 1)
    
    # Compute VLR
    vlr = _compute_vlr(data)  
    
    # Calculate variance of log abundance for each feature
    clr_var = np.nanvar(clr_data, axis=0)
    sum_log_var = np.sum(clr_var) * d / (d-1)
    log_var = (clr_var - 1/d**2 * sum_log_var) * d / (d-2)
    
    # Calculate correlation between log abundances for each feature pair
    log_var_matrix1 = np.tile(log_var, (d, 1))
    log_var_matrix2 = np.tile(log_var.reshape(-1, 1), (1, d))
    log_std_prod = np.sqrt(log_var_matrix1 * log_var_matrix2)
    rho = (vlr - log_var_matrix1 - log_var_matrix2) / (-2 * log_std_prod)
    
    return rho

# Helper functions for hypothesis testing based on VLR

def _norm_vlr(t):
    d = t.shape[0]
    t_colmed = np.median(t, axis=0)
    t_med = np.median(t)

    colmed_matrix1 = np.tile(t_colmed, (d, 1))
    colmed_matrix2 = np.tile((t_colmed).reshape(-1, 1), (1, d))

    norm_t = (t - colmed_matrix1 - colmed_matrix2 + t_med) / np.sqrt(colmed_matrix1 * colmed_matrix2)
    return norm_t

def _by_sample_permute(data):
    n, d = data.shape
    permuted_data = np.apply_along_axis(np.random.permutation, axis=0, arr=data)
    permuted_data = permuted_data.reshape(n, d)
    return permuted_data

def _es_direction(p_below, p_above):
    d = p_below.shape[0]
    direct = np.full((d, d), "increase")  # Default: increased co-occurrence

    # Update direction based on comparisons
    direct[p_below > p_above] = "decrease"  # Decreased co-occurrence

    # Set the diagonal to "unchanged"
    np.fill_diagonal(direct, "unchanged")

    return direct