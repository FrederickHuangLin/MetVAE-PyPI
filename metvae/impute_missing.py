import numpy as np
from scipy import stats
from scipy.optimize import minimize

# Helper functions for missing value imputation

def _log_likelihood(params, meta, data, th):
    b = np.array(params[:-1]).reshape(-1, 1)  # Reshape b to a column vector
    sigma = np.exp(params[-1])  # Exponential transform for log std
    
    if meta is None:
        X = np.ones((len(data), 1))  # Intercept only model
    else:
        X = np.hstack((np.ones((meta.shape[0], 1)), meta))  # Add intercept
    
    mu = np.dot(X, b) # Shape: (n, 1)
    e = data.reshape(-1, 1) - mu # Shape: (n, 1)
    
    uncensored_idx = ~np.isnan(data).flatten() # n1 uncensored data
    censored_idx = np.isnan(data).flatten() # n2 censored data, n = n1 + n2

    z_censored = (th[censored_idx] - mu.flatten()[censored_idx]) / sigma # Shape: (n1, )

    # Calculate log probability for uncensored observations
    ll_uncensored = - np.log(sigma) * np.nansum(uncensored_idx) - \
                    (1 / (2 * sigma**2)) * np.dot(e[uncensored_idx, :].T, e[uncensored_idx, :]) # Shape: (1, 1)
    ll_uncensored = ll_uncensored.item() # Return the scalar value
    
    # Calculate log probability for censored observations
    ll_censored = np.nansum(stats.norm.logcdf(z_censored))
    
    ll = ll_uncensored + ll_censored
    return -ll

def _gradient(params, meta, data, th):
    b = np.array(params[:-1]).reshape(-1, 1)  # Reshape b to a column vector
    sigma = np.exp(params[-1])  # Exponential transform for log std
    
    if meta is None:
        X = np.ones((len(data), 1))  # Intercept only model
    else:
        X = np.hstack((np.ones((meta.shape[0], 1)), meta))  # Add intercept
    
    mu = np.dot(X, b)
    e = data.reshape(-1, 1) - mu

    uncensored_idx = ~np.isnan(data).flatten()
    censored_idx = np.isnan(data).flatten()

    z_censored = (th[censored_idx] - mu[censored_idx].flatten()) / sigma
    z_censored_pdf = stats.norm.pdf(z_censored)
    z_censored_cdf = stats.norm.cdf(z_censored)
    z_censored_ratio = np.nan_to_num(z_censored_pdf / (z_censored_cdf+1e-10), nan=0.0) # add small constant to prevent division by zero
    z_censored = np.nan_to_num(z_censored, nan=0.0)

    grad_b_uncensored = (-1 / sigma**2) * np.dot(X[uncensored_idx, :].T, e[uncensored_idx, :])
    grad_b_censored = (1 / sigma) * np.dot(X[censored_idx, :].T, z_censored_ratio.reshape(-1, 1))
    grad_b = grad_b_uncensored + grad_b_censored

    grad_log_sigma_uncensored = np.nansum(uncensored_idx) - \
                                (1 / sigma**2) * np.dot(e[uncensored_idx, :].T, e[uncensored_idx, :])
    grad_log_sigma_censored = np.dot(z_censored_ratio.reshape(1, -1), z_censored.reshape(-1, 1))

    grad_log_sigma = grad_log_sigma_uncensored + grad_log_sigma_censored

    return np.concatenate((grad_b.flatten(), grad_log_sigma.flatten()))


def _censor_normal_mle(init_params, meta, data, th, fn=_log_likelihood, gr=_gradient, method='BFGS'):
    def objective(params):
        return fn(params, meta, data, th)
    
    def _gradient(params):
        return gr(params, meta, data, th)
    
    try:
        optim_results = minimize(fun=objective, x0=init_params, jac=_gradient, method=method)
        outputs = optim_results.x if optim_results.success else init_params
    except Exception as e:
        outputs = init_params
    
    return outputs