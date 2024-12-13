import numpy as np
import pandas as pd
from scipy import stats

def sim_data(n, d, cor_pairs=0, mu=None, sigma=1, x=None, cont_list=None, cat_list=None, da_prop=0.1):
    """
    Simulate abundance data with confounding effects and correlation structure.
    
    Parameters
    ----------
    n : int
        Number of samples to simulate
    d : int
        Number of features (e.g., metabolites) to simulate
    cor_pairs : int, default=0
        Number of correlated feature pairs to generate. Must not exceed (d * d - d) / 2
    mu : array-like or float, default=None
        Mean values for log-abundance. If None, must be provided
    sigma : float, default=1
        Standard deviation for all features
    x : pandas.DataFrame, default=None
        Confounder data matrix. Required if confounding effects are desired
    cont_list : list, default=None
        Column names of continuous confounders in x
    cat_list : list, default=None
        Column names of categorical confounders in x
    da_prop : float, default=0.1
        Proportion of features affected (differentially abundant) by confounders (0 to 1)
        
    Returns
    -------
    dict
        A dictionary containing:
        - 'cor_matrix': Correlation matrix (d × d)
        - 'y': Simulated abundance matrix (n × d)
        - 'x': Processed confounder matrix
        - 'beta': Effect sizes matrix for confounders
        
    Examples
    --------
    # Simple simulation without confounders
    >>> result = sim_data(n=100, d=50, cor_pairs=10, mu=[2, 3, 4])
    
    # Simulation with confounding effects
    >>> confounders = pd.DataFrame({
    ...     'age': np.random.normal(60, 10, 100),
    ...     'sex': np.random.choice(['M', 'F'], 100),
    ...     'treatment': np.random.choice(['A', 'B', 'C'], 100)
    ... })
    >>> result = sim_data(n=100, d=50, mu=[2, 3, 4], x=confounders,
    ...                   cont_list=['age'], cat_list=['sex', 'treatment'])
    """
    # Generate sample and feature names
    sample_name = ["s" + str(i) for i in range(n)]
    feature_name = ["f" + str(i) for i in range(d)]

    # Initialize mean and standard deviation vectors
    if mu is None:
        raise ValueError("mu parameter must be provided")
    mu_vector = np.random.choice(mu, size=d, replace=True)
    sd_vector = np.full(d, sigma)

    # Create correlation matrix and handle correlated pairs
    cor_matrix = np.eye(d)
    max_cor_pairs = (d * d - d) / 2
    if cor_pairs > max_cor_pairs:
        raise ValueError(f"The maximum number of correlated pairs is: {max_cor_pairs}. "
                         f"Please reduce the number of correlated pairs.")

    if cor_pairs != 0:
        # Generate pairs of correlated features
        idx1 = np.arange(0, 2 * cor_pairs, 2)  # First features in pairs
        idx2 = np.arange(1, 2 * cor_pairs + 1, 2)  # Second features in pairs

        # Sample correlation values from predefined set
        cor_values = np.random.choice([-0.7, -0.6, -0.5, 0.5, 0.6, 0.7], 
                                      size=cor_pairs, replace=True)

        # Combine idx1, idx2 and cor_values
        cor_pairs_matrix = np.column_stack((idx1, idx2, cor_values))

        # Fill correlation matrix with sampled values
        for i in range(cor_pairs):
            row_index = int(cor_pairs_matrix[i, 0])
            col_index = int(cor_pairs_matrix[i, 1])
            corr_value = cor_pairs_matrix[i, 2]
            cor_matrix[row_index, col_index] = corr_value
            cor_matrix[col_index, row_index] = corr_value # Symmetric matrix

    # Calculate covariance matrix from correlation matrix
    cov_matrix = cor_matrix * np.outer(sd_vector, sd_vector)

    # Generate log-scale abundances
    log_y = stats.multivariate_normal.rvs(mean=mu_vector, cov=cov_matrix, size=n)

    # Process confounders if provided
    if x is not None:
        # Handle continuous confounders
        x_cont = x[cont_list] if cont_list is not None else None

        # Handle categorical confounders
        if cat_list is not None:
            x_cat = pd.get_dummies(x[cat_list], drop_first=True)
            x_cat = x_cat.astype(int)
        else:
            x_cat = None

        # Combine confounders
        if x_cont is not None and x_cat is not None:
            x = pd.concat([x_cont, x_cat], axis=1)
        elif x_cont is not None:
            x = x_cont
        elif x_cat is not None:
            x = x_cat
        else:
            raise ValueError('At least one of `cont_list` and `cat_list` should be not None.')

    # Generate confounder effect sizes
    es = [0, -2, -1, 1, 2] # Possible effect sizes
    if x is not None:
        p = x.shape[1]

        # Effect sizes for correlated pairs (always affected by confounders)
        beta1 = np.random.choice(es, size=2 * cor_pairs * p, replace=True, p=[0, 0.25, 0.25, 0.25, 0.25])
        beta1 = beta1.reshape((2 * cor_pairs, p))

        # Effect sizes for uncorrelated features (affected based on da_prop)
        beta2_prob = [1 - da_prop] + [da_prop / 4] * 4
        beta2 = np.random.choice(es, size=(d - 2 * cor_pairs) * p, replace=True, p=beta2_prob)
        beta2 = beta2.reshape((d - 2 * cor_pairs, p))

        # Combine and apply confounder effects
        beta = np.vstack([beta1, beta2])
        log_y = log_y + np.dot(x.values, beta.T)
    else:
        beta = None

    # Convert log abundances to original scale
    y = np.exp(log_y)
    y = pd.DataFrame(y, index=sample_name, columns=feature_name)

    return {'cor_matrix': cor_matrix, 'y': y, 'x': x, 'beta': beta}




