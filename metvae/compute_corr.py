from typing import Optional

import torch

def _torch_nanvar(x: torch.Tensor, dim=None, keepdim=False, unbiased=False):
    """
    Variance ignoring non-finite entries.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    dim : int, optional
        Dimension to reduce. If None, reduces over all elements.
    keepdim : bool, default=False
        Whether the reduced dimension is retained.
    unbiased : bool, default=False
        If True, divides by the count minus one.

    Returns
    -------
    torch.Tensor
        Variance computed over the finite entries.
    """
    mask = torch.isfinite(x)
    count = mask.sum(dim=dim, keepdim=True).clamp_min(1)
    x_filled = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    mean = (x_filled.sum(dim=dim, keepdim=True) / count)
    sq_diff = (x_filled - mean) ** 2 * mask
    var = sq_diff.sum(dim=dim, keepdim=True) / (count - (1 if unbiased else 0)).clamp_min(1)
    if not keepdim and dim is not None:
        var = var.squeeze(dim)
    return var

@torch.no_grad()
def _compute_vlr(data: torch.Tensor) -> torch.Tensor:
    """
    Variance of pairwise log ratios, ignoring non-finite log values.

    Parameters
    ----------
    data : torch.Tensor
        Non-negative abundance matrix of shape (n, d).

    Returns
    -------
    torch.Tensor
        Symmetric matrix of shape (d, d) with zero diagonal. Pairs with no overlapping
        finite rows are NaN.
    """
    if not torch.is_floating_point(data):
        data = data.float()

    n, d = data.shape

    log_x = torch.log(data)
    log_x = log_x.clone()
    log_x[~torch.isfinite(log_x)] = torch.nan

    # Center the log data by feature.
    shift = torch.nanmean(log_x, dim=0, keepdim=True)
    clr = log_x - shift

    M = torch.isfinite(clr).to(clr.dtype)
    Y = torch.nan_to_num(clr, nan=0.0,
                         posinf=0.0, neginf=0.0)
    A = Y.square()

    # Number of rows where both features are finite.
    K = M.transpose(0, 1) @ M

    # Sum of squared differences restricted to the overlapping rows.
    S = A.transpose(0, 1) @ M
    P = Y.transpose(0, 1) @ Y
    sumsq = S + S.transpose(0, 1) - 2.0 * P

    vlr = sumsq / K.clamp_min(1)
    vlr = torch.where(K > 0, vlr, torch.nan)

    vlr = 0.5 * (vlr + vlr.transpose(0, 1))
    vlr = vlr.clamp_min(0.0)
    vlr.fill_diagonal_(0.0)
    return vlr

@torch.no_grad()
def _compute_correlation_general(
        data: torch.Tensor,
        threshold: float = 0.2,
        eps: float = 1e-12
    ) -> torch.Tensor:
    """
    Dense proportionality-based correlation for data whose log may be non-finite.

    Parameters
    ----------
    data : torch.Tensor
        Abundance matrix of shape (n, d) with d >= 3.
    threshold : float, default=0.2
        Entries with absolute value below this threshold are set to zero.
    eps : float, default=1e-12
        Lower bound applied to the per-feature variance and to the product of standard
        deviations.

    Returns
    -------
    torch.Tensor
        Symmetric matrix of shape (d, d) with unit diagonal and entries in [-1, 1].

    Raises
    ------
    ValueError
        If d < 3.
    """
    if not torch.is_floating_point(data):
        data = data.float()
    n, d = data.shape
    if d < 3:
        raise ValueError("Need at least d >= 3 features to compute stable correlations.")

    log_x = torch.log(data)
    log_x[~torch.isfinite(log_x)] = torch.nan

    # Center the log data by sample.
    shift = torch.nanmean(log_x, dim=1, keepdim=True)
    clr = log_x - shift

    vlr = _compute_vlr(data)

    clr_var = _torch_nanvar(clr, dim=0, unbiased=False)

    sum_log_var = torch.nansum(clr_var) * d / (d - 1)
    log_var = (clr_var - (1 / d**2) * sum_log_var) * d / (d - 2)
    log_var = torch.nan_to_num(log_var, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(eps)

    std = torch.sqrt(log_var)
    log_std_prod = (std.unsqueeze(0) * std.unsqueeze(1)).clamp_min(eps)

    lv1 = log_var.expand(d, d)
    lv2 = log_var.view(-1, 1).expand(d, d)
    numer = (vlr - lv1 - lv2)
    denom = (-2.0 * log_std_prod)

    rho = numer / denom
    rho = torch.nan_to_num(rho, nan=0.0, posinf=0.0, neginf=0.0)

    rho = 0.5 * (rho + rho.T)
    rho = rho.clamp(-1.0, 1.0)
    rho.fill_diagonal_(1.0)

    rho = torch.where(rho.abs() >= threshold, rho, torch.zeros_like(rho))

    return rho

@torch.no_grad()
def _compute_correlation_dense(
        data: torch.Tensor,
        threshold: float = 0.2,
        eps: float = 1e-12,
        *,
        out: Optional[torch.Tensor] = None,
        scratch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
    """
    Dense proportionality-based correlation matrix with hard thresholding.

    When every entry of log(data) is finite the matrix is formed from a single
    Gram matrix and two (d, d) buffers. Otherwise the computation falls back to the
    general path that masks non-finite log values.

    Parameters
    ----------
    data : torch.Tensor
        Abundance matrix of shape (n, d) with d >= 3.
    threshold : float, default=0.2
        Entries with absolute value below this threshold are set to zero.
    eps : float, default=1e-12
        Lower bound applied to the per-feature variance and to the product of standard
        deviations.
    out : torch.Tensor, optional
        Preallocated (d, d) buffer receiving the result.
    scratch : torch.Tensor, optional
        Preallocated (d, d) buffer used for intermediate products.

    Returns
    -------
    torch.Tensor
        Symmetric matrix of shape (d, d) with unit diagonal and entries in [-1, 1].
        Equal to ``out`` when ``out`` is given.

    Raises
    ------
    ValueError
        If d < 3.

    Notes
    -----
    The correlation is derived from the variance of pairwise log ratios,
    rho_ij = (vlr_ij - v_i - v_j) / (-2 sqrt(v_i v_j)), where v is the normalized
    per-feature variance of the sample-centered log data.
    """
    if not torch.is_floating_point(data):
        data = data.float()
    n, d = data.shape
    if d < 3:
        raise ValueError("Need at least d >= 3 features to compute stable correlations.")

    log_x = torch.log(data)
    if not bool(torch.isfinite(log_x).all()):
        rho = _compute_correlation_general(data, threshold=threshold, eps=eps)
        if out is not None:
            out.copy_(rho)
            return out
        return rho

    dtype, device = log_x.dtype, log_x.device
    G = torch.empty((d, d), dtype=dtype, device=device) if out is None else out
    T = torch.empty((d, d), dtype=dtype, device=device) if scratch is None else scratch

    # Variance of pairwise log ratios from the Gram matrix of the feature-centered logs.
    Yc = log_x - torch.nanmean(log_x, dim=0, keepdim=True)
    q = Yc.square().sum(0)
    torch.mm(Yc.t(), Yc, out=G)
    del Yc
    G.mul_(-2).add_(q.view(-1, 1)).add_(q.view(1, -1)).div_(n)
    G.clamp_min_(0.0)
    G.fill_diagonal_(0.0)

    # Per-feature variance of the sample-centered log data.
    clr = log_x - torch.nanmean(log_x, dim=1, keepdim=True)
    clr_mean = clr.sum(0, keepdim=True) / n
    clr_var = ((clr - clr_mean) ** 2).sum(0) / n
    del clr, log_x

    sum_log_var = torch.nansum(clr_var) * d / (d - 1)
    log_var = (clr_var - (1 / d**2) * sum_log_var) * d / (d - 2)
    log_var = torch.nan_to_num(log_var, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(eps)
    std = torch.sqrt(log_var)

    G.sub_(log_var.view(1, -1)).sub_(log_var.view(-1, 1))
    torch.mul(std.view(1, -1), std.view(-1, 1), out=T)
    T.clamp_min_(eps).mul_(-2.0)
    G.div_(T)

    G.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
    G.clamp_(-1.0, 1.0)
    G.fill_diagonal_(1.0)
    G.masked_fill_((G > -threshold) & (G < threshold), 0.0)
    return G

@torch.no_grad()
def _compute_correlation(data: torch.Tensor, threshold: float = 0.2, eps: float = 1e-12) -> torch.Tensor:
    """
    Sparse proportionality-based correlation matrix with hard thresholding.

    Parameters
    ----------
    data : torch.Tensor
        Abundance matrix of shape (n, d) with d >= 3.
    threshold : float, default=0.2
        Entries with absolute value below this threshold are set to zero.
    eps : float, default=1e-12
        Lower bound applied to the per-feature variance and to the product of standard
        deviations.

    Returns
    -------
    torch.Tensor
        Sparse COO tensor of shape (d, d) with unit diagonal.

    Raises
    ------
    ValueError
        If d < 3.
    """
    return _compute_correlation_dense(data, threshold=threshold, eps=eps).to_sparse_coo()
