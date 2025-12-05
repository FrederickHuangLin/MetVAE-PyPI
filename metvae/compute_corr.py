import torch

# Helper functions for correlation computation

def _torch_nanvar(x: torch.Tensor, dim=None, keepdim=False, unbiased=False):
    """Nan-aware variance (replacement for torch.nanvar)."""
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
    if not torch.is_floating_point(data):
        data = data.float()

    n, d = data.shape

    # log + treat ±inf as NaN
    log_x = torch.log(data)
    log_x = log_x.clone()
    log_x[~torch.isfinite(log_x)] = torch.nan

    # CLR by feature (column-wise) using nanmean
    # (keepdim=True to broadcast correctly)
    shift = torch.nanmean(log_x, dim=0, keepdim=True)  # (1, d)
    clr = log_x - shift                                # (n, d) with NaNs preserved

    # Masks & zero-fill for algebra
    M = torch.isfinite(clr).to(clr.dtype)              # (n, d) 1 where finite else 0
    Y = torch.nan_to_num(clr, nan=0.0, 
                         posinf=0.0, neginf=0.0)       # replace NaN with 0 only for algebra
    A = Y.square()                                     # (n, d)

    # Overlap counts for each (i, j): K = M^T M
    K = M.transpose(0, 1) @ M                          # (d, d)

    # Sum of squared diffs over overlapping rows (all pairs)
    # Σ (x_i - x_j)^2 over overlap = Σ x_i^2 (only rows where j present)
    #                               + Σ x_j^2 (only rows where i present)
    #                               - 2 Σ x_i x_j (only overlap contributes since NaNs->0)
    S = A.transpose(0, 1) @ M                          # S[i, j] = Σ_k x_i(k)^2 * 1{j present at k}
    P = Y.transpose(0, 1) @ Y                          # P[i, j] = Σ_k x_i(k) x_j(k) (zeros where any NaN)
    sumsq = S + S.transpose(0, 1) - 2.0 * P            # (d, d)

    # VLR = sumsq / K, with K==0 -> NaN
    vlr = sumsq / K.clamp_min(1)
    vlr = torch.where(K > 0, vlr, torch.nan)

    # numeric hygiene: symmetry, non-negativity, zero diagonal
    vlr = 0.5 * (vlr + vlr.transpose(0, 1))
    vlr = vlr.clamp_min(0.0)
    vlr.fill_diagonal_(0.0)
    return vlr

@torch.no_grad()
def _compute_correlation(data: torch.Tensor, threshold: float = 0.2, eps: float = 1e-12) -> torch.Tensor:
    if not torch.is_floating_point(data):
        data = data.float()
    n, d = data.shape
    if d < 3:
        raise ValueError("Need at least d >= 3 features to compute stable correlations.")

    # --- 1. Log-transform with NaN protection
    log_x = torch.log(data)
    log_x[~torch.isfinite(log_x)] = torch.nan

    # --- 2. CLR transform by samples (row-wise mean)
    shift = torch.nanmean(log_x, dim=1, keepdim=True)  # (n, 1)
    clr = log_x - shift                                # (n, d)

    # --- 3. Compute VLR matrix 
    vlr = _compute_vlr(data)                      # (d, d)

    # --- 4. Variance of CLR-transformed log abundances for each feature
    clr_var = _torch_nanvar(clr, dim=0, unbiased=False)  # (d,)

    # --- 5. Apply normalization
    sum_log_var = torch.nansum(clr_var) * d / (d - 1)
    log_var = (clr_var - (1 / d**2) * sum_log_var) * d / (d - 2)
    log_var = torch.nan_to_num(log_var, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(eps)

    # --- 6. Compute pairwise correlation matrix
    std = torch.sqrt(log_var)                          # (d,)
    log_std_prod = (std.unsqueeze(0) * std.unsqueeze(1)).clamp_min(eps)

    lv1 = log_var.expand(d, d)
    lv2 = log_var.view(-1, 1).expand(d, d)
    numer = (vlr - lv1 - lv2)                          # can be NaN where K==0
    denom = (-2.0 * log_std_prod)                      # finite and ≤ -2*eps

    rho = numer / denom                                # may produce NaN where numer is NaN
    # Map all non-finite correlations (incl. K==0 cases) to 0.0 (neutral)
    rho = torch.nan_to_num(rho, nan=0.0, posinf=0.0, neginf=0.0)

    # --- 7. Cleanup: clamp, symmetrize, set diagonal = 1
    rho = 0.5 * (rho + rho.T)
    rho = rho.clamp(-1.0, 1.0)
    rho.fill_diagonal_(1.0)
    
    # --- 8. Apply hard threshold 
    rho = torch.where(rho.abs() >= threshold, rho, torch.zeros_like(rho))
    
    # Convert to sparse tensor (COO format)
    sparse_rho = rho.to_sparse_coo()

    return sparse_rho

# Helper functions for hypothesis testing based on VLR

@torch.no_grad()
def _norm_vlr(t: torch.Tensor) -> torch.Tensor:
    if not torch.is_floating_point(t):
        t = t.float()
    d = t.shape[0]

    # medians
    t_colmed = t.median(dim=0).values              # (d,)
    t_med = t.median()                             # scalar

    colmed_matrix1 = t_colmed.unsqueeze(0).expand(d, -1)   # (d, d)
    colmed_matrix2 = t_colmed.unsqueeze(1).expand(-1, d)   # (d, d)

    denom = torch.sqrt(colmed_matrix1 * colmed_matrix2)
    # match numpy semantics: if denom==0, result -> inf/NaN; clamp if you prefer stability
    norm_t = (t - colmed_matrix1 - colmed_matrix2 + t_med) / denom
    return norm_t

@torch.no_grad()
def _by_sample_permute(data: torch.Tensor) -> torch.Tensor:
    if not torch.is_floating_point(data):
        data = data.float()
    n, d = data.shape

    # Column-wise random permutations via argsort of random noise
    idx = torch.argsort(torch.rand(n, d, device=data.device), dim=0)   # (n, d)
    permuted = data.gather(0, idx)                                     # (n, d)
    return permuted

@torch.no_grad()
def _es_direction(p_below: torch.Tensor, p_above: torch.Tensor):
    if p_below.dtype != torch.float32 and p_below.dtype != torch.float64:
        p_below = p_below.float()
    if p_above.dtype != torch.float32 and p_above.dtype != torch.float64:
        p_above = p_above.float()

    d = p_below.shape[0]
    direct = torch.zeros((d, d), dtype=torch.int8, device=p_below.device)  # 0='increase'
    direct[p_below > p_above] = 1                                          # 1='decrease'
    direct.fill_diagonal_(2)                                               # 2='unchanged'

    code_map = {0: "increase", 1: "decrease", 2: "unchanged"}
    return direct, code_map