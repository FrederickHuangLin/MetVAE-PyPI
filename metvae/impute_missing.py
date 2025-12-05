import torch
from typing import Optional, Tuple

# Helper functions for missing value imputation

def _build_X(
        meta: Optional[torch.Tensor], 
        n: int, dtype: torch.dtype, 
        device: torch.device
        ) -> torch.Tensor:
    if meta is None:
        return torch.ones(n, 1, dtype=dtype, device=device)     # intercept only
    X = meta.to(dtype=dtype, device=device)
    if X.ndim == 1:
        X = X.unsqueeze(1)
    return torch.cat([torch.ones(n, 1, dtype=dtype, device=device), X], dim=1)  # add intercept

def _ols_estimate(
        Y: torch.Tensor,                      # (n, d) with NaNs marking “missing”
        meta: Optional[torch.Tensor] = None,  # None or (n, p)
        ridge_eps: float = 1e-8,              # optional tiny ridge like 1e-8 for stability
        cond_warn: float = 1e12               # ill-conditioning threshold
        ) -> torch.Tensor:
    """
    Batched OLS handling NaNs per column via weights W = 1{observed}.
    Returns zeros for columns that cannot be estimated.
    
    Returns:
        estimates: (p'+1, d) = [beta rows; log(scale)]
    """
    device, dtype = Y.device, Y.dtype
    n, d = Y.shape

    # --- Design matrix with intercept ---
    X = _build_X(meta, n, dtype, device)    # (n, p')
    p_prime = X.shape[1]

    # --- Weights & filled Y ---
    W = (~torch.isnan(Y)).to(dtype=dtype)   # (n, d) 1 if observed else 0
    Y_filled = torch.nan_to_num(Y, nan=0.0) # (n, d)

    # --- Compute XtWX and XtWy in batch WITHOUT einsum bugs ---
    # For each feature j: XtWX_j = X^T (diag(w_j) X) = X^T (X * w_j[:,None])
    X_t = X.transpose(0, 1)                               # (p', n)
    Xw = X.unsqueeze(2) * W.unsqueeze(1)                  # (n, p', d)
    # Bring feature dim forward: (d, n, p')
    Xw_d = Xw.permute(2, 0, 1)                            # (d, n, p')
    # Batched matmul: (1, p', n) @ (d, n, p') -> (d, p', p')  (broadcast on leading dim)
    XtWX = torch.matmul(X_t.unsqueeze(0), Xw_d)           # (d, p', p')
    # (p', d): X^T (W * Y)
    XtWy = X_t @ (W * Y_filled)                           # (p', d)

    # Optional ridge (stabilize if badly conditioned)
    if ridge_eps > 0:
        I = torch.eye(p_prime, dtype=dtype, device=device).unsqueeze(0) # (1, p', p')
        XtWX = XtWX + ridge_eps * I                                     # (d, p', p')

    # --- Solve (XtWX_j) beta_j = XtWy_j for all j ---
    A = XtWX                                                         # (d, p', p')
    B = XtWy.T.unsqueeze(2)                                          # (d, p', 1)

    # Condition numbers (per feature) for diagnostics
    svals = torch.linalg.svdvals(A)                                  # (d, min(p',p'))
    cond = (svals[..., 0] / svals[..., -1].clamp_min(torch.finfo(dtype).eps))  # (d,)

    beta_d = torch.zeros_like(B)                                     # (d, p', 1)
    # Try direct solve where well-conditioned; fallback to pinv
    well = torch.isfinite(cond) & (cond < 1/torch.finfo(dtype).eps)
    if well.any():
        beta_d[well] = torch.linalg.solve(A[well], B[well])
    if (~well).any():
        A_pinv = torch.linalg.pinv(A[~well])
        beta_d[~well] = A_pinv @ B[~well]

    beta = beta_d.squeeze(2).T                                       # (p', d)

    # --- Residuals / scale computed on observed rows only ---
    Y_hat = X @ beta                                                 # (n, d)
    resid = torch.where(W.bool(), Y_filled - Y_hat, torch.zeros_like(Y_filled))
    sse = (resid ** 2).sum(dim=0)                                    # (d,)

    # df_resid per column: (#observed_j - rank_j), clamp at 1
    # Rank of XtWX_j equals rank of masked design for column j
    rank = torch.linalg.matrix_rank(A)                               # (d,)
    n_obs = W.sum(dim=0)                                             # (d,)
    df_resid = torch.clamp((n_obs - rank).to(dtype=dtype), min=1)

    scale = sse / df_resid                                           # (d,)
    log_scale = torch.log(scale).unsqueeze(0)                        # (1, d)

    estimates = torch.cat([beta, log_scale], dim=0)                  # (p'+1, d)
    
    # --- Identify bad columns and zero them out ---
    # Reasons: any NaN/Inf, ill-conditioned, rank deficiency, or too few obs (< p')
    per_col_bad = torch.isnan(estimates).any(dim=0) | torch.isinf(estimates).any(dim=0)
    ill = cond > cond_warn
    low_rank = rank < p_prime
    few_obs = n_obs < p_prime
    bad = per_col_bad | ill | low_rank | few_obs

    if bad.any():
        estimates[:, bad] = 0.0

    return estimates

@torch.no_grad()
def _tobit_em_warmstart(
    Y: torch.Tensor,                 # (n, d), NaN marks left-censored
    meta: Optional[torch.Tensor],    # None or (n, p)
    th: torch.Tensor,                # (n, d) or broadcastable to it
    init_estimates: torch.Tensor,    # (p′+1, d) from your OLS initializer (rows: beta; log(scale))
    steps: int = 20,
    tol: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Batched EM for censored normal (Tobit). Returns estimates stacked by rows:
    (p′+1, d) = [beta rows; log(sigma)].
    """
    device, dtype = Y.device, Y.dtype
    n, d = Y.shape
    X = _build_X(meta, n, dtype, device)             # (n, p′)
    p_prime = X.shape[1]

    # Broadcast threshold to (n, d)
    th = th.to(dtype=dtype, device=device)
    if th.ndim == 0:
        th = th.expand_as(Y)
    elif th.ndim == 1:
        th = th.view(1, -1).expand_as(Y)
    else:
        th = th.expand_as(Y)

    # Masks
    unc = ~torch.isnan(Y)                             # (n, d)
    cens = ~unc

    # Initial parameters from OLS: last row is log(scale) with scale = sigma^2
    est0 = init_estimates.to(dtype=dtype, device=device)
    assert est0.shape == (p_prime + 1, d), f"init_estimates must be {(p_prime+1, d)}"
    beta = est0[:-1, :]                               # (p′, d)
    sigma = torch.exp(0.5 * est0[-1:, :]).clamp_min(1e-8)  # (1, d)  sigma = sqrt(scale)

    # Precompute QR for fast OLS each M-step: beta = R^{-1} Q^T Ybar
    Q, R = torch.linalg.qr(X, mode="reduced")         # Q: (n,p′), R: (p′,p′)

    normal = torch.distributions.Normal(
        loc=torch.zeros((), device=device, dtype=dtype),
        scale=torch.ones((), device=device, dtype=dtype)
    )

    prev_obj = torch.tensor(float("inf"), device=device, dtype=dtype)

    for it in range(steps):
        # ----- E-step using current (beta, sigma) -----
        mu = X @ beta                                  # (n, d)
        a = (th - mu) / sigma                          # (n, d)
        # Clamp Phi to avoid 0; pdf is fine
        Phi = normal.cdf(a).clamp_min(torch.finfo(dtype).eps)
        phi = torch.exp(normal.log_prob(a))
        lam = (phi / Phi)                              # (n, d)

        # Expected y* for censored; observed y for uncensored
        y_bar = torch.where(unc, Y, mu - sigma * lam)  # (n, d)

        # Var term for censored: E[(y*-mu)^2 | cens, old]
        var_cens = (sigma ** 2) * (1.0 - a * lam - lam * lam)  # (n, d)

        # ----- M-step -----
        # Update beta (OLS on y_bar) using precomputed QR
        QtY = Q.transpose(0, 1) @ y_bar               # (p′, d)
        beta_new = torch.linalg.solve_triangular(R, QtY, upper=True)  # (p′, d)

        # Update sigma (per feature)
        mu_new = X @ beta_new                         # (n, d)
        # Uncensored residuals
        res_unc = torch.where(unc, (Y - mu_new), torch.zeros_like(Y))
        sse_unc = (res_unc ** 2).sum(dim=0)           # (d,)

        # Censored expected squared residuals:
        # E[(y* - mu_new)^2] = Var_old + (E[y*]_old - mu_new)^2
        mean_offset2 = torch.where(cens, (y_bar - mu_new) ** 2, torch.zeros_like(Y))
        var_term = torch.where(cens, var_cens, torch.zeros_like(Y))
        sse_cens = (mean_offset2 + var_term).sum(dim=0)   # (d,)

        sse_total = sse_unc + sse_cens                    # (d,)
        sigma_new = torch.sqrt((sse_total / n).clamp_min(1e-16)).unsqueeze(0)  # (1, d)

        # Convergence check (relative change in params or obj)
        # Use total negative Q-function proxy: sum of sse_total + log(sigma) terms
        obj = sse_total.sum() + torch.log(sigma_new).sum()
        rel_change = torch.abs(obj - prev_obj) / (torch.abs(prev_obj) + 1e-12)
        prev_obj = obj

        beta, sigma = beta_new, sigma_new

        if rel_change.item() < tol:
            break
    return X, th, beta, torch.log(sigma)

def _fit_censored_normal(
    Y: torch.Tensor,                       # (n, d), NaN marks censored obs
    meta: Optional[torch.Tensor],          # None or (n, p) (intercept added inside)
    th: torch.Tensor,                      # (n,) or (n,1) censoring thresholds (same scale as Y)
    init_estimates: torch.Tensor,          # (p'+1, d) from _ols_estimate = [beta; log(variance)]
    max_iter: int = 100,
    tol: float = 1e-6,
    sigma_floor: float = 1e-6,
    line_search: str = "strong_wolfe",
    em_steps: int = 20
) -> torch.Tensor:
    """
    Vectorized censored Normal (Tobit) MLE via LBFGS, initialized from _ols_estimate.
    Returns (beta_hat, sigma_hat, estimates) where estimates = [beta_hat; log(sigma_hat^2)].
    """
    device, dtype = Y.device, Y.dtype
    n, d = Y.shape
    mask_obs = ~torch.isnan(Y)                           # (n, d)
    
    # --- Unpack initial estimates ---
    X, thu, beta0, logsigma0 = _tobit_em_warmstart(
        Y=Y, meta=meta, th=th, 
        init_estimates=init_estimates, 
        steps=em_steps, tol=tol
    )
    
    p_prime = X.shape[1] # X shape: (n, p')
    
    # --- Validate init_estimates ---
    est0 = init_estimates.to(dtype=dtype, device=device)
    assert est0.shape == (p_prime + 1, d), f"init_estimates must be {(p_prime + 1, d)}"

    # Trainable parameters: stack [beta; rho]
    params = torch.nn.Parameter(torch.cat([beta0, logsigma0], dim=0).clone())  # (p′ + 1, d)

    def nll():
        beta = params[:-1, :]                         # (p′, d)
        logsigma = params[-1:, :]                     # (1, d)
        sigma = torch.exp(logsigma).clamp_min(sigma_floor)  # (1, d)
        inv_sigma = 1.0 / sigma                       # (1, d)
        
        mu = X @ beta                                 # (n, d)

        # --- Uncensored observations ---
        U = mask_obs
        e = torch.where(U, Y - mu, torch.zeros_like(Y))
        ll_unc = (U * (-torch.log(sigma) - 0.5 * (e * inv_sigma) ** 2)).sum()

        # --- Censored observations ---
        C = ~U
        z = (th - mu) * inv_sigma                     # (n, d)
        zC = torch.masked_select(z, C)
        ll_cens = torch.special.log_ndtr(zC).sum() if zC.numel() > 0 else torch.zeros((), device=device, dtype=dtype)

        return -(ll_unc + ll_cens)

    optimizer = torch.optim.LBFGS(
        [params],
        max_iter=max_iter,
        tolerance_grad=tol,
        tolerance_change=tol,
        line_search_fn=line_search,
    )

    def closure():
        optimizer.zero_grad(set_to_none=True)
        loss = nll()
        loss.backward()
        return loss

    try:
        optimizer.step(closure)
    except Exception:
        # Fallback to EM-only result if LBFGS fails
        with torch.no_grad():
            params.copy_(torch.cat([beta0, logsigma0], dim=0))

    # --- Extract estimates ---
    params = params.detach()

    # Safety: replace any NaN/Inf with 0
    bad = ~torch.isfinite(params)
    if bad.any():
        params[bad] = est0[bad]
    return params
    
    
    
    
    
    
    
    
    