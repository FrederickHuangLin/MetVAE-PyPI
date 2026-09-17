import torch
from typing import Optional, Tuple

# Helper functions for missing value imputation

def _build_X(
        meta: Optional[torch.Tensor],
        n: int, dtype: torch.dtype,
        device: torch.device
        ) -> torch.Tensor:
    """
    Build the design matrix with an intercept column.

    Parameters
    ----------
    meta : torch.Tensor or None
        Covariate matrix of shape (n, p) or (n,). None gives an intercept-only
        design.
    n : int
        Number of observations.
    dtype : torch.dtype
        Dtype of the returned matrix.
    device : torch.device
        Device of the returned matrix.

    Returns
    -------
    torch.Tensor
        Design matrix of shape (n, p + 1).
    """
    if meta is None:
        return torch.ones(n, 1, dtype=dtype, device=device)
    X = meta.to(dtype=dtype, device=device)
    if X.ndim == 1:
        X = X.unsqueeze(1)
    return torch.cat([torch.ones(n, 1, dtype=dtype, device=device), X], dim=1)

def _ols_estimate(
        Y: torch.Tensor,
        meta: Optional[torch.Tensor] = None,
        ridge_eps: float = 1e-8,
        cond_warn: float = 1e12,
        chunk_bytes: int = 256 * 1024 * 1024
        ) -> torch.Tensor:
    """
    Weighted least squares fit of each column of Y on a common design matrix.

    A missing entry of Y is marked by NaN and receives weight zero. The residual
    scale is computed on the observed rows only. Columns that cannot be
    estimated are returned as zero.

    Parameters
    ----------
    Y : torch.Tensor
        Response matrix of shape (n, d) with NaN marking missing entries.
    meta : torch.Tensor or None
        Covariate matrix of shape (n, p). None gives an intercept-only design.
    ridge_eps : float
        Ridge term added to the diagonal of each normal-equation matrix.
    cond_warn : float
        Condition number above which a column is treated as not estimable.
    chunk_bytes : int
        Memory budget of the intermediate weighted design used to accumulate the
        normal equations.

    Returns
    -------
    torch.Tensor
        Estimates of shape (p + 2, d) holding the regression coefficients in the
        leading rows and log(sigma) in the last row.
    """
    device, dtype = Y.device, Y.dtype
    n, d = Y.shape

    X = _build_X(meta, n, dtype, device)
    p_prime = X.shape[1]

    W = (~torch.isnan(Y)).to(dtype=dtype)
    Y_filled = torch.nan_to_num(Y, nan=0.0)

    # Normal equations per column, accumulated over chunks of columns
    X_t = X.transpose(0, 1)
    itemsize = torch.finfo(dtype).bits // 8
    chunk = max(1, int(chunk_bytes) // max(1, n * p_prime * itemsize))
    XtWX = torch.empty((d, p_prime, p_prime), dtype=dtype, device=device)
    for j0 in range(0, d, chunk):
        j1 = min(j0 + chunk, d)
        Xw_d = (X.unsqueeze(2) * W[:, j0:j1].unsqueeze(1)).permute(2, 0, 1)
        XtWX[j0:j1] = torch.matmul(X_t.unsqueeze(0), Xw_d)
    XtWy = X_t @ (W * Y_filled)

    if ridge_eps > 0:
        I = torch.eye(p_prime, dtype=dtype, device=device).unsqueeze(0)
        XtWX = XtWX + ridge_eps * I

    A = XtWX
    B = XtWy.T.unsqueeze(2)

    # Singular values give both the condition number and the rank
    svals = torch.linalg.svdvals(A)
    cond = (svals[..., 0] / svals[..., -1].clamp_min(torch.finfo(dtype).eps))
    rank = (svals > p_prime * torch.finfo(dtype).eps * svals[..., :1]).sum(-1)

    beta_d = torch.zeros_like(B)
    well = torch.isfinite(cond) & (cond < 1/torch.finfo(dtype).eps)
    if well.any():
        beta_d[well] = torch.linalg.solve(A[well], B[well])
    if (~well).any():
        A_pinv = torch.linalg.pinv(A[~well])
        beta_d[~well] = A_pinv @ B[~well]

    beta = beta_d.squeeze(2).T

    Y_hat = X @ beta
    resid = torch.where(W.bool(), Y_filled - Y_hat, torch.zeros_like(Y_filled))
    sse = (resid ** 2).sum(dim=0)

    n_obs = W.sum(dim=0)
    df_resid = torch.clamp((n_obs - rank).to(dtype=dtype), min=1)

    scale = sse / df_resid
    log_sigma = (0.5 * torch.log(scale)).unsqueeze(0)

    estimates = torch.cat([beta, log_sigma], dim=0)

    # Columns that are not estimable are set to zero
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
    Y: torch.Tensor,
    meta: Optional[torch.Tensor],
    th: torch.Tensor,
    init_estimates: torch.Tensor,
    steps: int = 20,
    tol: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Expectation maximization for the left-censored normal model.

    Parameters
    ----------
    Y : torch.Tensor
        Response matrix of shape (n, d) with NaN marking a left-censored entry.
    meta : torch.Tensor or None
        Covariate matrix of shape (n, p). None gives an intercept-only design.
    th : torch.Tensor
        Censoring thresholds, broadcastable to the shape of Y.
    init_estimates : torch.Tensor
        Starting values of shape (p + 2, d) holding the regression coefficients
        in the leading rows and log(sigma) in the last row.
    steps : int
        Maximum number of iterations.
    tol : float
        Relative change of the monitored objective below which the iteration
        stops.

    Returns
    -------
    X : torch.Tensor
        Design matrix of shape (n, p + 1).
    th : torch.Tensor
        Censoring thresholds broadcast to the shape of Y.
    beta : torch.Tensor
        Regression coefficients of shape (p + 1, d).
    log_sigma : torch.Tensor
        Logarithm of the residual standard deviation, shape (1, d).

    Raises
    ------
    ValueError
        If init_estimates does not have shape (p + 2, d).
    """
    device, dtype = Y.device, Y.dtype
    n, d = Y.shape
    X = _build_X(meta, n, dtype, device)
    p_prime = X.shape[1]

    th = th.to(dtype=dtype, device=device)
    if th.ndim == 0:
        th = th.expand_as(Y)
    elif th.ndim == 1:
        th = th.view(1, -1).expand_as(Y)
    else:
        th = th.expand_as(Y)

    unc = ~torch.isnan(Y)
    cens = ~unc

    est0 = init_estimates.to(dtype=dtype, device=device)
    if est0.shape != (p_prime + 1, d):
        raise ValueError(
            f"init_estimates must have shape {(p_prime + 1, d)}; "
            f"got {tuple(est0.shape)}."
        )
    beta = est0[:-1, :]
    sigma = torch.exp(est0[-1:, :]).clamp_min(1e-8)

    # QR of the design matrix for the least squares step of each iteration
    Q, R = torch.linalg.qr(X, mode="reduced")

    normal = torch.distributions.Normal(
        loc=torch.zeros((), device=device, dtype=dtype),
        scale=torch.ones((), device=device, dtype=dtype)
    )

    prev_obj = torch.tensor(float("inf"), device=device, dtype=dtype)

    for it in range(steps):
        # E-step
        mu = X @ beta
        a = (th - mu) / sigma
        Phi = normal.cdf(a).clamp_min(torch.finfo(dtype).eps)
        phi = torch.exp(normal.log_prob(a))
        lam = (phi / Phi)

        # Conditional mean of the latent response
        y_bar = torch.where(unc, Y, mu - sigma * lam)

        # Conditional variance of the latent response at a censored entry
        var_cens = (sigma ** 2) * (1.0 - a * lam - lam * lam)

        # M-step
        QtY = Q.transpose(0, 1) @ y_bar
        beta_new = torch.linalg.solve_triangular(R, QtY, upper=True)

        mu_new = X @ beta_new
        res_unc = torch.where(unc, (Y - mu_new), torch.zeros_like(Y))
        sse_unc = (res_unc ** 2).sum(dim=0)

        mean_offset2 = torch.where(cens, (y_bar - mu_new) ** 2, torch.zeros_like(Y))
        var_term = torch.where(cens, var_cens, torch.zeros_like(Y))
        sse_cens = (mean_offset2 + var_term).sum(dim=0)

        sse_total = sse_unc + sse_cens
        sigma_new = torch.sqrt((sse_total / n).clamp_min(1e-16)).unsqueeze(0)

        # Monitored quantity: total expected sum of squares plus summed log(sigma)
        obj = sse_total.sum() + torch.log(sigma_new).sum()
        rel_change = torch.abs(obj - prev_obj) / (torch.abs(prev_obj) + 1e-12)
        prev_obj = obj

        beta, sigma = beta_new, sigma_new

        if rel_change.item() < tol:
            break
    return X, th, beta, torch.log(sigma)

def _fit_censored_normal(
    Y: torch.Tensor,
    meta: Optional[torch.Tensor],
    th: torch.Tensor,
    init_estimates: torch.Tensor,
    max_iter: int = 100,
    tol: float = 1e-6,
    sigma_floor: float = 1e-6,
    line_search: str = "strong_wolfe",
    em_steps: int = 20
) -> torch.Tensor:
    """
    Maximum likelihood fit of the left-censored normal model by LBFGS.

    The optimizer is started from the expectation maximization solution returned
    by _tobit_em_warmstart. If the optimizer raises, the starting values are
    returned. Non-finite entries of the solution are replaced by the
    corresponding entries of init_estimates.

    Parameters
    ----------
    Y : torch.Tensor
        Response matrix of shape (n, d) with NaN marking a left-censored entry.
    meta : torch.Tensor or None
        Covariate matrix of shape (n, p). None gives an intercept-only design.
    th : torch.Tensor
        Censoring thresholds, broadcastable to the shape of Y.
    init_estimates : torch.Tensor
        Starting values of shape (p + 2, d) holding the regression coefficients
        in the leading rows and log(sigma) in the last row.
    max_iter : int
        Maximum number of LBFGS iterations.
    tol : float
        Tolerance on the gradient and on the parameter change.
    sigma_floor : float
        Lower bound on the residual standard deviation.
    line_search : str
        Line search used by LBFGS.
    em_steps : int
        Maximum number of expectation maximization iterations.

    Returns
    -------
    torch.Tensor
        Estimates of shape (p + 2, d) holding the regression coefficients in the
        leading rows and log(sigma) in the last row.
    """
    device, dtype = Y.device, Y.dtype
    mask_obs = ~torch.isnan(Y)

    X, thu, beta0, logsigma0 = _tobit_em_warmstart(
        Y=Y, meta=meta, th=th,
        init_estimates=init_estimates,
        steps=em_steps, tol=tol
    )

    est0 = init_estimates.to(dtype=dtype, device=device)

    params = torch.nn.Parameter(torch.cat([beta0, logsigma0], dim=0).clone())

    def nll():
        beta = params[:-1, :]
        logsigma = params[-1:, :]
        sigma = torch.exp(logsigma).clamp_min(sigma_floor)
        inv_sigma = 1.0 / sigma

        mu = X @ beta

        U = mask_obs
        e = torch.where(U, Y - mu, torch.zeros_like(Y))
        ll_unc = (U * (-torch.log(sigma) - 0.5 * (e * inv_sigma) ** 2)).sum()

        C = ~U
        z = (th - mu) * inv_sigma
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
        with torch.no_grad():
            params.copy_(torch.cat([beta0, logsigma0], dim=0))

    params = params.detach()

    bad = ~torch.isfinite(params)
    if bad.any():
        params[bad] = est0[bad]
    return params
