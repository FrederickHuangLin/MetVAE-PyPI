import os
import warnings
from typing import Optional, Dict, Sequence, Tuple, List
import pandas as pd
import math
import torch
from concurrent.futures import ThreadPoolExecutor
from statsmodels.stats.multitest import multipletests
from .compute_corr import _compute_correlation
from .utils import _thread_limit

# Helper functions for p-values filtering

def _p_filter(mat: torch.Tensor,
              mat_p: torch.Tensor,
              max_p: float,
              impute_value: float = 0.0) -> torch.Tensor:
    """
    Replace entries whose p-value exceeds a cutoff.

    Parameters
    ----------
    mat : torch.Tensor
        Matrix of estimates.
    mat_p : torch.Tensor
        Matrix of p-values with the same shape as mat.
    max_p : float
        Cutoff above which an entry of mat is replaced.
    impute_value : float
        Value written in place of a filtered entry.

    Returns
    -------
    torch.Tensor
        Copy of mat with filtered entries set to impute_value.
    """
    return torch.where(mat_p > max_p, mat.new_full((), impute_value), mat)

def _matrix_p_adjust(p_matrix: torch.Tensor,
                     method: str = 'fdr_bh') -> torch.Tensor:
    """
    Adjust the p-values of a symmetric matrix for multiple comparisons.

    The strict lower triangle is adjusted as a single vector and the result is
    mirrored into the upper triangle. The diagonal is zero.

    Parameters
    ----------
    p_matrix : torch.Tensor
        Square matrix of p-values.
    method : str
        Method passed to statsmodels.stats.multitest.multipletests.

    Returns
    -------
    torch.Tensor
        Symmetric matrix of adjusted p-values with the same shape, dtype and
        device as p_matrix.
    """
    device = p_matrix.device
    dtype = p_matrix.dtype
    n = p_matrix.shape[0]

    # Strict lower triangle in row-major order
    tril_mask = torch.ones((n, n), device=device, dtype=torch.bool).tril_(-1)
    p_vec = p_matrix.masked_select(tril_mask).detach().to('cpu').numpy()

    _, q_vec, _, _ = multipletests(p_vec, method=method)

    q_mat = torch.zeros((n, n), device=device, dtype=dtype)
    q_mat.masked_scatter_(tril_mask,
                          torch.from_numpy(q_vec).to(device=device, dtype=dtype))
    q_mat.add_(q_mat.T.contiguous())
    return q_mat

# Helper functions for SEC

@torch.no_grad()
def _projection_psd(
        A: torch.Tensor, *,
        jitter0: float = 1e-12,
        max_retries: int = 5,
        identity: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Project a matrix onto the positive semidefinite cone.

    The input is symmetrized, non-finite entries are replaced by finite values,
    and the eigendecomposition is taken in float64 after adding a diagonal
    jitter proportional to the mean absolute entry. The jitter is multiplied by
    ten after a failed decomposition.

    Parameters
    ----------
    A : torch.Tensor
        Square matrix, or a batch of square matrices.
    jitter0 : float
        Relative size of the initial diagonal jitter.
    max_retries : int
        Number of jitter escalations after the first attempt.
    identity : torch.Tensor or None
        Precomputed identity matrix. It is used only when its shape, dtype and
        device match the float64 working copy of A.

    Returns
    -------
    torch.Tensor
        Positive semidefinite matrix in the dtype of A.

    Raises
    ------
    RuntimeError
        If the eigendecomposition fails at every jitter level.
    """
    orig_dtype = A.dtype
    A = 0.5 * (A + A.transpose(-1, -2))
    A.nan_to_num_()

    # The eigendecomposition is taken in float64
    A64 = A if A.dtype == torch.float64 else A.to(torch.float64)

    if (identity is not None
            and identity.dtype == A64.dtype
            and identity.device == A64.device
            and identity.shape == A64.shape[-2:]):
        I = identity
    else:
        I = torch.eye(A64.shape[-1], device=A64.device, dtype=A64.dtype)

    scale = A64.abs().mean()
    base = (scale if torch.isfinite(scale) and scale > 0 else 1.0)
    jitter = jitter0 * base

    last_attempt = max_retries + 1
    for attempt in range(last_attempt + 1):
        try:
            evals, evecs = torch.linalg.eigh(A64 + jitter * I)
            evals = evals.clamp_min(0.0)
            out64 = (evecs * evals.unsqueeze(-2)) @ evecs.transpose(-1, -2)
            out64 = 0.5 * (out64 + out64.transpose(-1, -2))
            return out64.to(orig_dtype)
        except RuntimeError as err:
            if attempt == last_attempt:
                raise RuntimeError(
                    f"PSD projection failed after {attempt + 1} eigh attempts; "
                    f"final jitter {float(jitter):.3e}."
                ) from err
            jitter *= 10.0

@torch.no_grad()
def _SEC_dense(
    Rn: torch.Tensor,
    rho: float,
    *,
    epsilon: float = 1e-5,
    tol: float = 1e-3,
    max_iter: int = 1000,
    restart: Optional[int] = 50,
    line_search_apg: bool = True,
    delta: Optional[float] = None,
    n_samples: Optional[int] = None,
    c_delta: float = .1,
    threshold: float = 0.1
) -> torch.Tensor:
    """
    Sparse estimation of a correlation matrix, dense output.

    The problem solved is

        min_R 0.5 * ||R - Rn||_F^2 + rho * ||W * R||_1
        subject to R - epsilon * I positive semidefinite and R_ii = 1,

    where W is the entrywise weight matrix W_ij = 1 / |Rn_ij| for |Rn_ij| > delta
    and W_ij = 0 otherwise, and W * R is the entrywise product. The solver is an
    accelerated proximal gradient scheme with Nesterov restarts and projection
    onto the positive semidefinite cone at every iteration. The iterate is
    rescaled to unit diagonal on exit and off-diagonal entries below threshold in
    absolute value are set to zero.

    Parameters
    ----------
    Rn : torch.Tensor
        Square sample correlation matrix, dense or sparse COO.
    rho : float
        Penalty parameter of the weighted L1 term.
    epsilon : float
        Lower bound on the eigenvalues of the solution.
    tol : float
        Convergence tolerance on the relative gradient residual.
    max_iter : int
        Maximum number of proximal gradient iterations.
    restart : int or None
        Period of the Nesterov momentum restart. None or a value below 1
        disables restarts.
    line_search_apg : bool
        If True, the step size is adapted from the residual history; otherwise a
        fixed step size is used.
    delta : float or None
        Cutoff below which an off-diagonal entry of Rn is treated as zero. If
        None, it is set to c_delta * sqrt(log(p) / n_samples) when n_samples is
        given and to 1e-6 otherwise.
    n_samples : int or None
        Sample size used to derive delta.
    c_delta : float
        Constant in the expression for delta.
    threshold : float
        Hard threshold applied to the off-diagonal entries of the solution. A
        value of zero or less disables thresholding.

    Returns
    -------
    torch.Tensor
        Dense symmetric matrix of shape (p, p) with unit diagonal up to the
        epsilon offset.

    Raises
    ------
    ValueError
        If Rn is not square or if max_iter is less than one.

    Notes
    -----
    The sample correlation matrix is standardized by column mean and column
    standard deviation before the proximal gradient iterations. In Cui, Leng and
    Sun (2016) the standardization applies to the data variables rather than to
    the correlation matrix. The implementation retains the standardization of
    the correlation matrix.

    References
    ----------
    Cui, Y., Leng, C. and Sun, D. (2016). Sparse estimation of high-dimensional
    correlation matrices. Computational Statistics and Data Analysis, 93, 390-403.
    Reference implementation:
    https://warwick.ac.uk/fac/sci/statistics/staff/academic-research/leng/publications/sec.m
    """
    if Rn.is_sparse:
        Rn = Rn.coalesce().to_dense()
    if Rn.ndim != 2 or Rn.shape[0] != Rn.shape[1]:
        raise ValueError(f"Rn must be square; got shape {tuple(Rn.shape)}.")
    if max_iter < 1:
        raise ValueError(f"max_iter must be at least 1; got {max_iter}.")
    p = Rn.shape[0]
    device, dtype = Rn.device, Rn.dtype
    zero = Rn.new_zeros(())

    # Cutoff below which an off-diagonal entry of Rn is treated as zero
    if delta is None:
        if n_samples is not None and n_samples > 0:
            delta = float(c_delta * math.sqrt(max(math.log(p) / n_samples, 0.0)))
        else:
            delta = 1e-6

    Z = torch.zeros((p, p), device=device, dtype=dtype)
    b_vec = torch.ones((p,), device=device, dtype=dtype)

    abs_Rn = torch.abs(Rn)
    eye_mask = torch.eye(p, device=device, dtype=torch.bool)
    offdiag_mask = ~eye_mask
    tiny_mask = (abs_Rn < delta) & offdiag_mask
    W = torch.where(abs_Rn <= delta, zero, 1.0 / abs_Rn.clamp_min(1e-300))
    W.fill_diagonal_(0.0)

    # Column standardization of Rn followed by symmetrization
    col_mean = Rn.mean(dim=0, keepdim=True)
    col_std = Rn.std(dim=0, unbiased=True, keepdim=True).clamp_min(1e-12)
    Rn_work = (Rn - col_mean) / col_std
    Rn_work = 0.5 * (Rn_work + Rn_work.T)

    R = torch.zeros((p, p), device=device, dtype=dtype)
    Y = Z
    t = 1.0
    L = 1.0
    tau = 0.75
    eta = 0.9
    I = torch.eye(p, device=device, dtype=dtype)
    rhoW = rho * W
    epsI = epsilon * I

    res_old: Optional[float] = None

    for k in range(1, max_iter + 1):
        Yold = Y
        told = t

        X = Z + Rn_work

        R = torch.sign(X) * torch.clamp(torch.abs(X) - rhoW, min=0.0)
        R.masked_fill_(tiny_mask, 0.0)
        R.diagonal().fill_(1.0)
        R = 0.5 * (R + R.T)

        if line_search_apg:
            Y = _projection_psd(Z - (R - epsI) / tau, identity=I)
            res_gradient = (tau * torch.linalg.norm(Z - Y, ord='fro') /
                            (1.0 + torch.linalg.norm(Z, ord='fro')))
        else:
            Y = _projection_psd(Z - (R - epsI) / L, identity=I)
            res_gradient = (torch.linalg.norm(Z - Y, ord='fro') /
                            (1.0 + torch.linalg.norm(Z, ord='fro')))

        res_val = float(res_gradient)

        if line_search_apg:
            if (k % 5 == 0) and (tau < L) and (res_old is not None) and (res_val > res_old):
                tau = min(L, tau / eta)

        if k > 1:
            res_old = res_val

        if res_val <= tol:
            break

        if (restart is not None) and (restart > 0) and (k % restart == 0):
            t = 1.0
            told = t

        t = (1.0 + math.sqrt(1.0 + 4.0 * told * told)) / 2.0
        Z = Y + ((told - 1.0) / t) * (Y - Yold)

    # Rescaling to unit diagonal
    R_cal = R - epsI
    lam_min = torch.linalg.eigvalsh(R_cal).min().item()
    if lam_min < 0.0:
        R_cal = R_cal + (-lam_min) * I

    d = torch.diag(R_cal)
    if bool((d <= 0).any()):
        warnings.warn(
            "The calibrated matrix has nonpositive diagonal entries; the "
            "corresponding scaling factors are set to zero.",
            RuntimeWarning
        )
    d = d.clamp_min(1e-300)
    d = ((b_vec - epsilon) / d).clamp_min(0.0).sqrt()
    D = torch.diag(d)

    R_out = D @ R_cal @ D
    R_out = 0.5 * (R_out + R_out.T) + epsI

    # Hard threshold on the off-diagonal entries, diagonal preserved
    if threshold > 0.0:
        diag_vals = R_out.diagonal().clone()
        R_out = torch.where(R_out.abs() >= threshold, R_out, zero)
        R_out.diagonal().copy_(diag_vals)

    return R_out

@torch.no_grad()
def _SEC(Rn: torch.Tensor, rho: float, **kwargs) -> torch.Tensor:
    """
    Sparse estimation of a correlation matrix, sparse output.

    Parameters
    ----------
    Rn : torch.Tensor
        Square sample correlation matrix, dense or sparse COO.
    rho : float
        Penalty parameter of the weighted L1 term.
    **kwargs
        Keyword arguments of _SEC_dense.

    Returns
    -------
    torch.Tensor
        Coalesced sparse COO tensor holding the solution of _SEC_dense.
    """
    return _SEC_dense(Rn, rho, **kwargs).to_sparse_coo().coalesce()

@torch.no_grad()
def _SEC_cv(
    X: torch.Tensor,
    Rn: torch.Tensor,
    *,
    c_grid: Sequence[float] = tuple(float(x) for x in range(1, 11)),
    n_splits: int = 5,
    seed: int = 0,
    workers: int = -1,
    refine: bool = True,
    refine_points: int = 10,
    **sec_kwargs
) -> Tuple[float, "pd.DataFrame", torch.Tensor]:
    """
    Select the penalty parameter of _SEC_dense by K-fold cross-validation.

    The penalty is parameterized as rho = c * sqrt(log(p) / n). The grid c_grid
    is evaluated first, and the interval between the immediate neighbors of the
    best c is then evaluated on refine_points equally spaced values. The
    cross-validation score of a given rho is the mean over folds of the squared
    Frobenius norm of the difference between the estimate on the training fold
    and the sample correlation matrix of the validation fold.

    Parameters
    ----------
    X : torch.Tensor
        Data matrix of shape (n, p).
    Rn : torch.Tensor
        Sample correlation matrix on all n observations, dense or sparse COO.
    c_grid : sequence of float
        Values of c evaluated in the first pass.
    n_splits : int
        Number of folds.
    seed : int
        Seed of the permutation defining the folds.
    workers : int
        Number of worker threads used on CPU. None or a negative value means one
        worker per core. A value of one disables threading.
    refine : bool
        If True, a second pass is run between the neighbors of the best c.
    refine_points : int
        Number of equally spaced values in the refined interval.
    **sec_kwargs
        Keyword arguments of _SEC_dense.

    Returns
    -------
    best_rho : float
        Penalty with the smallest score, ties broken by the smaller rho.
    scores_df : pandas.DataFrame
        Columns c, rho and score, sorted by rho.
    R_hat_best : torch.Tensor
        Dense estimate obtained at best_rho from Rn.

    Raises
    ------
    ValueError
        If X is not two-dimensional, if n_splits is outside [2, n], or if a
        correlation matrix or an estimate has non-finite entries.
    """
    if X.ndim != 2:
        raise ValueError(f"X must be 2D (n x p); got shape {tuple(X.shape)}.")
    n, p = X.shape
    device, dtype = X.device, X.dtype

    if n_splits < 2 or n_splits > n:
        raise ValueError(f"n_splits must be in [2, n]; got {n_splits} for n={n}.")

    base = math.sqrt(max(math.log(p) / n, 0.0))
    def c_to_rho(c: float) -> float:
        return float(c) * base

    # Deterministic folds
    idx = torch.arange(n, device='cpu')
    g = torch.Generator(device='cpu').manual_seed(int(seed))
    perm = idx[torch.randperm(n, generator=g)]
    folds = []
    fold_sizes = [n // n_splits] * n_splits
    for i in range(n % n_splits):
        fold_sizes[i] += 1
    start = 0
    for fs in fold_sizes:
        val_idx = perm[start:start+fs]
        train_mask = torch.ones(n, dtype=torch.bool)
        train_mask[val_idx] = False
        train_idx = torch.nonzero(train_mask, as_tuple=False).squeeze(1)
        folds.append((train_idx, val_idx))
        start += fs

    # Fold correlation matrices, computed once
    R_tr_list = []
    R_val_list = []
    for train_idx, val_idx in folds:
        X_tr = X.index_select(0, train_idx.to(X.device))
        R_tr = _compute_correlation(X_tr).coalesce().to_dense()
        if not torch.isfinite(R_tr).all():
            raise ValueError("R_tr had non-finite entries")
        R_tr_list.append(R_tr)
        X_val = X.index_select(0, val_idx.to(X.device))
        R_val = _compute_correlation(X_val).coalesce().to_dense()
        if not torch.isfinite(R_val).all():
            raise ValueError("R_val had non-finite entries")
        R_val_list.append(R_val)

    def _score_one_rho(rho: float) -> float:
        errs = []
        for R_tr, R_val in zip(R_tr_list, R_val_list):
            R_hat = _SEC_dense(Rn=R_tr, rho=rho, **sec_kwargs)
            if not torch.isfinite(R_hat).all():
                raise ValueError("R_hat had non-finite entries")
            diff = (R_hat - R_val).to(dtype)
            err = torch.linalg.norm(diff, ord='fro') ** 2
            errs.append(float(err.item()))
        return float(sum(errs) / len(errs))

    rows: List[Dict[str, float]] = []
    scores_by_rho: Dict[float, float] = {}

    def _score_many_pairs(c_list: Sequence[float]) -> None:
        rhos_batch = [c_to_rho(c) for c in c_list]
        if device.type == "cpu" and (workers is None or workers < 0 or workers > 1):
            max_workers = (os.cpu_count() or 1) if workers in (None, -1) else int(workers)
            # One intra-op thread budget per worker
            with _thread_limit(max(1, (os.cpu_count() or 1) // max_workers)):
                with ThreadPoolExecutor(max_workers=max_workers) as ex:
                    scores = list(ex.map(_score_one_rho, rhos_batch))
        else:
            scores = [_score_one_rho(r) for r in rhos_batch]
        for c, r, score in zip(c_list, rhos_batch, scores):
            scores_by_rho[r] = score
            rows.append({"c": float(c), "rho": float(r), "score": float(score)})

    c_coarse = sorted(set(float(c) for c in c_grid))
    _score_many_pairs(c_coarse)

    def _best_c_from(c_list: Sequence[float]) -> float:
        best_c = None
        best_score = None
        for c in sorted(c_list):
            r = c_to_rho(c)
            s = scores_by_rho[r]
            if (best_score is None) or (s < best_score) or (s == best_score and c < best_c):
                best_c, best_score = c, s
        return best_c

    best_c_coarse = _best_c_from(c_coarse)

    if refine and refine_points >= 2 and len(c_coarse) >= 2:
        i = c_coarse.index(best_c_coarse)
        if i == 0:
            c_left, c_right = c_coarse[0], c_coarse[1]
        elif i == len(c_coarse) - 1:
            c_left, c_right = c_coarse[-2], c_coarse[-1]
        else:
            c_left, c_right = c_coarse[i - 1], c_coarse[i + 1]

        if c_right > c_left:
            step = (c_right - c_left) / (refine_points - 1)
            c_refined = [c_left + j * step for j in range(refine_points)]
            # Values already evaluated in the first pass are skipped
            coarse_keys = {round(c, 12) for c in c_coarse}
            c_new = [c for c in c_refined if round(c, 12) not in coarse_keys]
            if c_new:
                _score_many_pairs(c_new)

    best_rho = min(scores_by_rho.items(), key=lambda kv: (kv[1], kv[0]))[0]

    # Warning when the selected c sits at an edge of c_grid
    if base > 0.0:
        best_c = best_rho / base
        c_min, c_max = c_coarse[0], c_coarse[-1]
        if math.isclose(best_c, c_min, rel_tol=0.0, abs_tol=1e-12):
            warnings.warn(
                f"Best c = {best_c:.3g} occurs at the LOWER edge of c_grid [{c_min}, {c_max}]. "
                "Consider expanding c_grid to include smaller c values.",
                RuntimeWarning
            )
        elif math.isclose(best_c, c_max, rel_tol=0.0, abs_tol=1e-12):
            warnings.warn(
                f"Best c = {best_c:.3g} occurs at the UPPER edge of c_grid [{c_min}, {c_max}]. "
                "Consider expanding c_grid to include larger c values.",
                RuntimeWarning
            )

    # Final fit at best_rho on the full-sample correlation matrix
    R_hat_best = _SEC_dense(Rn=Rn, rho=best_rho, **sec_kwargs)

    scores_df = pd.DataFrame(rows).sort_values("rho", kind="mergesort").reset_index(drop=True)
    return best_rho, scores_df, R_hat_best
