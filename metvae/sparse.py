import os
import warnings
from typing import Optional, Dict, Sequence, Tuple, List
import pandas as pd
import math
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
from statsmodels.stats.multitest import multipletests
from .compute_corr import _compute_correlation

# Helper functions for p-values filtering

def _p_filter(mat: torch.Tensor,
              mat_p: torch.Tensor,
              max_p: float,
              impute_value: float = 0.0) -> torch.Tensor:
    out = mat.clone()
    out[mat_p > max_p] = impute_value
    return out

def _matrix_p_adjust(p_matrix: torch.Tensor,
                     method: str = 'fdr_bh') -> torch.Tensor:
    device = p_matrix.device
    dtype  = p_matrix.dtype
    n = p_matrix.shape[0]

    # Extract lower triangular part of the matrix into a vector
    tril = torch.tril_indices(n, n, offset=-1, device=device)
    p_vec = p_matrix[tril[0], tril[1]].detach().to('cpu').numpy()

    # Adjust the p-values
    _, q_vec, _, _ = multipletests(p_vec, method=method)

    # Back to Torch and form symmetric matrix
    q_mat = torch.zeros((n, n), device=device, dtype=dtype)
    q_mat[tril[0], tril[1]] = torch.from_numpy(q_vec).to(device=device, dtype=dtype)
    q_mat = q_mat + q_mat.T
    return q_mat

# Helper functions for SEC

@torch.no_grad()
def _projection_psd(
        A: torch.Tensor, *, 
        jitter0: float = 1e-12, 
        max_retries: int = 5
) -> torch.Tensor:
    """
    Robust PSD projection: symmetrize, sanitize, add tiny diagonal jitter,
    retry eigh with escalating jitter if needed. Works on CPU/GPU.

    Returns a matrix in the *original dtype* of A.
    """
    orig_dtype = A.dtype
    A = 0.5 * (A + A.transpose(-1, -2))

    # sanitize NaN/Inf (rare, but cheap insurance)
    A = torch.nan_to_num(A)

    # do the heavy linear algebra in float64 for stability
    if A.dtype != torch.float64:
        A64 = A.to(torch.float64)
    else:
        A64 = A

    I = torch.eye(A64.shape[-1], device=A64.device, dtype=A64.dtype)

    # base jitter scaled to matrix magnitude
    scale = A64.abs().mean()
    base = (scale if torch.isfinite(scale) and scale > 0 else 1.0)
    jitter = jitter0 * base

    for _ in range(max_retries + 1):
        try:
            evals, evecs = torch.linalg.eigh(A64 + jitter * I)
            evals = evals.clamp_min(0.0)
            out64 = (evecs * evals.unsqueeze(-2)) @ evecs.transpose(-1, -2)
            out64 = 0.5 * (out64 + out64.transpose(-1, -2))
            return out64.to(orig_dtype)
        except RuntimeError as e:
            # escalate jitter by x10 and retry
            jitter *= 10.0
            continue

    # last resort: project by zeroing negative eigenvalues via eigvalsh fallback
    evals, evecs = torch.linalg.eigh(A64 + jitter * I)
    evals = evals.clamp_min(0.0)
    out64 = (evecs * evals.unsqueeze(-2)) @ evecs.transpose(-1, -2)
    out64 = 0.5 * (out64 + out64.transpose(-1, -2))
    return out64.to(orig_dtype)

@torch.no_grad()
def _SEC(
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
    Sparse Estimation of the Correlation matrix (SEC) solver (APG + PSD projection),
    adapted from the MATLAB reference implementation:
      https://warwick.ac.uk/fac/sci/statistics/staff/academic-research/leng/publications/sec.m

    Solves (schematically):
        min_R  0.5||R - Rn||_F^2 + rho * ||W ∘ R||_1
        s.t.   R ⪰ epsilon*I,  R_ii = 1  (via calibration)

    Notes
    -----
    * Uses an APG (accelerated proximal gradient) scheme with optional
      Nesterov restarts and a simple backtracking-like linesearch toggle.
    * PSD feasibility is enforced by projection onto the PSD cone.
    * A final “calibration” rescales to (approx.) unit diagonal.
    * At the very end, a hard threshold is applied and the result is
      returned as a sparse COO tensor (diagonal kept).
    """
    # --- basic checks and shapes ---
    if Rn.is_sparse:
        Rn = Rn.coalesce().to_dense()
    assert Rn.ndim == 2 and Rn.shape[0] == Rn.shape[1], "Rn must be square."
    p = Rn.shape[0]
    device, dtype = Rn.device, Rn.dtype

    # --- delta (theoretical tiny-entry cutoff) ---
    # δ ≈ c_delta * sqrt(log p / n); falls back to a tiny numeric if n unknown.
    if delta is None:
        if n_samples is not None and n_samples > 0:
            delta = float(c_delta * math.sqrt(max(math.log(p) / n_samples, 0.0)))
        else:
            delta = 1e-6

    # Dual/init vars
    Z = torch.zeros((p, p), device=device, dtype=dtype)
    b_vec = torch.ones((p,), device=device, dtype=dtype)

    # --- masks from |Rn| and identity ---
    abs_Rn = torch.abs(Rn)
    eye_mask = torch.eye(p, device=device, dtype=torch.bool)
    offdiag_mask = ~eye_mask
    tiny_mask = (abs_Rn < delta) & offdiag_mask # “very small” off-diagonals
    Omega = eye_mask | tiny_mask # entries fixed to b (diag) or 0 (tiny)
    
    # Target b-matrix (diag = 1, tiny off-diagonals = 0 by construction)
    b_mat = torch.diag(b_vec)

    # --- weights W: 0 where |Rn| <= δ; else 1/|Rn| ---
    Wmat = torch.zeros((p, p), device=device, dtype=dtype)
    Wmat = torch.where(abs_Rn <= delta, Wmat, 1.0 / abs_Rn.clamp_min(1e-300))
    Wmat.fill_diagonal_(0.0)
    W = Wmat

    # --- preprocessing on Rn (centering & scaling by column) and symmetrize ---
    Rn = Rn.to(device=device, dtype=dtype)
    col_mean = Rn.mean(dim=0, keepdim=True)
    col_std = Rn.std(dim=0, unbiased=True, keepdim=True).clamp_min(1e-12)
    Rn_work = (Rn - col_mean) / col_std
    Rn_work = 0.5 * (Rn_work + Rn_work.T)

    # --- APG state ---    
    R = torch.zeros((p, p), device=device, dtype=dtype)
    Y = Z.clone()
    t = 1.0
    L = 1.0
    tau = 0.75
    eta = 0.9
    I = torch.eye(p, device=device, dtype=dtype)

    res_old: Optional[float] = None

    for k in range(1, max_iter + 1):
        Yold = Y.clone()
        told = t

        X = Z + Rn_work

        R_tmp = torch.sign(X) * torch.clamp(torch.abs(X) - rho * W, min=0.0)
        R = R_tmp.clone()
        R[Omega] = b_mat[Omega]
        R = 0.5 * (R + R.T)

        if line_search_apg:
            if (k % 5 == 0) and (tau < L) and (res_old is not None):
                pass

            Y = _projection_psd(Z - (R - epsilon * I) / tau)
            res_gradient = (tau * torch.linalg.norm(Z - Y, ord='fro') /
                            (1.0 + torch.linalg.norm(Z, ord='fro')))
            if (k % 5 == 0) and (tau < L) and (res_old is not None) and (res_gradient > res_old):
                tau = min(L, tau / eta)
        else:
            Y = _projection_psd(Z - (R - epsilon * I) / L)
            res_gradient = (torch.linalg.norm(Z - Y, ord='fro') /
                            (1.0 + torch.linalg.norm(Z, ord='fro')))

        if k > 1:
            res_old = float(res_gradient)

        # Early convergence
        if float(res_gradient) <= tol:
            break

        if (restart is not None) and (restart > 0) and (k % restart == 0):
            t = 1.0
            told = t

        t = (1.0 + math.sqrt(1.0 + 4.0 * told * told)) / 2.0
        Z = Y + ((told - 1.0) / t) * (Y - Yold)

    # Calibration
    R_cal = R - epsilon * I
    lam_min = torch.linalg.eigvalsh(R_cal).min().item()
    if lam_min < 0.0:
        R_cal = R_cal + (-lam_min) * I

    d = torch.diag(R_cal).clamp_min(1e-300)
    d = ((b_vec - epsilon) / d).clamp_min(0.0).sqrt()
    D = torch.diag(d)

    R_out = D @ R_cal @ D
    R_out = 0.5 * (R_out + R_out.T) + epsilon * I
    
    # --- postprocessing: hard threshold + sparse output --- 
    # Keep symmetry and always keep the diagonal, even if threshold is large.
    if threshold > 0.0:
        # keep diagonal exactly as produced by calibration
        diag_vals = R_out.diagonal().clone()
        # zero out sub-threshold off-diagonals
        R_out = torch.where(R_out.abs() >= threshold,
                            R_out,
                            torch.zeros_like(R_out))
        # restore diagonal
        R_out.diagonal().copy_(diag_vals)
    
    return R_out.to_sparse_coo().coalesce()

@torch.no_grad()
def _SEC_cv(
    X: torch.Tensor,
    Rn: torch.Tensor,
    *,
    c_grid: Sequence[float] = tuple(float(x) for x in range(1, 11)),  # 1.0, 2.0, ..., 10.0 (coarse)
    n_splits: int = 5,
    seed: int = 0,
    workers: int = -1,             # only used on CPU
    refine: bool = True,           # zoom after coarse pass
    refine_points: int = 10,       # number of points in the refined bracket (inclusive)
    **sec_kwargs
) -> Tuple[float, "pd.DataFrame", torch.Tensor]:
    """
    Choose rho for SEC via K-fold cross-validation with a single adaptive refinement:
    1) Evaluate coarse c_grid (e.g., 3..10).
    2) Find best c and refine once between its immediate neighbors (left/right).
    Returns (best_rho, scores_by_rho, R_hat_best_dense).
    """

    assert X.ndim == 2, "X must be 2D (n x p)."
    n, p = X.shape
    device, dtype = X.device, X.dtype

    if n_splits < 2 or n_splits > n:
        raise ValueError(f"n_splits must be in [2, n]; got {n_splits} for n={n}.")

    base = math.sqrt(max(math.log(p) / n, 0.0))
    def c_to_rho(c: float) -> float:
        return float(c) * base

    # Build K folds (deterministic)
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

    # Pre-cache validation correlations
    R_val_list = []
    for _, val_idx in folds:
        X_val = X.index_select(0, val_idx.to(X.device))
        R_val = _compute_correlation(X_val).coalesce().to_dense()
        assert torch.isfinite(R_val).all(), "R_val had non-finite entries"
        R_val_list.append(R_val)

    def _score_one_rho(rho: float) -> Tuple[float, float]:
        errs = []
        for (train_idx, _), R_val in zip(folds, R_val_list):
            X_tr = X.index_select(0, train_idx.to(X.device))
            R_tr = _compute_correlation(X_tr).coalesce().to_dense()
            assert torch.isfinite(R_tr).all(), "R_tr had non-finite entries"
            R_hat_sparse = _SEC(Rn=R_tr, rho=rho, **sec_kwargs)
            R_hat = R_hat_sparse.to_dense()
            assert torch.isfinite(R_hat).all(), "R_hat had non-finite entries"
            diff = (R_hat - R_val).to(dtype)
            err = torch.linalg.norm(diff, ord='fro') ** 2
            errs.append(float(err.item()))
        return rho, float(sum(errs) / len(errs))

    # Storage for results and quick lookup by rho
    rows: List[Dict[str, float]] = []
    scores_by_rho: Dict[float, float] = {}  # rho -> score

    def _score_many_pairs(c_list: Sequence[float]) -> None:
        """Score a batch of c's (and their rhos), append rows, and fill scores_by_rho."""
        rhos_batch = [c_to_rho(c) for c in c_list]
        if device.type == "cpu" and (workers is None or workers < 0 or workers > 1):
            max_workers = (os.cpu_count() or 1) if workers in (None, -1) else int(workers)
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                # Map each future directly to its (c, r) so we can recover them safely.
                futures = {ex.submit(_score_one_rho, r): (c, r) for c, r in zip(c_list, rhos_batch)}
                for fut in as_completed(futures):
                    c, r = futures[fut]
                    r_ret, score = fut.result()  # r_ret should equal r
                    # Use r_ret from the result to avoid any mismatch
                    scores_by_rho[r_ret] = score
                    rows.append({"c": float(c), "rho": float(r_ret), "score": float(score)})
        else:
            for c, r in zip(c_list, rhos_batch):
                r_ret, score = _score_one_rho(r)
                scores_by_rho[r_ret] = score
                rows.append({"c": float(c), "rho": float(r_ret), "score": float(score)})

    # ---- Coarse pass
    c_coarse = sorted(set(float(c) for c in c_grid))
    _score_many_pairs(c_coarse)

    # Helper to pick best c from a list of c's using scores_by_rho
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

    # ---- Single refinement (between immediate neighbors)
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
            # Skip any c we already evaluated in coarse
            c_new = [c for c in c_refined if c not in c_coarse]
            if c_new:
                _score_many_pairs(c_new)

    # ---- Final selection across all evaluated rhos
    best_rho = min(scores_by_rho.items(), key=lambda kv: (kv[1], kv[0]))[0]
    
    # --- Edge warning (relative to the provided coarse c_grid)
    # Only meaningful when base > 0 so that c = rho/base is defined.
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

    # Final fit at best_rho on full-sample correlation
    Rn_dense = Rn.coalesce().to_dense() if getattr(Rn, "is_sparse", False) and Rn.is_sparse else Rn
    R_hat_best = _SEC(Rn=Rn_dense, rho=best_rho, **sec_kwargs)

    # Make DataFrame
    scores_df = pd.DataFrame(rows).sort_values("rho", kind="mergesort").reset_index(drop=True)
    return best_rho, scores_df, R_hat_best.to_dense()









