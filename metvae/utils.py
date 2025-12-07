from typing import Optional, Sequence
import numpy as np
import pandas as pd
import torch

def _make_valid_column_name(name):
    if not isinstance(name, str):
        name = str(name)
    name = name.replace(' ', '_')
    name = name.replace('-', '_')
    name = name.replace('.', '_')
    name = name.replace('(', '_')
    name = name.replace(')', '_')
    name = name.replace(',', '_')
    name = name.replace('/', '_')
    name = name.replace('\\', '_')
    name = name.replace('&', 'and')
    # Add more replacements if necessary
    name = ''.join(char if char.isalnum() or char == '_' else '' for char in name)
    return name

def _torch_to_df(t: torch.Tensor,
                 names: Optional[Sequence[str]] = None) -> pd.DataFrame:
    arr = t.detach().to('cpu').numpy()
    if names is None:
        return pd.DataFrame(arr)
    return pd.DataFrame(arr, index=names, columns=names)

def _corr_to_long(
    corr: pd.DataFrame,
    *,
    use_upper: bool = True,
    sort_by_abs: bool = True,
) -> pd.DataFrame:
    """
    Convert a symmetric correlation matrix into a long (edge list) DataFrame.

    This is a small convenience utility that takes a square correlation matrix
    (a ``pandas.DataFrame``), extracts the off-diagonal entries, and returns a table of the form ::

        node1  node2  correlation

    Parameters
    ----------
    corr : pandas.DataFrame
        Symmetric correlation matrix. 

    use_upper : bool, default=True
        If ``True`` (default), use the strict upper triangle (i < j).
        If ``False``, use the strict lower triangle.

    sort_by_abs : bool, default=True
        If ``True``, rows are sorted by ``abs(correlation)`` in descending
        order. If ``False``, the original stacking order is kept.

    Returns
    -------
    df_long : pandas.DataFrame
        DataFrame with columns ``["node1", "node2", "correlation"]``.
    """
    
    # Clean up names (removes index/column names)
    df_cor = corr.copy()
    df_cor.index.name = None
    df_cor.columns.name = None

    p = df_cor.shape[0]
    if use_upper:
        mask = np.triu(np.ones((p, p), dtype=bool), k=1)
    else:
        mask = np.tril(np.ones((p, p), dtype=bool), k=-1)

    df_masked = df_cor.where(mask)
    # stack() automatically drops NaNs (i.e., the masked-out entries)
    df_long = df_masked.stack().reset_index()
    df_long.columns = ["node1", "node2", "correlation"]

    if sort_by_abs:
        order = df_long["correlation"].abs().sort_values(ascending=False).index
        df_long = df_long.loc[order].reset_index(drop=True)

    return df_long






