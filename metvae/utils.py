from typing import Optional, Sequence
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








