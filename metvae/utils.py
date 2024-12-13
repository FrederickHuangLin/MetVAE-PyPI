import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate
import pytorch_lightning as pl

# Helper function for the nonlinear transformation

class _NonlinearTransform(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool = True,
                 dropout_rate: float = 0,
                 depth: int = 1):
        super().__init__()
        assert depth >= 1
        layers = [nn.Linear(in_features=in_features, out_features=out_features, bias=bias),
                  nn.Softplus(),
                  nn.BatchNorm1d(num_features=out_features),
                  nn.Dropout(p=dropout_rate)]

        if depth > 1:
            for _ in range(depth - 1):
                layers.append(nn.Linear(in_features=out_features, out_features=out_features, bias=bias))
                layers.append(nn.Softplus())
                layers.append(nn.BatchNorm1d(num_features=out_features))
                layers.append(nn.Dropout(p=dropout_rate))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        y = self.layers(x)
        return y

# Helper functions for data processing that needs before batching

def _custom_collate(batch):
    batch_x, batch_y = zip(*batch)

    # Check if any x is None
    if any(x is None for x in batch_x):
        batch_x = None
    else:
        batch_x = default_collate(batch_x)

    batch_y = default_collate(batch_y)

    return batch_x, batch_y


class _CustomDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        # Fetch x, y, mask, and L at the given index
        x = self.x[idx] if self.x is not None else None
        y = self.y[idx]
        return x, y


def _CustomDataLoader(dataset, batch_size, num_workers):
    return torch.utils.data.DataLoader(dataset,
                                       batch_size=batch_size,
                                       collate_fn=_custom_collate,
                                       num_workers=num_workers)


class _CustomDataModule(pl.LightningDataModule):

    def __init__(self, x, y, batch_size=32, num_workers=0):
        super().__init__()
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        dataset = _CustomDataset(self.x, self.y)
        train_length = int(len(dataset) * 1)
        self.train_dataset = dataset

    def train_dataloader(self):
        return _CustomDataLoader(self.train_dataset,
                                 batch_size=self.batch_size,
                                 num_workers=self.num_workers)

# Helper function to make valid column names
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











