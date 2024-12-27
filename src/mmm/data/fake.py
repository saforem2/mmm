"""
mmm/data/fake.py
"""
from typing import Optional

import torch

from torch.utils.data import Dataset
from mmm.configs import TORCH_DTYPES


class FakeImageDataset(Dataset):
    def __init__(
        self,
        size: int,
        dtype: Optional[str | torch.dtype] = None,
    ):
        super().__init__()
        self.size = size
        self.dtype = torch.float32 if dtype is None else (
            TORCH_DTYPES[dtype] if isinstance(dtype, str) else dtype
        )

    def __len__(self):
        return int(1e6)

    def __getitem__(self, index):
        rand_image = torch.randn(
            [3, self.size, self.size],
            dtype=(torch.float32 if self.dtype is None else self.dtype)
        )
        label = torch.tensor(data =(index % 1000), dtype=torch.int64)
        return rand_image, label

