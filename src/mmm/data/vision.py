"""
mmm/data/vision.py

Sam Foreman
2024-12-27
"""

from pathlib import Path
from typing import Optional

import ezpz
import torch
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms

from torch.utils.data import Dataset
from mmm import OUTPUTS_DIR
from mmm.configs import TORCH_DTYPES_MAP


RANK = ezpz.get_rank()
WORLD_SIZE = ezpz.get_world_size()


class FakeImageDataset(Dataset):
    def __init__(
        self,
        size: int,
        dtype: Optional[str | torch.dtype] = None,
    ):
        super().__init__()
        self.size = size
        self.dtype = (
            torch.float32
            if dtype is None
            else (TORCH_DTYPES_MAP[dtype] if isinstance(dtype, str) else dtype)
        )

    def __len__(self):
        return int(1e6)

    def __getitem__(self, index):
        rand_image = torch.randn(
            [3, self.size, self.size],
            dtype=(torch.float32 if self.dtype is None else self.dtype),
        )
        label = torch.tensor(data=(index % 1000), dtype=torch.int64)
        return rand_image, label


def get_mnist(
    train_batch_size: int = 128,
    test_batch_size: int = 128,
    outdir: Optional[str | Path] = None,
    num_workers: int = 1,
    download: bool = True,
    shuffle: bool = False,
    pin_memory: bool = True,
) -> dict:
    outdir = OUTPUTS_DIR if outdir is None else outdir
    datadir = Path(outdir).joinpath('data', 'mnist')
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    if RANK == 0:
        _ = datasets.MNIST(
            datadir.as_posix(),
            train=True,
            download=download,
            transform=transform,
        )
    torch.distributed.barrier()  # type:ignore

    dataset1 = datasets.MNIST(
        datadir.as_posix(),
        train=True,
        download=False,
        transform=transform,
    )
    dataset2 = datasets.MNIST(datadir, train=False, transform=transform)
    train_kwargs = {'batch_size': train_batch_size}
    test_kwargs = {'batch_size': test_batch_size}
    sampler1, sampler2 = None, None
    if WORLD_SIZE > 1:
        sampler1 = DistributedSampler(
            dataset1, rank=RANK, num_replicas=WORLD_SIZE, shuffle=True
        )
        sampler2 = DistributedSampler(
            dataset2, rank=RANK, num_replicas=WORLD_SIZE
        )
        train_kwargs |= {'sampler': sampler1}
        test_kwargs |= {'sampler': sampler2}
    kwargs = {
        'pin_memory': pin_memory,
        'shuffle': shuffle,
        'num_workers': num_workers,
    }
    train_kwargs.update(kwargs)
    test_kwargs.update(kwargs)
    train_loader = torch.utils.data.DataLoader(  # type:ignore
        dataset1, **train_kwargs
    )
    test_loader = torch.utils.data.DataLoader(  # type:ignore
        dataset2, **test_kwargs
    )
    return {
        'train': {
            'data': dataset1,
            'loader': train_loader,
            'sampler': sampler1,
        },
        'test': {
            'data': dataset2,
            'loader': test_loader,
            'sampler': sampler2,
        },
    }


def get_fake_data(
    img_size: int,
    batch_size: int,
    num_workers: int = 1,
    pin_memory: bool = True,
    drop_last: bool = True,
    shuffle: bool = True,
    dtype: Optional[str | torch.dtype] = None,
) -> dict:
    dataset = FakeImageDataset(size=img_size, dtype=dtype)
    # loader = torch.utils.data.DataLoader(  # type:ignore
    #     dataset, batch_size=batch_size, shuffle=shuffle
    # )
    kwargs = {
        'batch_size': batch_size,
        'pin_memory': pin_memory,
        'num_workers': num_workers,
        'drop_last': drop_last,
    }
    sampler = None
    if WORLD_SIZE > 1:  # use DistributedSampler when > 1 device
        sampler = DistributedSampler(
            dataset, rank=RANK, num_replicas=WORLD_SIZE, shuffle=shuffle
        )
        kwargs |= {'sampler': sampler}
    train_loader = torch.utils.data.DataLoader(  # type:ignore
        dataset, **kwargs
    )
    return {
        'train': {
            'data': dataset,
            'loader': train_loader,
            'sampler': sampler,
        },
    }
