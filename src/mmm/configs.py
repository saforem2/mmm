"""
mmm/configs.py
"""

from typing import Callable, Optional
import ezpz

import torch
from dataclasses import dataclass, field


TORCH_DTYPES = {
    'bf16': torch.bfloat16,
    'bfloat16': torch.bfloat16,
    'fp16': torch.float16,
    'float16': torch.float16,
    'half': torch.float16,
    'fp32': torch.float32,
    'float32': torch.float32,
}


@dataclass
class ViTConfig:
    img_size: int = 224
    batch_size: int = 128
    num_heads: int = 16
    head_dim: int = 64
    depth: int = 24
    patch_size: int = 16

    def __post_init__(self):
        self.seq_len = (self.img_size // self.patch_size) ** 2  # 196, default


@dataclass
class TrainArgs:
    img_size: int
    batch_size: int
    num_heads: int
    head_dim: int
    depth: int
    patch_size: int
    dtype: str
    compile: bool
    attn_type: str
    num_workers: int
    max_iters: int
    format: Optional[str] = field(default_factory=str)
    cuda_sdpa_backend: Optional[str] = field(default_factory=str)
