"""
mmm/trainer/vit.py
"""

import argparse
import functools
import logging
import os
import time
from typing import Any, Optional, Sequence

import ezpz
from timm.models.vision_transformer import VisionTransformer
import torch
import torch._dynamo
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision
from torch.utils.data import DataLoader

from mmm.configs import TORCH_DTYPES, TrainArgs, ViTConfig
from mmm.data.fake import FakeImageDataset
from mmm.models.vit.attention import AttentionBlock

# torch._dynamo.config.suppress_errors = True  # type:ignore

SEED = int(os.environ.get('SEED', '0'))
RANK = ezpz.setup(backend='DDP', seed=SEED)
WORLD_SIZE = ezpz.get_world_size()

LOCAL_RANK = ezpz.get_local_rank()
DEVICE_TYPE = str(ezpz.get_torch_device(as_torch_device=False))
DEVICE = torch.device(f'{DEVICE_TYPE}:{LOCAL_RANK}')

LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO') if RANK == 0 else 'CRITICAL'

logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVEL)


def parse_args(argv: Optional[Sequence[str]] = None) -> TrainArgs:
    parser = argparse.ArgumentParser(
        prog='mmm.train.vit',
        description='Train a simple ViT',
    )
    parser.add_argument('--img_size', default=224, help='Image size')
    parser.add_argument('--batch_size', default=128, help='Batch size')
    parser.add_argument('--num_heads', default=16, help='Number of heads')
    parser.add_argument('--head_dim', default=64, help='Hidden Dimension')
    parser.add_argument('--depth', default=24, help='Depth')
    parser.add_argument('--patch_size', default=16, help='Patch size')
    parser.add_argument('--dtype', default='bf16', help='Data type')
    parser.add_argument('--compile', action='store_true', help='Compile model')
    parser.add_argument('--num_workers', default=0, help='Number of workers')
    parser.add_argument('--max_iters', default=None, help='Maximum iterations')
    parser.add_argument(
        '--attn_type',
        default='native',
        choices=['native', 'sdpa'],
        help='Attention function to use.',
    )
    parser.add_argument(
        '--cuda_sdpa_backend',
        default='all',
        choices=[
            'flash_sdp',
            'mem_efficient_sdp',
            'math_sdp',
            'cudnn_sdp',
            'all',
        ],
        help='CUDA SDPA backend to use.',
    )
    return TrainArgs(**parser.parse_args(argv).__dict__)


def train_fn(block_fn: Any, args: TrainArgs) -> ezpz.History:
    config = ViTConfig(
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_heads=args.num_heads,
        head_dim=args.head_dim,
        depth=args.depth,
        patch_size=args.patch_size,
    )
    logger.info(f'{config=}')
    train_set = FakeImageDataset(config.img_size)
    logger.info(f'{len(train_set)=}')
    train_loader = DataLoader(
        train_set,
        batch_size=config.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    model = VisionTransformer(
        img_size=config.img_size,
        patch_size=config.patch_size,
        embed_dim=(config.num_heads * config.head_dim),
        depth=config.depth,
        num_heads=config.num_heads,
        class_token=False,
        global_pool='avg',
        block_fn=block_fn,
    )
    try:
        from torchinfo import summary

        summary_str = summary(
            model,
            input_size=(config.batch_size, 3, config.img_size, config.img_size),
            depth=1,
            verbose=0,
        )
        logger.info(f'\n{summary_str}')

    except (ImportError, ModuleNotFoundError):
        logger.warning(
            'torchinfo not installed, unable to print model summary!'
        )
    model.to(DEVICE)
    # num_params = sum(
    #     [
    #         sum(
    #             [
    #                 getattr(p, 'ds_numel', 0)
    #                 if hasattr(p, 'ds_id')
    #                 else p.nelement()
    #                 for p in model_module.parameters()
    #             ]
    #         )
    #         for model_module in model.modules()
    #     ]
    # )
    # model_size_in_billions = num_params / 1e9
    # logger.info(f'Model size: nparams={model_size_in_billions:.2f} B')
    # except Exception:
    #     import pudb; pudb.set_trace()

    # dtypes = {
    #     "fp16": torch.float16,
    #     "bf16": torch.bfloat16,
    #     "bfloat16": torch.bfloat16,
    #     "fp32": torch.float32,
    # }
    # dtype = dtypes[args.dtype]

    if WORLD_SIZE > 1:
        if args.dtype in {'fp16', 'bf16', 'fp32'}:
            model = FSDP(
                model,
                mixed_precision=MixedPrecision(
                    param_dtype=TORCH_DTYPES[args.dtype],
                    cast_forward_inputs=True,
                ),
            )
        else:
            model = FSDP(model)

    if args.compile:
        logger.info('Compiling model')
        model = torch.compile(model)

    torch_dtype = TORCH_DTYPES[args.dtype]
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters())  # type:ignore
    model.train()  # type:ignore

    history = ezpz.History()
    logger.info(
        f'Training with {WORLD_SIZE} x {DEVICE_TYPE} (s), using {torch_dtype=}'
    )
    for step, data in enumerate(train_loader):
        if args.max_iters is not None and step > int(args.max_iters):
            break
        t0 = time.perf_counter()
        inputs = data[0].to(device=DEVICE, non_blocking=True)
        label = data[1].to(device=DEVICE, non_blocking=True)
        t1 = time.perf_counter()
        with torch.autocast(device_type=DEVICE_TYPE, dtype=torch_dtype):
            outputs = model(inputs)
            loss = criterion(outputs, label)
        t2 = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        t3 = time.perf_counter()
        metrics = {
            'train/iter': step,
            'train/loss': loss.item(),
            'train/dt': t3 - t0,
            'train/dtf': t2 - t1,
            'train/dtb': t3 - t2,
        }
        _ = history.update(metrics)
        summary = ezpz.summarize_dict(metrics)
        logger.info(summary.replace('train/', ''))

    if RANK == 0:
        from mmm import OUTPUTS_DIR

        outdir = OUTPUTS_DIR.joinpath('plots', 'vit')
        tplotdir = outdir.joinpath('tplot')
        mplotdir = outdir.joinpath('mplot')
        tplotdir.mkdir(exist_ok=True, parents=True)
        mplotdir.mkdir(exist_ok=True, parents=True)

        import matplotlib.pyplot as plt
        import ambivalent

        plt.style.use(ambivalent.STYLES['ambivalent'])

        dataset = history.plot_all(outdir=mplotdir)
        _ = history.tplot_all(
            outdir=tplotdir, append=True, xkey='train/iter', dataset=dataset
        )
        logger.info(f'{dataset=}')

    return history


def main(argv: Optional[Sequence[str]] = None):
    args: TrainArgs = parse_args(argv)
    config = ViTConfig(
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_heads=args.num_heads,
        head_dim=args.head_dim,
        depth=args.depth,
        patch_size=args.patch_size,
    )

    def attn_fn(
        q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        scale = config.head_dim ** (-0.5)
        q = q * scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        x = attn @ v
        return x

    logger.info(f'Using {args.attn_type} for SDPA backend')
    if args.attn_type == 'native':
        block_fn = functools.partial(AttentionBlock, attn_fn=attn_fn)
    # if args.sdpa_backend == 'by_hand':
    elif args.attn_type == 'sdpa':
        if torch.cuda.is_available():
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(False)
            torch.backends.cuda.enable_cudnn_sdp(False)

            if args.cuda_sdpa_backend in ['flash_sdp', 'all']:
                torch.backends.cuda.enable_flash_sdp(True)
            if args.cuda_sdpa_backend in ['mem_efficient_sdp', 'all']:
                torch.backends.cuda.enable_mem_efficient_sdp(True)
            if args.cuda_sdpa_backend in ['math_sdp', 'all']:
                torch.backends.cuda.enable_math_sdp(True)
            if args.cuda_sdpa_backend in ['cudnn_sdp', 'all']:
                torch.backends.cuda.enable_cudnn_sdp(True)

        block_fn = functools.partial(
            AttentionBlock,
            attn_fn=torch.nn.functional.scaled_dot_product_attention,
        )
    else:
        raise ValueError(f'Unknown attention type: {args.attn_type}')
    logger.info(f'Using AttentionBlock Attention with {args.compile=}')
    train_fn(block_fn, args)


if __name__ == '__main__':
    t0 = time.perf_counter()
    main()
    logger.info(f'Took {time.perf_counter() - t0:.2f} seconds')
