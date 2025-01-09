"""
mmm/examples/fsdp_tp.py

Sam Foreman
2024-12-31

Modified from:
<https://pytorch.org/tutorials/intermediate/TP_tutorial.html>


This is the script to test 2D Parallel which combines Tensor/Sequence
parallel with Fully Sharded Data Parallel (TP/SP + FSDP) on a example
Llama2 model. We show an E2E working flow from forward, backward
and optimization.

We enabled Fully Sharded Data Parallel + Tensor Parallel in
separate parallel dimensions:
    Data Parallel ("dp") across hosts
    Tensor Parallel ("tp") within each host

 We use a simple diagram to illustrate below:

┌──────────┐       ┌──────────┐       ┌──────────┐       ┌──────────┐
│ Host 1   │       │ Host 2   │       │          │       │ Host N   │
│ 8 GPUs   │       │ 8 GPUs   │       │          │       │ 8 GPUs   │
│          │       │          │       │    ...   │       │          │
│ (TP)     │       │ (TP)     │       │          │       │ (TP)     │
│[0,1,..,7]│       │[8,9..,15]│       │          │       │[8N-8,8N-7│
│          │       │          │       │          │       │ .., 8N-1]│
│          │       │          │       │          │       │          │
└──────────┘       └──────────┘       └──────────┘       └──────────┘
FSDP:
[0, 8, ..., 8N-8], [1, 9, ..., 8N-7], ..., [7, 15, ..., 8N-1]
"""

import argparse
from time import perf_counter

import ezpz

import torch

import torch.nn as nn
import torch.nn.functional as F

from mmm.models import summarize_model

# from mmm.models.llama import Transformer, ModelArgs
# from mmm.models.llama import Transformer, ModelArgs
from mmm.models.llama2 import Transformer, ModelArgs
from mmm.data.text import RandomTokenDataset


from torch.utils.data import DataLoader
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
)
from torch.distributed._tensor import Shard, Replicate
# from torch.distributed.tensor.parallel import loss_parallel

from torch.distributed.tensor.parallel import (
    parallelize_module,
    ColwiseParallel,
    RowwiseParallel,
    PrepareModuleInput,
    SequenceParallel,
)

logger = ezpz.get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='2D Parallel Training')
    parser.add_argument('--dim', type=int, default=256)
    parser.add_argument('--n_layers', type=int, default=24)
    parser.add_argument('--n_heads', type=int, default=32)
    parser.add_argument('--n_kv_heads', type=int, default=8)
    parser.add_argument('--multiple_of', type=int, default=360)
    parser.add_argument('--ffn_dim_multiplier', type=float, default=None)
    parser.add_argument('--norm_eps', type=float, default=1e-5)
    parser.add_argument('--vocab_size', type=int, default=32_000)
    parser.add_argument('--seq_length', type=int, default=128)
    parser.add_argument('--lr', type=float, default=3e-3)
    parser.add_argument('--num_iterations', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--tpsize', type=int, default=2)
    # parser.add_argument('--max_batch_size', type=int, default=None)
    parser.add_argument('--max_seq_len', type=int, default=32768)
    parser.add_argument('--depth_init', type=bool, default=True)
    # max_batch_size: int = 32
    # max_seq_len: int = 32768
    # depth_init: bool = True
    return parser.parse_args()


def parallelize(model: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
    tp_mesh = device_mesh['tp']
    dp_mesh = device_mesh['dp']

    model.init_weights()
    model = parallelize_module(
        model,
        tp_mesh,
        {
            'tok_embeddings': RowwiseParallel(
                input_layouts=Replicate(),
                output_layouts=Shard(1),
            ),
            'norm': SequenceParallel(),
            'output': ColwiseParallel(
                input_layouts=Shard(1),
                output_layouts=Replicate(),
            ),
        },
    )

    for _, transformer_block in enumerate(model.layers):
        layer_tp_plan = {
            'attention_norm': SequenceParallel(),
            'attention': PrepareModuleInput(
                input_layouts=(Shard(1), None),  # type:ignore
                desired_input_layouts=(Replicate(), None),  # type:ignore
            ),
            'attention.wq': ColwiseParallel(),
            'attention.wk': ColwiseParallel(),
            'attention.wv': ColwiseParallel(),
            'attention.wo': RowwiseParallel(output_layouts=Shard(1)),
            'ffn_norm': SequenceParallel(),
            'feed_forward': PrepareModuleInput(
                input_layouts=(Shard(1),),
                desired_input_layouts=(Replicate(),),
            ),
            'feed_forward.w1': ColwiseParallel(),
            'feed_forward.w2': RowwiseParallel(output_layouts=Shard(1)),
            'feed_forward.w3': ColwiseParallel(),
        }

        attn_layer = transformer_block.attention
        attn_layer.n_heads = attn_layer.n_heads // tp_mesh.size()
        attn_layer.n_kv_heads = attn_layer.n_kv_heads // tp_mesh.size()
        parallelize_module(
            module=transformer_block,
            device_mesh=tp_mesh,
            parallelize_plan=layer_tp_plan,
        )
    sharded_model = FSDP(
        model,
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16,
            cast_forward_inputs=True,
            reduce_dtype=torch.float32,
        ),
        device_mesh=dp_mesh,
    )
    logger.info(f'Model after parallelization:\n{sharded_model=}\n')
    return sharded_model


def train(args: argparse.Namespace):
    _ = ezpz.setup_torch('DDP', tpsize=args.tpsize)
    world_size = ezpz.get_world_size()
    assert (
        world_size % args.tpsize == 0
    ), 'WORLD_SIZE must be divisible by TPSIZE'
    dpsize = world_size // args.tpsize
    device_mesh = init_device_mesh(
        str(ezpz.get_torch_device()),
        (dpsize, args.tpsize),
        mesh_dim_names=('dp', 'tp'),
    )
    logger.info(f'Device mesh created:\n{device_mesh=}')

    config = ModelArgs(
        dim=args.dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        n_kv_heads=args.n_kv_heads,
        vocab_size=args.vocab_size,
        multiple_of=args.multiple_of,
    )
    device_type = str(ezpz.get_torch_device(as_torch_device=False))
    device_id = f'{device_type}:{ezpz.get_local_rank()}'
    model = Transformer.from_model_args(config)
    logger.info(
        f'\n{summarize_model(model, verbose=False, depth=2)}'  #', input_size=inshape)}'
    )
    model.to(device_id)
    model = parallelize(model, device_mesh)
    logger.info(f'Creating AdamW optimizer with lr={args.lr}')
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, foreach=True)

    device = ezpz.get_torch_device(as_torch_device=False)

    dataset = RandomTokenDataset(
        vocab_size=args.vocab_size, seq_length=args.seq_length
    )
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size
    )  # , num_workers=0)
    logger.info('\nStarting 2D training...')
    model.train()
    history = ezpz.History()

    # # model.init_weights()
    # tdist.barrier()
    # For TP, input needs to be the same across all TP ranks.
    # while for SP, input can be different across all ranks
    # We will use dp_rank for setting the random seed
    # to mimic the behavior of the dataloader
    for idx, batch in enumerate(dataloader):
        t0 = perf_counter()
        batch = batch.to(device)
        batch.to(torch.long)
        inp = batch[:, :-1]
        labels = batch[:, 1:]
        if idx == 0:
            logger.info(f'{inp.shape=}')
        output = model(inp)
        t1 = perf_counter()
        loss = F.cross_entropy(
            output.reshape(-1, output.size(-1)), labels.reshape(-1)
        )
        loss.backward()
        optimizer.step()
        t2 = perf_counter()
        logger.info(
            history.update(
                {
                    'iter': idx,
                    'loss': loss.item(),
                    'dt': t2 - t0,
                    'dtf': t1 - t0,
                    'dtb': t2 - t1,
                }
            )
        )
    logger.info('Finished 2D training')
    if ezpz.get_rank() == 0:
        dataset = history.finalize(
            run_name='mmm-fsdp-tp', dataset_fname='train', therm_frac=0.1
        )
        logger.info(f'{dataset=}')


if __name__ == '__main__':
    args = parse_args()
    train(args)
