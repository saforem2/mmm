"""
mmm/examples/fsdp_tp.py

Sam Foreman
2024-12-31

Modified from:
<https://pytorch.org/tutorials/intermediate/TP_tutorial.html>
"""
import argparse

import ezpz

import torch

import torch.nn as nn

from mmm.models.llama import Transformer, ModelArgs

from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed._tensor import Shard, Replicate

from torch.distributed.tensor.parallel import (
    parallelize_module,
    ColwiseParallel,
    RowwiseParallel,
    PrepareModuleInput,
    SequenceParallel,
)

"""
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


logger = ezpz.get_logger(__name__)


def parse_args():
    args = argparse.ArgumentParser(description='2D Parallel Training')
    # args: dim, n_layers, n_heads, vocab_size
    args.add_argument('--dim', type=int, default=256)
    args.add_argument('--n_layers', type=int, default=2)
    args.add_argument('--n_heads', type=int, default=16)
    args.add_argument('--vocab_size', type=int, default=32000)
    args.add_argument('--lr', type=float, default=3e-3)
    args.add_argument('--num_iterations', type=int, default=10)
    args.add_argument('--batch_size', type=int, default=2)
    args.add_argument('--tpsize', type=int, default=2)
    return args.parse_args()


def parallelize(
    model: nn.Module, device_mesh: DeviceMesh
) -> nn.Module:
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
    sharded_model = FSDP(model, device_mesh=dp_mesh)
    logger.info(f'Model after parallelization: {sharded_model=}\n')
    return sharded_model


def train(args: argparse.Namespace):
    _ = ezpz.setup_torch('DDP')  # , tensor_parallel_size=args.tpsize)
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
        vocab_size=args.vocab_size,
    )
    model = Transformer.from_model_args(config).to(ezpz.get_torch_device())
    model = parallelize(model, device_mesh)
    logger.info(f'Creating AdamW optimizer with lr={args.lr}')
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, foreach=True
    )

    logger.info('\nStarting 2D training...')
    device = ezpz.get_torch_device(as_torch_device=False)

    # For TP, input needs to be the same across all TP ranks.
    # while for SP, input can be different across all ranks
    # We will use dp_rank for setting the random seed
    # to mimic the behavior of the dataloader
    for i in range(args.num_iterations):
        torch.manual_seed(i + device_mesh['dp'].get_local_rank())
        inp = torch.randint(
            0,
            config.vocab_size,
            (args.batch_size, 10),
        )
        inp.to(device)
        output = model(inp)
        loss = output.sum()
        loss.backward()
        optimizer.step()
        logger.info(f'iter={i}, loss={loss.item()}')

    logger.info('Finished 2D training')


if __name__ == '__main__':
    args = parse_args()
    train(args)
