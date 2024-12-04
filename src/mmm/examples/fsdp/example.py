# Based on: https://github.com/pytorch/examples/blob/master/mnist/main.py
import argparse
import logging
import os
from pathlib import Path
import time
from typing import Optional

import ezpz as ez
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms

logger = logging.getLogger(__name__)

RANK = ez.setup_torch(
    backend=os.environ.get("BACKEND", "DDP"),
)
WORLD_SIZE = ez.get_world_size()
DEVICE = ez.get_torch_device()
DEVICE_ID = f"{DEVICE}:{ez.get_local_rank()}"

logger.setLevel("INFO") if RANK == 0 else logger.setLevel("CRITICAL")

if torch.cuda.is_available():
    torch.cuda.set_device(ez.get_local_rank())


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(model, train_loader, optimizer, epoch, sampler=None):
    model.train()
    ddp_loss = torch.zeros(2).to(DEVICE_ID)
    if sampler:
        sampler.set_epoch(epoch)
    t0 = time.perf_counter()
    for _, (batch, target) in enumerate(train_loader):
        batch, target = batch.to(DEVICE_ID), target.to(DEVICE_ID)
        optimizer.zero_grad()
        output = model(batch)
        loss = F.nll_loss(output, target, reduction="sum")
        loss.backward()
        optimizer.step()
        ddp_loss[0] += loss.item()
        ddp_loss[1] += len(batch)
    t1 = time.perf_counter()

    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
    return {
        "epoch": epoch,
        "dt": t1 - t0,
        "train_loss": ddp_loss[0] / ddp_loss[1],
    }


def test(model, test_loader):
    model.eval()
    # correct = 0
    ddp_loss = torch.zeros(3).to(DEVICE_ID)
    with torch.no_grad():
        for batch, target in test_loader:
            batch, target = batch.to(DEVICE_ID), target.to(DEVICE_ID)
            output = model(batch)
            ddp_loss[0] += F.nll_loss(
                output, target, reduction="sum"
            ).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            ddp_loss[1] += pred.eq(target.view_as(pred)).sum().item()
            ddp_loss[2] += len(batch)

    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)

    test_loss = ddp_loss[0] / ddp_loss[2]

    return {
        "test_loss": test_loss,
        "test_acc": 100.0 * ddp_loss[1] / ddp_loss[2],
    }


def prepare_data(outdir: Optional[str] = None) -> dict:
    outdir = "./data" if outdir is None else outdir
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    dataset1 = datasets.MNIST(
        outdir, train=True, download=True, transform=transform
    )
    dataset2 = datasets.MNIST(outdir, train=False, transform=transform)

    sampler1 = DistributedSampler(
        dataset1, rank=RANK, num_replicas=WORLD_SIZE, shuffle=True
    )
    sampler2 = DistributedSampler(dataset2, rank=RANK, num_replicas=WORLD_SIZE)

    train_kwargs = {"batch_size": args.batch_size, "sampler": sampler1}
    test_kwargs = {"batch_size": args.test_batch_size, "sampler": sampler2}
    cuda_kwargs = {"num_workers": 1, "pin_memory": True, "shuffle": False}
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)

    train_loader = torch.utils.data.DataLoader(  # type:ignore
        dataset1, **train_kwargs
    )
    test_loader = torch.utils.data.DataLoader(  # type:ignore
        dataset2, **test_kwargs
    )
    return {
        "train": {
            "data": dataset1,
            "loader": train_loader,
            "sampler": sampler1,
        },
        "test": {
            "data": dataset2,
            "loader": test_loader,
            "sampler": sampler2,
        },
    }


def prepare_model_optimizer_and_scheduler(args: argparse.Namespace) -> dict:
    model = Net().to(DEVICE_ID)
    dtypes = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp32": torch.float32,
    }
    dtype = dtypes[args.dtype]

    if args.dtype in {"fp16", "bf16", "fp32"}:
        model = FSDP(
            model,
            mixed_precision=MixedPrecision(
                param_dtype=dtype,
                cast_forward_inputs=True,
            ),
        )
    else:
        model = FSDP(model)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    logger.info(f"{model=}")

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    return {
        "model": model,
        "optimizer": optimizer,
        "scheduler": scheduler,
    }


def fsdp_main(args: argparse.Namespace) -> None:
    data = prepare_data()
    train_loader = data["train"]["loader"]
    test_loader = data["test"]["loader"]

    tmp = prepare_model_optimizer_and_scheduler(args)
    model = tmp["model"]
    optimizer = tmp["optimizer"]
    scheduler = tmp["scheduler"]

    history = ez.History()
    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        train_metrics = train(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            epoch=epoch,
            sampler=data["train"]["sampler"],
        )
        test_metrics = test(model, test_loader)
        scheduler.step()
        metrics = {**train_metrics, **test_metrics}
        _ = history.update(metrics)
        summary = ez.summarize_dict(metrics)
        logger.info(f"{summary}")

    logger.info(f"{args.epochs + 1} took {time.perf_counter() - start:.1f}s")
    dist.barrier()

    if args.save_model:
        dist.barrier()  # wait for slowpokes
        states = model.state_dict()
        if RANK == 0:
            torch.save(states, "mnist_cnn.pt")

    if RANK == 0:
        mplotdir = Path("plots")
        tplotdir = mplotdir.joinpath("tplots")
        tplotdir.mkdir(exist_ok=True, parents=True)
        dataset = history.plot_all(outdir=mplotdir)
        _ = history.tplot_all(
            outdir=tplotdir, append=True, xkey="epoch", dataset=dataset
        )
        logger.info(f"{dataset=}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PyTorch MNIST Example using FSDP"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        metavar="D",
        help="Datatype for training (default=bf16).",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        metavar="N",
        help="number of epochs to train (default: 10)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        metavar="LR",
        help="learning rate (default: 1e-3)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.7,
        metavar="M",
        help="Learning rate step gamma (default: 0.7)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        metavar="S",
        help="random seed (default: 1)",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        default=False,
        help="For Saving the current Model",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    torch.manual_seed(args.seed)
    fsdp_main(args=args)
