# `mmm`: Multi-Model Models

## 🐣 Getting Started

### 🏡 Setup Environment

We use [saforem2/`ezpz`](https://github.com/saforem2/ezpz)
for setting up, launching, and orchestrating our distributed training.

In particular, we can use the `ezpz_setup_env` helper function from
[`ezpz/bin/utils.sh`](https://github.com/saforem2/ezpz/blob/main/src/ezpz/bin/utils.sh):

```bash
source /dev/stdin <<< $(curl 'https://raw.githubusercontent.com/saforem2/ezpz/refs/heads/main/src/ezpz/bin/utils.sh')
ezpz_setup_env
```

🪄 This will, _automagically_:

- Setup + activate python environment
- Determine available resources (i.e. `NHOSTS`, `NGPU_PER_HOST`, `NGPUS`)
- Define a `launch` alias to launch our application across them

For additional information, see [`ezpz`](https://github.com/saforem2/ezpz).

**Note**: This is _technically_ optional, but highly recommended as it will allow you to automatically launch on any[^any] distributed setup with a compatible MPI.

[^any]: This has been tested and confirmed to work on:

    - Any job behind a {PBS, slurm} job scheduler
    - All [ALCF](https://alcf.anl.gov) systems (Aurora, Polaris, Sunspot, Sophia, etc.)
    - Frontier (AMD system) @ OLCF
    - Perlmutter (NVIDIA system) @ NERSC
    - Distributed CPUs via `mpirun`
    - Distributed `mps` devices via `mpirun`
    
    Both PBS and Slurm job schedulers are supported and the specifics of the running job will be used to populate the corresponding `launch` command.

### 📦 Install

From local clone (**recommended**):

```bash
git clone https://github.com/saforem2/mmm
python3 -m pip install -e mmm
```

<details><summary>From git:</summary>

```bash
python3 -m pip install -e "git+https://github.com/saforem2/mmm#egg=mmm"
```

</details>

<!--
> [!TIP]
> We use [`ezpz_setup_env`](https://github.com/saforem2/ezpz) to 🪄 _automagically_:
> - Setup + activate python environment
> - Determine available resources (i.e. `NHOSTS`, `NGPU_PER_HOST`, `NGPUS`)
> - Define a `launch` alias to launch our application across them
>
> This is optional, but will allow you to _automatically_ launch on any[^any] distributed setup with MPI.
> ```bash
> source /dev/stdin <<< $(curl 'https://raw.githubusercontent.com/saforem2/ezpz/refs/heads/main/src/ezpz/bin/utils.sh')
> ezpz_setup_env
> ```
> 
> [^any]: Both PBS and Slurm job schedulers are supported and the specifics of the running job will be used to populate the corresponding `launch` command.
> 
> - This will:
>  - automatically setup python
>  - create a `launch` alias for launching applications
>  
> For additional information, see: \[🍋 [saforem2/`ezpz`](https://github.com/saforem2/ezpz)\]
-->

<!--
1. Clone repo:

   ```bash
   git clone https://github.com/saforem2/mmm
   cd mmm
   ```

1. Install `mmm`:

   ```bash
   python3 -m pip install -e . --require-virtualenv
   ```
-->

## 🖼️ Example: ViT

We can now `launch` the example in
[`src/mmm/trainer/vit.py`](/src/mmm/trainer/vit.py):

```bash
launch python3 -m mmm.trainer.vit
```

<details closed><summary>Output:</summary>

```bash

from typing import Optional
```

</details>

## 📝 Example: FSDP

```bash
launch python3 -Wignore -m mmm.trainer.fsdp
```

<details closed><summary>Output:</summary>

#[🐍 aurora_nre_models_frameworks-2024.2.1_u1](👻 aurora_nre_models_frameworks-2024.2.1_u1)
#[🤖][02:27:58 PM][foremans@x4211c7s0b0n0][…/mmm/src/mmm][🌱 main][!]
$ CCL_LOG_LEVEL=ERROR launch python3 -Wignore -m mmm.trainer.fsdp
Disabling local launch: multi-node application
Connected to tcp://x4211c7s0b0n0.hostmgmt2211.cm.aurora.alcf.anl.gov:7919
Found executable /lus/flare/projects/Aurora_deployment/foremans/projects/saforem2/mmm/venvs/aurora_nre_models_frameworks-2024.2.1_u1/bin/python3
Launching application cfa9ce41-cd3f-45ce-b0a4-ff6b5d4fc67c
[2024-12-26 14:28:08.822672][INFO][dist.py:348] - [device='xpu'][rank=4/23][local_rank=4/11][node=0/1]
[2024-12-26 14:28:08.843632][INFO][dist.py:348] - [device='xpu'][rank=1/23][local_rank=1/11][node=1/1]
[2024-12-26 14:28:08.989100][INFO][dist.py:348] - [device='xpu'][rank=8/23][local_rank=8/11][node=0/1]
[2024-12-26 14:28:09.006545][INFO][dist.py:348] - [device='xpu'][rank=6/23][local_rank=6/11][node=0/1]
[2024-12-26 14:28:09.015927][INFO][dist.py:348] - [device='xpu'][rank=10/23][local_rank=10/11][node=0/1]
[2024-12-26 14:28:09.016602][INFO][dist.py:348] - [device='xpu'][rank=7/23][local_rank=7/11][node=1/1]
[2024-12-26 14:28:09.018056][INFO][dist.py:348] - [device='xpu'][rank=11/23][local_rank=11/11][node=1/1]
[2024-12-26 14:28:09.035759][INFO][dist.py:348] - [device='xpu'][rank=9/23][local_rank=9/11][node=1/1]
[2024-12-26 14:28:09.063471][INFO][dist.py:348] - [device='xpu'][rank=2/23][local_rank=2/11][node=0/1]
[2024-12-26 14:28:09.089491][INFO][dist.py:348] - [device='xpu'][rank=5/23][local_rank=5/11][node=1/1]
[2024-12-26 14:28:09.095103][INFO][dist.py:348] - [device='xpu'][rank=3/23][local_rank=3/11][node=1/1]
[2024-12-26 14:28:11.114689][INFO][dist.py:348] - [device='xpu'][rank=21/23][local_rank=9/11][node=1/1]
[2024-12-26 14:28:11.117251][INFO][dist.py:348] - [device='xpu'][rank=20/23][local_rank=8/11][node=0/1]
[2024-12-26 14:28:11.121203][INFO][dist.py:348] - [device='xpu'][rank=13/23][local_rank=1/11][node=1/1]
[2024-12-26 14:28:11.121739][INFO][dist.py:348] - [device='xpu'][rank=12/23][local_rank=0/11][node=0/1]
[2024-12-26 14:28:11.122393][INFO][dist.py:348] - [device='xpu'][rank=23/23][local_rank=11/11][node=1/1]
[2024-12-26 14:28:11.127747][INFO][dist.py:348] - [device='xpu'][rank=22/23][local_rank=10/11][node=0/1]
[2024-12-26 14:28:11.131268][INFO][dist.py:348] - [device='xpu'][rank=16/23][local_rank=4/11][node=0/1]
[2024-12-26 14:28:11.170519][INFO][dist.py:348] - [device='xpu'][rank=17/23][local_rank=5/11][node=1/1]
[2024-12-26 14:28:11.180653][INFO][dist.py:348] - [device='xpu'][rank=14/23][local_rank=2/11][node=0/1]
[2024-12-26 14:28:11.197443][INFO][dist.py:348] - [device='xpu'][rank=15/23][local_rank=3/11][node=1/1]
[2024-12-26 14:28:11.268430][INFO][dist.py:348] - [device='xpu'][rank=19/23][local_rank=7/11][node=1/1]
[2024-12-26 14:28:11.270448][INFO][dist.py:348] - [device='xpu'][rank=18/23][local_rank=6/11][node=0/1]
[2024-12-26 14:28:11.279857][INFO][dist.py:92] -

[dist_info]:
  • DEVICE=xpu
  • DEVICE_ID=xpu:0
  • DISTRIBUTED_BACKEND=ccl
  • GPUS_PER_NODE=12
  • HOSTS=['x4211c7s0b0n0.hostmgmt2211.cm.aurora.alcf.anl.gov', 'x4211c7s1b0n0.hostmgmt2211.cm.aurora.alcf.anl.gov']
  • HOSTFILE=/var/spool/pbs/aux/1227800.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov
  • HOSTNAME=x4211c7s0b0n0.hostmgmt2211.cm.aurora.alcf.anl.gov
  • LOCAL_RANK=0
  • MACHINE=Aurora
  • NUM_NODES=2
  • NGPUS=24
  • NGPUS_AVAILABLE=24
  • NODE_ID=0
  • RANK=0
  • SCHEDULER=PBS
  • WORLD_SIZE_TOTAL=24
  • WORLD_SIZE_IN_USE=24
  • LAUNCH_CMD=mpiexec --verbose --envall -n 24 -ppn 12 --hostfile /var/spool/pbs/aux/1227800.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov --cpu-bind depth -d 16


[2024-12-26 14:28:11.284103][INFO][dist.py:725] - Using oneccl_bindings from: /opt/aurora/24.180.1/frameworks/aurora_nre_models_frameworks-2024.2.1_u1/lib/python3.10/site-packages/oneccl_bindings_for_pytorch/__init__.py
[2024-12-26 14:28:11.284551][INFO][dist.py:727] - Using ipex from: /opt/aurora/24.180.1/frameworks/aurora_nre_models_frameworks-2024.2.1_u1/lib/python3.10/site-packages/intel_extension_for_pytorch/__init__.py
[2024-12-26 14:28:11.284933][INFO][dist.py:728] - [0/24] Using device='xpu' with backend='DDP' + 'ccl' for distributed training.
[2024-12-26 14:28:11.290154][INFO][dist.py:348] - [device='xpu'][rank=0/23][local_rank=0/11][node=0/1]
[2024-12-26 14:28:11.290709][WARNING][_logger.py:68] - Using [24 / 24] available "xpu" devices !!
2024:12:26-14:28:11:(43460) |CCL_WARN| value of CCL_LOG_LEVEL changed to be error (default:warn)
[2024-12-26 14:28:12.499812][INFO][fsdp.py:185] - model=FullyShardedDataParallel(
  (_fsdp_wrapped_module): Net(
    (conv1): Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1))
    (conv2): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1))
    (dropout1): Dropout(p=0.25, inplace=False)
    (dropout2): Dropout(p=0.5, inplace=False)
    (fc1): Linear(in_features=9216, out_features=128, bias=True)
    (fc2): Linear(in_features=128, out_features=10, bias=True)
  )
)
[2024-12-26 14:28:26.084048][INFO][fsdp.py:220] - epoch=1 dt=13.054375 train_loss=0.641844 test_loss=0.151953 test_acc=95.353714
[2024-12-26 14:28:27.072836][INFO][fsdp.py:220] - epoch=2 dt=0.669714 train_loss=0.176456 test_loss=0.077788 test_acc=97.631897
[2024-12-26 14:28:28.110445][INFO][fsdp.py:220] - epoch=3 dt=0.710506 train_loss=0.117828 test_loss=0.061101 test_acc=98.121506
[2024-12-26 14:28:29.133584][INFO][fsdp.py:220] - epoch=4 dt=0.704271 train_loss=0.098093 test_loss=0.050803 test_acc=98.321342
[2024-12-26 14:28:30.054510][INFO][fsdp.py:220] - epoch=5 dt=0.602212 train_loss=0.084964 test_loss=0.046719 test_acc=98.481216
[2024-12-26 14:28:31.060956][INFO][fsdp.py:220] - epoch=6 dt=0.612104 train_loss=0.077979 test_loss=0.044652 test_acc=98.521179
[2024-12-26 14:28:32.000895][INFO][fsdp.py:220] - epoch=7 dt=0.626015 train_loss=0.072789 test_loss=0.043784 test_acc=98.571144
[2024-12-26 14:28:32.927694][INFO][fsdp.py:220] - epoch=8 dt=0.630451 train_loss=0.071375 test_loss=0.042176 test_acc=98.621101
[2024-12-26 14:28:33.926055][INFO][fsdp.py:220] - epoch=9 dt=0.582817 train_loss=0.069234 test_loss=0.041798 test_acc=98.641090
[2024-12-26 14:28:34.847603][INFO][fsdp.py:220] - epoch=10 dt=0.632328 train_loss=0.065591 test_loss=0.041320 test_acc=98.571144
[2024-12-26 14:28:34.848981][INFO][fsdp.py:222] - 11 epochs took 22.3s
[2024-12-26 14:28:35.172290][INFO][history.py:696] - Saving epoch plot to: /lus/flare/projects/Aurora_deployment/foremans/projects/saforem2/mmm/src/mmm/plots
[2024-12-26 14:28:35.173081][INFO][history.py:700] - Saving epoch plot to: /lus/flare/projects/Aurora_deployment/foremans/projects/saforem2/mmm/src/mmm/plots/pngs/epoch.png
[2024-12-26 14:28:35.318850][INFO][history.py:700] - Saving epoch plot to: /lus/flare/projects/Aurora_deployment/foremans/projects/saforem2/mmm/src/mmm/plots/svgs/epoch.svg
[2024-12-26 14:28:35.393567][INFO][history.py:696] - Saving dt plot to: /lus/flare/projects/Aurora_deployment/foremans/projects/saforem2/mmm/src/mmm/plots
[2024-12-26 14:28:35.394396][INFO][history.py:700] - Saving dt plot to: /lus/flare/projects/Aurora_deployment/foremans/projects/saforem2/mmm/src/mmm/plots/pngs/dt.png
[2024-12-26 14:28:35.531726][INFO][history.py:700] - Saving dt plot to: /lus/flare/projects/Aurora_deployment/foremans/projects/saforem2/mmm/src/mmm/plots/svgs/dt.svg
[2024-12-26 14:28:35.605478][INFO][history.py:696] - Saving train_loss plot to: /lus/flare/projects/Aurora_deployment/foremans/projects/saforem2/mmm/src/mmm/plots
[2024-12-26 14:28:35.606234][INFO][history.py:700] - Saving train_loss plot to: /lus/flare/projects/Aurora_deployment/foremans/projects/saforem2/mmm/src/mmm/plots/pngs/train_loss.png
[2024-12-26 14:28:35.751727][INFO][history.py:700] - Saving train_loss plot to: /lus/flare/projects/Aurora_deployment/foremans/projects/saforem2/mmm/src/mmm/plots/svgs/train_loss.svg
[2024-12-26 14:28:35.835458][INFO][history.py:696] - Saving test_loss plot to: /lus/flare/projects/Aurora_deployment/foremans/projects/saforem2/mmm/src/mmm/plots
[2024-12-26 14:28:35.836373][INFO][history.py:700] - Saving test_loss plot to: /lus/flare/projects/Aurora_deployment/foremans/projects/saforem2/mmm/src/mmm/plots/pngs/test_loss.png
[2024-12-26 14:28:35.986343][INFO][history.py:700] - Saving test_loss plot to: /lus/flare/projects/Aurora_deployment/foremans/projects/saforem2/mmm/src/mmm/plots/svgs/test_loss.svg
[2024-12-26 14:28:36.060131][INFO][history.py:696] - Saving test_acc plot to: /lus/flare/projects/Aurora_deployment/foremans/projects/saforem2/mmm/src/mmm/plots
[2024-12-26 14:28:36.060910][INFO][history.py:700] - Saving test_acc plot to: /lus/flare/projects/Aurora_deployment/foremans/projects/saforem2/mmm/src/mmm/plots/pngs/test_acc.png
[2024-12-26 14:28:36.199877][INFO][history.py:700] - Saving test_acc plot to: /lus/flare/projects/Aurora_deployment/foremans/projects/saforem2/mmm/src/mmm/plots/svgs/test_acc.svg
                            dt [2024-12-26-142836]
    ┌─────────────────────────────────────────────────────────────────────┐
13.1┤▌                                                                    │
    │▚                                                                    │
    │▝▖                                                                   │
11.0┤ ▚                                                                   │
    │ ▐                                                                   │
    │  ▌                                                                  │
 8.9┤  ▐                                                                  │
    │   ▌                                                                 │
 6.8┤   ▚                                                                 │
    │   ▝▖                                                                │
    │    ▚                                                                │
 4.7┤    ▐                                                                │
    │     ▌                                                               │
    │     ▐                                                               │
 2.7┤      ▌                                                              │
    │      ▚                                                              │
    │      ▝▖                                                             │
 0.6┤       ▚▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄│
    └┬───────┬──────┬───────┬──────┬───────┬──────┬───────┬──────┬───────┬┘
     1       2      3       4      5       6      7       8      9      10
dt                                   epoch
[2024-12-26 14:28:36.304403][INFO][plot.py:220] - Appending plot to: plots/tplots/dt.txt
text saved in /lus/flare/projects/Aurora_deployment/foremans/projects/saforem2/mmm/src/mmm/plots/tplots/dt.txt
                         train_loss [2024-12-26-142836]
     ┌────────────────────────────────────────────────────────────────────┐
0.642┤▌                                                                   │
     │▐                                                                   │
     │ ▌                                                                  │
0.546┤ ▐                                                                  │
     │  ▌                                                                 │
     │  ▐                                                                 │
0.450┤   ▌                                                                │
     │   ▝▖                                                               │
0.354┤    ▚                                                               │
     │    ▝▖                                                              │
     │     ▚                                                              │
0.258┤     ▝▖                                                             │
     │      ▚                                                             │
     │      ▝▖                                                            │
0.162┤       ▝▄▖                                                          │
     │         ▝▀▚▄▖                                                      │
     │             ▝▀▚▄▄▄▄▄▄▄                                             │
0.066┤                       ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄│
     └┬──────┬───────┬──────┬───────┬──────┬───────┬──────┬───────┬──────┬┘
      1      2       3      4       5      6       7      8       9     10
train_loss                            epoch
[2024-12-26 14:28:36.310045][INFO][plot.py:220] - Appending plot to: plots/tplots/train_loss.txt
text saved in /lus/flare/projects/Aurora_deployment/foremans/projects/saforem2/mmm/src/mmm/plots/tplots/train_loss.txt
                          test_loss [2024-12-26-142836]
     ┌────────────────────────────────────────────────────────────────────┐
0.152┤▌                                                                   │
     │▐                                                                   │
     │ ▚                                                                  │
0.134┤ ▝▖                                                                 │
     │  ▐                                                                 │
     │   ▚                                                                │
0.115┤   ▝▖                                                               │
     │    ▐                                                               │
0.097┤     ▚                                                              │
     │     ▝▖                                                             │
     │      ▐                                                             │
0.078┤       ▚                                                            │
     │        ▀▄▖                                                         │
     │          ▝▚▄                                                       │
0.060┤             ▀▄▖                                                    │
     │               ▝▀▀▄▄▖                                               │
     │                    ▝▀▀▄▄▄▄▄▄▄▖                                     │
0.041┤                              ▝▀▀▀▀▀▀▀▀▀▀▀▀▀▀▚▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄│
     └┬──────┬───────┬──────┬───────┬──────┬───────┬──────┬───────┬──────┬┘
      1      2       3      4       5      6       7      8       9     10
test_loss                             epoch
[2024-12-26 14:28:36.316097][INFO][plot.py:220] - Appending plot to: plots/tplots/test_loss.txt
text saved in /lus/flare/projects/Aurora_deployment/foremans/projects/saforem2/mmm/src/mmm/plots/tplots/test_loss.txt
                          test_acc [2024-12-26-142836]
     ┌────────────────────────────────────────────────────────────────────┐
98.64┤                                     ▗▄▄▄▄▄▄▄▄▄▄▄▄▄▄▞▀▀▀▀▀▀▀▚▄▄▄▄▄▄▄│
     │                      ▗▄▄▄▄▄▄▄▀▀▀▀▀▀▀▘                              │
     │                 ▗▄▄▀▀▘                                             │
98.09┤             ▗▄▀▀▘                                                  │
     │          ▗▄▀▘                                                      │
     │       ▗▄▀▘                                                         │
97.55┤      ▗▘                                                            │
     │      ▞                                                             │
97.00┤     ▐                                                              │
     │     ▌                                                              │
     │    ▞                                                               │
96.45┤   ▗▘                                                               │
     │   ▌                                                                │
     │  ▐                                                                 │
95.90┤ ▗▘                                                                 │
     │ ▞                                                                  │
     │▐                                                                   │
95.35┤▌                                                                   │
     └┬──────┬───────┬──────┬───────┬──────┬───────┬──────┬───────┬──────┬┘
      1      2       3      4       5      6       7      8       9     10
test_acc                              epoch
[2024-12-26 14:28:36.321999][INFO][plot.py:220] - Appending plot to: plots/tplots/test_acc.txt
text saved in /lus/flare/projects/Aurora_deployment/foremans/projects/saforem2/mmm/src/mmm/plots/tplots/test_acc.txt
[2024-12-26 14:28:36.328707][INFO][fsdp.py:246] - dataset=<xarray.Dataset> Size: 360B
Dimensions:     (draw: 10)
Coordinates:
  * draw        (draw) int64 80B 0 1 2 3 4 5 6 7 8 9
Data variables:
    epoch       (draw) int64 80B 1 2 3 4 5 6 7 8 9 10
    dt          (draw) float64 80B 13.05 0.6697 0.7105 ... 0.6305 0.5828 0.6323
    train_loss  (draw) float32 40B 0.6418 0.1765 0.1178 ... 0.06923 0.06559
    test_loss   (draw) float32 40B 0.152 0.07779 0.0611 ... 0.0418 0.04132
    test_acc    (draw) float32 40B 95.35 97.63 98.12 98.32 ... 98.62 98.64 98.57
Application cfa9ce41 resources: utime=1241s stime=257s maxrss=2839580KB inblock=298774 oublock=688 minflt=15316801 majflt=541210 nvcsw=745685 nivcsw=411483
took: 0h:00m:38s
```

</details>
