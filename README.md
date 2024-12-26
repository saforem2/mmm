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
Disabling local launch: multi-node application
Connected to tcp://x4712c2s1b0n0.hostmgmt2712.cm.aurora.alcf.anl.gov:7919
Found executable /flare/Aurora_deployment/foremans/projects/saforem2/mmm/venvs/aurora_nre_models_frameworks-2024.2.1_u1/bin/python3
Launching application fbf7ec7d-2a79-4443-bda3-32ce2add2ea0
[2024-12-26 12:58:35.056566][INFO][dist.py:348] - [device='xpu'][rank=17/23][local_rank=5/11][node=1/1]
[2024-12-26 12:58:35.057975][INFO][dist.py:348] - [device='xpu'][rank=15/23][local_rank=3/11][node=1/1]
[2024-12-26 12:58:35.063629][INFO][dist.py:348] - [device='xpu'][rank=4/23][local_rank=4/11][node=0/1]
[2024-12-26 12:58:35.066485][INFO][dist.py:348] - [device='xpu'][rank=1/23][local_rank=1/11][node=1/1]
[2024-12-26 12:58:35.069585][INFO][dist.py:348] - [device='xpu'][rank=12/23][local_rank=0/11][node=0/1]
[2024-12-26 12:58:35.072192][INFO][dist.py:348] - [device='xpu'][rank=13/23][local_rank=1/11][node=1/1]
[2024-12-26 12:58:35.074019][INFO][dist.py:348] - [device='xpu'][rank=14/23][local_rank=2/11][node=0/1]
[2024-12-26 12:58:35.075079][INFO][dist.py:348] - [device='xpu'][rank=3/23][local_rank=3/11][node=1/1]
[2024-12-26 12:58:35.074998][INFO][dist.py:348] - [device='xpu'][rank=16/23][local_rank=4/11][node=0/1]
[2024-12-26 12:58:35.082450][INFO][dist.py:348] - [device='xpu'][rank=5/23][local_rank=5/11][node=1/1]
[2024-12-26 12:58:35.086127][INFO][dist.py:348] - [device='xpu'][rank=2/23][local_rank=2/11][node=0/1]
[2024-12-26 12:58:35.096759][INFO][dist.py:348] - [device='xpu'][rank=8/23][local_rank=8/11][node=0/1]
[2024-12-26 12:58:35.141914][INFO][dist.py:348] - [device='xpu'][rank=9/23][local_rank=9/11][node=1/1]
[2024-12-26 12:58:35.160933][INFO][dist.py:348] - [device='xpu'][rank=10/23][local_rank=10/11][node=0/1]
[2024-12-26 12:58:35.165743][INFO][dist.py:348] - [device='xpu'][rank=11/23][local_rank=11/11][node=1/1]
[2024-12-26 12:58:35.265369][INFO][dist.py:348] - [device='xpu'][rank=22/23][local_rank=10/11][node=0/1]
[2024-12-26 12:58:35.270633][INFO][dist.py:348] - [device='xpu'][rank=20/23][local_rank=8/11][node=0/1]
[2024-12-26 12:58:35.270760][INFO][dist.py:348] - [device='xpu'][rank=23/23][local_rank=11/11][node=1/1]
[2024-12-26 12:58:35.334601][INFO][dist.py:348] - [device='xpu'][rank=6/23][local_rank=6/11][node=0/1]
[2024-12-26 12:58:35.334370][INFO][dist.py:348] - [device='xpu'][rank=7/23][local_rank=7/11][node=1/1]
[2024-12-26 12:58:35.335810][INFO][dist.py:348] - [device='xpu'][rank=21/23][local_rank=9/11][node=1/1]
[2024-12-26 12:58:35.337893][INFO][dist.py:348] - [device='xpu'][rank=19/23][local_rank=7/11][node=1/1]
[2024-12-26 12:58:35.344547][INFO][dist.py:348] - [device='xpu'][rank=18/23][local_rank=6/11][node=0/1]

[2024-12-26 12:58:35.349227][INFO][dist.py:92] -
[dist_info]:
  • DEVICE=xpu
  • DEVICE_ID=xpu:0
  • DISTRIBUTED_BACKEND=ccl
  • GPUS_PER_NODE=12
  • HOSTS=['x4712c2s1b0n0.hostmgmt2712.cm.aurora.alcf.anl.gov', 'x4712c2s2b0n0.hostmgmt2712.cm.aurora.alcf.anl.gov']
  • HOSTFILE=/var/spool/pbs/aux/1227576.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov
  • HOSTNAME=x4712c2s1b0n0.hostmgmt2712.cm.aurora.alcf.anl.gov
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
  • LAUNCH_CMD=mpiexec --verbose --envall -n 24 -ppn 12 --hostfile /var/spool/pbs/aux/1227576.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov --cpu-bind depth -d 16

[2024-12-26 12:58:35.354055][INFO][dist.py:725] - Using oneccl_bindings from: /opt/aurora/24.180.1/frameworks/aurora_nre_models_frameworks-2024.2.1_u1/lib/python3.10/site-packages/oneccl_bindings_for_pytorch/__init__.py
[2024-12-26 12:58:35.354482][INFO][dist.py:727] - Using ipex from: /opt/aurora/24.180.1/frameworks/aurora_nre_models_frameworks-2024.2.1_u1/lib/python3.10/site-packages/intel_extension_for_pytorch/__init__.py
[2024-12-26 12:58:35.354850][INFO][dist.py:728] - [0/24] Using device='xpu' with backend='DDP' + 'ccl' for distributed training.
[2024-12-26 12:58:35.359349][INFO][dist.py:348] - [device='xpu'][rank=0/23][local_rank=0/11][node=0/1]
[2024-12-26 12:58:35.359893][WARNING][_logger.py:68] - Using [24 / 24] available "xpu" devices !!
2024:12:26-12:58:35:(124163) |CCL_WARN| value of CCL_PROCESS_LAUNCHER changed to be pmix (default:hydra)
2024:12:26-12:58:35:(124163) |CCL_WARN| MPI was initialized externally, CCL-MPI specific environment is ignored
[2024-12-26 12:58:36.145325][INFO][vit.py:243] - Using native for SDPA backend
[2024-12-26 12:58:36.145798][INFO][vit.py:269] - Using AttentionBlock Attention with args.compile=False
[2024-12-26 12:58:36.146212][INFO][vit.py:85] - config=ViTConfig(img_size=224, batch_size=128, num_heads=16, head_dim=64, depth=24, patch_size=16)
[2024-12-26 12:58:36.146755][INFO][vit.py:87] - len(train_set)=1000000
[2024-12-26 12:59:05.251449][INFO][vit.py:115] -
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
VisionTransformer                        [128, 1000]               200,704
├─PatchEmbed: 1-1                        [128, 196, 1024]          787,456
├─Dropout: 1-2                           [128, 196, 1024]          --
├─Identity: 1-3                          [128, 196, 1024]          --
├─Identity: 1-4                          [128, 196, 1024]          --
├─Sequential: 1-5                        [128, 196, 1024]          302,235,648
├─Identity: 1-6                          [128, 196, 1024]          --
├─LayerNorm: 1-7                         [128, 1024]               2,048
├─Dropout: 1-8                           [128, 1024]               --
├─Linear: 1-9                            [128, 1000]               1,025,000
==========================================================================================
Total params: 304,250,856
Trainable params: 304,250,856
Non-trainable params: 0
Total mult-adds (G): 58.57
==========================================================================================
Input size (MB): 77.07
Forward/backward pass size (MB): 54465.11
Params size (MB): 1216.20
Estimated Total Size (MB): 55758.38
==========================================================================================
[2024-12-26 12:59:05.449000][INFO][vit.py:170] - Training with 24 x xpu (s), using torch_dtype=torch.bfloat16
[2024-12-26 12:59:19.651525][INFO][vit.py:197] - iter=0 loss=7.102205 dt=14.105401 dtf=13.301080 dtb=0.787732
[2024-12-26 12:59:20.693789][INFO][vit.py:197] - iter=1 loss=7.464778 dt=0.917288 dtf=0.303060 dtb=0.605716
[2024-12-26 12:59:21.724462][INFO][vit.py:197] - iter=2 loss=7.242332 dt=0.905948 dtf=0.291535 dtb=0.605900
[2024-12-26 12:59:22.756742][INFO][vit.py:197] - iter=3 loss=6.974534 dt=0.910001 dtf=0.296447 dtb=0.605194
[2024-12-26 12:59:23.781231][INFO][vit.py:197] - iter=4 loss=7.322947 dt=0.896517 dtf=0.289726 dtb=0.598474
[2024-12-26 12:59:24.804153][INFO][vit.py:197] - iter=5 loss=7.471062 dt=0.900163 dtf=0.295749 dtb=0.596026
[2024-12-26 12:59:25.836680][INFO][vit.py:197] - iter=6 loss=7.666119 dt=0.908761 dtf=0.293596 dtb=0.606492
[2024-12-26 12:59:26.855877][INFO][vit.py:197] - iter=7 loss=8.378406 dt=0.895732 dtf=0.293892 dtb=0.593241
[2024-12-26 12:59:27.877752][INFO][vit.py:197] - iter=8 loss=10.235675 dt=0.897727 dtf=0.293244 dtb=0.595893
[2024-12-26 12:59:28.899810][INFO][vit.py:197] - iter=9 loss=8.988402 dt=0.898550 dtf=0.293711 dtb=0.596322
[2024-12-26 12:59:29.919233][INFO][vit.py:197] - iter=10 loss=7.622493 dt=0.894075 dtf=0.291858 dtb=0.593592
                        train_iter [2024-12-26-125932]
    ┌─────────────────────────────────────────────────────────────────────┐
10.0┤                                                                  ▗▄▞│
    │                                                             ▗▄▄▀▀▘  │
    │                                                          ▗▄▀▘       │
 8.3┤                                                       ▄▄▀▘          │
    │                                                  ▗▄▄▀▀              │
    │                                              ▗▄▀▀▘                  │
 6.7┤                                           ▄▞▀▘                      │
    │                                       ▄▄▀▀                          │
 5.0┤                                  ▗▄▄▀▀                              │
    │                               ▄▄▀▘                                  │
    │                           ▗▄▞▀                                      │
 3.3┤                       ▄▄▞▀▘                                         │
    │                   ▄▞▀▀                                              │
    │               ▗▄▀▀                                                  │
 1.7┤           ▗▄▞▀▘                                                     │
    │       ▄▄▞▀▘                                                         │
    │   ▗▄▞▀                                                              │
 0.0┤▄▄▀▘                                                                 │
    └┬──────┬──────┬─────┬──────┬──────┬──────┬──────┬─────┬──────┬───────┘
     1      2      3     4      5      6      7      8     9     10
train_iter                        train/iter
[2024-12-26 12:59:32.352154][INFO][plot.py:220] - Appending plot to: /flare/Aurora_deployment/foremans/projects/saforem2/mmm/outputs/plots/vit/tplot/train_iter.txt
text saved in /lus/flare/projects/Aurora_deployment/foremans/projects/saforem2/mmm/outputs/plots/vit/tplot/train_iter.txt
                         train_loss [2024-12-26-125932]
     ┌────────────────────────────────────────────────────────────────────┐
10.24┤                                                     ▗▚             │
     │                                                     ▌ ▚            │
     │                                                    ▞   ▚           │
 9.69┤                                                   ▗▘    ▚          │
     │                                                   ▌      ▚         │
     │                                                  ▞        ▚        │
 9.15┤                                                 ▗▘         ▚▖      │
     │                                                 ▌           ▚      │
 8.61┤                                                ▞             ▚     │
     │                                               ▗▘              ▚    │
     │                                               ▞                ▌   │
 8.06┤                                             ▄▀                 ▝▖  │
     │                                           ▗▀                    ▝▖ │
     │                                         ▗▞▘                      ▝▖│
 7.52┤                                     ▗▄▄▞▘                         ▝│
     │     ▗▄▚▄▄▖                ▄▄▄▄▄▄▄▀▀▀▘                              │
     │  ▄▞▀▘    ▝▀▀▀▄▄        ▄▞▀                                         │
 6.97┤▀▀              ▀▀▚▄▄▄▞▀                                            │
     └┬──────┬─────┬──────┬──────┬──────┬─────┬──────┬──────┬─────┬───────┘
      1      2     3      4      5      6     7      8      9    10
train_loss                         train/iter
[2024-12-26 12:59:32.357394][INFO][plot.py:220] - Appending plot to: /flare/Aurora_deployment/foremans/projects/saforem2/mmm/outputs/plots/vit/tplot/train_loss.txt
text saved in /lus/flare/projects/Aurora_deployment/foremans/projects/saforem2/mmm/outputs/plots/vit/tplot/train_loss.txt
                         train_dt [2024-12-26-125932]
    ┌─────────────────────────────────────────────────────────────────────┐
14.1┤▌                                                                    │
    │▚                                                                    │
    │▝▖                                                                   │
11.9┤ ▌                                                                   │
    │ ▐                                                                   │
    │  ▌                                                                  │
 9.7┤  ▚                                                                  │
    │  ▝▖                                                                 │
 7.5┤   ▌                                                                 │
    │   ▐                                                                 │
    │    ▌                                                                │
 5.3┤    ▚                                                                │
    │    ▝▖                                                               │
    │     ▌                                                               │
 3.1┤     ▐                                                               │
    │      ▌                                                              │
    │      ▚                                                              │
 0.9┤      ▝▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄│
    └┬──────┬──────┬─────┬──────┬──────┬──────┬──────┬─────┬──────┬───────┘
     1      2      3     4      5      6      7      8     9     10
train_dt                          train/iter
[2024-12-26 12:59:32.363355][INFO][plot.py:220] - Appending plot to: /flare/Aurora_deployment/foremans/projects/saforem2/mmm/outputs/plots/vit/tplot/train_dt.txt
text saved in /lus/flare/projects/Aurora_deployment/foremans/projects/saforem2/mmm/outputs/plots/vit/tplot/train_dt.txt
                         train_dtf [2024-12-26-125932]
    ┌─────────────────────────────────────────────────────────────────────┐
13.3┤▌                                                                    │
    │▚                                                                    │
    │▝▖                                                                   │
11.1┤ ▌                                                                   │
    │ ▐                                                                   │
    │  ▌                                                                  │
 9.0┤  ▚                                                                  │
    │  ▝▖                                                                 │
 6.8┤   ▌                                                                 │
    │   ▐                                                                 │
    │    ▌                                                                │
 4.6┤    ▚                                                                │
    │    ▝▖                                                               │
    │     ▌                                                               │
 2.5┤     ▐                                                               │
    │      ▌                                                              │
    │      ▚                                                              │
 0.3┤      ▝▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄│
    └┬──────┬──────┬─────┬──────┬──────┬──────┬──────┬─────┬──────┬───────┘
     1      2      3     4      5      6      7      8     9     10
train_dtf                         train/iter
[2024-12-26 12:59:32.369629][INFO][plot.py:220] - Appending plot to: /flare/Aurora_deployment/foremans/projects/saforem2/mmm/outputs/plots/vit/tplot/train_dtf.txt
text saved in /lus/flare/projects/Aurora_deployment/foremans/projects/saforem2/mmm/outputs/plots/vit/tplot/train_dtf.txt
                          train_dtb [2024-12-26-125932]
     ┌────────────────────────────────────────────────────────────────────┐
0.788┤▌                                                                   │
     │▚                                                                   │
     │▝▖                                                                  │
0.755┤ ▌                                                                  │
     │ ▐                                                                  │
     │  ▌                                                                 │
0.723┤  ▐                                                                 │
     │  ▝▖                                                                │
0.690┤   ▚                                                                │
     │   ▝▖                                                               │
     │    ▌                                                               │
0.658┤    ▐                                                               │
     │     ▌                                                              │
     │     ▐                                                              │
0.626┤     ▝▖                                                             │
     │      ▚                                                             │
     │      ▝▄▄▄▄▄▄▄▄▄▄▄▄▄▄                   ▗                           │
0.593┤                     ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▘▀▀▀▚▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▚▄▄▄▄▄▄│
     └┬──────┬─────┬──────┬──────┬──────┬─────┬──────┬──────┬─────┬───────┘
      1      2     3      4      5      6     7      8      9    10
train_dtb                          train/iter
[2024-12-26 12:59:32.375669][INFO][plot.py:220] - Appending plot to: /flare/Aurora_deployment/foremans/projects/saforem2/mmm/outputs/plots/vit/tplot/train_dtb.txt
text saved in /lus/flare/projects/Aurora_deployment/foremans/projects/saforem2/mmm/outputs/plots/vit/tplot/train_dtb.txt
[2024-12-26 12:59:32.382346][INFO][vit.py:217] - dataset=<xarray.Dataset> Size: 528B
Dimensions:     (draw: 11)
Coordinates:
  * draw        (draw) int64 88B 0 1 2 3 4 5 6 7 8 9 10
Data variables:
    train_iter  (draw) int64 88B 0 1 2 3 4 5 6 7 8 9 10
    train_loss  (draw) float64 88B 7.102 7.465 7.242 6.975 ... 10.24 8.988 7.622
    train_dt    (draw) float64 88B 14.11 0.9173 0.9059 ... 0.8977 0.8986 0.8941
    train_dtf   (draw) float64 88B 13.3 0.3031 0.2915 ... 0.2932 0.2937 0.2919
    train_dtb   (draw) float64 88B 0.7877 0.6057 0.6059 ... 0.5959 0.5963 0.5936
[2024-12-26 12:59:32.384948][INFO][vit.py:276] - Took 56.24 seconds
Application fbf7ec7d resources: utime=5417s stime=893s maxrss=4521792KB inblock=285066 oublock=712 minflt=18332211 majflt=13497 nvcsw=460770 nivcsw=471955
```

</details>

## 📝 Example: FSDP

```bash
launch python3 \
  src/mmm/trainer/fsdp.py \
  --lr 1e-4 \
  --epochs 20 \
  --batch-size 1024 \
  --dtype bf16 \
  --gamma 0.99
```

<details closed><summary>Output:</summary>

```bash
# e.g. on Sunspot:
$ launch python3 -Wignore src/mmm/examples/fsdp/example.py --lr 1e-4 --batch-size 128 --epochs 10
Disabling local launch: multi-node application
Connected to tcp://x1921c0s2b0n0.hostmgmt2000.cm.americas.sgi.com:7919
Found executable /lus/gila/projects/Aurora_deployment/foremans/projects/saforem2/mmm/venvs/aurora_nre_models_frameworks-2024.2.1_u1/bin/python3
Launching application 515ccad8-5eee-498e-8782-74612e854f7c
[2024-12-04 08:29:36,983][INFO][ezpz.dist]: [device='xpu'][rank=1/23][local_rank=1/11][node=1/1]
[2024-12-04 08:29:36,987][INFO][ezpz.dist]: [device='xpu'][rank=4/23][local_rank=4/11][node=0/1]
[2024-12-04 08:29:37,228][INFO][ezpz.dist]: [device='xpu'][rank=20/23][local_rank=8/11][node=0/1]
[2024-12-04 08:29:37,230][INFO][ezpz.dist]: [device='xpu'][rank=16/23][local_rank=4/11][node=0/1]
[2024-12-04 08:29:37,231][INFO][ezpz.dist]: [device='xpu'][rank=5/23][local_rank=5/11][node=1/1]
[2024-12-04 08:29:37,233][INFO][ezpz.dist]: [device='xpu'][rank=9/23][local_rank=9/11][node=1/1]
[2024-12-04 08:29:37,233][INFO][ezpz.dist]: [device='xpu'][rank=10/23][local_rank=10/11][node=0/1]
[2024-12-04 08:29:37,239][INFO][ezpz.dist]: [device='xpu'][rank=12/23][local_rank=0/11][node=0/1]
[2024-12-04 08:29:37,240][INFO][ezpz.dist]: [device='xpu'][rank=13/23][local_rank=1/11][node=1/1]
[2024-12-04 08:29:37,240][INFO][ezpz.dist]: [device='xpu'][rank=3/23][local_rank=3/11][node=1/1]
[2024-12-04 08:29:37,242][INFO][ezpz.dist]: [device='xpu'][rank=8/23][local_rank=8/11][node=0/1]
[2024-12-04 08:29:37,243][INFO][ezpz.dist]: [device='xpu'][rank=2/23][local_rank=2/11][node=0/1]
[2024-12-04 08:29:37,247][INFO][ezpz.dist]: [device='xpu'][rank=11/23][local_rank=11/11][node=1/1]
[2024-12-04 08:29:37,280][INFO][ezpz.dist]: [device='xpu'][rank=23/23][local_rank=11/11][node=1/1]
[2024-12-04 08:29:37,286][INFO][ezpz.dist]: [device='xpu'][rank=21/23][local_rank=9/11][node=1/1]
[2024-12-04 08:29:37,289][INFO][ezpz.dist]: [device='xpu'][rank=22/23][local_rank=10/11][node=0/1]
[2024-12-04 08:29:37,331][INFO][ezpz.dist]: [device='xpu'][rank=14/23][local_rank=2/11][node=0/1]
[2024-12-04 08:29:37,332][INFO][ezpz.dist]: [device='xpu'][rank=17/23][local_rank=5/11][node=1/1]
[2024-12-04 08:29:37,332][INFO][ezpz.dist]: [device='xpu'][rank=15/23][local_rank=3/11][node=1/1]
[2024-12-04 08:29:37,360][INFO][ezpz.dist]: [device='xpu'][rank=7/23][local_rank=7/11][node=1/1]
[2024-12-04 08:29:37,361][INFO][ezpz.dist]: [device='xpu'][rank=6/23][local_rank=6/11][node=0/1]
[2024-12-04 08:29:37,362][INFO][ezpz.dist]: [device='xpu'][rank=18/23][local_rank=6/11][node=0/1]
[2024-12-04 08:29:37,367][INFO][ezpz.dist]: [device='xpu'][rank=19/23][local_rank=7/11][node=1/1]
[2024-12-04 08:29:37,380][INFO][ezpz.dist]:

[dist_info]:
  • DEVICE=xpu
  • DEVICE_ID=xpu:0
  • DISTRIBUTED_BACKEND=ccl
  • GPUS_PER_NODE=12
  • HOSTS=['x1921c0s2b0n0.hostmgmt2000.cm.americas.sgi.com', 'x1921c1s2b0n0.hostmgmt2000.cm.americas.sgi.com']
  • HOSTFILE=/var/spool/pbs/aux/10284362.amn-0001
  • HOSTNAME=x1921c0s2b0n0.hostmgmt2000.cm.americas.sgi.com
  • LOCAL_RANK=0
  • MACHINE=SunSpot
  • NUM_NODES=2
  • NGPUS=24
  • NGPUS_AVAILABLE=24
  • NODE_ID=0
  • RANK=0
  • SCHEDULER=PBS
  • WORLD_SIZE_TOTAL=24
  • WORLD_SIZE_IN_USE=24
  • LAUNCH_CMD=mpiexec --verbose --envall -n 24 -ppn 12 --hostfile /var/spool/pbs/aux/10284362.amn-0001 --cpu-bind depth -d 16


[2024-12-04 08:29:37,380][INFO][ezpz.dist]: Using oneccl_bindings from: /opt/aurora/24.180.1/frameworks/aurora_nre_models_frameworks-2024.2.1_u1/lib/python3.10/site-packages/oneccl_bindings_for_pytorch/__init__.py
[2024-12-04 08:29:37,380][INFO][ezpz.dist]: Using ipex from: /opt/aurora/24.180.1/frameworks/aurora_nre_models_frameworks-2024.2.1_u1/lib/python3.10/site-packages/intel_extension_for_pytorch/__init__.py
[2024-12-04 08:29:37,380][INFO][ezpz.dist]: [0/24] Using device='xpu' with backend='DDP' + 'ccl' for distributed training.
[2024-12-04 08:29:37,385][INFO][ezpz.dist]: [device='xpu'][rank=0/23][local_rank=0/11][node=0/1]
[2024-12-04 08:29:37,385][WARNING][ezpz.dist]: Using [24 / 24] available "xpu" devices !!
2024:12:04-08:29:37:(158466) |CCL_WARN| value of CCL_LOG_LEVEL changed to be error (default:warn)
[2024-12-04 08:29:38,578][INFO][__main__]: model=FullyShardedDataParallel(
  (_fsdp_wrapped_module): Net(
    (conv1): Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1))
    (conv2): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1))
    (dropout1): Dropout(p=0.25, inplace=False)
    (dropout2): Dropout(p=0.5, inplace=False)
    (fc1): Linear(in_features=9216, out_features=128, bias=True)
    (fc2): Linear(in_features=128, out_features=10, bias=True)
  )
)
[2024-12-04 08:29:51,996][INFO][__main__]: epoch=1 dt=12.933120 train_loss=1.801742 test_loss=1.092526 test_acc=80.725418
[2024-12-04 08:29:52,931][INFO][__main__]: epoch=2 dt=0.579176 train_loss=0.975217 test_loss=0.607414 test_acc=86.820541
[2024-12-04 08:29:53,894][INFO][__main__]: epoch=3 dt=0.674108 train_loss=0.689858 test_loss=0.469025 test_acc=89.158676
[2024-12-04 08:29:54,953][INFO][__main__]: epoch=4 dt=0.709618 train_loss=0.588500 test_loss=0.412870 test_acc=90.257797
[2024-12-04 08:29:56,152][INFO][__main__]: epoch=5 dt=0.884786 train_loss=0.542037 test_loss=0.382394 test_acc=90.927261
[2024-12-04 08:29:56,937][INFO][__main__]: epoch=6 dt=0.502172 train_loss=0.513188 test_loss=0.364508 test_acc=91.247002
[2024-12-04 08:29:57,840][INFO][__main__]: epoch=7 dt=0.541722 train_loss=0.494552 test_loss=0.353867 test_acc=91.456833
[2024-12-04 08:29:58,702][INFO][__main__]: epoch=8 dt=0.513719 train_loss=0.477829 test_loss=0.344874 test_acc=91.566750
[2024-12-04 08:29:59,632][INFO][__main__]: epoch=9 dt=0.511596 train_loss=0.473746 test_loss=0.340128 test_acc=91.676659
[2024-12-04 08:30:00,589][INFO][__main__]: epoch=10 dt=0.612377 train_loss=0.467006 test_loss=0.336481 test_acc=91.696640
[2024-12-04 08:30:00,589][INFO][__main__]: 11 epochs took 22.0s
[2024-12-04 08:30:01,133][INFO][ezpz.history]: Saving epoch plot to: /lus/gila/projects/Aurora_deployment/foremans/projects/saforem2/mmm/plots
[2024-12-04 08:30:01,370][INFO][ezpz.history]: Saving dt plot to: /lus/gila/projects/Aurora_deployment/foremans/projects/saforem2/mmm/plots
[2024-12-04 08:30:01,582][INFO][ezpz.history]: Saving train_loss plot to: /lus/gila/projects/Aurora_deployment/foremans/projects/saforem2/mmm/plots
[2024-12-04 08:30:01,786][INFO][ezpz.history]: Saving test_loss plot to: /lus/gila/projects/Aurora_deployment/foremans/projects/saforem2/mmm/plots
[2024-12-04 08:30:01,982][INFO][ezpz.history]: Saving test_acc plot to: /lus/gila/projects/Aurora_deployment/foremans/projects/saforem2/mmm/plots
[2024-12-04 08:30:02,214][INFO][ezpz.plot]: Appending plot to: plots/tplots/dt.txt
[2024-12-04 08:30:02,219][INFO][ezpz.plot]: Appending plot to: plots/tplots/train_loss.txt
[2024-12-04 08:30:02,225][INFO][ezpz.plot]: Appending plot to: plots/tplots/test_loss.txt

                            dt [2024-12-04-083002]
    ┌─────────────────────────────────────────────────────────────────────┐
12.9┤▌                                                                    │
    │▚                                                                    │
    │▝▖                                                                   │
10.9┤ ▚                                                                   │
    │ ▐                                                                   │
    │  ▌                                                                  │
 8.8┤  ▐                                                                  │
    │   ▌                                                                 │
 6.7┤   ▚                                                                 │
    │   ▝▖                                                                │
    │    ▚                                                                │
 4.6┤    ▐                                                                │
    │     ▌                                                               │
    │     ▐                                                               │
 2.6┤      ▌                                                              │
    │      ▚                                                              │
    │      ▝▖                                                             │
 0.5┤       ▚▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▀▀▀▀▀▀▀▀▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄│
    └┬───────┬──────┬───────┬──────┬───────┬──────┬───────┬──────┬───────┬┘
     1       2      3       4      5       6      7       8      9      10
dt                                   epoch
text saved in /lus/gila/projects/Aurora_deployment/foremans/projects/saforem2/mmm/plots/tplots/dt.txt

                        train_loss [2024-12-04-083002]
    ┌─────────────────────────────────────────────────────────────────────┐
1.80┤▌                                                                    │
    │▝▖                                                                   │
    │ ▚                                                                   │
1.58┤  ▌                                                                  │
    │  ▝▖                                                                 │
    │   ▚                                                                 │
1.36┤    ▌                                                                │
    │    ▝▖                                                               │
1.13┤     ▚                                                               │
    │      ▌                                                              │
    │      ▝▖                                                             │
0.91┤       ▝▄                                                            │
    │         ▀▄                                                          │
    │           ▀▄                                                        │
0.69┤             ▀▄▖                                                     │
    │               ▝▀▀▄▄▖                                                │
    │                    ▝▀▀▚▄▄▄▄▄▄▄                                      │
0.47┤                               ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄│
    └┬───────┬──────┬───────┬──────┬───────┬──────┬───────┬──────┬───────┬┘
     1       2      3       4      5       6      7       8      9      10
train_loss                           epoch
text saved in /lus/gila/projects/Aurora_deployment/foremans/projects/saforem2/mmm/plots/tplots/train_loss.txt

                         test_loss [2024-12-04-083002]
    ┌─────────────────────────────────────────────────────────────────────┐
1.09┤▌                                                                    │
    │▝▖                                                                   │
    │ ▚                                                                   │
0.97┤  ▌                                                                  │
    │  ▝▖                                                                 │
    │   ▚                                                                 │
0.84┤    ▌                                                                │
    │    ▝▖                                                               │
0.71┤     ▚                                                               │
    │      ▌                                                              │
    │      ▝▖                                                             │
0.59┤       ▝▄                                                            │
    │         ▀▄                                                          │
    │           ▀▄                                                        │
0.46┤             ▀▄▖                                                     │
    │               ▝▀▀▀▚▄▄▄▖                                             │
    │                       ▝▀▀▀▄▄▄▄                                      │
0.34┤                               ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄│
    └┬───────┬──────┬───────┬──────┬───────┬──────┬───────┬──────┬───────┬┘
     1       2      3       4      5       6      7       8      9      10
test_loss                            epoch
text saved in /lus/gila/projects/Aurora_deployment/foremans/projects/saforem2/mmm/plots/tplots/test_loss.txt
[2024-12-04 08:30:02,237][INFO][__main__]: dataset=<xarray.Dataset> Size: 360B
Dimensions:     (draw: 10)
Coordinates:
  * draw        (draw) int64 80B 0 1 2 3 4 5 6 7 8 9
Data variables:
    epoch       (draw) int64 80B 1 2 3 4 5 6 7 8 9 10
    dt          (draw) float64 80B 12.93 0.5792 0.6741 ... 0.5137 0.5116 0.6124
    train_loss  (draw) float32 40B 1.802 0.9752 0.6899 ... 0.4778 0.4737 0.467
    test_loss   (draw) float32 40B 1.093 0.6074 0.469 ... 0.3449 0.3401 0.3365
    test_acc    (draw) float32 40B 80.73 86.82 89.16 90.26 ... 91.57 91.68 91.7
                         test_acc [2024-12-04-083002]
    ┌─────────────────────────────────────────────────────────────────────┐
91.7┤                                      ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▞▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀│
    │                            ▄▄▞▀▀▀▀▀▀▀                               │
    │                       ▄▄▞▀▀                                         │
89.9┤                  ▄▄▞▀▀                                              │
    │              ▗▀▀▀                                                   │
    │            ▗▞▘                                                      │
88.0┤          ▗▞▘                                                        │
    │        ▗▞▘                                                          │
86.2┤       ▞▘                                                            │
    │      ▞                                                              │
    │     ▐                                                               │
84.4┤    ▗▘                                                               │
    │   ▗▘                                                                │
    │   ▞                                                                 │
82.6┤  ▞                                                                  │
    │ ▐                                                                   │
    │▗▘                                                                   │
80.7┤▌                                                                    │
    └┬───────┬──────┬───────┬──────┬───────┬──────┬───────┬──────┬───────┬┘
     1       2      3       4      5       6      7       8      9      10
test_acc                             epoch
text saved in /lus/gila/projects/Aurora_deployment/foremans/projects/saforem2/mmm/plots/tplots/test_acc.txt
Application 515ccad8 resources: utime=1232s stime=259s maxrss=2950696KB inblock=220230 oublock=640 minflt=14481515 majflt=544754 nvcsw=637540 nivcsw=251406
took: 0h:00m:35s
```

</details>
