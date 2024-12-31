# 🐫 `mmm`: Multi-Model Modeling

## 🐣 Getting Started

### 🏡 Setup Environment

- 🍋 We use [saforem2/`ezpz`](https://github.com/saforem2/ezpz)
  for setting up, launching, and orchestrating our distributed training.

  In particular, we can use the `ezpz_setup_env` helper function from
  [`ezpz/bin/utils.sh`](https://github.com/saforem2/ezpz/blob/main/src/ezpz/bin/utils.sh):

  ```bash
  source <(curl -s https://raw.githubusercontent.com/saforem2/ezpz/refs/heads/main/src/ezpz/bin/utils.sh)
  ezpz_setup_env
  ```

- 🪄 This will, _automagically_:

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

## 🕸️ Example: TP + FSDP on Aurora

We can use a combination of:

1. Tensor Parallelism (TP)
1. Fully Sharded Data Parallelism (FSDP)

```bash
$ CCL_LOG_LEVEL=ERROR launch python3 -Wignore -m mmm.examples.fsdp_tp --tpsize 2
```

<details closed><summary>Output:</summary>

<details closed><summary>Aurora @ ALCF:</summary>

```python
Disabling local launch: multi-node application
Connected to tcp://x4515c2s4b0n0.hostmgmt2515.cm.aurora.alcf.anl.gov:7919
Found executable /flare/Aurora_deployment/foremans/projects/saforem2/mmm/venvs/aurora_nre_models_frameworks-2024.2.1_u1/bin/python3
Launching application 30e8e012-cbab-4003-9a22-4c4ac20dc088
[2024-12-31 15:56:39.103362][INFO][__init__.py:146] - > initializing tensor parallel with size 2
[2024-12-31 15:56:39.106619][INFO][__init__.py:151] - > initializing context parallel with size 1
[2024-12-31 15:56:39.107182][INFO][__init__.py:156] - > initializing pipeline with size 1
[2024-12-31 15:56:39.107680][INFO][__init__.py:159] - > initializing ddp with size 12
2024:12:31-15:56:39:(25048) |CCL_WARN| value of CCL_LOG_LEVEL changed to be error (default:warn)
[2024-12-31 15:56:40.271801][INFO][dist.py:824] - Using device='xpu' with backend='DDP' + 'ccl' for distributed training.
[2024-12-31 15:56:40.272560][INFO][dist.py:846] - [ 0/23]: [tp:0/1][dp: 0/11]
[2024-12-31 15:56:40.271772][INFO][dist.py:846] - [ 6/23]: [tp:0/1][dp: 3/11]
[2024-12-31 15:56:40.271773][INFO][dist.py:846] - [ 7/23]: [tp:1/1][dp: 3/11]
[2024-12-31 15:56:40.271776][INFO][dist.py:846] - [ 8/23]: [tp:0/1][dp: 4/11]
[2024-12-31 15:56:40.271801][INFO][dist.py:846] - [ 9/23]: [tp:1/1][dp: 4/11]
[2024-12-31 15:56:40.271775][INFO][dist.py:846] - [10/23]: [tp:0/1][dp: 5/11]
[2024-12-31 15:56:40.271775][INFO][dist.py:846] - [11/23]: [tp:1/1][dp: 5/11]
[2024-12-31 15:56:40.271825][INFO][dist.py:846] - [ 1/23]: [tp:1/1][dp: 0/11]
[2024-12-31 15:56:40.271806][INFO][dist.py:846] - [ 2/23]: [tp:0/1][dp: 1/11]
[2024-12-31 15:56:40.271816][INFO][dist.py:846] - [ 3/23]: [tp:1/1][dp: 1/11]
[2024-12-31 15:56:40.271805][INFO][dist.py:846] - [ 4/23]: [tp:0/1][dp: 2/11]
[2024-12-31 15:56:40.271807][INFO][dist.py:846] - [ 5/23]: [tp:1/1][dp: 2/11]
[2024-12-31 15:56:40.271706][INFO][dist.py:846] - [19/23]: [tp:1/1][dp: 9/11]
[2024-12-31 15:56:40.271708][INFO][dist.py:846] - [20/23]: [tp:0/1][dp:10/11]
[2024-12-31 15:56:40.271712][INFO][dist.py:846] - [21/23]: [tp:1/1][dp:10/11]
[2024-12-31 15:56:40.271706][INFO][dist.py:846] - [22/23]: [tp:0/1][dp:11/11]
[2024-12-31 15:56:40.271705][INFO][dist.py:846] - [23/23]: [tp:1/1][dp:11/11]
[2024-12-31 15:56:40.271733][INFO][dist.py:846] - [12/23]: [tp:0/1][dp: 6/11]
[2024-12-31 15:56:40.271731][INFO][dist.py:846] - [13/23]: [tp:1/1][dp: 6/11]
[2024-12-31 15:56:40.271731][INFO][dist.py:846] - [14/23]: [tp:0/1][dp: 7/11]
[2024-12-31 15:56:40.271735][INFO][dist.py:846] - [15/23]: [tp:1/1][dp: 7/11]
[2024-12-31 15:56:40.271736][INFO][dist.py:846] - [16/23]: [tp:0/1][dp: 8/11]
[2024-12-31 15:56:40.271763][INFO][dist.py:846] - [17/23]: [tp:1/1][dp: 8/11]
[2024-12-31 15:56:40.271704][INFO][dist.py:846] - [18/23]: [tp:0/1][dp: 9/11]
[2024-12-31 15:56:40.438371][INFO][fsdp_tp.py:151] - Device mesh created:
device_mesh=DeviceMesh([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13], [14, 15], [16, 17], [18, 19], [20, 21], [22, 23]], mesh_dim_names=('dp', 'tp'))
[2024-12-31 15:56:40.825747][INFO][fsdp_tp.py:131] - Model after parallelization: sharded_model=FullyShardedDataParallel(
  (_fsdp_wrapped_module): Transformer(
    (tok_embeddings): Embedding(32000, 256)
    (layers): ModuleList(
      (0-1): 2 x TransformerBlock(
        (attention): Attention(
          (wq): Linear(in_features=256, out_features=256, bias=False)
          (wk): Linear(in_features=256, out_features=256, bias=False)
          (wv): Linear(in_features=256, out_features=256, bias=False)
          (wo): Linear(in_features=256, out_features=256, bias=False)
        )
        (feed_forward): FeedForward(
          (w1): Linear(in_features=256, out_features=768, bias=False)
          (w2): Linear(in_features=768, out_features=256, bias=False)
          (w3): Linear(in_features=256, out_features=768, bias=False)
        )
        (attention_norm): RMSNorm()
        (ffn_norm): RMSNorm()
      )
    )
    (norm): RMSNorm()
    (output): Linear(in_features=256, out_features=32000, bias=False)
  )
)

[2024-12-31 15:56:40.828674][INFO][fsdp_tp.py:132] - Creating AdamW optimizer with lr=0.003
[2024-12-31 15:56:40.829325][INFO][fsdp_tp.py:161] -
Starting 2D training...
[2024-12-31 15:56:52.118143][INFO][fsdp_tp.py:180] - iter=0, loss=924.0859375
[2024-12-31 15:56:52.174829][INFO][fsdp_tp.py:180] - iter=1, loss=-39734.98828125
[2024-12-31 15:56:52.226686][INFO][fsdp_tp.py:180] - iter=2, loss=-208714.5
[2024-12-31 15:56:52.276847][INFO][fsdp_tp.py:180] - iter=3, loss=-816428.9375
[2024-12-31 15:56:52.327237][INFO][fsdp_tp.py:180] - iter=4, loss=-1490284.375
[2024-12-31 15:56:52.377006][INFO][fsdp_tp.py:180] - iter=5, loss=-1950706.0
[2024-12-31 15:56:52.425511][INFO][fsdp_tp.py:180] - iter=6, loss=-2355900.5
[2024-12-31 15:56:52.475568][INFO][fsdp_tp.py:180] - iter=7, loss=-2753351.0
[2024-12-31 15:56:52.525781][INFO][fsdp_tp.py:180] - iter=8, loss=-3167700.5
[2024-12-31 15:56:52.575289][INFO][fsdp_tp.py:180] - iter=9, loss=-3609225.0
[2024-12-31 15:56:52.575860][INFO][fsdp_tp.py:182] - Finished 2D training
Application 30e8e012 resources: utime=648s stime=150s maxrss=2747408KB inblock=251200 oublock=1536 minflt=7129626 majflt=8632 nvcsw=198555 nivcsw=345273
took: 0h:00m:26s
```

## 🖼️ Example: ViT

We can now `launch` the example in
[`src/mmm/trainer/vit.py`](/src/mmm/trainer/vit.py):

```bash
launch python3 -m mmm.trainer.vit
```

<details closed><summary>Output:</summary>

<details closed><summary>Aurora @ ALCF:</summary>

```python
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

<details closed><summary>Polaris @ ALCF:</summary>

Command:

```bash
(👻 2024-04-29)
#[10:13:32 AM][x3005c0s7b1n0][/e/a/f/p/s/mmm][🌱 main][⏱️ 1m2s]
$ launch python3 -m mmm.trainer.vit --max_iters 50
```

Output:

```python
Connected to tcp://x3005c0s7b0n0.hsn.cm.polaris.alcf.anl.gov:7919
Found executable /lus/eagle/projects/argonne_tpc/foremans/projects/saforem2/mmm/venvs/2024-04-29/bin/python3
Launching application 018d194b-df5f-4006-b3c0-b7abaa38fcfe
Using PMI port 47475,47476
[2024-12-28 10:14:00.608057][INFO][dist.py:348] - [device='cuda'][rank=1/7][local_rank=1/3][node=1/1]
[2024-12-28 10:14:00.610074][INFO][dist.py:348] - [device='cuda'][rank=3/7][local_rank=3/3][node=1/1]
[2024-12-28 10:14:00.610515][INFO][dist.py:348] - [device='cuda'][rank=2/7][local_rank=2/3][node=0/1]
[2024-12-28 10:14:00.953769][INFO][dist.py:92] -

[dist_info]:
  • DEVICE=cuda
  • DEVICE_ID=cuda:0
  • DISTRIBUTED_BACKEND=nccl
  • GPUS_PER_NODE=4
  • HOSTS=['x3005c0s7b0n0.hsn.cm.polaris.alcf.anl.gov', 'x3005c0s7b1n0.hsn.cm.polaris.alcf.anl.gov']
  • HOSTFILE=/var/spool/pbs/aux/3121900.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov
  • HOSTNAME=x3005c0s7b0n0.hsn.cm.polaris.alcf.anl.gov
  • LOCAL_RANK=0
  • MACHINE=Polaris
  • NUM_NODES=2
  • NGPUS=8
  • NGPUS_AVAILABLE=8
  • NODE_ID=0
  • RANK=0
  • SCHEDULER=PBS
  • WORLD_SIZE_TOTAL=8
  • WORLD_SIZE_IN_USE=8
  • LAUNCH_CMD=mpiexec --verbose --envall -n 8 -ppn 4 --hostfile /var/spool/pbs/aux/3121900.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov --cpu-bind depth -d 16


[2024-12-28 10:14:00.957594][INFO][dist.py:728] - [0/8] Using device='cuda' with backend='DDP' + 'nccl' for distributed training.
[2024-12-28 10:14:00.957545][INFO][dist.py:348] - [device='cuda'][rank=4/7][local_rank=0/3][node=0/1]
[2024-12-28 10:14:00.962118][INFO][dist.py:348] - [device='cuda'][rank=0/7][local_rank=0/3][node=0/1]
[2024-12-28 10:14:00.962640][WARNING][dist.py:352] - Using [8 / 8] available "cuda" devices !!
[2024-12-28 10:14:01.503274][INFO][dist.py:348] - [device='cuda'][rank=6/7][local_rank=2/3][node=0/1]
[2024-12-28 10:14:01.504172][INFO][dist.py:348] - [device='cuda'][rank=7/7][local_rank=3/3][node=1/1]
[2024-12-28 10:14:01.504931][INFO][dist.py:348] - [device='cuda'][rank=5/7][local_rank=1/3][node=1/1]
[2024-12-28 10:14:02.320322][INFO][vit.py:243] - Using native for SDPA backend
[2024-12-28 10:14:02.320742][INFO][vit.py:269] - Using AttentionBlock Attention with args.compile=False
[2024-12-28 10:14:02.321158][INFO][vit.py:85] - config=ViTConfig(img_size=224, batch_size=128, num_heads=16, head_dim=64, depth=24, patch_size=16)
[2024-12-28 10:14:02.321731][INFO][vit.py:87] - len(train_set)=1000000
[2024-12-28 10:14:06.712686][INFO][vit.py:115] -
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
Total mult-adds (Units.GIGABYTES): 58.57
==========================================================================================
Input size (MB): 77.07
Forward/backward pass size (MB): 54465.11
Params size (MB): 1216.20
Estimated Total Size (MB): 55758.38
==========================================================================================
[2024-12-28 10:14:06.730727][INFO][vit.py:170] - Training with 8 x cuda (s), using torch_dtype=torch.bfloat16
[2024-12-28 10:14:07.609732][INFO][vit.py:197] - iter=0 loss=7.100830 dt=0.729478 dtf=0.282426 dtb=0.446978
[2024-12-28 10:14:08.318874][INFO][vit.py:197] - iter=1 loss=7.463623 dt=0.219952 dtf=0.017852 dtb=0.202019
[2024-12-28 10:14:08.932269][INFO][vit.py:197] - iter=2 loss=7.246826 dt=0.222642 dtf=0.018405 dtb=0.203614
[2024-12-28 10:14:09.558809][INFO][vit.py:197] - iter=3 loss=6.971924 dt=0.235078 dtf=0.018312 dtb=0.216144
[2024-12-28 10:14:10.189978][INFO][vit.py:197] - iter=4 loss=7.366455 dt=0.237982 dtf=0.018586 dtb=0.218771
[2024-12-28 10:14:10.800485][INFO][vit.py:197] - iter=5 loss=7.468506 dt=0.220255 dtf=0.018452 dtb=0.201179
[2024-12-28 10:14:11.412288][INFO][vit.py:197] - iter=6 loss=7.635498 dt=0.221028 dtf=0.018446 dtb=0.201948
[2024-12-28 10:14:12.023196][INFO][vit.py:197] - iter=7 loss=8.348389 dt=0.219977 dtf=0.018441 dtb=0.200913
[2024-12-28 10:14:12.632000][INFO][vit.py:197] - iter=8 loss=10.300293 dt=0.218624 dtf=0.018380 dtb=0.199598
[2024-12-28 10:14:13.240470][INFO][vit.py:197] - iter=9 loss=8.890137 dt=0.217283 dtf=0.018368 dtb=0.198277
[2024-12-28 10:14:13.850009][INFO][vit.py:197] - iter=10 loss=7.557617 dt=0.218973 dtf=0.018442 dtb=0.199910
[2024-12-28 10:14:14.460874][INFO][vit.py:197] - iter=11 loss=7.795654 dt=0.219587 dtf=0.018369 dtb=0.200558
[2024-12-28 10:14:15.071426][INFO][vit.py:197] - iter=12 loss=7.594971 dt=0.219704 dtf=0.018500 dtb=0.200562
[2024-12-28 10:14:15.683757][INFO][vit.py:197] - iter=13 loss=6.972900 dt=0.222515 dtf=0.018471 dtb=0.203425
[2024-12-28 10:14:16.297867][INFO][vit.py:197] - iter=14 loss=6.898682 dt=0.222503 dtf=0.018515 dtb=0.203340
[2024-12-28 10:14:16.911537][INFO][vit.py:197] - iter=15 loss=7.268066 dt=0.223067 dtf=0.018372 dtb=0.204067
[2024-12-28 10:14:17.521205][INFO][vit.py:197] - iter=16 loss=7.569092 dt=0.219474 dtf=0.018513 dtb=0.200341
[2024-12-28 10:14:18.131493][INFO][vit.py:197] - iter=17 loss=7.310059 dt=0.219029 dtf=0.018444 dtb=0.199968
[2024-12-28 10:14:18.740917][INFO][vit.py:197] - iter=18 loss=7.435547 dt=0.219226 dtf=0.018433 dtb=0.200174
[2024-12-28 10:14:19.349284][INFO][vit.py:197] - iter=19 loss=7.121094 dt=0.218426 dtf=0.018344 dtb=0.199448
[2024-12-28 10:14:19.958744][INFO][vit.py:197] - iter=20 loss=7.122314 dt=0.219093 dtf=0.018461 dtb=0.199998
[2024-12-28 10:14:20.571327][INFO][vit.py:197] - iter=21 loss=7.341064 dt=0.222530 dtf=0.018423 dtb=0.203487
[2024-12-28 10:14:21.181196][INFO][vit.py:197] - iter=22 loss=7.297607 dt=0.219753 dtf=0.018444 dtb=0.200691
[2024-12-28 10:14:21.789146][INFO][vit.py:197] - iter=23 loss=6.865479 dt=0.218066 dtf=0.018332 dtb=0.199104
[2024-12-28 10:14:22.397758][INFO][vit.py:197] - iter=24 loss=6.850830 dt=0.218093 dtf=0.018525 dtb=0.198946
[2024-12-28 10:14:23.007309][INFO][vit.py:197] - iter=25 loss=7.235596 dt=0.219788 dtf=0.018468 dtb=0.200686
[2024-12-28 10:14:23.616575][INFO][vit.py:197] - iter=26 loss=7.146240 dt=0.219198 dtf=0.018442 dtb=0.200127
[2024-12-28 10:14:24.225950][INFO][vit.py:197] - iter=27 loss=7.146729 dt=0.219128 dtf=0.018538 dtb=0.199970
[2024-12-28 10:14:24.835281][INFO][vit.py:197] - iter=28 loss=7.238037 dt=0.218991 dtf=0.018479 dtb=0.199888
[2024-12-28 10:14:25.445283][INFO][vit.py:197] - iter=29 loss=7.225830 dt=0.220475 dtf=0.018417 dtb=0.201428
[2024-12-28 10:14:26.053357][INFO][vit.py:197] - iter=30 loss=7.141113 dt=0.217549 dtf=0.018551 dtb=0.198373
[2024-12-28 10:14:26.663588][INFO][vit.py:197] - iter=31 loss=7.093262 dt=0.219957 dtf=0.018410 dtb=0.200912
[2024-12-28 10:14:27.272752][INFO][vit.py:197] - iter=32 loss=7.089844 dt=0.219036 dtf=0.018496 dtb=0.199916
[2024-12-28 10:14:27.881781][INFO][vit.py:197] - iter=33 loss=6.967773 dt=0.218686 dtf=0.018411 dtb=0.199638
[2024-12-28 10:14:28.491795][INFO][vit.py:197] - iter=34 loss=7.002686 dt=0.219045 dtf=0.018489 dtb=0.199922
[2024-12-28 10:14:29.101144][INFO][vit.py:197] - iter=35 loss=7.168701 dt=0.218585 dtf=0.018428 dtb=0.199525
[2024-12-28 10:14:29.710525][INFO][vit.py:197] - iter=36 loss=7.083252 dt=0.219552 dtf=0.018508 dtb=0.200425
[2024-12-28 10:14:30.320813][INFO][vit.py:197] - iter=37 loss=7.026855 dt=0.219426 dtf=0.018411 dtb=0.200394
[2024-12-28 10:14:30.931049][INFO][vit.py:197] - iter=38 loss=7.024170 dt=0.220127 dtf=0.018518 dtb=0.200974
[2024-12-28 10:14:31.540885][INFO][vit.py:197] - iter=39 loss=7.295166 dt=0.219916 dtf=0.018411 dtb=0.200893
[2024-12-28 10:14:32.156859][INFO][vit.py:197] - iter=40 loss=7.184814 dt=0.225526 dtf=0.018485 dtb=0.206416
[2024-12-28 10:14:32.764830][INFO][vit.py:197] - iter=41 loss=7.108154 dt=0.218227 dtf=0.018425 dtb=0.199184
[2024-12-28 10:14:33.372941][INFO][vit.py:197] - iter=42 loss=7.039551 dt=0.218725 dtf=0.018496 dtb=0.199575
[2024-12-28 10:14:33.980313][INFO][vit.py:197] - iter=43 loss=7.017578 dt=0.217501 dtf=0.018366 dtb=0.198506
[2024-12-28 10:14:34.588849][INFO][vit.py:197] - iter=44 loss=6.994873 dt=0.218611 dtf=0.018420 dtb=0.199561
[2024-12-28 10:14:35.199744][INFO][vit.py:197] - iter=45 loss=7.003662 dt=0.220470 dtf=0.018433 dtb=0.201405
[2024-12-28 10:14:35.808761][INFO][vit.py:197] - iter=46 loss=7.071045 dt=0.218984 dtf=0.018433 dtb=0.199921
[2024-12-28 10:14:36.417607][INFO][vit.py:197] - iter=47 loss=7.121338 dt=0.218782 dtf=0.018587 dtb=0.199551
[2024-12-28 10:14:37.030363][INFO][vit.py:197] - iter=48 loss=7.159424 dt=0.222677 dtf=0.018649 dtb=0.203399
[2024-12-28 10:14:37.644456][INFO][vit.py:197] - iter=49 loss=7.189697 dt=0.223807 dtf=0.018569 dtb=0.204615
[2024-12-28 10:14:38.252624][INFO][vit.py:197] - iter=50 loss=7.047852 dt=0.219155 dtf=0.018644 dtb=0.199880
Failed to download font: IBM Plex Sans, skipping!
Failed to download font: IBM Plex Sans Condensed, skipping!
Failed to download font: IBM Plex Serif, skipping!
[2024-12-28 10:14:39.752134][INFO][history.py:709] - Saving train_iter plot to: /lus/eagle/projects/argonne_tpc/foremans/projects/saforem2/mmm/outputs/plots/vit/mplot
[2024-12-28 10:14:40.027863][INFO][history.py:709] - Saving train_loss plot to: /lus/eagle/projects/argonne_tpc/foremans/projects/saforem2/mmm/outputs/plots/vit/mplot
[2024-12-28 10:14:40.283827][INFO][history.py:709] - Saving train_dt plot to: /lus/eagle/projects/argonne_tpc/foremans/projects/saforem2/mmm/outputs/plots/vit/mplot
[2024-12-28 10:14:40.545507][INFO][history.py:709] - Saving train_dtf plot to: /lus/eagle/projects/argonne_tpc/foremans/projects/saforem2/mmm/outputs/plots/vit/mplot
[2024-12-28 10:14:40.805544][INFO][history.py:709] - Saving train_dtb plot to: /lus/eagle/projects/argonne_tpc/foremans/projects/saforem2/mmm/outputs/plots/vit/mplot
                        train_iter [2024-12-28-101442]
    ┌─────────────────────────────────────────────────────────────────────┐
50.0┤                                                                  ▄▄▞│
    │                                                             ▗▄▄▞▀   │
    │                                                           ▄▞▘       │
41.7┤                                                       ▄▀▀▀          │
    │                                                  ▗▄▀▀▀              │
    │                                              ▗▄▀▀▘                  │
33.3┤                                          ▗▄▄▀▘                      │
    │                                      ▗▄▄▀▘                          │
25.0┤                                  ▗▄▄▀▘                              │
    │                               ▗▄▀▘                                  │
    │                           ▗▄▀▀▘                                     │
16.7┤                       ▗▞▀▀▘                                         │
    │                   ▄▞▀▀▘                                             │
    │               ▄▄▄▀                                                  │
 8.3┤           ▄▄▞▀                                                      │
    │       ▄▄▞▀                                                          │
    │    ▄▞▀                                                              │
 0.0┤▄▞▀▀                                                                 │
    └─┬──┬──┬────┬──┬────┬───┬──┬──┬────┬──┬──┬────┬──┬────┬──┬───┬──┬──┬─┘
    0 2  4  6   10 12   16  19 21 23   27 29 31   35 37   41 43  46 48 50
train_iter                        train/iter
[2024-12-28 10:14:42.778051][INFO][plot.py:220] - Appending plot to: /lus/eagle/projects/argonne_tpc/foremans/projects/saforem2/mmm/outputs/plots/vit/tplot/train_iter.txt
text saved in /lus/eagle/projects/argonne_tpc/foremans/projects/saforem2/mmm/outputs/plots/vit/tplot/train_iter.txt
                         train_loss [2024-12-28-101442]
     ┌────────────────────────────────────────────────────────────────────┐
10.30┤          ▗▌                                                        │
     │          ▐▌                                                        │
     │          ▐▌                                                        │
 9.73┤          ▞▚                                                        │
     │          ▌▐                                                        │
     │          ▌▐                                                        │
 9.15┤          ▌▐                                                        │
     │         ▐  ▌                                                       │
 8.58┤         ▐  ▌                                                       │
     │         ▐  ▚                                                       │
     │         ▞  ▐                                                       │
 8.00┤        ▗▘  ▐                                                       │
     │        ▞    ▌ ▖                                                    │
     │        ▌    ▌▞▝▖                                                   │
 7.43┤ ▗     ▞     ▝  ▌    ▞▖ ▗                                           │
     │▗▘▚  ▞▀         ▐   ▐ ▝▀▀▖  ▞▜    ▖   ▄▖            ▞▖              │
     │▘  ▚▞            ▌ ▗▘    ▝▀▀  ▌  ▞▝▀▀▀ ▝▀▄▄▖  ▄▞▄▄▄▟ ▝▀▀▄▄▖  ▄▄▞▀▀▚▄│
 6.85┤    ▘            ▝▄▌          ▝▄▟          ▝▀▀            ▝▀▀       │
     └─┬──┬──┬─┬──┬──┬────┬───┬────┬──┬──┬────┬──┬────┬──┬──┬─────┬──┬──┬─┘
     0 2  4  6 8 10 12   16  19   23 25 27   31 33   37 39 41    46 48 50
train_loss                         train/iter
[2024-12-28 10:14:42.785192][INFO][plot.py:220] - Appending plot to: /lus/eagle/projects/argonne_tpc/foremans/projects/saforem2/mmm/outputs/plots/vit/tplot/train_loss.txt
text saved in /lus/eagle/projects/argonne_tpc/foremans/projects/saforem2/mmm/outputs/plots/vit/tplot/train_loss.txt
                          train_dt [2024-12-28-101442]
     ┌────────────────────────────────────────────────────────────────────┐
0.729┤▌                                                                   │
     │▌                                                                   │
     │▌                                                                   │
0.644┤▌                                                                   │
     │▌                                                                   │
     │▌                                                                   │
0.559┤▐                                                                   │
     │▐                                                                   │
0.473┤▐                                                                   │
     │▐                                                                   │
     │▐                                                                   │
0.388┤▐                                                                   │
     │ ▌                                                                  │
     │ ▌                                                                  │
0.303┤ ▌                                                                  │
     │ ▌                                                                  │
     │ ▌                                                                  │
0.217┤ ▚▄▄▀▀▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▚▄▄▄▄▄▄▄▄▄▄▄▄▄│
     └─┬──┬──┬─┬──┬──┬────┬───┬────┬──┬──┬────┬──┬────┬──┬──┬─────┬──┬──┬─┘
     0 2  4  6 8 10 12   16  19   23 25 27   31 33   37 39 41    46 48 50
train_dt                           train/iter
[2024-12-28 10:14:42.791361][INFO][plot.py:220] - Appending plot to: /lus/eagle/projects/argonne_tpc/foremans/projects/saforem2/mmm/outputs/plots/vit/tplot/train_dt.txt
text saved in /lus/eagle/projects/argonne_tpc/foremans/projects/saforem2/mmm/outputs/plots/vit/tplot/train_dt.txt
                          train_dtf [2024-12-28-101442]
     ┌────────────────────────────────────────────────────────────────────┐
0.282┤▌                                                                   │
     │▌                                                                   │
     │▌                                                                   │
0.238┤▌                                                                   │
     │▌                                                                   │
     │▌                                                                   │
0.194┤▐                                                                   │
     │▐                                                                   │
0.150┤▐                                                                   │
     │▐                                                                   │
     │▐                                                                   │
0.106┤▐                                                                   │
     │ ▌                                                                  │
     │ ▌                                                                  │
0.062┤ ▌                                                                  │
     │ ▌                                                                  │
     │ ▌                                                                  │
0.018┤ ▚▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄│
     └─┬──┬──┬─┬──┬──┬────┬───┬────┬──┬──┬────┬──┬────┬──┬──┬─────┬──┬──┬─┘
     0 2  4  6 8 10 12   16  19   23 25 27   31 33   37 39 41    46 48 50
train_dtf                          train/iter
[2024-12-28 10:14:42.797558][INFO][plot.py:220] - Appending plot to: /lus/eagle/projects/argonne_tpc/foremans/projects/saforem2/mmm/outputs/plots/vit/tplot/train_dtf.txt
text saved in /lus/eagle/projects/argonne_tpc/foremans/projects/saforem2/mmm/outputs/plots/vit/tplot/train_dtf.txt
                          train_dtb [2024-12-28-101442]
     ┌────────────────────────────────────────────────────────────────────┐
0.447┤▌                                                                   │
     │▌                                                                   │
     │▌                                                                   │
0.406┤▌                                                                   │
     │▌                                                                   │
     │▌                                                                   │
0.364┤▐                                                                   │
     │▐                                                                   │
0.323┤▐                                                                   │
     │▐                                                                   │
     │▐                                                                   │
0.281┤▝▖                                                                  │
     │ ▌                                                                  │
     │ ▌                                                                  │
0.240┤ ▌                                                                  │
     │ ▌                                                                  │
     │ ▌ ▗▀▀▖                                                             │
0.198┤ ▝▀▘  ▝▄▚▄▄▄▄▄▄▄▄▞▀▀▀▄▄▄▄▄▄▄▞▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▚▄▄▄▄▄▄▄▄▄▄▀▚▄│
     └─┬──┬──┬─┬──┬──┬────┬───┬────┬──┬──┬────┬──┬────┬──┬──┬─────┬──┬──┬─┘
     0 2  4  6 8 10 12   16  19   23 25 27   31 33   37 39 41    46 48 50
train_dtb                          train/iter
[2024-12-28 10:14:42.803708][INFO][plot.py:220] - Appending plot to: /lus/eagle/projects/argonne_tpc/foremans/projects/saforem2/mmm/outputs/plots/vit/tplot/train_dtb.txt
text saved in /lus/eagle/projects/argonne_tpc/foremans/projects/saforem2/mmm/outputs/plots/vit/tplot/train_dtb.txt
[2024-12-28 10:14:42.808034][INFO][vit.py:217] - dataset=<xarray.Dataset> Size: 2kB
Dimensions:     (draw: 51)
Coordinates:
  * draw        (draw) int64 408B 0 1 2 3 4 5 6 7 8 ... 43 44 45 46 47 48 49 50
Data variables:
    train_iter  (draw) int64 408B 0 1 2 3 4 5 6 7 8 ... 43 44 45 46 47 48 49 50
    train_loss  (draw) float64 408B 7.101 7.464 7.247 6.972 ... 7.159 7.19 7.048
    train_dt    (draw) float64 408B 0.7295 0.22 0.2226 ... 0.2227 0.2238 0.2192
    train_dtf   (draw) float64 408B 0.2824 0.01785 0.01841 ... 0.01857 0.01864
    train_dtb   (draw) float64 408B 0.447 0.202 0.2036 ... 0.2034 0.2046 0.1999
[2024-12-28 10:14:42.810548][INFO][vit.py:276] - Took 40.49 seconds
Application 018d194b resources: utime=387s stime=169s maxrss=2508996KB inblock=635062 oublock=944 minflt=2610764 majflt=5176 nvcsw=390711 nivcsw=3806
Time: 0h:00m:57s
```

</details>

</details>


## 📝 Example: FSDP

```bash
launch python3 -m mmm.trainer.fsdp
```

<details closed><summary>Output:</summary>

<details closed><summary>Aurora @ ALCF:</summary>

Command:

```bash
#[🐍 aurora_nre_models_frameworks-2024.2.1_u1](👻 aurora_nre_models_frameworks-2024.2.1_u1)
#[🤖][02:27:58 PM][foremans@x4211c7s0b0n0][…/mmm/src/mmm][🌱 main][!]
$ CCL_LOG_LEVEL=ERROR launch python3 -Wignore -m mmm.trainer.fsdp
```

Output:

```python
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

<details closed><summary>Polaris @ ALCF:</summary>

Command:

```bash
# (👻 2024-04-29)
#[10:24:19 AM][x3005c0s7b1n0][/e/a/f/p/s/mmm][🌱 main][?][⏱️ 21s]
$ launch python3 -m mmm.trainer.fsdp --epochs 20
```

Output:

```python
Connected to tcp://x3005c0s7b0n0.hsn.cm.polaris.alcf.anl.gov:7919
Found executable /lus/eagle/projects/argonne_tpc/foremans/projects/saforem2/mmm/venvs/2024-04-29/bin/python3
Launching application 79292806-9c62-47ad-a497-98a67d6ada50
Using PMI port 57431,57432
[2024-12-28 10:24:33.839266][INFO][dist.py:348] - [device='cuda'][rank=4/7][local_rank=0/3][node=0/1]
[2024-12-28 10:24:33.842556][INFO][dist.py:92] -

[dist_info]:
  • DEVICE=cuda
  • DEVICE_ID=cuda:0
  • DISTRIBUTED_BACKEND=nccl
  • GPUS_PER_NODE=4
  • HOSTS=['x3005c0s7b0n0.hsn.cm.polaris.alcf.anl.gov', 'x3005c0s7b1n0.hsn.cm.polaris.alcf.anl.gov']
  • HOSTFILE=/var/spool/pbs/aux/3121900.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov
  • HOSTNAME=x3005c0s7b0n0.hsn.cm.polaris.alcf.anl.gov
  • LOCAL_RANK=0
  • MACHINE=Polaris
  • NUM_NODES=2
  • NGPUS=8
  • NGPUS_AVAILABLE=8
  • NODE_ID=0
  • RANK=0
  • SCHEDULER=PBS
  • WORLD_SIZE_TOTAL=8
  • WORLD_SIZE_IN_USE=8
  • LAUNCH_CMD=mpiexec --verbose --envall -n 8 -ppn 4 --hostfile /var/spool/pbs/aux/3121900.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov --cpu-bind depth -d 16


[2024-12-28 10:24:33.846506][INFO][dist.py:728] - [0/8] Using device='cuda' with backend='DDP' + 'nccl' for distributed training.
[2024-12-28 10:24:33.851271][INFO][dist.py:348] - [device='cuda'][rank=0/7][local_rank=0/3][node=0/1]
[2024-12-28 10:24:33.851822][WARNING][dist.py:352] - Using [8 / 8] available "cuda" devices !!
[2024-12-28 10:24:34.375094][INFO][dist.py:348] - [device='cuda'][rank=1/7][local_rank=1/3][node=1/1]
[2024-12-28 10:24:34.381428][INFO][dist.py:348] - [device='cuda'][rank=2/7][local_rank=2/3][node=0/1]
[2024-12-28 10:24:34.382345][INFO][dist.py:348] - [device='cuda'][rank=3/7][local_rank=3/3][node=1/1]
[2024-12-28 10:24:34.385237][INFO][dist.py:348] - [device='cuda'][rank=7/7][local_rank=3/3][node=1/1]
[2024-12-28 10:24:34.385717][INFO][dist.py:348] - [device='cuda'][rank=6/7][local_rank=2/3][node=0/1]
[2024-12-28 10:24:34.386163][INFO][dist.py:348] - [device='cuda'][rank=5/7][local_rank=1/3][node=1/1]
[2024-12-28 10:24:35.265916][INFO][fsdp.py:185] - model=FullyShardedDataParallel(
  (_fsdp_wrapped_module): Net(
    (conv1): Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1))
    (conv2): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1))
    (dropout1): Dropout(p=0.25, inplace=False)
    (dropout2): Dropout(p=0.5, inplace=False)
    (fc1): Linear(in_features=9216, out_features=128, bias=True)
    (fc2): Linear(in_features=128, out_features=10, bias=True)
  )
)
/soft/applications/conda/2024-04-29/mconda3/lib/python3.11/multiprocessing/popen_fork.py:66: RuntimeWarning: os.fork() was called. os.fork() is incompatible with multithreaded code, and JAX is multithreaded, so this will likely lead to a deadlock.
  self.pid = os.fork()
/soft/applications/conda/2024-04-29/mconda3/lib/python3.11/multiprocessing/popen_fork.py:66: RuntimeWarning: os.fork() was called. os.fork() is incompatible with multithreaded code, and JAX is multithreaded, so this will likely lead to a deadlock.
  self.pid = os.fork()
/soft/applications/conda/2024-04-29/mconda3/lib/python3.11/multiprocessing/popen_fork.py:66: RuntimeWarning: os.fork() was called. os.fork() is incompatible with multithreaded code, and JAX is multithreaded, so this will likely lead to a deadlock.
  self.pid = os.fork()
/soft/applications/conda/2024-04-29/mconda3/lib/python3.11/multiprocessing/popen_fork.py:66: RuntimeWarning: os.fork() was called. os.fork() is incompatible with multithreaded code, and JAX is multithreaded, so this will likely lead to a deadlock.
  self.pid = os.fork()
/soft/applications/conda/2024-04-29/mconda3/lib/python3.11/multiprocessing/popen_fork.py:66: RuntimeWarning: os.fork() was called. os.fork() is incompatible with multithreaded code, and JAX is multithreaded, so this will likely lead to a deadlock.
  self.pid = os.fork()
/soft/applications/conda/2024-04-29/mconda3/lib/python3.11/multiprocessing/popen_fork.py:66: RuntimeWarning: os.fork() was called. os.fork() is incompatible with multithreaded code, and JAX is multithreaded, so this will likely lead to a deadlock.
  self.pid = os.fork()
/soft/applications/conda/2024-04-29/mconda3/lib/python3.11/multiprocessing/popen_fork.py:66: RuntimeWarning: os.fork() was called. os.fork() is incompatible with multithreaded code, and JAX is multithreaded, so this will likely lead to a deadlock.
  self.pid = os.fork()
/soft/applications/conda/2024-04-29/mconda3/lib/python3.11/multiprocessing/popen_fork.py:66: RuntimeWarning: os.fork() was called. os.fork() is incompatible with multithreaded code, and JAX is multithreaded, so this will likely lead to a deadlock.
  self.pid = os.fork()
[2024-12-28 10:24:37.041326][INFO][fsdp.py:220] - epoch=1 dt=1.537077 train_loss=0.332522 test_loss=0.063956 test_acc=98.010002
[2024-12-28 10:24:38.319908][INFO][fsdp.py:220] - epoch=2 dt=1.039602 train_loss=0.095827 test_loss=0.046772 test_acc=98.510002
[2024-12-28 10:24:39.607701][INFO][fsdp.py:220] - epoch=3 dt=1.046263 train_loss=0.069456 test_loss=0.036497 test_acc=98.709999
[2024-12-28 10:24:40.897047][INFO][fsdp.py:220] - epoch=4 dt=1.050575 train_loss=0.057079 test_loss=0.034091 test_acc=98.739998
[2024-12-28 10:24:42.208014][INFO][fsdp.py:220] - epoch=5 dt=1.069041 train_loss=0.049113 test_loss=0.030856 test_acc=98.860001
[2024-12-28 10:24:43.506457][INFO][fsdp.py:220] - epoch=6 dt=1.058201 train_loss=0.044704 test_loss=0.031061 test_acc=98.870003
[2024-12-28 10:24:44.794764][INFO][fsdp.py:220] - epoch=7 dt=1.046943 train_loss=0.039734 test_loss=0.029716 test_acc=99.019997
[2024-12-28 10:24:46.099857][INFO][fsdp.py:220] - epoch=8 dt=1.063810 train_loss=0.038983 test_loss=0.028920 test_acc=99.000000
[2024-12-28 10:24:47.387096][INFO][fsdp.py:220] - epoch=9 dt=1.046904 train_loss=0.037915 test_loss=0.028552 test_acc=99.080002
[2024-12-28 10:24:48.674634][INFO][fsdp.py:220] - epoch=10 dt=1.047954 train_loss=0.035310 test_loss=0.029600 test_acc=99.029999
[2024-12-28 10:24:49.981004][INFO][fsdp.py:220] - epoch=11 dt=1.066301 train_loss=0.034965 test_loss=0.029157 test_acc=99.050003
[2024-12-28 10:24:51.295468][INFO][fsdp.py:220] - epoch=12 dt=1.073833 train_loss=0.035009 test_loss=0.029088 test_acc=99.040001
[2024-12-28 10:24:52.599066][INFO][fsdp.py:220] - epoch=13 dt=1.059257 train_loss=0.034744 test_loss=0.028920 test_acc=99.019997
[2024-12-28 10:24:53.907407][INFO][fsdp.py:220] - epoch=14 dt=1.063748 train_loss=0.034605 test_loss=0.028827 test_acc=99.040001
[2024-12-28 10:24:55.219236][INFO][fsdp.py:220] - epoch=15 dt=1.070160 train_loss=0.034538 test_loss=0.028670 test_acc=99.029999
[2024-12-28 10:24:56.518929][INFO][fsdp.py:220] - epoch=16 dt=1.057357 train_loss=0.032730 test_loss=0.028812 test_acc=99.059998
[2024-12-28 10:24:57.818063][INFO][fsdp.py:220] - epoch=17 dt=1.058079 train_loss=0.033438 test_loss=0.028770 test_acc=99.059998
[2024-12-28 10:24:59.121623][INFO][fsdp.py:220] - epoch=18 dt=1.062857 train_loss=0.033123 test_loss=0.028738 test_acc=99.059998
[2024-12-28 10:25:00.438261][INFO][fsdp.py:220] - epoch=19 dt=1.073954 train_loss=0.032591 test_loss=0.028736 test_acc=99.059998
[2024-12-28 10:25:01.729533][INFO][fsdp.py:220] - epoch=20 dt=1.049736 train_loss=0.033869 test_loss=0.028739 test_acc=99.050003
[2024-12-28 10:25:01.730550][INFO][fsdp.py:222] - 21 epochs took 26.5s
[2024-12-28 10:25:02.522897][INFO][history.py:709] - Saving epoch plot to: /lus/eagle/projects/argonne_tpc/foremans/projects/saforem2/mmm/plots
[2024-12-28 10:25:02.523536][INFO][history.py:713] - Saving epoch plot to: /lus/eagle/projects/argonne_tpc/foremans/projects/saforem2/mmm/plots/pngs/epoch.png
[2024-12-28 10:25:02.687711][INFO][history.py:713] - Saving epoch plot to: /lus/eagle/projects/argonne_tpc/foremans/projects/saforem2/mmm/plots/svgs/epoch.svg
[2024-12-28 10:25:02.769424][INFO][history.py:709] - Saving dt plot to: /lus/eagle/projects/argonne_tpc/foremans/projects/saforem2/mmm/plots
[2024-12-28 10:25:02.769946][INFO][history.py:713] - Saving dt plot to: /lus/eagle/projects/argonne_tpc/foremans/projects/saforem2/mmm/plots/pngs/dt.png
[2024-12-28 10:25:02.922523][INFO][history.py:713] - Saving dt plot to: /lus/eagle/projects/argonne_tpc/foremans/projects/saforem2/mmm/plots/svgs/dt.svg
[2024-12-28 10:25:03.003017][INFO][history.py:709] - Saving train_loss plot to: /lus/eagle/projects/argonne_tpc/foremans/projects/saforem2/mmm/plots
[2024-12-28 10:25:03.003590][INFO][history.py:713] - Saving train_loss plot to: /lus/eagle/projects/argonne_tpc/foremans/projects/saforem2/mmm/plots/pngs/train_loss.png
[2024-12-28 10:25:03.162992][INFO][history.py:713] - Saving train_loss plot to: /lus/eagle/projects/argonne_tpc/foremans/projects/saforem2/mmm/plots/svgs/train_loss.svg
[2024-12-28 10:25:03.245831][INFO][history.py:709] - Saving test_loss plot to: /lus/eagle/projects/argonne_tpc/foremans/projects/saforem2/mmm/plots
[2024-12-28 10:25:03.246346][INFO][history.py:713] - Saving test_loss plot to: /lus/eagle/projects/argonne_tpc/foremans/projects/saforem2/mmm/plots/pngs/test_loss.png
[2024-12-28 10:25:03.413526][INFO][history.py:713] - Saving test_loss plot to: /lus/eagle/projects/argonne_tpc/foremans/projects/saforem2/mmm/plots/svgs/test_loss.svg
[2024-12-28 10:25:03.500235][INFO][history.py:709] - Saving test_acc plot to: /lus/eagle/projects/argonne_tpc/foremans/projects/saforem2/mmm/plots
[2024-12-28 10:25:03.500763][INFO][history.py:713] - Saving test_acc plot to: /lus/eagle/projects/argonne_tpc/foremans/projects/saforem2/mmm/plots/pngs/test_acc.png
[2024-12-28 10:25:03.660713][INFO][history.py:713] - Saving test_acc plot to: /lus/eagle/projects/argonne_tpc/foremans/projects/saforem2/mmm/plots/svgs/test_acc.svg
                             dt [2024-12-28-102504]
     ┌────────────────────────────────────────────────────────────────────┐
1.537┤▌                                                                   │
     │▌                                                                   │
     │▚                                                                   │
1.454┤▐                                                                   │
     │▐                                                                   │
     │ ▌                                                                  │
1.371┤ ▌                                                                  │
     │ ▚                                                                  │
1.288┤ ▐                                                                  │
     │ ▐                                                                  │
     │  ▌                                                                 │
1.205┤  ▌                                                                 │
     │  ▚                                                                 │
     │  ▐                                                                 │
1.123┤  ▐                                                                 │
     │   ▌                                                                │
     │   ▌          ▖          ▖         ▗▄▄▄▖      ▄▄▄▄          ▗▄▄▄▖   │
1.040┤   ▚▄▄▄▄▄▄▞▀▀▀▝▀▀▀▀▀▀▀▀▀▀▝▀▀▀▀▀▀▀▀▀▘   ▝▀▀▀▀▀▀    ▀▀▀▀▀▀▀▀▀▀▘   ▝▀▀▀│
     └┬───┬──┬───┬──┬───┬──┬───┬──┬───┬──┬───┬──┬───┬──┬───┬──┬───┬──┬───┬┘
      1   2  3   4  5   6  7   8  9  10 11  12 13  14 15  16 17  18 19  20
dt                                    epoch
[2024-12-28 10:25:04.530970][INFO][plot.py:220] - Appending plot to: plots/tplots/dt.txt
text saved in /lus/eagle/projects/argonne_tpc/foremans/projects/saforem2/mmm/plots/tplots/dt.txt
                         train_loss [2024-12-28-102504]
     ┌────────────────────────────────────────────────────────────────────┐
0.333┤▌                                                                   │
     │▌                                                                   │
     │▐                                                                   │
0.283┤▐                                                                   │
     │ ▌                                                                  │
     │ ▌                                                                  │
0.233┤ ▐                                                                  │
     │ ▐                                                                  │
0.183┤  ▌                                                                 │
     │  ▌                                                                 │
     │  ▐                                                                 │
0.133┤  ▐                                                                 │
     │   ▌                                                                │
     │   ▌                                                                │
0.083┤   ▝▄                                                               │
     │     ▀▄▖                                                            │
     │       ▝▀▀▀▄▄▄▖                                                     │
0.033┤              ▝▀▀▀▀▀▀▀▀▀▀▀▀▀▀▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄│
     └┬───┬──┬───┬──┬───┬──┬───┬──┬───┬──┬───┬──┬───┬──┬───┬──┬───┬──┬───┬┘
      1   2  3   4  5   6  7   8  9  10 11  12 13  14 15  16 17  18 19  20
train_loss                            epoch
[2024-12-28 10:25:04.535626][INFO][plot.py:220] - Appending plot to: plots/tplots/train_loss.txt
text saved in /lus/eagle/projects/argonne_tpc/foremans/projects/saforem2/mmm/plots/tplots/train_loss.txt
                          test_loss [2024-12-28-102504]
      ┌───────────────────────────────────────────────────────────────────┐
0.0640┤▌                                                                  │
      │▚                                                                  │
      │▝▖                                                                 │
0.0581┤ ▌                                                                 │
      │ ▐                                                                 │
      │  ▌                                                                │
0.0522┤  ▚                                                                │
      │  ▝▖                                                               │
0.0463┤   ▚                                                               │
      │   ▝▖                                                              │
      │    ▐                                                              │
0.0404┤     ▚                                                             │
      │     ▝▖                                                            │
      │      ▝▖                                                           │
0.0345┤       ▝▚▖                                                         │
      │         ▝▀▄                                                       │
      │            ▀▄▄▄▄▄                                                 │
0.0286┤                  ▀▀▀▚▄▄▄▄▄▄▄▄▄▞▀▀▀▀▀▀▀▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄│
      └┬──┬───┬──┬───┬──┬───┬──┬───┬──┬───┬──┬───┬──┬───┬──┬───┬──┬───┬──┬┘
       1  2   3  4   5  6   7  8   9 10  11 12  13 14  15 16  17 18  19 20
test_loss                             epoch
[2024-12-28 10:25:04.540326][INFO][plot.py:220] - Appending plot to: plots/tplots/test_loss.txt
text saved in /lus/eagle/projects/argonne_tpc/foremans/projects/saforem2/mmm/plots/tplots/test_loss.txt
                          test_acc [2024-12-28-102504]
     ┌────────────────────────────────────────────────────────────────────┐
99.08┤                           ▗▞▄▖    ▗▄▄▄▖      ▖      ▗▄▄▄▄▄▄▄▄▄▄▄▄▄▄│
     │                     ▞▄▄▄▄▞▘  ▝▀▀▀▀▘   ▝▀▀▀▀▀▀▝▀▀▀▀▀▀▘              │
     │                   ▗▞                                               │
98.90┤              ▄▄▄▄▄▘                                                │
     │            ▗▞                                                      │
     │          ▗▞▘                                                       │
98.72┤      ▗▀▀▀▘                                                         │
     │     ▗▘                                                             │
98.55┤    ▗▘                                                              │
     │   ▗▘                                                               │
     │   ▌                                                                │
98.37┤  ▐                                                                 │
     │  ▌                                                                 │
     │ ▐                                                                  │
98.19┤ ▞                                                                  │
     │▗▘                                                                  │
     │▞                                                                   │
98.01┤▌                                                                   │
     └┬───┬──┬───┬──┬───┬──┬───┬──┬───┬──┬───┬──┬───┬──┬───┬──┬───┬──┬───┬┘
      1   2  3   4  5   6  7   8  9  10 11  12 13  14 15  16 17  18 19  20
test_acc                              epoch
[2024-12-28 10:25:04.544967][INFO][plot.py:220] - Appending plot to: plots/tplots/test_acc.txt
text saved in /lus/eagle/projects/argonne_tpc/foremans/projects/saforem2/mmm/plots/tplots/test_acc.txt
[2024-12-28 10:25:04.548079][INFO][fsdp.py:246] - dataset=<xarray.Dataset> Size: 720B
Dimensions:     (draw: 20)
Coordinates:
  * draw        (draw) int64 160B 0 1 2 3 4 5 6 7 8 ... 12 13 14 15 16 17 18 19
Data variables:
    epoch       (draw) int64 160B 1 2 3 4 5 6 7 8 9 ... 13 14 15 16 17 18 19 20
    dt          (draw) float64 160B 1.537 1.04 1.046 1.051 ... 1.063 1.074 1.05
    train_loss  (draw) float32 80B 0.3325 0.09583 0.06946 ... 0.03259 0.03387
    test_loss   (draw) float32 80B 0.06396 0.04677 0.0365 ... 0.02874 0.02874
    test_acc    (draw) float32 80B 98.01 98.51 98.71 98.74 ... 99.06 99.06 99.05
Application 79292806 resources: utime=398s stime=124s maxrss=1856804KB inblock=15464 oublock=720 minflt=4500156 majflt=1053998 nvcsw=2168618 nivcsw=6864
Time: 0h:00m:42s
```

</details>

</details>
