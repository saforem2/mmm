# `mmm`: Multi-Model Models

## 🐣 Getting Started


1. Clone repo:
  1. 
  ```bash
  git clone https://github.com/saforem2/mmm
  cd mmm
  ```

2. Setup env:

  ```bash
  export PBS_O_WORKDIR=$(pwd)
  source /dev/stdin <<< $(curl 'https://raw.githubusercontent.com/saforem2/ezpz/refs/heads/main/src/ezpz/bin/utils.sh')
  ezpz_setup_env
  ```
  This will automatically setup python and create a `launch` alias for
  launching applications.  
  For additional information, see
  🍋 [saforem2/`ezpz`](https://github.com/saforem2/ezpz)

3. Install `mmm`:

  ```bash
  python3 -m pip install -e . --require-virtualenv
  ```

### Example: ViT

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
Launching application d67f5af4-4ef0-495e-a1b2-96836b545274
[2024-12-26 13:03:03,538][INFO][ezpz.dist]: [device='xpu'][rank=21/23][local_rank=9/11][node=1/1]
[2024-12-26 13:03:03,538][INFO][ezpz.dist]: [device='xpu'][rank=19/23][local_rank=7/11][node=1/1]
[2024-12-26 13:03:03,539][INFO][ezpz.dist]: [device='xpu'][rank=22/23][local_rank=10/11][node=0/1]
[2024-12-26 13:03:03,542][INFO][ezpz.dist]: [device='xpu'][rank=5/23][local_rank=5/11][node=1/1]
[2024-12-26 13:03:03,543][INFO][ezpz.dist]: [device='xpu'][rank=6/23][local_rank=6/11][node=0/1]
[2024-12-26 13:03:03,545][INFO][ezpz.dist]: [device='xpu'][rank=2/23][local_rank=2/11][node=0/1]
[2024-12-26 13:03:03,546][INFO][ezpz.dist]: [device='xpu'][rank=4/23][local_rank=4/11][node=0/1]
[2024-12-26 13:03:03,547][INFO][ezpz.dist]: [device='xpu'][rank=9/23][local_rank=9/11][node=1/1]
[2024-12-26 13:03:03,547][INFO][ezpz.dist]: [device='xpu'][rank=14/23][local_rank=2/11][node=0/1]
[2024-12-26 13:03:03,548][INFO][ezpz.dist]: [device='xpu'][rank=18/23][local_rank=6/11][node=0/1]
[2024-12-26 13:03:03,548][INFO][ezpz.dist]: [device='xpu'][rank=13/23][local_rank=1/11][node=1/1]
[2024-12-26 13:03:03,549][INFO][ezpz.dist]: [device='xpu'][rank=20/23][local_rank=8/11][node=0/1]
[2024-12-26 13:03:03,551][INFO][ezpz.dist]: [device='xpu'][rank=11/23][local_rank=11/11][node=1/1]
[2024-12-26 13:03:03,553][INFO][ezpz.dist]: [device='xpu'][rank=1/23][local_rank=1/11][node=1/1]
[2024-12-26 13:03:03,553][INFO][ezpz.dist]: [device='xpu'][rank=8/23][local_rank=8/11][node=0/1]
[2024-12-26 13:03:03,554][INFO][ezpz.dist]: [device='xpu'][rank=12/23][local_rank=0/11][node=0/1]
[2024-12-26 13:03:03,555][INFO][ezpz.dist]: [device='xpu'][rank=3/23][local_rank=3/11][node=1/1]
[2024-12-26 13:03:03,556][INFO][ezpz.dist]: [device='xpu'][rank=15/23][local_rank=3/11][node=1/1]
[2024-12-26 13:03:03,602][INFO][ezpz.dist]: [device='xpu'][rank=7/23][local_rank=7/11][node=1/1]
[2024-12-26 13:03:03,602][INFO][ezpz.dist]: [device='xpu'][rank=10/23][local_rank=10/11][node=0/1]
[2024-12-26 13:03:03,603][INFO][ezpz.dist]: [device='xpu'][rank=17/23][local_rank=5/11][node=1/1]
[2024-12-26 13:03:03,603][INFO][ezpz.dist]: [device='xpu'][rank=23/23][local_rank=11/11][node=1/1]
[2024-12-26 13:03:03,604][INFO][ezpz.dist]: [device='xpu'][rank=16/23][local_rank=4/11][node=0/1]
[2024-12-26 13:03:03,607][INFO][ezpz.dist]: 

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


[2024-12-26 13:03:03,607][INFO][ezpz.dist]: Using oneccl_bindings from: /opt/aurora/24.180.1/frameworks/aurora_nre_models_frameworks-2024.2.1_u1/lib/python3.10/site-packages/oneccl_bindings_for_pytorch/__init__.py
[2024-12-26 13:03:03,607][INFO][ezpz.dist]: Using ipex from: /opt/aurora/24.180.1/frameworks/aurora_nre_models_frameworks-2024.2.1_u1/lib/python3.10/site-packages/intel_extension_for_pytorch/__init__.py
[2024-12-26 13:03:03,607][INFO][ezpz.dist]: [0/24] Using device='xpu' with backend='DDP' + 'ccl' for distributed training.
[2024-12-26 13:03:03,611][INFO][ezpz.dist]: [device='xpu'][rank=0/23][local_rank=0/11][node=0/1]
[2024-12-26 13:03:03,611][WARNING][ezpz.dist]: Using [24 / 24] available "xpu" devices !!
2024:12:26-13:03:03:(129002) |CCL_WARN| value of CCL_LOG_LEVEL changed to be error (default:warn)
[2024-12-26 13:03:04,801][INFO][__main__]: Using native for SDPA backend
[2024-12-26 13:03:04,801][INFO][__main__]: Using AttentionBlock Attention with args.compile=False
[2024-12-26 13:03:04,801][INFO][__main__]: config=ViTConfig(img_size=224, batch_size=128, num_heads=16, head_dim=64, depth=24, patch_size=16)
[2024-12-26 13:03:04,801][INFO][__main__]: len(train_set)=1000000
[2024-12-26 13:03:33,857][INFO][__main__]: 
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
[2024-12-26 13:03:34,045][INFO][__main__]: Training with 24 x xpu (s), using torch_dtype=torch.bfloat16
[2024-12-26 13:03:48,429][INFO][__main__]: iter=0 loss=7.102205 dt=14.290959 dtf=13.501260 dtb=0.772920
[2024-12-26 13:03:49,467][INFO][__main__]: iter=1 loss=7.464774 dt=0.912323 dtf=0.295670 dtb=0.607295
[2024-12-26 13:03:50,489][INFO][__main__]: iter=2 loss=7.242321 dt=0.899001 dtf=0.290576 dtb=0.599036
[2024-12-26 13:03:51,512][INFO][__main__]: iter=3 loss=6.974370 dt=0.900507 dtf=0.289955 dtb=0.601244
[2024-12-26 13:03:52,533][INFO][__main__]: iter=4 loss=7.320842 dt=0.897502 dtf=0.289744 dtb=0.598354
[2024-12-26 13:03:53,556][INFO][__main__]: iter=5 loss=7.471265 dt=0.897839 dtf=0.287929 dtb=0.600329
[2024-12-26 13:03:54,574][INFO][__main__]: iter=6 loss=7.666187 dt=0.893821 dtf=0.287799 dtb=0.596269
[2024-12-26 13:03:55,600][INFO][__main__]: iter=7 loss=8.376398 dt=0.899029 dtf=0.286100 dtb=0.603470
[2024-12-26 13:03:56,618][INFO][__main__]: iter=8 loss=10.248241 dt=0.891929 dtf=0.287084 dtb=0.595473
[2024-12-26 13:03:57,635][INFO][__main__]: iter=9 loss=9.002174 dt=0.892073 dtf=0.288276 dtb=0.594184
[2024-12-26 13:03:58,650][INFO][__main__]: iter=10 loss=7.627015 dt=0.889942 dtf=0.287662 dtb=0.592798
[2024-12-26 13:03:59,908][INFO][ezpz.history]: Saving train_iter plot to: /lus/flare/projects/Aurora_deployment/foremans/projects/saforem2/mmm/outputs/plots/vit/mplot
[2024-12-26 13:03:59,908][INFO][ezpz.history]: Saving train_iter plot to: /lus/flare/projects/Aurora_deployment/foremans/projects/saforem2/mmm/outputs/plots/vit/mplot/pngs/train_iter.png
[2024-12-26 13:04:00,051][INFO][ezpz.history]: Saving train_iter plot to: /lus/flare/projects/Aurora_deployment/foremans/projects/saforem2/mmm/outputs/plots/vit/mplot/svgs/train_iter.svg
[2024-12-26 13:04:00,122][INFO][ezpz.history]: Saving train_loss plot to: /lus/flare/projects/Aurora_deployment/foremans/projects/saforem2/mmm/outputs/plots/vit/mplot
[2024-12-26 13:04:00,122][INFO][ezpz.history]: Saving train_loss plot to: /lus/flare/projects/Aurora_deployment/foremans/projects/saforem2/mmm/outputs/plots/vit/mplot/pngs/train_loss.png
[2024-12-26 13:04:00,262][INFO][ezpz.history]: Saving train_loss plot to: /lus/flare/projects/Aurora_deployment/foremans/projects/saforem2/mmm/outputs/plots/vit/mplot/svgs/train_loss.svg
[2024-12-26 13:04:00,329][INFO][ezpz.history]: Saving train_dt plot to: /lus/flare/projects/Aurora_deployment/foremans/projects/saforem2/mmm/outputs/plots/vit/mplot
[2024-12-26 13:04:00,329][INFO][ezpz.history]: Saving train_dt plot to: /lus/flare/projects/Aurora_deployment/foremans/projects/saforem2/mmm/outputs/plots/vit/mplot/pngs/train_dt.png
[2024-12-26 13:04:00,465][INFO][ezpz.history]: Saving train_dt plot to: /lus/flare/projects/Aurora_deployment/foremans/projects/saforem2/mmm/outputs/plots/vit/mplot/svgs/train_dt.svg
[2024-12-26 13:04:00,723][INFO][ezpz.history]: Saving train_dtf plot to: /lus/flare/projects/Aurora_deployment/foremans/projects/saforem2/mmm/outputs/plots/vit/mplot
[2024-12-26 13:04:00,723][INFO][ezpz.history]: Saving train_dtf plot to: /lus/flare/projects/Aurora_deployment/foremans/projects/saforem2/mmm/outputs/plots/vit/mplot/pngs/train_dtf.png
[2024-12-26 13:04:00,864][INFO][ezpz.history]: Saving train_dtf plot to: /lus/flare/projects/Aurora_deployment/foremans/projects/saforem2/mmm/outputs/plots/vit/mplot/svgs/train_dtf.svg
[2024-12-26 13:04:00,932][INFO][ezpz.history]: Saving train_dtb plot to: /lus/flare/projects/Aurora_deployment/foremans/projects/saforem2/mmm/outputs/plots/vit/mplot
[2024-12-26 13:04:00,933][INFO][ezpz.history]: Saving train_dtb plot to: /lus/flare/projects/Aurora_deployment/foremans/projects/saforem2/mmm/outputs/plots/vit/mplot/pngs/train_dtb.png
[2024-12-26 13:04:01,069][INFO][ezpz.history]: Saving train_dtb plot to: /lus/flare/projects/Aurora_deployment/foremans/projects/saforem2/mmm/outputs/plots/vit/mplot/svgs/train_dtb.svg
[2024-12-26 13:04:01,166][INFO][ezpz.plot]: Appending plot to: /flare/Aurora_deployment/foremans/projects/saforem2/mmm/outputs/plots/vit/tplot/train_iter.txt
[2024-12-26 13:04:01,172][INFO][ezpz.plot]: Appending plot to: /flare/Aurora_deployment/foremans/projects/saforem2/mmm/outputs/plots/vit/tplot/train_loss.txt
[2024-12-26 13:04:01,178][INFO][ezpz.plot]: Appending plot to: /flare/Aurora_deployment/foremans/projects/saforem2/mmm/outputs/plots/vit/tplot/train_dt.txt
Failed to download font: IBM Plex Sans, skipping!
Failed to download font: IBM Plex Sans Condensed, skipping!
Failed to download font: IBM Plex Serif, skipping!
                        train_iter [2024-12-26-130401]                     ␛[0m
    ┌─────────────────────────────────────────────────────────────────────┐␛[0m
10.0┤                                                                  ▗▄▞│␛[0m
    │                                                             ▗▄▄▀▀▘  │␛[0m
    │                                                          ▗▄▀▘       │␛[0m
 8.3┤                                                       ▄▄▀▘          │␛[0m
    │                                                  ▗▄▄▀▀              │␛[0m
    │                                              ▗▄▀▀▘                  │␛[0m
 6.7┤                                           ▄▞▀▘                      │␛[0m
    │                                       ▄▄▀▀                          │␛[0m
 5.0┤                                  ▗▄▄▀▀                              │␛[0m
    │                               ▄▄▀▘                                  │␛[0m
    │                           ▗▄▞▀                                      │␛[0m
 3.3┤                       ▄▄▞▀▘                                         │␛[0m
    │                   ▄▞▀▀                                              │␛[0m
    │               ▗▄▀▀                                                  │␛[0m
 1.7┤           ▗▄▞▀▘                                                     │␛[0m
    │       ▄▄▞▀▘                                                         │␛[0m
    │   ▗▄▞▀                                                              │␛[0m
 0.0┤▄▄▀▘                                                                 │␛[0m
    └┬──────┬──────┬─────┬──────┬──────┬──────┬──────┬─────┬──────┬───────┘␛[0m
     1      2      3     4      5      6      7      8     9     10        ␛[0m
train_iter                        train/iter                               ␛[0m
␛[1m␛[38;5;10mtext saved in␛[0m␛[0m ␛[2m/lus/flare/projects/Aurora_deployment/foremans/projects/saforem2/mmm/outputs/plots/vit/tplot/train_iter.txt␛[0m
                         train_loss [2024-12-26-130401]                    ␛[0m
     ┌────────────────────────────────────────────────────────────────────┐␛[0m
10.25┤                                                     ▗▚             │␛[0m
     │                                                     ▌ ▚            │␛[0m
     │                                                    ▞   ▚           │␛[0m
 9.70┤                                                   ▗▘    ▚          │␛[0m
     │                                                   ▌      ▚         │␛[0m
     │                                                  ▞        ▚        │␛[0m
 9.16┤                                                 ▗▘         ▚▖      │␛[0m
     │                                                 ▌           ▚      │␛[0m
 8.61┤                                                ▞             ▚     │␛[0m
     │                                               ▗▘              ▚    │␛[0m
     │                                               ▞                ▌   │␛[0m
 8.07┤                                             ▄▀                 ▝▖  │␛[0m
     │                                           ▗▀  [2024-12-26 13:04:01,184][INFO][ezpz.plot]: Appending plot to: /flare/Aurora_deployment/foremans/projects/saforem2/mmm/outputs/plots/vit/tplot/train_dtf.txt
                  ▝▖ │␛[0m
     │                                         ▗▞▘                      ▝▖│␛[0m
 7.52┤                                     ▗▄▄▞▘                         ▝│␛[0m
     │     ▗▄▚▄▄▖                ▄▄▄▄▄▄▄▀▀▀▘                              │␛[0m
     │  ▄▞▀▘    ▝▀▀▀▄▄        ▄▞▀                                         │␛[0m
 6.97┤▀▀              ▀▀▚▄▄▄▞▀                                            │␛[0m
     └┬──────┬─────┬──────┬──────┬──────┬─────┬──────┬──────┬─────┬───────┘␛[0m
      1      2     3      4      5      6     7      8      9    10        ␛[0m
train_loss                         train/iter                              ␛[0m
␛[1m␛[38;5;10mtext saved in␛[0m␛[0m ␛[2m/lus/flare/projects/Aurora_deployment/foremans/projects/saforem2/mmm/outputs/plots/vit/tplot/train_loss.txt␛[0m
                         train_dt [2024-12-26-130401]                      ␛[0m
    ┌─────────────────────────────────────────────────────────────────────┐␛[0m
14.3┤▌                                                                    │␛[0m
    │▚                                                                    │␛[0m
    │▝▖                                                                   │␛[0m
12.1┤ ▌                                                                   │␛[0m
    │ ▐                                                                   │␛[0m
    │  ▌                                                                  │␛[0m
 9.8┤  ▚                                                                  │␛[0m
    │  ▝▖                                                                 │␛[0m
 7.6┤   ▌                                                                 │␛[0m
    │   ▐                                                                 │␛[0m
    │    ▌                                                                │␛[0m
 5.4┤    ▚                                                                │␛[0m
    │    ▝▖                                                               │␛[0m
    │     ▌                                                               │␛[0m
 3.1┤     ▐                                                               │␛[0m
    │      ▌                                                              │␛[0m
    │      ▚                                                              │␛[0m
 0.9┤      ▝▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄│␛[0m
    └┬──────┬──────┬─────┬──────┬──────┬──────┬──────┬─────┬──────┬───────┘␛[0m
     1      2      3     4      5      6      7      8     9     10        ␛[0m
train_dt                          train/iter                               ␛[0m
␛[1m␛[38;5;10mtext saved in␛[0m␛[0m ␛[2m/lus/flare/projects/Aurora_deployment/foremans/projects/saforem2/mmm/outputs/plots/vit/tplot/train_dt.txt␛[0m
[2024-12-26 13:04:01,189][INFO][ezpz.plot]: Appending plot to: /flare/Aurora_deployment/foremans/projects/saforem2/mmm/outputs/plots/vit/tplot/train_dtb.txt
[2024-12-26 13:04:01,196][INFO][__main__]: dataset=<xarray.Dataset> Size: 528B
Dimensions:     (draw: 11)
Coordinates:
  * draw        (draw) int64 88B 0 1 2 3 4 5 6 7 8 9 10
Data variables:
    train_iter  (draw) int64 88B 0 1 2 3 4 5 6 7 8 9 10
    train_loss  (draw) float64 88B 7.102 7.465 7.242 6.974 ... 10.25 9.002 7.627
    train_dt    (draw) float64 88B 14.29 0.9123 0.899 ... 0.8919 0.8921 0.8899
    train_dtf   (draw) float64 88B 13.5 0.2957 0.2906 ... 0.2871 0.2883 0.2877
    train_dtb   (draw) float64 88B 0.7729 0.6073 0.599 ... 0.5955 0.5942 0.5928
[2024-12-26 13:04:01,197][INFO][__main__]: Took 56.40 seconds
                         train_dtf [2024-12-26-130401]                     ␛[0m
    ┌─────────────────────────────────────────────────────────────────────┐␛[0m
13.5┤▌                                                                    │␛[0m
    │▚                                                                    │␛[0m
    │▝▖                                                                   │␛[0m
11.3┤ ▌                                                                   │␛[0m
    │ ▐                                                                   │␛[0m
    │  ▌                                                                  │␛[0m
 9.1┤  ▚                                                                  │␛[0m
    │  ▝▖                                                                 │␛[0m
 6.9┤   ▌                                                                 │␛[0m
    │   ▐                                                                 │␛[0m
    │    ▌                                                                │␛[0m
 4.7┤    ▚                                                                │␛[0m
    │    ▝▖                                                               │␛[0m
    │     ▌                                                               │␛[0m
 2.5┤     ▐                                                               │␛[0m
    │      ▌                                                              │␛[0m
    │      ▚                                                              │␛[0m
 0.3┤      ▝▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄│␛[0m
    └┬──────┬──────┬─────┬──────┬──────┬──────┬──────┬─────┬──────┬───────┘␛[0m
     1      2      3     4      5      6      7      8     9     10        ␛[0m
train_dtf                         train/iter                               ␛[0m
␛[1m␛[38;5;10mtext saved in␛[0m␛[0m ␛[2m/lus/flare/projects/Aurora_deployment/foremans/projects/saforem2/mmm/outputs/plots/vit/tplot/train_dtf.txt␛[0m
                          train_dtb [2024-12-26-130401]                    ␛[0m
     ┌────────────────────────────────────────────────────────────────────┐␛[0m
0.773┤▌                                                                   │␛[0m
     │▚                                                                   │␛[0m
     │▝▖                                                                  │␛[0m
0.743┤ ▚                                                                  │␛[0m
     │ ▐                                                                  │␛[0m
     │  ▌                                                                 │␛[0m
0.713┤  ▐                                                                 │␛[0m
     │   ▌                                                                │␛[0m
0.683┤   ▐                                                                │␛[0m
     │   ▝▖                                                               │␛[0m
     │    ▚                                                               │␛[0m
0.653┤    ▝▖                                                              │␛[0m
     │     ▌                                                              │␛[0m
     │     ▐                                                              │␛[0m
0.623┤      ▌                                                             │␛[0m
     │      ▐                                                             │␛[0m
     │       ▚▄▄▖         ▗                          ▗                    │␛[0m
0.593┤          ▝▀▀▀▀▀▀▀▀▀▘▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▘▀▀▀▀▀▀▚▄▄▄▄▄▄▄▄▄▄▄▄▄│␛[0m
     └┬──────┬─────┬──────┬──────┬──────┬─────┬──────┬──────┬─────┬───────┘␛[0m
      1      2     3      4      5      6     7      8      9    10        ␛[0m
train_dtb                          train/iter                              ␛[0m
␛[1m␛[38;5;10mtext saved in␛[0m␛[0m ␛[2m/lus/flare/projects/Aurora_deployment/foremans/projects/saforem2/mmm/outputs/plots/vit/tplot/train_dtb.txt␛[0m
Application d67f5af4 resources: utime=5400s stime=901s maxrss=4510072KB inblock=243770 oublock=752 minflt=18278180 majflt=11718 nvcsw=450176 nivcsw=154328
```
  ```

</details>


### 📝 Example: FSDP

```bash
git clone https://github.com/saforem2/mmm
cd mmm

export PBS_O_WORKDIR=$(pwd)
source /dev/stdin <<< $(curl 'https://raw.githubusercontent.com/saforem2/ezpz/refs/heads/main/src/ezpz/bin/utils.sh')
ezpz_setup_env

python3 -m pip install -e . --require-virtualenv

# ---- [smoke-test] ------------------------------------
#test ability to launch simple distributed training job:
launch python3 -m ezpz.test_dist

# ---- [FSDP Example] -----------------------------------
launch python3 \
  src/mmm/examples/fsdp/example.py \
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
