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
launch python3 -m mmm.examples.fsdp_tp --n_layers 24 --tpsize 4
```

<details closed><summary>Output:</summary>

<details closed><summary>Aurora @ ALCF:</summary>

```python
#[🐍 aurora_nre_models_frameworks-2024.2.1_u1](👻 aurora_nre_models_frameworks-2024.2.1_u1)
#[09:57:34 AM][x4404c5s2b0n0][/f/A/f/p/s/mmm][🌱 main][$!?][⏱️ 26s]
# $ launch python3 -Wignore -m mmm.examples.fsdp_tp --n_layers 24 --tpsize 4
Disabling local launch: multi-node application
Connected to tcp://x4509c0s2b0n0.hostmgmt2509.cm.aurora.alcf.anl.gov:7919
Found executable /flare/Aurora_deployment/foremans/projects/saforem2/mmm/venvs/aurora_nre_models_frameworks-2024.2.1_u1/bin/python3
Launching application 15ace429-216f-4cad-afa5-773dd63ef116
[2025-01-03 09:57:47,333] [INFO] [real_accelerator.py:222:get_accelerator] Setting ds_accelerator to xpu (auto detect)
# ...clipped...
[2025-01-03 09:57:47,972] [INFO] [real_accelerator.py:222:get_accelerator] Setting ds_accelerator to xpu (auto detect)
[2025-01-03 09:57:52.544034][INFO][__init__.py:146] - > initializing tensor parallel with size 4
[2025-01-03 09:57:52.546527][INFO][__init__.py:151] - > initializing context parallel with size 1
[2025-01-03 09:57:52.546977][INFO][__init__.py:156] - > initializing pipeline with size 1
[2025-01-03 09:57:52.547377][INFO][__init__.py:159] - > initializing ddp with size 12
2025:01:03-09:57:52:(92348) |CCL_WARN| value of CCL_LOG_LEVEL changed to be error (default:warn)
[2025-01-03 09:57:53.665930][INFO][dist.py:818] - Using device='xpu' with backend='DDP' + 'ccl' for distributed training.
[2025-01-03 09:57:53.780030][INFO][dist.py:859] - ['x4509c0s2b0n0'][ 0/47] [tp:0/3][dp: 0/11]
[2025-01-03 09:57:53.779551][INFO][dist.py:859] - ['x4509c0s2b0n0'][11/47] [tp:3/3][dp: 2/11]
[2025-01-03 09:57:53.779562][INFO][dist.py:859] - ['x4509c0s2b0n0'][ 3/47] [tp:3/3][dp: 0/11]
[2025-01-03 09:57:53.779566][INFO][dist.py:859] - ['x4509c0s2b0n0'][ 7/47] [tp:3/3][dp: 1/11]
[2025-01-03 09:57:53.780021][INFO][dist.py:859] - ['x4509c0s2b0n0'][ 8/47] [tp:0/3][dp: 2/11]
[2025-01-03 09:57:53.780023][INFO][dist.py:859] - ['x4509c0s2b0n0'][ 2/47] [tp:2/3][dp: 0/11]
[2025-01-03 09:57:53.780010][INFO][dist.py:859] - ['x4509c0s2b0n0'][ 5/47] [tp:1/3][dp: 1/11]
[2025-01-03 09:57:53.780021][INFO][dist.py:859] - ['x4509c0s2b0n0'][10/47] [tp:2/3][dp: 2/11]
[2025-01-03 09:57:53.780023][INFO][dist.py:859] - ['x4509c0s2b0n0'][ 1/47] [tp:1/3][dp: 0/11]
[2025-01-03 09:57:53.780039][INFO][dist.py:859] - ['x4509c0s2b0n0'][ 4/47] [tp:0/3][dp: 1/11]
[2025-01-03 09:57:53.780014][INFO][dist.py:859] - ['x4509c0s2b0n0'][ 9/47] [tp:1/3][dp: 2/11]
[2025-01-03 09:57:53.779500][INFO][dist.py:859] - ['x4206c2s1b0n0'][35/47] [tp:3/3][dp: 8/11]
[2025-01-03 09:57:53.779507][INFO][dist.py:859] - ['x4206c2s1b0n0'][27/47] [tp:3/3][dp: 6/11]
[2025-01-03 09:57:53.779509][INFO][dist.py:859] - ['x4206c2s1b0n0'][31/47] [tp:3/3][dp: 7/11]
[2025-01-03 09:57:53.779540][INFO][dist.py:859] - ['x4404c5s2b0n0'][43/47] [tp:3/3][dp:10/11]
[2025-01-03 09:57:53.779982][INFO][dist.py:859] - ['x4206c2s1b0n0'][24/47] [tp:0/3][dp: 6/11]
[2025-01-03 09:57:53.779550][INFO][dist.py:859] - ['x4404c5s2b0n0'][47/47] [tp:3/3][dp:11/11]
[2025-01-03 09:57:53.779971][INFO][dist.py:859] - ['x4206c2s1b0n0'][25/47] [tp:1/3][dp: 6/11]
[2025-01-03 09:57:53.779558][INFO][dist.py:859] - ['x4206c0s0b0n0'][15/47] [tp:3/3][dp: 3/11]
[2025-01-03 09:57:53.779972][INFO][dist.py:859] - ['x4206c2s1b0n0'][26/47] [tp:2/3][dp: 6/11]
[2025-01-03 09:57:53.779545][INFO][dist.py:859] - ['x4206c0s0b0n0'][19/47] [tp:3/3][dp: 4/11]
[2025-01-03 09:57:53.780016][INFO][dist.py:859] - ['x4404c5s2b0n0'][36/47] [tp:0/3][dp: 9/11]
[2025-01-03 09:57:53.780066][INFO][dist.py:859] - ['x4509c0s2b0n0'][ 6/47] [tp:2/3][dp: 1/11]
[2025-01-03 09:57:53.779552][INFO][dist.py:859] - ['x4206c0s0b0n0'][23/47] [tp:3/3][dp: 5/11]
[2025-01-03 09:57:53.779979][INFO][dist.py:859] - ['x4206c2s1b0n0'][28/47] [tp:0/3][dp: 7/11]
[2025-01-03 09:57:53.779996][INFO][dist.py:859] - ['x4404c5s2b0n0'][37/47] [tp:1/3][dp: 9/11]
[2025-01-03 09:57:53.780021][INFO][dist.py:859] - ['x4206c0s0b0n0'][12/47] [tp:0/3][dp: 3/11]
[2025-01-03 09:57:53.779957][INFO][dist.py:859] - ['x4206c2s1b0n0'][29/47] [tp:1/3][dp: 7/11]
[2025-01-03 09:57:53.780000][INFO][dist.py:859] - ['x4206c0s0b0n0'][13/47] [tp:1/3][dp: 3/11]
[2025-01-03 09:57:53.779978][INFO][dist.py:859] - ['x4206c2s1b0n0'][32/47] [tp:0/3][dp: 8/11]
[2025-01-03 09:57:53.780022][INFO][dist.py:859] - ['x4206c0s0b0n0'][14/47] [tp:2/3][dp: 3/11]
[2025-01-03 09:57:53.779964][INFO][dist.py:859] - ['x4206c2s1b0n0'][33/47] [tp:1/3][dp: 8/11]
[2025-01-03 09:57:53.780032][INFO][dist.py:859] - ['x4206c0s0b0n0'][16/47] [tp:0/3][dp: 4/11]
[2025-01-03 09:57:53.780014][INFO][dist.py:859] - ['x4206c0s0b0n0'][17/47] [tp:1/3][dp: 4/11]
[2025-01-03 09:57:53.779970][INFO][dist.py:859] - ['x4206c2s1b0n0'][34/47] [tp:2/3][dp: 8/11]
[2025-01-03 09:57:53.780024][INFO][dist.py:859] - ['x4404c5s2b0n0'][38/47] [tp:2/3][dp: 9/11]
[2025-01-03 09:57:53.780014][INFO][dist.py:859] - ['x4206c0s0b0n0'][20/47] [tp:0/3][dp: 5/11]
[2025-01-03 09:57:53.780031][INFO][dist.py:859] - ['x4206c2s1b0n0'][30/47] [tp:2/3][dp: 7/11]
[2025-01-03 09:57:53.779557][INFO][dist.py:859] - ['x4404c5s2b0n0'][39/47] [tp:3/3][dp: 9/11]
[2025-01-03 09:57:53.779990][INFO][dist.py:859] - ['x4206c0s0b0n0'][21/47] [tp:1/3][dp: 5/11]
[2025-01-03 09:57:53.780024][INFO][dist.py:859] - ['x4206c0s0b0n0'][22/47] [tp:2/3][dp: 5/11]
[2025-01-03 09:57:53.780065][INFO][dist.py:859] - ['x4206c0s0b0n0'][18/47] [tp:2/3][dp: 4/11]
[2025-01-03 09:57:53.780019][INFO][dist.py:859] - ['x4404c5s2b0n0'][40/47] [tp:0/3][dp:10/11]
[2025-01-03 09:57:53.780004][INFO][dist.py:859] - ['x4404c5s2b0n0'][41/47] [tp:1/3][dp:10/11]
[2025-01-03 09:57:53.780046][INFO][dist.py:859] - ['x4404c5s2b0n0'][42/47] [tp:2/3][dp:10/11]
[2025-01-03 09:57:53.780005][INFO][dist.py:859] - ['x4404c5s2b0n0'][44/47] [tp:0/3][dp:11/11]
[2025-01-03 09:57:53.779988][INFO][dist.py:859] - ['x4404c5s2b0n0'][45/47] [tp:1/3][dp:11/11]
[2025-01-03 09:57:53.780001][INFO][dist.py:859] - ['x4404c5s2b0n0'][46/47] [tp:2/3][dp:11/11]
[2025-01-03 09:57:53.784438][INFO][fsdp_tp.py:172] - Device mesh created:
device_mesh=DeviceMesh([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23], [24, 25, 26, 27], [28, 29, 30, 31], [32, 33, 34, 35], [36, 37, 38, 39], [40, 41, 42, 43], [44, 45, 46, 47]], mesh_dim_names=('dp', 'tp'))
[2025-01-03 09:57:54.354114][INFO][fsdp_tp.py:185] -
=================================================================
Layer (type:depth-idx)                   Param #
=================================================================
Transformer                              --
├─Embedding: 1-1                         8,192,000
├─ModuleList: 1-2                        --
│    └─TransformerBlock: 2-1             717,312
│    └─TransformerBlock: ...             717,312
│    └─TransformerBlock: 2-24            717,312
├─RMSNorm: 1-3                           256
├─Linear: 1-4                            8,192,000
=================================================================
Total params: 33,599,744
Trainable params: 33,599,744
Non-trainable params: 0
=================================================================
[2025-01-03 09:57:54.989514][INFO][fsdp_tp.py:156] - Model after parallelization:
sharded_model=FullyShardedDataParallel(
  (_fsdp_wrapped_module): Transformer(
    (tok_embeddings): Embedding(32000, 256)
    (layers): ModuleList(
      (0-23): 24 x TransformerBlock(
        (attention): Attention(
          (wq): Linear(in_features=256, out_features=256, bias=False)
          (wk): Linear(in_features=256, out_features=64, bias=False)
          (wv): Linear(in_features=256, out_features=64, bias=False)
          (wo): Linear(in_features=256, out_features=256, bias=False)
        )
        (feed_forward): FeedForward(
          (w1): Linear(in_features=256, out_features=720, bias=False)
          (w2): Linear(in_features=720, out_features=256, bias=False)
          (w3): Linear(in_features=256, out_features=720, bias=False)
        )
        (attention_norm): RMSNorm()
        (ffn_norm): RMSNorm()
      )
    )
    (norm): RMSNorm()
    (output): Linear(in_features=256, out_features=32000, bias=False)
  )
)

[2025-01-03 09:57:54.992906][INFO][fsdp_tp.py:190] - Creating AdamW optimizer with lr=0.003
[2025-01-03 09:57:54.994139][INFO][fsdp_tp.py:201] -
Starting 2D training...
[2025-01-03 09:57:54.995754][INFO][fsdp_tp.py:218] - inp.shape=torch.Size([16, 128])
[2025-01-03 09:58:07.084567][INFO][fsdp_tp.py:227] - iter=0 loss=10.863029 dt=12.088833 dtf=11.562250 dtb=0.526583
[2025-01-03 09:58:07.491328][INFO][fsdp_tp.py:227] - iter=1 loss=10.827834 dt=0.405578 dtf=0.161620 dtb=0.243958
[2025-01-03 09:58:07.893360][INFO][fsdp_tp.py:227] - iter=2 loss=10.783049 dt=0.400887 dtf=0.162063 dtb=0.238825
[2025-01-03 09:58:08.294581][INFO][fsdp_tp.py:227] - iter=3 loss=10.761011 dt=0.400123 dtf=0.161115 dtb=0.239009
[2025-01-03 09:58:08.694642][INFO][fsdp_tp.py:227] - iter=4 loss=10.702212 dt=0.399039 dtf=0.160752 dtb=0.238287
[2025-01-03 09:58:09.095107][INFO][fsdp_tp.py:227] - iter=5 loss=10.734871 dt=0.399371 dtf=0.159905 dtb=0.239466
[2025-01-03 09:58:09.494406][INFO][fsdp_tp.py:227] - iter=6 loss=10.697002 dt=0.398326 dtf=0.160210 dtb=0.238116
[2025-01-03 09:58:09.893997][INFO][fsdp_tp.py:227] - iter=7 loss=10.633245 dt=0.398598 dtf=0.160773 dtb=0.237825
[2025-01-03 09:58:09.894818][INFO][fsdp_tp.py:238] - Finished 2D training
[2025-01-03 09:58:11.014390][INFO][history.py:723] - Saving iter plot to: /lus/flare/projects/Aurora_deployment/foremans/projects/saforem2/mmm/outputs/mmm-fsdp-tp/plots/mplot
[2025-01-03 09:58:11.503575][INFO][history.py:723] - Saving loss plot to: /lus/flare/projects/Aurora_deployment/foremans/projects/saforem2/mmm/outputs/mmm-fsdp-tp/plots/mplot
[2025-01-03 09:58:11.734301][INFO][history.py:723] - Saving dt plot to: /lus/flare/projects/Aurora_deployment/foremans/projects/saforem2/mmm/outputs/mmm-fsdp-tp/plots/mplot
[2025-01-03 09:58:11.962392][INFO][history.py:723] - Saving dtf plot to: /lus/flare/projects/Aurora_deployment/foremans/projects/saforem2/mmm/outputs/mmm-fsdp-tp/plots/mplot
[2025-01-03 09:58:12.191307][INFO][history.py:723] - Saving dtb plot to: /lus/flare/projects/Aurora_deployment/foremans/projects/saforem2/mmm/outputs/mmm-fsdp-tp/plots/mplot
[2025-01-03 09:58:12.403876][INFO][history.py:603] - Saving tplots to /lus/flare/projects/Aurora_deployment/foremans/projects/saforem2/mmm/outputs/mmm-fsdp-tp/plots/tplot
                           iter [2025-01-03-095812]
   ┌──────────────────────────────────────────────────────────────────────┐
7.0┤                                                                   ▗▄▞│
   │                                                               ▗▄▞▀▘  │
   │                                                           ▗▄▞▀▘      │
5.8┤                                                       ▗▄▞▀▘          │
   │                                                   ▗▄▞▀▘              │
   │                                               ▗▄▞▀▘                  │
4.7┤                                           ▗▄▞▀▘                      │
   │                                       ▗▄▞▀▘                          │
3.5┤                                    ▄▄▀▘                              │
   │                                ▄▄▀▀                                  │
   │                            ▄▄▀▀                                      │
2.3┤                        ▄▄▀▀                                          │
   │                    ▄▄▀▀                                              │
   │                ▄▄▀▀                                                  │
1.2┤            ▄▄▀▀                                                      │
   │        ▄▄▀▀                                                          │
   │    ▄▄▀▀                                                              │
0.0┤▄▄▀▀                                                                  │
   └┬─────────┬─────────┬─────────┬────────┬─────────┬─────────┬──────────┘
    1         2         3         4        5         6         7
iter
text saved in /lus/flare/projects/Aurora_deployment/foremans/projects/saforem2/mmm/outputs/mmm-fsdp-tp/plots/tplot/iter.txt
                            loss [2025-01-03-095812]
      ┌───────────────────────────────────────────────────────────────────┐
10.863┤▚▄                                                                 │
      │  ▀▀▄▄                                                             │
      │      ▀▀▄▄                                                         │
10.825┤          ▀▄▖                                                      │
      │            ▝▀▄                                                    │
      │               ▀▚▄                                                 │
10.786┤                  ▀▚▄▖                                             │
      │                     ▝▀▀▄▄▖                                        │
10.748┤                          ▝▀▀▄                                     │
      │                              ▀▄▖                                  │
      │                                ▝▚▖          ▗▄▞▄▖                 │
10.710┤                                  ▝▀▄   ▗▄▄▀▀▘   ▝▀▚▄▖             │
      │                                     ▀▀▀▘            ▝▀▚▄▖         │
      │                                                         ▝▄        │
10.672┤                                                           ▀▄      │
      │                                                             ▀▄    │
      │                                                               ▀▄  │
10.633┤                                                                 ▀▄│
      └┬────────┬─────────┬────────┬─────────┬────────┬─────────┬─────────┘
       1        2         3        4         5        6         7
loss
text saved in /lus/flare/projects/Aurora_deployment/foremans/projects/saforem2/mmm/outputs/mmm-fsdp-tp/plots/tplot/loss.txt
                            dt [2025-01-03-095812]
    ┌─────────────────────────────────────────────────────────────────────┐
12.1┤▌                                                                    │
    │▐                                                                    │
    │ ▌                                                                   │
10.1┤ ▝▖                                                                  │
    │  ▚                                                                  │
    │  ▝▖                                                                 │
 8.2┤   ▚                                                                 │
    │    ▌                                                                │
 6.2┤    ▐                                                                │
    │     ▌                                                               │
    │     ▝▖                                                              │
 4.3┤      ▚                                                              │
    │      ▝▖                                                             │
    │       ▚                                                             │
 2.3┤        ▌                                                            │
    │        ▐                                                            │
    │         ▌                                                           │
 0.4┤         ▝▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄│
    └┬─────────┬────────┬─────────┬─────────┬─────────┬────────┬──────────┘
     1         2        3         4         5         6        7
dt
text saved in /lus/flare/projects/Aurora_deployment/foremans/projects/saforem2/mmm/outputs/mmm-fsdp-tp/plots/tplot/dt.txt
                            dtf [2025-01-03-095812]
    ┌─────────────────────────────────────────────────────────────────────┐
11.6┤▌                                                                    │
    │▐                                                                    │
    │ ▌                                                                   │
 9.7┤ ▝▖                                                                  │
    │  ▚                                                                  │
    │  ▝▖                                                                 │
 7.8┤   ▚                                                                 │
    │    ▌                                                                │
 5.9┤    ▐                                                                │
    │     ▌                                                               │
    │     ▝▖                                                              │
 4.0┤      ▚                                                              │
    │      ▝▖                                                             │
    │       ▚                                                             │
 2.1┤        ▌                                                            │
    │        ▐                                                            │
    │         ▌                                                           │
 0.2┤         ▝▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄│
    └┬─────────┬────────┬─────────┬─────────┬─────────┬────────┬──────────┘
     1         2        3         4         5         6        7
dtf
text saved in /lus/flare/projects/Aurora_deployment/foremans/projects/saforem2/mmm/outputs/mmm-fsdp-tp/plots/tplot/dtf.txt
                             dtb [2025-01-03-095812]
     ┌────────────────────────────────────────────────────────────────────┐
0.527┤▌                                                                   │
     │▐                                                                   │
     │ ▌                                                                  │
0.478┤ ▐                                                                  │
     │  ▚                                                                 │
     │  ▝▖                                                                │
0.430┤   ▚                                                                │
     │   ▝▖                                                               │
0.382┤    ▚                                                               │
     │     ▌                                                              │
     │     ▐                                                              │
0.334┤      ▌                                                             │
     │      ▐                                                             │
     │       ▚                                                            │
0.286┤       ▝▖                                                           │
     │        ▚                                                           │
     │        ▝▖                                                          │
0.238┤         ▝▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄│
     └┬─────────┬────────┬─────────┬────────┬─────────┬────────┬──────────┘
      1         2        3         4        5         6        7
dtb
text saved in /lus/flare/projects/Aurora_deployment/foremans/projects/saforem2/mmm/outputs/mmm-fsdp-tp/plots/tplot/dtb.txt
[2025-01-03 09:58:12.981346][INFO][utils.py:132] - Saving dataset to: /lus/flare/projects/Aurora_deployment/foremans/projects/saforem2/mmm/outputs/mmm-fsdp-tp/train_dataset.h5
[2025-01-03 09:58:13.054512][INFO][fsdp_tp.py:243] - dataset=<xarray.Dataset> Size: 384B
Dimensions:  (draw: 8)
Coordinates:
  * draw     (draw) int64 64B 0 1 2 3 4 5 6 7
Data variables:
    iter     (draw) int64 64B 0 1 2 3 4 5 6 7
    loss     (draw) float64 64B 10.86 10.83 10.78 10.76 10.7 10.73 10.7 10.63
    dt       (draw) float64 64B 12.09 0.4056 0.4009 ... 0.3994 0.3983 0.3986
    dtf      (draw) float64 64B 11.56 0.1616 0.1621 ... 0.1599 0.1602 0.1608
    dtb      (draw) float64 64B 0.5266 0.244 0.2388 ... 0.2395 0.2381 0.2378
Application 15ace429 resources: utime=1612s stime=434s maxrss=2901172KB inblock=777320 oublock=1488 minflt=15879731 majflt=23706 nvcsw=929293 nivcsw=2432655
took: 0h:00m:39s
```

</details>

</details>

## 🖼️ Example: ViT

We can now `launch` the example in
[`src/mmm/examples/vit.py`](/src/mmm/examples/vit.py):

```bash
launch python3 -m mmm.examples.vit --max_iters 10
```

<details closed><summary>Output:</summary>

<details closed><summary>Aurora @ ALCF:</summary>

```python
[🐍 aurora_nre_models_frameworks-2024.2.1_u1](👻 aurora_nre_models_frameworks-2024.2.1_u1)
#[10:02:16 AM][x4404c5s2b0n0][/f/A/f/p/s/mmm][🌱 main][$!?]
$ launch python3 -Wignore -m mmm.examples.vit --max_iters 10
Disabling local launch: multi-node application
Connected to tcp://x4509c0s2b0n0.hostmgmt2509.cm.aurora.alcf.anl.gov:7919
Found executable /flare/Aurora_deployment/foremans/projects/saforem2/mmm/venvs/aurora_nre_models_frameworks-2024.2.1_u1/bin/python3
Launching application 8ac3f76b-2b75-45c0-925d-3bfa3eee7633
[2025-01-03 10:04:41,306] [INFO] [real_accelerator.py:222:get_accelerator] Setting ds_accelerator to xpu (auto detect)
# ...[clipped]...
[2025-01-03 10:04:42,574] [INFO] [real_accelerator.py:222:get_accelerator] Setting ds_accelerator to xpu (auto detect)
[2025-01-03 10:05:03.848363][INFO][vit.py:246] - Using native for SDPA backend
[2025-01-03 10:05:03.851500][INFO][vit.py:272] - Using AttentionBlock Attention with args.compile=False
[2025-01-03 10:05:03.918476][INFO][__init__.py:146] - > initializing tensor parallel with size 1
[2025-01-03 10:05:03.918996][INFO][__init__.py:151] - > initializing context parallel with size 1
[2025-01-03 10:05:03.919362][INFO][__init__.py:156] - > initializing pipeline with size 1
[2025-01-03 10:05:03.919692][INFO][__init__.py:159] - > initializing ddp with size 48
2025:01:03-10:05:03:(94856) |CCL_WARN| value of CCL_LOG_LEVEL changed to be error (default:warn)
[2025-01-03 10:05:05.399317][INFO][dist.py:818] - Using device='xpu' with backend='DDP' + 'ccl' for distributed training.
[2025-01-03 10:05:05.400019][INFO][dist.py:859] - ['x4509c0s2b0n0'][ 0/47]
[2025-01-03 10:05:05.399397][INFO][dist.py:859] - ['x4509c0s2b0n0'][ 1/47]
[2025-01-03 10:05:05.399353][INFO][dist.py:859] - ['x4509c0s2b0n0'][11/47]
[2025-01-03 10:05:05.399368][INFO][dist.py:859] - ['x4509c0s2b0n0'][ 2/47]
[2025-01-03 10:05:05.399384][INFO][dist.py:859] - ['x4509c0s2b0n0'][ 3/47]
[2025-01-03 10:05:05.399373][INFO][dist.py:859] - ['x4509c0s2b0n0'][ 4/47]
[2025-01-03 10:05:05.399388][INFO][dist.py:859] - ['x4509c0s2b0n0'][ 5/47]
[2025-01-03 10:05:05.399352][INFO][dist.py:859] - ['x4509c0s2b0n0'][ 7/47]
[2025-01-03 10:05:05.399348][INFO][dist.py:859] - ['x4509c0s2b0n0'][ 8/47]
[2025-01-03 10:05:05.399372][INFO][dist.py:859] - ['x4509c0s2b0n0'][ 9/47]
[2025-01-03 10:05:05.399356][INFO][dist.py:859] - ['x4509c0s2b0n0'][10/47]
[2025-01-03 10:05:05.399312][INFO][dist.py:859] - ['x4404c5s2b0n0'][36/47]
[2025-01-03 10:05:05.399312][INFO][dist.py:859] - ['x4206c2s1b0n0'][24/47]
[2025-01-03 10:05:05.399333][INFO][dist.py:859] - ['x4206c0s0b0n0'][20/47]
[2025-01-03 10:05:05.399358][INFO][dist.py:859] - ['x4206c0s0b0n0'][12/47]
[2025-01-03 10:05:05.399358][INFO][dist.py:859] - ['x4206c0s0b0n0'][13/47]
[2025-01-03 10:05:05.399352][INFO][dist.py:859] - ['x4206c0s0b0n0'][14/47]
[2025-01-03 10:05:05.399325][INFO][dist.py:859] - ['x4206c2s1b0n0'][25/47]
[2025-01-03 10:05:05.399330][INFO][dist.py:859] - ['x4404c5s2b0n0'][37/47]
[2025-01-03 10:05:05.399363][INFO][dist.py:859] - ['x4206c0s0b0n0'][15/47]
[2025-01-03 10:05:05.399311][INFO][dist.py:859] - ['x4206c2s1b0n0'][26/47]
[2025-01-03 10:05:05.399333][INFO][dist.py:859] - ['x4404c5s2b0n0'][39/47]
[2025-01-03 10:05:05.399363][INFO][dist.py:859] - ['x4206c0s0b0n0'][16/47]
[2025-01-03 10:05:05.399333][INFO][dist.py:859] - ['x4206c2s1b0n0'][27/47]
[2025-01-03 10:05:05.399332][INFO][dist.py:859] - ['x4404c5s2b0n0'][41/47]
[2025-01-03 10:05:05.399386][INFO][dist.py:859] - ['x4206c0s0b0n0'][17/47]
[2025-01-03 10:05:05.399302][INFO][dist.py:859] - ['x4206c2s1b0n0'][28/47]
[2025-01-03 10:05:05.399306][INFO][dist.py:859] - ['x4404c5s2b0n0'][43/47]
[2025-01-03 10:05:05.399348][INFO][dist.py:859] - ['x4206c0s0b0n0'][19/47]
[2025-01-03 10:05:05.399351][INFO][dist.py:859] - ['x4206c0s0b0n0'][21/47]
[2025-01-03 10:05:05.399333][INFO][dist.py:859] - ['x4206c0s0b0n0'][22/47]
[2025-01-03 10:05:05.399444][INFO][dist.py:859] - ['x4509c0s2b0n0'][ 6/47]
[2025-01-03 10:05:05.399355][INFO][dist.py:859] - ['x4206c0s0b0n0'][23/47]
[2025-01-03 10:05:05.399294][INFO][dist.py:859] - ['x4404c5s2b0n0'][44/47]
[2025-01-03 10:05:05.399317][INFO][dist.py:859] - ['x4206c2s1b0n0'][29/47]
[2025-01-03 10:05:05.399405][INFO][dist.py:859] - ['x4206c0s0b0n0'][18/47]
[2025-01-03 10:05:05.399306][INFO][dist.py:859] - ['x4404c5s2b0n0'][45/47]
[2025-01-03 10:05:05.399297][INFO][dist.py:859] - ['x4206c2s1b0n0'][31/47]
[2025-01-03 10:05:05.399296][INFO][dist.py:859] - ['x4404c5s2b0n0'][46/47]
[2025-01-03 10:05:05.399286][INFO][dist.py:859] - ['x4206c2s1b0n0'][32/47]
[2025-01-03 10:05:05.399320][INFO][dist.py:859] - ['x4206c2s1b0n0'][33/47]
[2025-01-03 10:05:05.399289][INFO][dist.py:859] - ['x4206c2s1b0n0'][34/47]
[2025-01-03 10:05:05.399290][INFO][dist.py:859] - ['x4404c5s2b0n0'][47/47]
[2025-01-03 10:05:05.399290][INFO][dist.py:859] - ['x4206c2s1b0n0'][35/47]
[2025-01-03 10:05:05.399323][INFO][dist.py:859] - ['x4404c5s2b0n0'][38/47]
[2025-01-03 10:05:05.399324][INFO][dist.py:859] - ['x4206c2s1b0n0'][30/47]
[2025-01-03 10:05:05.399324][INFO][dist.py:859] - ['x4404c5s2b0n0'][40/47]
[2025-01-03 10:05:05.399373][INFO][dist.py:859] - ['x4404c5s2b0n0'][42/47]
[2025-01-03 10:05:05.404629][INFO][vit.py:83] - config=ViTConfig(img_size=224, batch_size=128, num_heads=16, head_dim=64, depth=24, patch_size=16)
[2025-01-03 10:05:33.726284][INFO][vit.py:122] -
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
[2025-01-03 10:05:33.838873][INFO][vit.py:138] - Model size: nparams=1.42 B
[2025-01-03 10:05:33.875672][INFO][vit.py:163] - Training with 48 x xpu (s), using torch_dtype=torch.bfloat16
[2025-01-03 10:05:51.652331][INFO][vit.py:181] - iter=0 loss=7.110761 dt=17.446471 dtf=16.439806 dtb=0.984764
[2025-01-03 10:05:52.375636][INFO][vit.py:181] - iter=1 loss=7.161612 dt=0.699467 dtf=0.047248 dtb=0.641429
[2025-01-03 10:05:53.105314][INFO][vit.py:181] - iter=2 loss=7.230668 dt=0.704598 dtf=0.048066 dtb=0.645473
[2025-01-03 10:05:53.825272][INFO][vit.py:181] - iter=3 loss=7.113066 dt=0.695661 dtf=0.042115 dtb=0.642792
[2025-01-03 10:05:54.546189][INFO][vit.py:181] - iter=4 loss=6.960967 dt=0.697206 dtf=0.042410 dtb=0.643944
[2025-01-03 10:05:55.266622][INFO][vit.py:181] - iter=5 loss=6.963243 dt=0.696353 dtf=0.044059 dtb=0.641429
[2025-01-03 10:05:55.988250][INFO][vit.py:181] - iter=6 loss=7.016630 dt=0.696779 dtf=0.043519 dtb=0.642242
[2025-01-03 10:05:56.706743][INFO][vit.py:181] - iter=7 loss=7.080251 dt=0.693541 dtf=0.043349 dtb=0.639498
[2025-01-03 10:05:57.426416][INFO][vit.py:181] - iter=8 loss=7.020833 dt=0.695024 dtf=0.043666 dtb=0.640416
[2025-01-03 10:05:58.147060][INFO][vit.py:181] - iter=9 loss=7.047978 dt=0.696053 dtf=0.045267 dtb=0.639672
[2025-01-03 10:05:58.865314][INFO][vit.py:181] - iter=10 loss=7.021929 dt=0.693284 dtf=0.043635 dtb=0.638591
[2025-01-03 10:05:59.218314][INFO][history.py:723] - Saving train_iter plot to: /lus/flare/projects/Aurora_deployment/foremans/projects/saforem2/mmm/outputs/mmm-vit/plots/mplot
[2025-01-03 10:05:59.499058][INFO][history.py:723] - Saving train_loss plot to: /lus/flare/projects/Aurora_deployment/foremans/projects/saforem2/mmm/outputs/mmm-vit/plots/mplot
[2025-01-03 10:05:59.740021][INFO][history.py:723] - Saving train_dt plot to: /lus/flare/projects/Aurora_deployment/foremans/projects/saforem2/mmm/outputs/mmm-vit/plots/mplot
[2025-01-03 10:05:59.984435][INFO][history.py:723] - Saving train_dtf plot to: /lus/flare/projects/Aurora_deployment/foremans/projects/saforem2/mmm/outputs/mmm-vit/plots/mplot
[2025-01-03 10:06:00.214398][INFO][history.py:723] - Saving train_dtb plot to: /lus/flare/projects/Aurora_deployment/foremans/projects/saforem2/mmm/outputs/mmm-vit/plots/mplot
[2025-01-03 10:06:00.433872][INFO][history.py:603] - Saving tplots to /lus/flare/projects/Aurora_deployment/foremans/projects/saforem2/mmm/outputs/mmm-vit/plots/tplot
                        train_iter [2025-01-03-100600]
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
train_iter
text saved in /lus/flare/projects/Aurora_deployment/foremans/projects/saforem2/mmm/outputs/mmm-vit/plots/tplot/train_iter.txt
                         train_loss [2025-01-03-100600]
     ┌────────────────────────────────────────────────────────────────────┐
7.231┤             ▟                                                      │
     │           ▗▞ ▚                                                     │
     │          ▄▘   ▚                                                    │
7.186┤        ▗▞      ▚                                                   │
     │       ▄▘        ▚                                                  │
     │     ▄▀           ▚                                                 │
7.141┤   ▄▀              ▚                                                │
     │ ▄▀                 ▚                                               │
7.096┤▀                   ▝▖                                              │
     │                     ▚                                              │
     │                      ▌                        ▞▖                   │
7.051┤                      ▝▖                     ▄▀ ▝▚▖                 │
     │                       ▚                   ▗▀     ▝▚▖      ▄▄▚▄▖    │
     │                        ▌                ▗▞▘        ▝▚▄▄▞▀▀    ▝▀▚▄▄│
7.006┤                        ▝▖              ▞▘                          │
     │                         ▚            ▄▀                            │
     │                          ▌         ▄▀                              │
6.961┤                          ▝▄▄▄▄▄▄▄▄▀                                │
     └┬──────┬─────┬──────┬──────┬──────┬─────┬──────┬──────┬─────┬───────┘
      1      2     3      4      5      6     7      8      9    10
train_loss
text saved in /lus/flare/projects/Aurora_deployment/foremans/projects/saforem2/mmm/outputs/mmm-vit/plots/tplot/train_loss.txt
                         train_dt [2025-01-03-100600]
    ┌─────────────────────────────────────────────────────────────────────┐
17.4┤▌                                                                    │
    │▚                                                                    │
    │▝▖                                                                   │
14.7┤ ▌                                                                   │
    │ ▐                                                                   │
    │  ▌                                                                  │
11.9┤  ▚                                                                  │
    │  ▝▖                                                                 │
 9.1┤   ▌                                                                 │
    │   ▐                                                                 │
    │    ▌                                                                │
 6.3┤    ▚                                                                │
    │    ▝▖                                                               │
    │     ▌                                                               │
 3.5┤     ▐                                                               │
    │      ▌                                                              │
    │      ▚                                                              │
 0.7┤      ▝▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄│
    └┬──────┬──────┬─────┬──────┬──────┬──────┬──────┬─────┬──────┬───────┘
     1      2      3     4      5      6      7      8     9     10
train_dt
text saved in /lus/flare/projects/Aurora_deployment/foremans/projects/saforem2/mmm/outputs/mmm-vit/plots/tplot/train_dt.txt
                         train_dtf [2025-01-03-100600]
    ┌─────────────────────────────────────────────────────────────────────┐
16.4┤▌                                                                    │
    │▚                                                                    │
    │▝▖                                                                   │
13.7┤ ▌                                                                   │
    │ ▐                                                                   │
    │  ▌                                                                  │
11.0┤  ▚                                                                  │
    │  ▝▖                                                                 │
 8.2┤   ▌                                                                 │
    │   ▐                                                                 │
    │    ▌                                                                │
 5.5┤    ▚                                                                │
    │    ▝▖                                                               │
    │     ▌                                                               │
 2.8┤     ▐                                                               │
    │      ▌                                                              │
    │      ▚                                                              │
 0.0┤      ▝▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄│
    └┬──────┬──────┬─────┬──────┬──────┬──────┬──────┬─────┬──────┬───────┘
     1      2      3     4      5      6      7      8     9     10
train_dtf
text saved in /lus/flare/projects/Aurora_deployment/foremans/projects/saforem2/mmm/outputs/mmm-vit/plots/tplot/train_dtf.txt
                          train_dtb [2025-01-03-100600]
     ┌────────────────────────────────────────────────────────────────────┐
0.985┤▌                                                                   │
     │▚                                                                   │
     │▝▖                                                                  │
0.927┤ ▌                                                                  │
     │ ▐                                                                  │
     │  ▌                                                                 │
0.869┤  ▚                                                                 │
     │  ▝▖                                                                │
0.812┤   ▌                                                                │
     │   ▐                                                                │
     │    ▌                                                               │
0.754┤    ▚                                                               │
     │    ▝▖                                                              │
     │     ▌                                                              │
0.696┤     ▐                                                              │
     │      ▌                                                             │
     │      ▚                                                             │
0.639┤      ▝▄▄▄▄▄▄▞▄▄▄▄▄▄▄▄▄▄▄▄▄▚▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄│
     └┬──────┬─────┬──────┬──────┬──────┬─────┬──────┬──────┬─────┬───────┘
      1      2     3      4      5      6     7      8      9    10
train_dtb
text saved in /lus/flare/projects/Aurora_deployment/foremans/projects/saforem2/mmm/outputs/mmm-vit/plots/tplot/train_dtb.txt
[2025-01-03 10:06:00.618027][INFO][utils.py:132] - Saving dataset to: /lus/flare/projects/Aurora_deployment/foremans/projects/saforem2/mmm/outputs/mmm-vit/train_dataset.h5
[2025-01-03 10:06:00.656925][INFO][vit.py:200] - dataset=<xarray.Dataset> Size: 528B
Dimensions:     (draw: 11)
Coordinates:
  * draw        (draw) int64 88B 0 1 2 3 4 5 6 7 8 9 10
Data variables:
    train_iter  (draw) int64 88B 0 1 2 3 4 5 6 7 8 9 10
    train_loss  (draw) float64 88B 7.111 7.162 7.231 7.113 ... 7.021 7.048 7.022
    train_dt    (draw) float64 88B 17.45 0.6995 0.7046 ... 0.695 0.6961 0.6933
    train_dtf   (draw) float64 88B 16.44 0.04725 0.04807 ... 0.04527 0.04364
    train_dtb   (draw) float64 88B 0.9848 0.6414 0.6455 ... 0.6404 0.6397 0.6386
[2025-01-03 10:06:00.664331][INFO][vit.py:279] - Took 56.82 seconds
Application 8ac3f76b resources: utime=10756s stime=1931s maxrss=4638452KB inblock=1297516 oublock=1920 minflt=38089683 majflt=12671790 nvcsw=1276384 nivcsw=1603670
took: 0h:01m:30s
```

</details>

<details closed><summary>Polaris @ ALCF:</summary>

Command:

```bash
launch python3 -m mmm.examples.vit --max_iters 10
```

Output:

```python
#[🐍 aurora_nre_models_frameworks-2024.2.1_u1](👻 aurora_nre_models_frameworks-2024.2.1_u1)
#[07:48:56 PM][x4603c7s2b0n0][/f/A/f/p/s/mmm][🌱 main][$!?][⏱️ 29s]
Disabling local launch: multi-node application
Connected to tcp://x4603c7s2b0n0.hostmgmt2603.cm.aurora.alcf.anl.gov:7919
Found executable /flare/Aurora_deployment/foremans/projects/saforem2/mmm/venvs/aurora_nre_models_frameworks-2024.2.1_u1/bin/python3
Launching application 66c85dca-4341-4edd-8100-9b8181ff3759
[2024-12-31 19:51:22.275852][INFO][__init__.py:146] - > initializing tensor parallel with size 1
[2024-12-31 19:51:22.278271][INFO][__init__.py:151] - > initializing context parallel with size 1
[2024-12-31 19:51:22.278651][INFO][__init__.py:156] - > initializing pipeline with size 1
[2024-12-31 19:51:22.278991][INFO][__init__.py:159] - > initializing ddp with size 24
2024:12:31-19:51:22:(148734) |CCL_WARN| value of CCL_LOG_LEVEL changed to be error (default:warn)
[2024-12-31 19:51:23.535446][INFO][dist.py:810] - Using device='xpu' with backend='DDP' + 'ccl' for distributed training.
[2024-12-31 19:51:23.536194][INFO][dist.py:848] - ['x4603c7s2b0n0'][ 0/23]
[2024-12-31 19:51:23.535458][INFO][dist.py:848] - ['x4603c7s2b0n0'][ 6/23]
[2024-12-31 19:51:23.535450][INFO][dist.py:848] - ['x4603c7s2b0n0'][ 8/23]
[2024-12-31 19:51:23.535448][INFO][dist.py:848] - ['x4603c7s2b0n0'][11/23]
[2024-12-31 19:51:23.535473][INFO][dist.py:848] - ['x4603c7s2b0n0'][ 1/23]
[2024-12-31 19:51:23.535474][INFO][dist.py:848] - ['x4603c7s2b0n0'][ 2/23]
[2024-12-31 19:51:23.535490][INFO][dist.py:848] - ['x4603c7s2b0n0'][ 3/23]
[2024-12-31 19:51:23.535487][INFO][dist.py:848] - ['x4603c7s2b0n0'][ 4/23]
[2024-12-31 19:51:23.535477][INFO][dist.py:848] - ['x4603c7s2b0n0'][ 5/23]
[2024-12-31 19:51:23.535449][INFO][dist.py:848] - ['x4603c7s2b0n0'][ 7/23]
[2024-12-31 19:51:23.535456][INFO][dist.py:848] - ['x4603c7s2b0n0'][ 9/23]
[2024-12-31 19:51:23.535460][INFO][dist.py:848] - ['x4603c7s2b0n0'][10/23]
[2024-12-31 19:51:23.535470][INFO][dist.py:848] - ['x4411c6s2b0n0'][12/23]
[2024-12-31 19:51:23.535457][INFO][dist.py:848] - ['x4411c6s2b0n0'][13/23]
[2024-12-31 19:51:23.535459][INFO][dist.py:848] - ['x4411c6s2b0n0'][14/23]
[2024-12-31 19:51:23.535461][INFO][dist.py:848] - ['x4411c6s2b0n0'][15/23]
[2024-12-31 19:51:23.535490][INFO][dist.py:848] - ['x4411c6s2b0n0'][16/23]
[2024-12-31 19:51:23.535474][INFO][dist.py:848] - ['x4411c6s2b0n0'][17/23]
[2024-12-31 19:51:23.535453][INFO][dist.py:848] - ['x4411c6s2b0n0'][19/23]
[2024-12-31 19:51:23.535456][INFO][dist.py:848] - ['x4411c6s2b0n0'][20/23]
[2024-12-31 19:51:23.535456][INFO][dist.py:848] - ['x4411c6s2b0n0'][21/23]
[2024-12-31 19:51:23.535458][INFO][dist.py:848] - ['x4411c6s2b0n0'][22/23]
[2024-12-31 19:51:23.535451][INFO][dist.py:848] - ['x4411c6s2b0n0'][23/23]
[2024-12-31 19:51:23.535465][INFO][dist.py:848] - ['x4411c6s2b0n0'][18/23]
[2024-12-31 19:51:23.781740][INFO][vit.py:250] - Using native for SDPA backend
[2024-12-31 19:51:23.782234][INFO][vit.py:276] - Using AttentionBlock Attention with args.compile=False
[2024-12-31 19:51:23.782641][INFO][vit.py:86] - config=ViTConfig(img_size=224, batch_size=128, num_heads=16, head_dim=64, depth=24, patch_size=16)
[2024-12-31 19:51:53.240524][INFO][vit.py:122] -
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
[2024-12-31 19:51:53.434376][INFO][vit.py:177] - Training with 24 x xpu (s), using torch_dtype=torch.bfloat16
[2024-12-31 19:52:09.714576][INFO][vit.py:204] - iter=0 loss=7.121100 dt=15.842163 dtf=14.856410 dtb=0.966353
[2024-12-31 19:52:10.656914][INFO][vit.py:204] - iter=1 loss=7.239427 dt=0.918050 dtf=0.299365 dtb=0.606726
[2024-12-31 19:52:11.593232][INFO][vit.py:204] - iter=2 loss=7.290485 dt=0.912320 dtf=0.295520 dtb=0.604882
[2024-12-31 19:52:12.525405][INFO][vit.py:204] - iter=3 loss=7.131482 dt=0.908596 dtf=0.294570 dtb=0.602297
[2024-12-31 19:52:13.454671][INFO][vit.py:204] - iter=4 loss=7.143745 dt=0.905077 dtf=0.290769 dtb=0.602704
[2024-12-31 19:52:14.376427][INFO][vit.py:204] - iter=5 loss=7.047148 dt=0.897958 dtf=0.291084 dtb=0.596143
[2024-12-31 19:52:15.299474][INFO][vit.py:204] - iter=6 loss=7.120049 dt=0.899810 dtf=0.292079 dtb=0.597195
[2024-12-31 19:52:16.218167][INFO][vit.py:204] - iter=7 loss=6.963861 dt=0.895402 dtf=0.293643 dtb=0.591206
[2024-12-31 19:52:17.138623][INFO][vit.py:204] - iter=8 loss=7.105645 dt=0.896737 dtf=0.290377 dtb=0.594644
[2024-12-31 19:52:18.065521][INFO][vit.py:204] - iter=9 loss=7.102675 dt=0.903153 dtf=0.292257 dtb=0.600275
[2024-12-31 19:52:18.992375][INFO][vit.py:204] - iter=10 loss=7.045384 dt=0.902353 dtf=0.291056 dtb=0.600508
                        train_iter [2024-12-31-195221]
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
[2024-12-31 19:52:21.800856][INFO][plot.py:220] - Appending plot to: /flare/Aurora_deployment/foremans/projects/saforem2/mmm/outputs/plots/vit/tplot/train_iter.txt
text saved in /lus/flare/projects/Aurora_deployment/foremans/projects/saforem2/mmm/outputs/plots/vit/tplot/train_iter.txt
                         train_loss [2024-12-31-195221]
     ┌────────────────────────────────────────────────────────────────────┐
7.290┤            ▗▟                                                      │
     │          ▄▀▘ ▚                                                     │
     │       ▄▞▀     ▚                                                    │
7.236┤      ▞        ▝▖                                                   │
     │     ▞          ▝▖                                                  │
     │    ▞            ▝▖                                                 │
7.182┤   ▞              ▚                                                 │
     │  ▞                ▚                                                │
7.127┤ ▞                  ▚▄▄▄▄▄▄▚                                        │
     │▀                           ▚▖          ▟                           │
     │                             ▝▄       ▗▀ ▚           ▗▀▀▀▀▀▀▀▚▖     │
7.073┤                               ▚    ▗▞▘   ▚         ▗▘        ▝▚▖   │
     │                                ▀▖ ▄▘     ▝▖       ▗▘           ▝▚▖ │
     │                                 ▝▀        ▝▖     ▗▘              ▝▀│
7.018┤                                            ▝▖    ▞                 │
     │                                             ▚   ▞                  │
     │                                              ▚ ▞                   │
6.964┤                                               ▜                    │
     └┬──────┬─────┬──────┬──────┬──────┬─────┬──────┬──────┬─────┬───────┘
      1      2     3      4      5      6     7      8      9    10
train_loss                         train/iter
[2024-12-31 19:52:21.834715][INFO][plot.py:220] - Appending plot to: /flare/Aurora_deployment/foremans/projects/saforem2/mmm/outputs/plots/vit/tplot/train_loss.txt
text saved in /lus/flare/projects/Aurora_deployment/foremans/projects/saforem2/mmm/outputs/plots/vit/tplot/train_loss.txt
                         train_dt [2024-12-31-195221]
    ┌─────────────────────────────────────────────────────────────────────┐
15.8┤▌                                                                    │
    │▚                                                                    │
    │▝▖                                                                   │
13.4┤ ▌                                                                   │
    │ ▐                                                                   │
    │  ▌                                                                  │
10.9┤  ▚                                                                  │
    │  ▝▖                                                                 │
 8.4┤   ▌                                                                 │
    │   ▐                                                                 │
    │    ▌                                                                │
 5.9┤    ▚                                                                │
    │    ▝▖                                                               │
    │     ▌                                                               │
 3.4┤     ▐                                                               │
    │      ▌                                                              │
    │      ▚                                                              │
 0.9┤      ▝▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄│
    └┬──────┬──────┬─────┬──────┬──────┬──────┬──────┬─────┬──────┬───────┘
     1      2      3     4      5      6      7      8     9     10
train_dt                          train/iter
[2024-12-31 19:52:21.872382][INFO][plot.py:220] - Appending plot to: /flare/Aurora_deployment/foremans/projects/saforem2/mmm/outputs/plots/vit/tplot/train_dt.txt
text saved in /lus/flare/projects/Aurora_deployment/foremans/projects/saforem2/mmm/outputs/plots/vit/tplot/train_dt.txt
                         train_dtf [2024-12-31-195221]
    ┌─────────────────────────────────────────────────────────────────────┐
14.9┤▌                                                                    │
    │▚                                                                    │
    │▝▖                                                                   │
12.4┤ ▌                                                                   │
    │ ▐                                                                   │
    │  ▌                                                                  │
10.0┤  ▚                                                                  │
    │  ▝▖                                                                 │
 7.6┤   ▌                                                                 │
    │   ▐                                                                 │
    │    ▌                                                                │
 5.1┤    ▚                                                                │
    │    ▝▖                                                               │
    │     ▌                                                               │
 2.7┤     ▐                                                               │
    │      ▌                                                              │
    │      ▚                                                              │
 0.3┤      ▝▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄│
    └┬──────┬──────┬─────┬──────┬──────┬──────┬──────┬─────┬──────┬───────┘
     1      2      3     4      5      6      7      8     9     10
train_dtf                         train/iter
[2024-12-31 19:52:21.896801][INFO][plot.py:220] - Appending plot to: /flare/Aurora_deployment/foremans/projects/saforem2/mmm/outputs/plots/vit/tplot/train_dtf.txt
text saved in /lus/flare/projects/Aurora_deployment/foremans/projects/saforem2/mmm/outputs/plots/vit/tplot/train_dtf.txt
                          train_dtb [2024-12-31-195221]
     ┌────────────────────────────────────────────────────────────────────┐
0.966┤▌                                                                   │
     │▚                                                                   │
     │▝▖                                                                  │
0.904┤ ▌                                                                  │
     │ ▐                                                                  │
     │  ▌                                                                 │
0.841┤  ▚                                                                 │
     │  ▝▖                                                                │
0.779┤   ▚                                                                │
     │   ▐                                                                │
     │    ▌                                                               │
0.716┤    ▐                                                               │
     │    ▝▖                                                              │
     │     ▚                                                              │
0.654┤     ▐                                                              │
     │      ▌                                                             │
     │      ▐                                                             │
0.591┤       ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▚▄▄▄▄▄▄▄▄▄▄▄▄▞▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▀▀▀▀▀▀▀│
     └┬──────┬─────┬──────┬──────┬──────┬─────┬──────┬──────┬─────┬───────┘
      1      2     3      4      5      6     7      8      9    10
train_dtb                          train/iter
[2024-12-31 19:52:21.921380][INFO][plot.py:220] - Appending plot to: /flare/Aurora_deployment/foremans/projects/saforem2/mmm/outputs/plots/vit/tplot/train_dtb.txt
text saved in /lus/flare/projects/Aurora_deployment/foremans/projects/saforem2/mmm/outputs/plots/vit/tplot/train_dtb.txt
[2024-12-31 19:52:21.956158][INFO][vit.py:224] - dataset=<xarray.Dataset> Size: 308B
Dimensions:     (draw: 11)
Coordinates:
  * draw        (draw) int64 88B 0 1 2 3 4 5 6 7 8 9 10
Data variables:
    train_iter  (draw) float32 44B 0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0
    train_loss  (draw) float32 44B 7.121 7.239 7.29 7.131 ... 7.106 7.103 7.045
    train_dt    (draw) float32 44B 15.84 0.9181 0.9123 ... 0.8967 0.9032 0.9024
    train_dtf   (draw) float32 44B 14.86 0.2994 0.2955 ... 0.2904 0.2923 0.2911
    train_dtb   (draw) float32 44B 0.9664 0.6067 0.6049 ... 0.5946 0.6003 0.6005
[2024-12-31 19:52:21.963271][INFO][vit.py:283] - Took 58.18 seconds
Application 66c85dca resources: utime=5443s stime=912s maxrss=4527808KB inblock=755188 oublock=712 minflt=18046931 majflt=6336043 nvcsw=562230 nivcsw=788543
took: 0h:01m:15s
```

</details>

</details>


## 📝 Example: FSDP

```bash
launch python3 -m mmm.examples.fsdp
```

<details closed><summary>Output:</summary>

<details closed><summary>Aurora @ ALCF:</summary>

Command:

```bash
launch python3 -Wignore -m mmm.examples.fsdp
```

Output:

```python
#[🐍 aurora_nre_models_frameworks-2024.2.1_u1](👻 aurora_nre_models_frameworks-2024.2.1_u1)
#[10:15:01 AM][x4404c5s2b0n0][/f/A/f/p/s/mmm][🌱 main][$!?][⏱️ 1m29s]
# $ launch python3 -Wignore -m mmm.examples.fsdp --epochs 10
Disabling local launch: multi-node application
Connected to tcp://x4509c0s2b0n0.hostmgmt2509.cm.aurora.alcf.anl.gov:7919
Found executable /flare/Aurora_deployment/foremans/projects/saforem2/mmm/venvs/aurora_nre_models_frameworks-2024.2.1_u1/bin/python3
Launching application d4145fbf-be87-460e-ac9a-366b2224df60
[2025-01-03 10:16:09,820] [INFO] [real_accelerator.py:222:get_accelerator] Setting ds_accelerator to xpu (auto detect)
# ...clipped...
[2025-01-03 10:16:11,118] [INFO] [real_accelerator.py:222:get_accelerator] Setting ds_accelerator to xpu (auto detect)
[2025-01-03 10:16:16.396841][INFO][__init__.py:146] - > initializing tensor parallel with size 1
[2025-01-03 10:16:16.399341][INFO][__init__.py:151] - > initializing context parallel with size 1
[2025-01-03 10:16:16.399726][INFO][__init__.py:156] - > initializing pipeline with size 1
[2025-01-03 10:16:16.400065][INFO][__init__.py:159] - > initializing ddp with size 48
2025:01:03-10:16:16:(99961) |CCL_WARN| value of CCL_LOG_LEVEL changed to be error (default:warn)
[2025-01-03 10:16:17.795345][INFO][dist.py:818] - Using device='xpu' with backend='DDP' + 'ccl' for distributed training.
[2025-01-03 10:16:17.796091][INFO][dist.py:859] - ['x4509c0s2b0n0'][ 0/47]
[2025-01-03 10:16:17.795365][INFO][dist.py:859] - ['x4509c0s2b0n0'][ 2/47]
[2025-01-03 10:16:17.795355][INFO][dist.py:859] - ['x4509c0s2b0n0'][ 3/47]
[2025-01-03 10:16:17.795344][INFO][dist.py:859] - ['x4509c0s2b0n0'][ 5/47]
[2025-01-03 10:16:17.795374][INFO][dist.py:859] - ['x4509c0s2b0n0'][ 1/47]
[2025-01-03 10:16:17.795396][INFO][dist.py:859] - ['x4509c0s2b0n0'][ 4/47]
[2025-01-03 10:16:17.795352][INFO][dist.py:859] - ['x4509c0s2b0n0'][ 6/47]
[2025-01-03 10:16:17.795331][INFO][dist.py:859] - ['x4509c0s2b0n0'][ 7/47]
[2025-01-03 10:16:17.795358][INFO][dist.py:859] - ['x4509c0s2b0n0'][ 8/47]
[2025-01-03 10:16:17.795331][INFO][dist.py:859] - ['x4509c0s2b0n0'][ 9/47]
[2025-01-03 10:16:17.795348][INFO][dist.py:859] - ['x4509c0s2b0n0'][10/47]
[2025-01-03 10:16:17.795332][INFO][dist.py:859] - ['x4509c0s2b0n0'][11/47]
[2025-01-03 10:16:17.795337][INFO][dist.py:859] - ['x4206c0s0b0n0'][20/47]
[2025-01-03 10:16:17.795322][INFO][dist.py:859] - ['x4206c0s0b0n0'][23/47]
[2025-01-03 10:16:17.795427][INFO][dist.py:859] - ['x4404c5s2b0n0'][36/47]
[2025-01-03 10:16:17.795368][INFO][dist.py:859] - ['x4206c0s0b0n0'][12/47]
[2025-01-03 10:16:17.795400][INFO][dist.py:859] - ['x4206c2s1b0n0'][24/47]
[2025-01-03 10:16:17.795402][INFO][dist.py:859] - ['x4404c5s2b0n0'][37/47]
[2025-01-03 10:16:17.795348][INFO][dist.py:859] - ['x4206c0s0b0n0'][13/47]
[2025-01-03 10:16:17.795375][INFO][dist.py:859] - ['x4206c2s1b0n0'][25/47]
[2025-01-03 10:16:17.795426][INFO][dist.py:859] - ['x4404c5s2b0n0'][38/47]
[2025-01-03 10:16:17.795356][INFO][dist.py:859] - ['x4206c0s0b0n0'][14/47]
[2025-01-03 10:16:17.795387][INFO][dist.py:859] - ['x4206c2s1b0n0'][26/47]
[2025-01-03 10:16:17.795350][INFO][dist.py:859] - ['x4206c0s0b0n0'][15/47]
[2025-01-03 10:16:17.795376][INFO][dist.py:859] - ['x4206c2s1b0n0'][27/47]
[2025-01-03 10:16:17.795391][INFO][dist.py:859] - ['x4206c0s0b0n0'][16/47]
[2025-01-03 10:16:17.795377][INFO][dist.py:859] - ['x4206c2s1b0n0'][28/47]
[2025-01-03 10:16:17.795341][INFO][dist.py:859] - ['x4206c0s0b0n0'][17/47]
[2025-01-03 10:16:17.795368][INFO][dist.py:859] - ['x4206c2s1b0n0'][29/47]
[2025-01-03 10:16:17.795429][INFO][dist.py:859] - ['x4206c0s0b0n0'][18/47]
[2025-01-03 10:16:17.795391][INFO][dist.py:859] - ['x4206c2s1b0n0'][30/47]
[2025-01-03 10:16:17.795406][INFO][dist.py:859] - ['x4404c5s2b0n0'][39/47]
[2025-01-03 10:16:17.795323][INFO][dist.py:859] - ['x4206c0s0b0n0'][19/47]
[2025-01-03 10:16:17.795362][INFO][dist.py:859] - ['x4206c2s1b0n0'][31/47]
[2025-01-03 10:16:17.795325][INFO][dist.py:859] - ['x4206c0s0b0n0'][21/47]
[2025-01-03 10:16:17.795381][INFO][dist.py:859] - ['x4206c2s1b0n0'][32/47]
[2025-01-03 10:16:17.795392][INFO][dist.py:859] - ['x4404c5s2b0n0'][43/47]
[2025-01-03 10:16:17.795338][INFO][dist.py:859] - ['x4206c0s0b0n0'][22/47]
[2025-01-03 10:16:17.795357][INFO][dist.py:859] - ['x4206c2s1b0n0'][33/47]
[2025-01-03 10:16:17.795403][INFO][dist.py:859] - ['x4404c5s2b0n0'][44/47]
[2025-01-03 10:16:17.795374][INFO][dist.py:859] - ['x4206c2s1b0n0'][34/47]
[2025-01-03 10:16:17.795396][INFO][dist.py:859] - ['x4404c5s2b0n0'][45/47]
[2025-01-03 10:16:17.795359][INFO][dist.py:859] - ['x4206c2s1b0n0'][35/47]
[2025-01-03 10:16:17.795402][INFO][dist.py:859] - ['x4404c5s2b0n0'][46/47]
[2025-01-03 10:16:17.795386][INFO][dist.py:859] - ['x4404c5s2b0n0'][47/47]
[2025-01-03 10:16:17.795442][INFO][dist.py:859] - ['x4404c5s2b0n0'][40/47]
[2025-01-03 10:16:17.795406][INFO][dist.py:859] - ['x4404c5s2b0n0'][41/47]
[2025-01-03 10:16:17.795542][INFO][dist.py:859] - ['x4404c5s2b0n0'][42/47]
[2025-01-03 10:16:18.127243][INFO][fsdp.py:168] -
=================================================================
Layer (type:depth-idx)                   Param #
=================================================================
Net                                      --
├─Conv2d: 1-1                            320
├─Conv2d: 1-2                            18,496
├─Dropout: 1-3                           --
├─Dropout: 1-4                           --
├─Linear: 1-5                            1,179,776
├─Linear: 1-6                            1,290
=================================================================
Total params: 1,199,882
Trainable params: 1,199,882
Non-trainable params: 0
=================================================================
[2025-01-03 10:16:18.158700][INFO][fsdp.py:187] - model=FullyShardedDataParallel(
  (_fsdp_wrapped_module): Net(
    (conv1): Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1))
    (conv2): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1))
    (dropout1): Dropout(p=0.25, inplace=False)
    (dropout2): Dropout(p=0.5, inplace=False)
    (fc1): Linear(in_features=9216, out_features=128, bias=True)
    (fc2): Linear(in_features=128, out_features=10, bias=True)
  )
)
[2025-01-03 10:16:31.371926][INFO][fsdp.py:228] - epoch=1 dt=12.708439 train_loss=0.907240 test_loss=0.251744 test_acc=92.723282
[2025-01-03 10:16:32.432899][INFO][fsdp.py:228] - epoch=2 dt=0.552378 train_loss=0.295683 test_loss=0.146544 test_acc=95.474480
[2025-01-03 10:16:33.919076][INFO][fsdp.py:228] - epoch=3 dt=1.008603 train_loss=0.196732 test_loss=0.108416 test_acc=96.700554
[2025-01-03 10:16:35.221881][INFO][fsdp.py:228] - epoch=4 dt=0.497740 train_loss=0.162368 test_loss=0.088102 test_acc=97.288673
[2025-01-03 10:16:37.285276][INFO][fsdp.py:228] - epoch=5 dt=1.478215 train_loss=0.140347 test_loss=0.078072 test_acc=97.647530
[2025-01-03 10:16:38.332123][INFO][fsdp.py:228] - epoch=6 dt=0.535241 train_loss=0.129397 test_loss=0.072069 test_acc=97.836922
[2025-01-03 10:16:39.187470][INFO][fsdp.py:228] - epoch=7 dt=0.456952 train_loss=0.122222 test_loss=0.069216 test_acc=97.876793
[2025-01-03 10:16:40.155302][INFO][fsdp.py:228] - epoch=8 dt=0.508528 train_loss=0.113515 test_loss=0.066888 test_acc=97.956535
[2025-01-03 10:16:41.000475][INFO][fsdp.py:228] - epoch=9 dt=0.504310 train_loss=0.112428 test_loss=0.065184 test_acc=98.006378
[2025-01-03 10:16:41.867020][INFO][fsdp.py:228] - epoch=10 dt=0.513946 train_loss=0.110064 test_loss=0.064282 test_acc=98.026314
[2025-01-03 10:16:41.868205][INFO][fsdp.py:234] - 11 epochs took 23.7s
[2025-01-03 10:16:42.037421][INFO][history.py:723] - Saving epoch plot to: /lus/flare/projects/Aurora_deployment/foremans/projects/saforem2/mmm/outputs/mmm-fsdp/plots/mplot
[2025-01-03 10:16:42.270616][INFO][history.py:723] - Saving dt plot to: /lus/flare/projects/Aurora_deployment/foremans/projects/saforem2/mmm/outputs/mmm-fsdp/plots/mplot
[2025-01-03 10:16:42.489108][INFO][history.py:723] - Saving train_loss plot to: /lus/flare/projects/Aurora_deployment/foremans/projects/saforem2/mmm/outputs/mmm-fsdp/plots/mplot
[2025-01-03 10:16:42.705195][INFO][history.py:723] - Saving test_loss plot to: /lus/flare/projects/Aurora_deployment/foremans/projects/saforem2/mmm/outputs/mmm-fsdp/plots/mplot
[2025-01-03 10:16:42.916880][INFO][history.py:723] - Saving test_acc plot to: /lus/flare/projects/Aurora_deployment/foremans/projects/saforem2/mmm/outputs/mmm-fsdp/plots/mplot
[2025-01-03 10:16:43.114870][INFO][history.py:603] - Saving tplots to /lus/flare/projects/Aurora_deployment/foremans/projects/saforem2/mmm/outputs/mmm-fsdp/plots/tplot
                           epoch [2025-01-03-101643]
    ┌─────────────────────────────────────────────────────────────────────┐
10.0┤                                                                   ▄▞│
    │                                                               ▄▄▀▀  │
    │                                                           ▗▄▀▀      │
 8.5┤                                                       ▗▄▞▀▘         │
    │                                                   ▗▄▞▀▘             │
    │                                               ▗▄▞▀▘                 │
 7.0┤                                            ▄▞▀▘                     │
    │                                        ▄▄▀▀                         │
 5.5┤                                   ▗▄▄▀▀                             │
    │                              ▗▄▄▀▀▘                                 │
    │                           ▄▄▀▘                                      │
 4.0┤                       ▄▄▀▀                                          │
    │                   ▄▄▀▀                                              │
    │               ▄▄▀▀                                                  │
 2.5┤           ▗▄▞▀                                                      │
    │       ▗▄▞▀▘                                                         │
    │    ▄▄▀▘                                                             │
 1.0┤▄▄▀▀                                                                 │
    └┬───────┬──────┬───────┬──────┬───────┬──────┬───────┬──────┬────────┘
     1       2      3       4      5       6      7       8      9
epoch
text saved in /lus/flare/projects/Aurora_deployment/foremans/projects/saforem2/mmm/outputs/mmm-fsdp/plots/tplot/epoch.txt
                            dt [2025-01-03-101643]
    ┌─────────────────────────────────────────────────────────────────────┐
12.7┤▌                                                                    │
    │▚                                                                    │
    │▝▖                                                                   │
10.7┤ ▚                                                                   │
    │ ▐                                                                   │
    │  ▌                                                                  │
 8.6┤  ▐                                                                  │
    │   ▌                                                                 │
 6.6┤   ▚                                                                 │
    │   ▝▖                                                                │
    │    ▚                                                                │
 4.5┤    ▐                                                                │
    │     ▌                                                               │
    │     ▐                                                               │
 2.5┤      ▌                                                              │
    │      ▚                                                              │
    │      ▝▖       ▖            ▄▄▞▄▄▖                                   │
 0.5┤       ▚▄▄▄▞▀▀▀▝▀▀▀▚▄▄▄▄▄▞▀▀     ▝▀▀▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄│
    └┬───────┬──────┬───────┬──────┬───────┬──────┬───────┬──────┬────────┘
     1       2      3       4      5       6      7       8      9
dt
text saved in /lus/flare/projects/Aurora_deployment/foremans/projects/saforem2/mmm/outputs/mmm-fsdp/plots/tplot/dt.txt
                        train_loss [2025-01-03-101643]
    ┌─────────────────────────────────────────────────────────────────────┐
0.91┤▌                                                                    │
    │▐                                                                    │
    │ ▌                                                                   │
0.77┤ ▐                                                                   │
    │  ▚                                                                  │
    │  ▝▖                                                                 │
0.64┤   ▚                                                                 │
    │   ▝▖                                                                │
0.51┤    ▚                                                                │
    │     ▌                                                               │
    │     ▐                                                               │
0.38┤      ▌                                                              │
    │      ▐                                                              │
    │       ▚                                                             │
0.24┤        ▀▚▄▖                                                         │
    │           ▝▀▚▄▖                                                     │
    │               ▝▀▀▀▚▄▄▄▖                                             │
0.11┤                       ▝▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄│
    └┬───────┬──────┬───────┬──────┬───────┬──────┬───────┬──────┬────────┘
     1       2      3       4      5       6      7       8      9
train_loss
text saved in /lus/flare/projects/Aurora_deployment/foremans/projects/saforem2/mmm/outputs/mmm-fsdp/plots/tplot/train_loss.txt
                          test_loss [2025-01-03-101643]
     ┌────────────────────────────────────────────────────────────────────┐
0.252┤▌                                                                   │
     │▝▖                                                                  │
     │ ▐                                                                  │
0.221┤  ▚                                                                 │
     │   ▌                                                                │
     │   ▝▖                                                               │
0.189┤    ▐                                                               │
     │     ▚                                                              │
0.158┤      ▌                                                             │
     │      ▝▖                                                            │
     │       ▝▄                                                           │
0.127┤         ▀▄                                                         │
     │           ▀▄                                                       │
     │             ▀▄▖                                                    │
0.096┤               ▝▀▄▄                                                 │
     │                   ▀▀▄▄                                             │
     │                       ▀▀▀▀▀▀▀▚▄▄▄                                  │
0.064┤                                  ▀▀▀▀▀▀▀▀▀▀▀▚▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄│
     └┬──────┬───────┬──────┬───────┬──────┬───────┬──────┬───────┬───────┘
      1      2       3      4       5      6       7      8       9
test_loss
text saved in /lus/flare/projects/Aurora_deployment/foremans/projects/saforem2/mmm/outputs/mmm-fsdp/plots/tplot/test_loss.txt
                          test_acc [2025-01-03-101643]
     ┌────────────────────────────────────────────────────────────────────┐
98.03┤                                     ▗▄▄▄▄▄▄▄▄▄▄▄▄▄▄▞▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀│
     │                           ▗▄▄▀▀▀▀▀▀▀▘                              │
     │                      ▗▄▄▀▀▘                                        │
97.14┤                   ▄▄▀▘                                             │
     │               ▄▄▀▀                                                 │
     │             ▗▞                                                     │
96.26┤           ▗▞▘                                                      │
     │         ▗▞▘                                                        │
95.37┤       ▗▞▘                                                          │
     │      ▗▘                                                            │
     │     ▗▘                                                             │
94.49┤     ▌                                                              │
     │    ▞                                                               │
     │   ▞                                                                │
93.61┤  ▐                                                                 │
     │ ▗▘                                                                 │
     │▗▘                                                                  │
92.72┤▌                                                                   │
     └┬──────┬───────┬──────┬───────┬──────┬───────┬──────┬───────┬───────┘
      1      2       3      4       5      6       7      8       9
test_acc
text saved in /lus/flare/projects/Aurora_deployment/foremans/projects/saforem2/mmm/outputs/mmm-fsdp/plots/tplot/test_acc.txt
[2025-01-03 10:16:43.287065][INFO][utils.py:132] - Saving dataset to: /lus/flare/projects/Aurora_deployment/foremans/projects/saforem2/mmm/outputs/mmm-fsdp/train_dataset.h5
[2025-01-03 10:16:43.333022][INFO][fsdp.py:252] - dataset=<xarray.Dataset> Size: 360B
Dimensions:     (draw: 10)
Coordinates:
  * draw        (draw) int64 80B 0 1 2 3 4 5 6 7 8 9
Data variables:
    epoch       (draw) int64 80B 1 2 3 4 5 6 7 8 9 10
    dt          (draw) float64 80B 12.71 0.5524 1.009 ... 0.5085 0.5043 0.5139
    train_loss  (draw) float32 40B 0.9072 0.2957 0.1967 ... 0.1135 0.1124 0.1101
    test_loss   (draw) float32 40B 0.2517 0.1465 0.1084 ... 0.06518 0.06428
    test_acc    (draw) float32 40B 92.72 95.47 96.7 97.29 ... 97.96 98.01 98.03
Application d4145fbf resources: utime=2668s stime=646s maxrss=2916524KB inblock=981664 oublock=2264 minflt=35730810 majflt=550491 nvcsw=1216582 nivcsw=1876932
took: 0h:00m:48s
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
