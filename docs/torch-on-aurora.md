# Torch on Aurora

## Current Issue

With the most recent `pytorch==2.5` on Aurora, the following error is thrown
when trying to run the [`mmm/train.py`](src/mmm/train.py) script.

- Executing:

    ```bash
    #[🐍 anl_2024_12_release_2](👻 anl_2024_12_release_2)
    #[01:27:36 PM][x4512c5s2b0n0][/f/A/f/p/s/mmm][🌱 main][📦📝🤷✓] [⏱️ 19s]
    $ mpiexec -n 12 --hostfile nodefile python3 -m mmm.train --job.config_file train_configs/debug_model.toml
    ```

- Crashes with:

    ```python
    [rank0]:   File "/lus/flare/projects/Aurora_deployment/foremans/micromamba/envs/anl_2024_12_release_2/lib/python3.10/site-packages/torch/cuda/__init__.py", line 310, in _lazy_init
    [rank0]:     raise AssertionError("Torch not compiled with CUDA enabled")
    [rank0]: AssertionError: Torch not compiled with CUDA enabled
    ```

    - <details closed><summary>Full Traceback:</summary>

      ```python
      [rank0]: Traceback (most recent call last):
      [rank0]:   File "/lus/flare/projects/Aurora_deployment/foremans/micromamba/envs/anl_2024_12_release_2/lib/python3.10/runpy.py", line 196, in _run_module_as_main
      [rank0]:     return _run_code(code, main_globals, None,
      [rank0]:   File "/lus/flare/projects/Aurora_deployment/foremans/micromamba/envs/anl_2024_12_release_2/lib/python3.10/runpy.py", line 86, in _run_code
      [rank0]:     exec(code, run_globals)
      [rank0]:   File "/lus/flare/projects/Aurora_deployment/foremans/projects/saforem2/mmm/src/mmm/train.py", line 564, in <module>
      [rank0]:     main(config)
      [rank0]:   File "/lus/flare/projects/Aurora_deployment/foremans/micromamba/envs/anl_2024_12_release_2/lib/python3.10/site-packages/torch/distributed/elastic/multiprocessing/errors/__init__.py", line 355, in wrapper
      [rank0]:     return f(*args, **kwargs)
      [rank0]:   File "/lus/flare/projects/Aurora_deployment/foremans/projects/saforem2/mmm/src/mmm/train.py", line 387, in main
      [rank0]:     pred = model(input_ids)
      [rank0]:   File "/lus/flare/projects/Aurora_deployment/foremans/micromamba/envs/anl_2024_12_release_2/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1736, in _wrapped_call_impl
      [rank0]:     return self._call_impl(*args, **kwargs)
      [rank0]:   File "/lus/flare/projects/Aurora_deployment/foremans/micromamba/envs/anl_2024_12_release_2/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1844, in _call_impl
      [rank0]:     return inner()
      [rank0]:   File "/lus/flare/projects/Aurora_deployment/foremans/micromamba/envs/anl_2024_12_release_2/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1769, in inner
      [rank0]:     args_kwargs_result = hook(self, args, kwargs)  # type: ignore[misc]
      [rank0]:   File "/lus/flare/projects/Aurora_deployment/foremans/micromamba/envs/anl_2024_12_release_2/lib/python3.10/site-packages/torch/distributed/_composable/fsdp/_fsdp_state.py", line 67, in fsdp_hook_wrapper
      [rank0]:     return torch._dynamo.disable(func, recursive=True)(*args, **kwargs)
      [rank0]:   File "/lus/flare/projects/Aurora_deployment/foremans/micromamba/envs/anl_2024_12_release_2/lib/python3.10/site-packages/torch/_dynamo/eval_frame.py", line 632, in _fn
      [rank0]:     return fn(*args, **kwargs)
      [rank0]:   File "/lus/flare/projects/Aurora_deployment/foremans/micromamba/envs/anl_2024_12_release_2/lib/python3.10/site-packages/torch/distributed/_composable/fsdp/_fsdp_state.py", line 237, in _pre_forward
      [rank0]:     args, kwargs = self._fsdp_param_group.pre_forward(module, args, kwargs)
      [rank0]:   File "/lus/flare/projects/Aurora_deployment/foremans/micromamba/envs/anl_2024_12_release_2/lib/python3.10/site-packages/torch/distributed/_composable/fsdp/_fsdp_param_group.py", line 337, in pre_forward
      [rank0]:     self.unshard()
      [rank0]:   File "/lus/flare/projects/Aurora_deployment/foremans/micromamba/envs/anl_2024_12_release_2/lib/python3.10/site-packages/torch/distributed/_composable/fsdp/_fsdp_param_group.py", line 262, in unshard
      [rank0]:     self._all_gather_result = foreach_all_gather(
      [rank0]:   File "/lus/flare/projects/Aurora_deployment/foremans/micromamba/envs/anl_2024_12_release_2/lib/python3.10/site-packages/torch/utils/_contextlib.py", line 116, in decorate_context
      [rank0]:     return func(*args, **kwargs)
      [rank0]:   File "/lus/flare/projects/Aurora_deployment/foremans/micromamba/envs/anl_2024_12_release_2/lib/python3.10/site-packages/torch/distributed/_composable/fsdp/_fsdp_collectives.py", line 136, in foreach_all_gather
      [rank0]:     with torch.cuda.stream(all_gather_copy_in_stream):
      [rank0]:   File "/lus/flare/projects/Aurora_deployment/foremans/micromamba/envs/anl_2024_12_release_2/lib/python3.10/site-packages/torch/cuda/__init__.py", line 575, in __enter__
      [rank0]:     self.src_prev_stream = torch.cuda.current_stream(None)
      [rank0]:   File "/lus/flare/projects/Aurora_deployment/foremans/micromamba/envs/anl_2024_12_release_2/lib/python3.10/site-packages/torch/cuda/__init__.py", line 979, in current_stream
      [rank0]:     _lazy_init()
      [rank0]:   File "/lus/flare/projects/Aurora_deployment/foremans/micromamba/envs/anl_2024_12_release_2/lib/python3.10/site-packages/torch/cuda/__init__.py", line 310, in _lazy_init
      [rank0]:     raise AssertionError("Torch not compiled with CUDA enabled")
      [rank0]: AssertionError: Torch not compiled with CUDA enabled
      wandb:
      wandb: 🚀 View run hearty-water-4 at: https://wandb.ai/aurora_gpt/mmm.train/runs/s74wejlp
      wandb: Find logs at: ../../../../../../lus/flare/projects/Aurora_deployment/foremans/projects/saforem2/mmm/wandb/run-20250204_132848-s74wejlp/logs
      x4512c5s2b0n0: rank 6 exited with code 1
      x4512c5s2b0n0: rank 11 died from signal 15
      [1]    79314 exit 143   mpiexec -n 12 --hostfile nodefile python3 -m mmm.train --job.config_file
      took: 0h:00m:18s
      ```

    </details>

## Envionment Information


- PyTorch Version: `2.5.1+cxx11.abi`

    ```bash
    $ python3 -c 'import torch; print(torch.__version__); print(torch.xpu.is_available())'
    2.5.1+cxx11.abi
    True
    ```

- Relevant python packages:

    ```bash
    $ python3 -m pip list | egrep "deep|torch|intel|ccl"
    intel-cmplr-lib-rt          2025.0.4
    intel-cmplr-lib-ur          2025.0.4
    intel-cmplr-lic-rt          2025.0.4
    intel_extension_for_pytorch 2.5.10+xpu
    intel-opencl-rt             2025.0.4
    intel-openmp                2025.0.4
    intel-pti                   0.10.0
    intel-sycl-rt               2025.0.4
    oneccl                      2021.14.1
    oneccl-bind-pt              2.5.0+xpu
    oneccl-devel                2021.14.1
    torch                       2.5.1+cxx11.abi
    torchaudio                  2.5.1+cxx11.abi
    torchdata                   0.10.1
    torchinfo                   1.8.0
    torchvision                 0.20.1+cxx11.abi
    ```

- Loaded Modules:

    ```bash
    $ module list

    Currently Loaded Modules:
      1) libfabric/1.20.1
      2) cray-pals/1.4.0
      3) cray-libpals/1.4.0
      4) gcc-runtime/12.2.0-267awrk
      5) gmp/6.2.1-yctcuid
      6) mpfr/4.2.1-fhgnwe7
      7) mpc/1.3.1-ygprpb4
      8) gcc/12.2.0
      9) hwloc/master-git.1793e43-level-zero
      10) yaksa/0.3-aw2kkvy
      11) mpich/opt/4.3.0rc3
      12) intel_compute_runtime/release/1057.13
      13) oneapi/release/2025.0.5
    ```


## Fix

In order to get the full [`mmm/train.py`](src/mmm/train.py) script to work on
Aurora we need to make some changes to the pytorch source code.

In particular we need to change:

- `torch/distributed/composable/fsdp/_fsdp_state.py`
- `torch/distributed/composable/fsdp/_fsdp_collectives.py`

to fallback to using `torch.xpu.Stream`s when `torch.xpu.is_available()` (and `torch.cuda.is_available()` _is not_).

Explicitly, in my case, these were located at:

```bash
/lus/flare/projects/Aurora_deployment/foremans/micromamba/envs/anl_2024_12_release_2/lib/python3.10/site-packages/torch/distributed/_composable/fsdp/
```

### Diff of `_fsdp_state.py`


```diff
14a15
>     Union,
59c60
<         self.post_optim_event: Optional[torch.cuda.Event] = None
---
>         self.post_optim_event: Optional[Union[torch.cuda.Event, torch.xpu.Event]] = None
130c131,135
<                 current_stream = torch.cuda.current_stream()
---
>                 current_stream = (
>                     torch.cuda.current_stream() if torch.cuda.is_available()
>                     else torch.xpu.current_stream()
>                 )
> 
294,296c299,306
<                     torch.cuda.current_stream().wait_event(
<                         self._comm_ctx.reduce_scatter_state.event
<                     )
---
>                     if torch.cuda.is_available():
>                         torch.cuda.current_stream().wait_event(
>                             self._comm_ctx.reduce_scatter_state.event
>                         )
>                     elif torch.xpu.is_available():
>                         torch.xpu.current_stream().wait_event(
>                             self._comm_ctx.reduce_scatter_state.event
>                         )
```


### Diff of `_fsdp_collectives.py`

```diff
diff _fsdp_collectives_orig.py _fsdp_collectives_new.py
20c20
<     all_gather_event: Optional[torch.cuda.Event]
---
>     all_gather_event: Optional[Union[torch.cuda.Event, torch.xpu.Event]]
131,132c131,132
<     all_gather_copy_in_stream: torch.cuda.Stream,
<     all_gather_stream: torch.cuda.Stream,
---
>     all_gather_copy_in_stream: Union[torch.cuda.Stream, torch.xpu.Stream],
>     all_gather_stream: Union[torch.cuda.Stream, torch.xpu.Stream],
136c136,144
<     with torch.cuda.stream(all_gather_copy_in_stream):
---
>     if torch.cuda.is_available():
>         agc_cm = torch.cuda.stream(all_gather_copy_in_stream)
>     elif torch.xpu.is_available():
>         agc_cm = torch.xpu.stream(all_gather_copy_in_stream)
>     else:
>         raise RuntimeError("No available device")
>     with agc_cm:
162c170,182
<     with torch.cuda.stream(all_gather_stream):
---
>     if torch.cuda.is_available():
>         ags = torch.cuda.stream(all_gather_stream)
>     elif torch.xpu.is_available():
>         ags = torch.xpu.stream(all_gather_stream)
>     else:
>         raise RuntimeError("No available device")
>     with ags:
247c267,271
<         torch.cuda.current_stream().wait_event(all_gather_event)
---
>         if torch.cuda.is_available():
>             torch.cuda.current_stream().wait_event(all_gather_event)
>         elif torch.xpu.is_available():
>             torch.xpu.current_stream().wait_event(all_gather_event)
285c309
<     reduce_scatter_stream: torch.cuda.Stream,
---
>     reduce_scatter_stream: Union[torch.cuda.Stream, torch.xpu.Stream],
291c315
<     all_reduce_stream: torch.cuda.Stream,
---
>     all_reduce_stream: Union[torch.cuda.Stream, torch.xpu.Stream],
321c345,349
<     current_stream = torch.cuda.current_stream()
---
>     current_stream = (
>             torch.cuda.current_stream() if torch.cuda.is_available()
>             else torch.xpu.current_stream() if torch.xpu.is_available()
>             else None
>     )
325c353,361
<     with torch.cuda.stream(reduce_scatter_stream):
---
>     if torch.cuda.is_available():
>         cm = torch.cuda.stream(reduce_scatter_stream)
>     elif torch.xpu.is_available():
>         reduce_scatter_reduce_op = ReduceOp.SUM
>         cm = torch.xpu.stream(reduce_scatter_stream)
>     else:
>         raise RuntimeError("No available device")
>     with cm:
358c394,409
<             with torch.cuda.stream(all_reduce_stream):
---
>             if torch.cuda.is_available():
>                 cm1 = torch.cuda.stream(all_reduce_stream)
>             elif torch.xpu.is_available():
>                 cm1 = torch.xpu.stream(all_reduce_stream)
>             else:
>                 raise RuntimeError("No available device")
>             # with torch.cuda.stream(all_reduce_stream):
>             with cm1:
364c415,429
<     with torch.cuda.stream(post_reduce_stream):
---
>     if torch.cuda.is_available():
>         cm2 = torch.cuda.stream(post_reduce_stream)
>     elif torch.xpu.is_available():
>         cm2 = torch.xpu.stream(post_reduce_stream)
>     else:
>         raise RuntimeError("No available device")
>     # with torch.cuda.stream(post_reduce_stream):
>     with cm2:
```

## Working

```bash
$ PYTORCH_ENABLE_XPU_FALLBACK=1 WORLD_SIZE=12 yeet python3 -m mmm.train --job.config_file train_configs/llama3_8b.toml | tee train-llama3-8b-$(tstamp).log
```

<details closed><summary>Full Output</summary>


- Executable:

    ```bash
    #[🐍 anl_2024_12_release_2](👻 anl_2024_12_release_2)
    #[04:39:12 PM][x4706c6s1b0n0][/f/A/f/p/s/mmm][🌱 main][📦📝🤷✓]
    $ PYTORCH_ENABLE_XPU_FALLBACK=1 WORLD_SIZE=12 yeet python3 -m mmm.train --job.config_file train_configs/llama3_8b.toml | tee train-llama3-8b-$(tstamp).log
    Couldn't open /etc/cray/nid: No such file or directory
    Couldn't open /etc/cray/xname: No such file or directory
    Using local launch
    Found executable /flare/Aurora_deployment/foremans/projects/saforem2/mmm/venvs/anl_2024_12_release_2/bin/python3
    Launching application 1bfa8ab0-3232-4f65-ac6f-c635a95314f3
    Executing local shepherd /usr/sbin/palsd -s 6
    evaluating:
        mpiexec --verbose --envall -n 12 -ppn 12 --hostfile /var/spool/pbs/aux/2132687.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov --cpu-bind depth -d 8 --no-vni
    with arguments:
        python3 -m mmm.train --job.config_file train_configs/llama3_8b.toml
    ```

- Output:

    ```python
    [W204 16:39:26.120505479 OperatorEntry.cpp:155] Warning: Warning only once for all operators,  other operators may also be overridden.
      Overriding a previously registered kernel for the same operator and the same dispatch key
      operator: aten::_cummax_helper(Tensor self, Tensor(a!) values, Tensor(b!) indices, int dim) -> ()
        registered at /build/pytorch/build/aten/src/ATen/RegisterSchema.cpp:6
      dispatch key: XPU
      previous kernel: registered at /build/pytorch/build/aten/src/ATen/RegisterCPU.cpp:30476
          new kernel: registered at /build/intel-pytorch-extension/build/Release/csrc/gpu/csrc/aten/generated/ATen/RegisterXPU.cpp:2971 (function operator())
    [2025-02-04 16:39:36][I][datasets/config:54:datasets] PyTorch version 2.5.1+cxx11.abi available.
    [2025-02-04 16:39:36][I][datasets/config:125:datasets] JAX version 0.5.0 available.
    [2025-02-04 16:39:47][W][utils/_logger:68:ezpz.dist] Mismatch between os_world_size='12' and world_size=12
    [2025-02-04 16:39:47][I][mmm/train:50:__main__] Starting job: Llama 3 8B training
    [2025-02-04 16:39:47][W][utils/_logger:68:ezpz.dist] Mismatch between os_world_size='12' and world_size=12
    [2025-02-04 16:39:47][W][utils/_logger:68:ezpz.dist] Mismatch between os_world_size='12' and world_size=12
    [2025-02-04 16:39:47][W][utils/_logger:68:ezpz.dist] Mismatch between os_world_size='12' and world_size=12
    [2025-02-04 16:39:47][W][utils/_logger:68:ezpz.dist] Mismatch between os_world_size='12' and world_size=12
    [2025-02-04 16:39:47][W][utils/_logger:68:ezpz.dist] Mismatch between os_world_size='12' and world_size=12
    [2025-02-04 16:39:47][W][utils/_logger:68:ezpz.dist] Mismatch between os_world_size='12' and world_size=12
    [2025-02-04 16:39:47][W][utils/_logger:68:ezpz.dist] Mismatch between os_world_size='12' and world_size=12
    [2025-02-04 16:39:47][W][utils/_logger:68:ezpz.dist] Mismatch between os_world_size='12' and world_size=12
    [2025-02-04 16:39:47][W][utils/_logger:68:ezpz.dist] Mismatch between os_world_size='12' and world_size=12
    [2025-02-04 16:39:47][W][utils/_logger:68:ezpz.dist] Mismatch between os_world_size='12' and world_size=12
    [2025-02-04 16:39:47][W][utils/_logger:68:ezpz.dist] Mismatch between os_world_size='12' and world_size=12
    [2025-02-04 16:39:47][I][ezpz/dist:521] Using get_torch_device_type()='xpu' with backend='ccl'
    [2025-02-04 16:39:47][I][ezpz/dist:882] ['x4706c6s1b0n0'][ 3/11]
    [2025-02-04 16:39:47][I][ezpz/dist:882] ['x4706c6s1b0n0'][ 5/11]
    [2025-02-04 16:39:47][I][ezpz/dist:882] ['x4706c6s1b0n0'][ 2/11]
    [2025-02-04 16:39:47][I][ezpz/dist:882] ['x4706c6s1b0n0'][ 4/11]
    [2025-02-04 16:39:47][I][ezpz/dist:882] ['x4706c6s1b0n0'][11/11]
    [2025-02-04 16:39:47][I][ezpz/dist:882] ['x4706c6s1b0n0'][10/11]
    [2025-02-04 16:39:47][I][ezpz/dist:882] ['x4706c6s1b0n0'][ 1/11]
    [2025-02-04 16:39:47][I][ezpz/dist:882] ['x4706c6s1b0n0'][ 6/11]
    [2025-02-04 16:39:47][I][ezpz/dist:882] ['x4706c6s1b0n0'][ 7/11]
    [2025-02-04 16:39:47][I][ezpz/dist:882] ['x4706c6s1b0n0'][ 8/11]
    [2025-02-04 16:39:47][I][ezpz/dist:882] ['x4706c6s1b0n0'][ 9/11]
    [2025-02-04 16:39:47][I][ezpz/dist:832] Using device='xpu' with backend='DDP' + 'ccl' for distributed training.
    [2025-02-04 16:39:47][I][ezpz/dist:882] ['x4706c6s1b0n0'][ 0/11]
    [2025-02-04 16:39:47][I][ezpz/dist:1057] Setting up wandb from rank=0
    [2025-02-04 16:39:47][I][ezpz/dist:1058] Using=WB PROJECT=mmm.train
    wandb: Currently logged in as: foremans (aurora_gpt). Use `wandb login --relogin` to force relogin
    wandb: Using wandb-core as the SDK backend.  Please refer to https://wandb.me/wandb-core for more information.
    wandb: Tracking run with wandb version 0.19.4
    wandb: Run data is saved locally in /lus/flare/projects/Aurora_deployment/foremans/projects/saforem2/mmm/wandb/run-20250204_163947-cw24jgs6
    wandb: Run `wandb offline` to turn off syncing.
    wandb: Syncing run wise-water-22
    wandb: ⭐️ View project at https://wandb.ai/aurora_gpt/mmm.train
    wandb: 🚀 View run at https://wandb.ai/aurora_gpt/mmm.train/runs/cw24jgs6
    [2025-02-04 16:39:48][I][ezpz/dist:1083] W&B RUN=[wise-water-22](https://wandb.ai/aurora_gpt/mmm.train/runs/cw24jgs6)
    [2025-02-04 16:39:48][I][ezpz/dist:301] Updating wandb.run: wise-water-22 config with "DIST_INFO"
    [2025-02-04 16:39:48][I][ezpz/dist:1123] Running on machine='Aurora'
    [2025-02-04 16:39:48][I][mmm/utils:311] Process group already initialized, skipping init_process_group
    [2025-02-04 16:39:48][W][utils/_logger:68:mmm.utils] Peak flops undefined for: xpu, fallback to A100
    [2025-02-04 16:39:48][I][mmm/train:110:__main__] Peak FLOPS used for computing MFU: 3.120e+14
    [2025-02-04 16:39:48][I][parallelisms/parallel_dims:80:mmm.parallelisms.parallel_dims] Building 1-D device mesh with ['dp_shard'], [12]
    [2025-02-04 16:39:49][I][tokenizer/__init__:16:mmm.data.tokenizer] Building tiktoken tokenizer locally from ./tests/assets/test_tiktoken.model
    [2025-02-04 16:39:50][I][tokenizer/tiktoken:91:mmm.data.tokenizer.tiktoken] TikTokenizer built: #words 2256, BOS ID 2000, EOS ID 2001
    [2025-02-04 16:39:50][I][data/hf_datasets:69:mmm.data.hf_datasets] Preparing c4 dataset from allenai/c4
    [2025-02-04 16:40:04][I][mmm/train:166:__main__] Building llama3 8B with ModelArgs(dim=4096, n_layers=32, n_heads=32, n_kv_heads=8, vocab_size=2256, multiple_of=1024, ffn_dim_multiplier=1.3, norm_eps=1e-05, rope_theta=500000, max_seq_len=8192, depth_init=True, norm_type='rmsnorm')
    [2025-02-04 16:40:05][I][mmm/train:184:__main__] Model llama3 8B size: 6,998,069,248 total parameters
    [2025-02-04 16:40:05][I][parallelisms/parallelize_llama:321:mmm.parallelisms.parallelize_llama] Applied selective activation checkpointing to the model
    [2025-02-04 16:40:05][I][parallelisms/parallelize_llama:119:mmm.parallelisms.parallelize_llama] Applied FSDP to the model
    2025:02:04-16:40:07:(44763) |CCL_WARN| value of CCL_OP_SYNC changed to be 1 (default:0)
    2025:02:04-16:40:07:(44763) |CCL_WARN| value of CCL_PROCESS_LAUNCHER changed to be pmix (default:hydra)
    [2025-02-04 16:40:10][E][mmm/train:254:__main__] Error getting memory stats: name 'device_memory_monitor' is not defined
    [2025-02-04 16:40:10][I][mmm/metrics:125] TensorBoard logging enabled. Logs will be saved at ./outputs/tb/20250204-1640
    [2025-02-04 16:40:10][E][mmm/train:313:__main__] Error resetting memory stats: name 'device_memory_monitor' is not defined
    [2025-02-04 16:40:10][I][mmm/train:317:__main__] Training starts at step 1, with local batch size 1, global batch size 12, sequence length 8192, total steps 1000 (warmup 200)
    [2025-02-04 16:40:10][I][mmm/profiling:60] Profiling active. Traces will be saved at ./outputs/profile_trace
    [rank0]:W0204 16:40:11.694000 44763 site-packages/torch/distributed/_composable/fsdp/_fsdp_param_group.py:50] Using XPU for Streams!
    [rank1]:[W204 16:40:13.733434151 RegisterXPU.cpp:7614] Warning: Aten Op fallback from XPU to CPU happends. This may have performance implications. If need debug the fallback ops please set environment variable `PYTORCH_DEBUG_XPU_FALLBACK=1`  (function operator())
    read_signalfd: got signal 18 (Continued)
    Forwarding signal 18
    read_signalfd: got signal 18 (Continued)
    Forwarding signal 18
    [2025-02-04 16:41:22][E][mmm/train:473:__main__] Error getting memory stats: name 'device_memory_monitor' is not defined
    [2025-02-04 16:41:22][I][mmm/train:493:__main__] step=1 global_avg_loss=8.214795 global_max_loss=8.254231 throughput(tps)=112.507887 mfu(%)=1.976745 end_to_end(s)=7.281267 data_loading(s)=1.644477 data_loading(%)=2.258504
    [2025-02-04 16:41:22][E][mmm/train:536:__main__] Error resetting memory stats: name 'device_memory_monitor' is not defined
    [2025-02-04 16:41:22][I][mmm/utils:180] Synchronizing and adjusting timeout for all ProcessGroups to 0:01:40
    /lus/flare/projects/Aurora_deployment/foremans/micromamba/envs/anl_2024_12_release_2/lib/python3.10/site-packages/torch/distributed/distributed_c10d.py:1354: UserWarning: Set
    timeout is now only supported for either nccl or gloo.
      warnings.warn("Set timeout is now only supported for either nccl or gloo.")
    read_signalfd: got signal 18 (Continued)
    Forwarding signal 18
    read_signalfd: got signal 18 (Continued)
    Forwarding signal 18
    [2025-02-04 16:45:57][I][mmm/train:493:__main__] step=10 global_avg_loss=6.886052 global_max_loss=8.090250 throughput(tps)=268.175160 mfu(%)=4.711792 end_to_end(s)=27.492479 data_loading(s)=0.007497 data_loading(%)=0.024541
    ```

   </details>


# Running on Polaris

### LLama3 on Polaris

I've confirmed independently that this code also runs as expected on Polaris.

See [🦙 Llama 3](https://github.com/saforem2/mmm/blob/main/README.md#-llama3)
for more details.

- Launch training:

    ```bash
    launch python3 -m mmm.train --job.config_file train_configs/llama3_8b.toml --training.seq_len=2048
    ```

    Note that `--training.seq_len=2048` is needed since `4096` didn't fit on
    the 40GB A100s of Polaris 

    <details closed><summary>Output:</summary>

    <details closed><summary>Polaris @ ALCF:</summary>

    ```bash
    launch python3 -Wignore -m mmm.train --job.config_file train_configs/llama3_8b.toml --training.seq_len=2048 | tee train-llama3-8b.log
    ```

    ```python
    Connected to tcp://x3005c0s37b1n0.hsn.cm.polaris.alcf.anl.gov:7919
    Found executable /eagle/argonne_tpc/foremans/projects/saforem2/mmm/venvs/2024-04-29/bin/python3
    Launching application a95db38c-5a1f-4f20-a362-6cd48cc0f009
    Using PMI port 40193,40194
    [2025-02-03 09:19:25][I][datasets/config:54:datasets] PyTorch version 2.6.0+cu124 available.
    [2025-02-03 09:19:25][I][datasets/config:112:datasets] TensorFlow version 2.16.1 available.
    [2025-02-03 09:19:25][I][datasets/config:125:datasets] JAX version 0.4.26 available.
    [2025-02-03 09:19:41][I][mmm/train:71:__main__] Starting job: Llama 3 8B training
    [rank4]:[W203 09:19:41.475782740 ProcessGroupNCCL.cpp:4561] [PG ID 0 PG GUID 0 Rank 4]  using GPU 0 to perform barrier as devices used by this process are currently unknown. This can potentially cause a hang if this rank to GPU mapping is incorrect. Specify device_ids in barrier() to force use of a particular device, or call init_process_group() with a device_id.
    [rank0]:[W203 09:19:41.501737516 ProcessGroupNCCL.cpp:4561] [PG ID 0 PG GUID 0 Rank 0]  using GPU 0 to perform barrier as devices used by this process are currently unknown. This can potentially cause a hang if this rank to GPU mapping is incorrect. Specify device_ids in barrier() to force use of a particular device, or call init_process_group() with a device_id.
    [rank7]:[W203 09:19:42.070718085 ProcessGroupNCCL.cpp:4561] [PG ID 0 PG GUID 0 Rank 7]  using GPU 3 to perform barrier as devices used by this process are currently unknown. This can potentially cause a hang if this rank to GPU mapping is incorrect. Specify device_ids in barrier() to force use of a particular device, or call init_process_group() with a device_id.
    [rank5]:[W203 09:19:42.071337647 ProcessGroupNCCL.cpp:4561] [PG ID 0 PG GUID 0 Rank 5]  using GPU 1 to perform barrier as devices used by this process are currently unknown. This can potentially cause a hang if this rank to GPU mapping is incorrect. Specify device_ids in barrier() to force use of a particular device, or call init_process_group() with a device_id.
    [rank6]:[W203 09:19:42.071862543 ProcessGroupNCCL.cpp:4561] [PG ID 0 PG GUID 0 Rank 6]  using GPU 2 to perform barrier as devices used by this process are currently unknown. This can potentially cause a hang if this rank to GPU mapping is incorrect. Specify device_ids in barrier() to force use of a particular device, or call init_process_group() with a device_id.
    [rank2]:[W203 09:19:42.067891022 ProcessGroupNCCL.cpp:4561] [PG ID 0 PG GUID 0 Rank 2]  using GPU 2 to perform barrier as devices used by this process are currently unknown. This can potentially cause a hang if this rank to GPU mapping is incorrect. Specify device_ids in barrier() to force use of a particular device, or call init_process_group() with a device_id.
    [rank3]:[W203 09:19:42.068768126 ProcessGroupNCCL.cpp:4561] [PG ID 0 PG GUID 0 Rank 3]  using GPU 3 to perform barrier as devices used by this process are currently unknown. This can potentially cause a hang if this rank to GPU mapping is incorrect. Specify device_ids in barrier() to force use of a particular device, or call init_process_group() with a device_id.
    [rank1]:[W203 09:19:42.069066595 ProcessGroupNCCL.cpp:4561] [PG ID 0 PG GUID 0 Rank 1]  using GPU 1 to perform barrier as devices used by this process are currently unknown. This can potentially cause a hang if this rank to GPU mapping is incorrect. Specify device_ids in ba
    [2025-02-03 09:19:43][I][ezpz/dist:823] Using device='cuda' with backend='DDP' + 'nccl' for distributed training.
    [2025-02-03 09:19:43][I][ezpz/dist:869] ['x3005c0s37b1n0'][2/7]
    [2025-02-03 09:19:43][I][ezpz/dist:869] ['x3005c0s37b1n0'][1/7]
    [2025-02-03 09:19:43][I][ezpz/dist:869] ['x3005c0s7b0n0'][4/7]
    [2025-02-03 09:19:43][I][ezpz/dist:869] ['x3005c0s7b0n0'][6/7]
    [2025-02-03 09:19:43][I][ezpz/dist:869] ['x3005c0s7b0n0'][5/7]
    [2025-02-03 09:19:43][I][ezpz/dist:869] ['x3005c0s37b1n0'][3/7]
    [2025-02-03 09:19:43][I][ezpz/dist:869] ['x3005c0s7b0n0'][7/7]
    [2025-02-03 09:19:43][I][ezpz/dist:869] ['x3005c0s37b1n0'][0/7]
    
    [2025-02-03 09:19:43][I][mmm/utils:298] Process group already initialized, skipping init_process_group
    [2025-02-03 09:19:43][W][mmm/utils:361] Peak flops undefined for: cuda, fallback to A100
    [2025-02-03 09:19:43][I][mmm/train:113:__main__] Peak FLOPS used for computing MFU: 3.120e+14
    [2025-02-03 09:19:43][I][parallelisms/parallel_dims:80:mmm.parallelisms.parallel_dims] Building 1-D device mesh with ['dp_shard'], [8]
    [2025-02-03 09:19:43][I][tokenizer/__init__:18:mmm.data.tokenizer] Building tiktoken tokenizer locally from ./tests/assets/test_tiktoken.model
    [2025-02-03 09:19:44][I][tokenizer/tiktoken:92:mmm.data.tokenizer.tiktoken] TikTokenizer built: #words 128256, BOS ID 128000, EOS ID 128001
    [2025-02-03 09:19:44][I][data/hf_datasets:73:mmm.data.hf_datasets] Preparing c4 dataset from allenai/c4
    [2025-02-03 09:19:49][I][mmm/train:165:__main__] Building llama3 8B with ModelArgs(dim=4096, n_layers=32, n_heads=32, n_kv_heads=8, vocab_size=128256, multiple_of=1024, ffn_dim_multiplier=1.3, norm_eps=1e-05, rope_theta=500000, max_seq_len=2048, depth_init=True, norm_type='rmsnorm')
    [2025-02-03 09:19:50][I][mmm/train:181:__main__] Model llama3 8B size: 8,030,261,248 total parameters
    [2025-02-03 09:19:50][I][parallelisms/parallelize_llama:309:mmm.parallelisms.parallelize_llama] Applied selective activation checkpointing to the model
    [2025-02-03 09:19:50][I][parallelisms/parallelize_llama:114:mmm.parallelisms.parallelize_llama] Applied FSDP to the model
    [2025-02-03 09:19:51][E][mmm/train:241:__main__] Error getting memory stats: name 'device_memory_monitor' is not defined
    [2025-02-03 09:19:51][I][mmm/metrics:128] TensorBoard logging enabled. Logs will be saved at ./outputs/tb/20250203-0919
    [2025-02-03 09:19:51][E][mmm/train:296:__main__] Error resetting memory stats: name 'device_memory_monitor' is not defined
    [2025-02-03 09:19:51][I][mmm/train:300:__main__] Training starts at step 1, with local batch size 1, global batch size 8, sequence length 2048, total steps 1000 (warmup 200)
    [2025-02-03 09:19:51][I][mmm/profiling:59] Profiling active. Traces will be saved at ./outputs/profile_trace
    [2025-02-03 09:20:14][E][mmm/train:438:__main__] Error getting memory stats: name 'device_memory_monitor' is not defined
    [2025-02-03 09:20:14][I][mmm/train:458:__main__] step=1 global_avg_loss=12.269001 global_max_loss=12.310555 throughput(tps)=86.019928 mfu(%)=1.330297 end_to_end(s)=2.380844 data_loading(s)=0.900774 data_loading(%)=3.783424
    [2025-02-03 09:20:14][E][mmm/train:498:__main__] Error resetting memory stats: name 'device_memory_monitor' is not defined
    [2025-02-03 09:20:14][I][mmm/utils:171] Synchronizing and adjusting timeout for all ProcessGroups to 0:01:40
    
    [2025-02-03 09:22:59][I][mmm/train:458:__main__] step=10 global_avg_loss=11.101408 global_max_loss=12.190623 throughput(tps)=112.290594 mfu(%)=1.736573 end_to_end(s)=16.414554 data_loading(s)=0.003078 data_loading(%)=0.016878
    [2025-02-03 09:26:05][I][mmm/train:458:__main__] step=20 global_avg_loss=9.304403 global_max_loss=10.041307 throughput(tps)=110.054333 mfu(%)=1.701989 end_to_end(s)=18.608990 data_loading(s)=0.002827 data_loading(%)=0.015190
    [2025-02-03 09:29:13][I][mmm/train:458:__main__] step=30 global_avg_loss=8.363521 global_max_loss=10.094500 throughput(tps)=109.026794 mfu(%)=1.686098 end_to_end(s)=18.784373 data_loading(s)=0.003182 data_loading(%)=0.016937
    [2025-02-03 09:32:18][I][mmm/train:458:__main__] step=40 global_avg_loss=7.754060 global_max_loss=10.717106 throughput(tps)=110.327183 mfu(%)=1.706209 end_to_end(s)=18.562968 data_loading(s)=0.003432 data_loading(%)=0.018489
    [2025-02-03 09:35:28][I][mmm/train:458:__main__] step=50 global_avg_loss=7.405908 global_max_loss=11.071878 throughput(tps)=108.140150 mfu(%)=1.672386 end_to_end(s)=18.938387 data_loading(s)=0.002782 data_loading(%)=0.014690
    [2025-02-03 09:38:32][I][mmm/train:458:__main__] step=60 global_avg_loss=7.379553 global_max_loss=16.768843 throughput(tps)=111.114522 mfu(%)=1.718385 end_to_end(s)=18.431434 data_loading(s)=0.002796 data_loading(%)=0.015172
    [2025-02-03 09:41:36][I][mmm/train:458:__main__] step=70 global_avg_loss=7.180877 global_max_loss=8.136713 throughput(tps)=111.323097 mfu(%)=1.721611 end_to_end(s)=18.396901 data_loading(s)=0.003316 data_loading(%)=0.018022
    [2025-02-03 09:44:40][I][mmm/train:458:__main__] step=80 global_avg_loss=6.949995 global_max_loss=8.055350 throughput(tps)=111.244684 mfu(%)=1.720398 end_to_end(s)=18.409868 data_loading(s)=0.003094 data_loading(%)=0.016804
    [2025-02-03 09:47:47][I][mmm/train:458:__main__] step=90 global_avg_loss=6.984058 global_max_loss=8.244547 throughput(tps)=109.736104 mfu(%)=1.697068 end_to_end(s)=18.662955 data_loading(s)=0.003001 data_loading(%)=0.016082
    [2025-02-03 09:50:52][I][mmm/train:458:__main__] step=100 global_avg_loss=6.869817 global_max_loss=8.179618 throughput(tps)=110.670108 mfu(%)=1.711512 end_to_end(s)=18.505449 data_loading(s)=0.008446 data_loading(%)=0.045638
    ```

    </details>

  </details>


## Complete Install Instructions on Aurora

```bash
# install micromamba
"${SHELL}" <(curl -L micro.mamba.pm/install.sh)

# create a new environment
mm create --prefix "/flare/Aurora_deployment/foremans/micromamba/${ENV_NAME}" python=3.12 -y
mm activate /lus/flare/projects/Aurora_deployment/foremans/micromamba/anl_2024_12_release_2.1
mm install -y git cmake ninja

# install mpi4py
gh repo clone mpi4py/mpi4py /tmp/mpi4py
cd /tmp/mpi4py/
python3 setup.py develop |& tee build.log
python3 setup.py Install

# install requirements
python3 -m pip install fixedint pudb flake8 regex pybind11 einops six transformers numpy setuptools

# install torch
USE_XPU=1 python3 -m pip install torch==2.5.1+xpu intel-extension-for-pytorch==2.5.1+xpu oneccl_bind_pt==2.5.1+xpu --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/

# uninstall intel-mpi runtime
python3 -m pip uninstall impi_rt

# load modules
module unload oneapi/eng-compiler/2024.07.30.002
module use /opt/aurora/24.180.3/spack/unified/0.8.0/install/modulefiles/oneapi/2024.07.30.002
module use /soft/preview/pe/24.347.0-RC2/modulefiles
module add oneapi/release

# export environment variables
export ZE_ENABLE_PCI_ID_DEVICE_ORDER=1
export CCL_PROCESS_LAUNCHER=pmix
export PALS_PMI=pmix
export CCL_ATL_TRANSPORT=mpi
export CCL_OP_SYNC=1
export FI_PROVIDER="cxi,tcp;ofi_rxm"
export I_MPI_OFI_LIBRARY="/opt/cray/libfabric/1.20.1/lib64/libfabric.so.1"
export ZE_FLAT_DEVICE_HIERARCHY=FLAT

# install deepspeed
python3 -m pip install deepspeed
```

### 🍋 Install `ezpz`

```bash
# ezpz_setup_env
source <(curl -s https://raw.githubusercontent.com/saforem2/ezpz/refs/heads/main/src/ezpz/bin/utils.sh)
ezpz_setup_env

# install ezpz
python3 -m pip install -e "git+https://github.com/saforem2/ezpz#egg=ezpz"

# run simple DDP test
launch python3 -m ezpz.test_dist
```


### Issue with Multi-Node MPICH (?)

There seems to be a subtle issue somehow related to multi-node MPICH.

I thought previously the issue might've been due to my conda environment living
in my home directory and thought I forgot to specify the `home` filesystem in
my job submission (thereby preventing the other nodes from seeing it), but
after trying again this morning with my environment explicitly placed on
`/flare/`, I no longer believe this is the case.

Will have to look into this further.

- [x] Works:

    ```bash
    mpiexec -n 12 -ppn 12 python3 -m ezpz.test_dist
    ```


- [ ] Hangs:

    ```bash
    launch python3 -m ezpz.test_dist
    ```

  🤔 hmmmm.......
