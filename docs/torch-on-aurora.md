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
diff --git a/torch/distributed/_composable/fsdp/_fsdp_state.py b/torch/distributed/_composable/fsdp/_fsdp_state.py
index ceb480fd239..c4fa73534ee 100644
--- a/torch/distributed/_composable/fsdp/_fsdp_state.py
+++ b/torch/distributed/_composable/fsdp/_fsdp_state.py
@@ -12,6 +12,7 @@ from typing import (
     Set,
     Tuple,
     TYPE_CHECKING,
+    Union,
 )
 
 import torch
@@ -56,7 +57,7 @@ class FSDPStateContext:
         self.is_last_backward: bool = True
         # Optional user-provided event recorded after optimizer for the
         # all-gather streams to wait on in the root pre-forward
-        self.post_optim_event: Optional[torch.cuda.Event] = None
+        self.post_optim_event: Optional[Union[torch.cuda.Event, torch.xpu.Event]] = None
 
 
 def disable_if_config_true(func):
@@ -127,7 +128,11 @@ class FSDPState(_State):
                 self._comm_ctx.all_gather_stream.wait_event(event)
                 self._state_ctx.post_optim_event = None
             else:
-                current_stream = torch.cuda.current_stream()
+                current_stream = (
+                    torch.cuda.current_stream() if torch.cuda.is_available()
+                    else torch.xpu.current_stream()
+                )
+
                 self._comm_ctx.all_gather_copy_in_stream.wait_stream(current_stream)
                 self._comm_ctx.all_gather_stream.wait_stream(current_stream)
             if self._device.type == "cuda":
@@ -291,9 +296,14 @@ class FSDPState(_State):
             if self._state_ctx.is_last_backward:
                 self._comm_ctx.post_forward_order.clear()
                 if self._comm_ctx.reduce_scatter_state is not None:
-                    torch.cuda.current_stream().wait_event(
-                        self._comm_ctx.reduce_scatter_state.event
-                    )
+                    if torch.cuda.is_available():
+                        torch.cuda.current_stream().wait_event(
+                            self._comm_ctx.reduce_scatter_state.event
+                        )
+                    elif torch.xpu.is_available():
+                        torch.xpu.current_stream().wait_event(
+                            self._comm_ctx.reduce_scatter_state.event
+                        )
                     self._comm_ctx.reduce_scatter_state = None
             self._state_ctx.post_backward_final_callback_queued = False

```

### Diff of `_fsdp_collectives.py`


```diff
diff --git a/torch/distributed/_composable/fsdp/_fsdp_collectives.py b/torch/distributed/_composable/fsdp/_fsdp_collectives.py
index 4e10f4594c1..45d66775d13 100644
--- a/torch/distributed/_composable/fsdp/_fsdp_collectives.py
+++ b/torch/distributed/_composable/fsdp/_fsdp_collectives.py
@@ -15,9 +15,28 @@ from ._fsdp_common import (
 from ._fsdp_param import FSDPParam, ShardedState
 
 
+Event = Union[torch.cuda.Event, torch.xpu.Event]
+Stream = Union[torch.cuda.Stream, torch.xpu.Stream]
+
+def get_current_stream() -> Stream:
+    if torch.cuda.is_available():
+        return torch.cuda.current_stream()
+    if torch.xpu.is_available():
+        return torch.xpu.current_stream()
+    raise RuntimeError("No CUDA or XPU device available")
+
+
+def get_stream(stream: Stream) -> Stream:
+    if torch.cuda.is_available():
+        return torch.cuda.stream(stream)
+    if torch.xpu.is_available():
+        return torch.xpu.stream(stream)
+    raise RuntimeError("No CUDA or XPU device available")
+
+
 class AllGatherResult(NamedTuple):
     all_gather_output: torch.Tensor
-    all_gather_event: Optional[torch.cuda.Event]
+    all_gather_event: Optional[Event]
     all_gather_work: Optional[dist.distributed_c10d.Work]
     # For each parameter, the all-gather input dtype for each input
     param_all_gather_input_dtypes: List[List[torch.dtype]]
@@ -113,6 +132,7 @@ lib.define(
 
 @torch.library.impl(lib, "chunk_cat", "Meta")
 @torch.library.impl(lib, "chunk_cat", "CUDA")
+@torch.library.impl(lib, "chunk_cat", "XPU")
 @torch.library.impl(lib, "chunk_cat", "CPU")
 def chunk_cat(
     tensors: List[torch.Tensor],
@@ -128,12 +148,12 @@ def foreach_all_gather(
     fsdp_params: List[FSDPParam],
     group: dist.ProcessGroup,
     async_op: bool,
-    all_gather_copy_in_stream: torch.cuda.Stream,
-    all_gather_stream: torch.cuda.Stream,
+    all_gather_copy_in_stream: Stream,
+    all_gather_stream: Stream,
     device: torch.device,
 ) -> Optional[AllGatherResult]:
     world_size, rank = group.size(), group.rank()
-    with torch.cuda.stream(all_gather_copy_in_stream):
+    with get_stream(all_gather_copy_in_stream):
         param_all_gather_inputs = _get_param_all_gather_inputs(fsdp_params)
         (
             param_all_gather_input_dtypes,
@@ -159,7 +179,7 @@ def foreach_all_gather(
         )
         del param_all_gather_inputs
     all_gather_stream.wait_stream(all_gather_copy_in_stream)
-    with torch.cuda.stream(all_gather_stream):
+    with get_stream(all_gather_stream):
         all_gather_work = dist.all_gather_into_tensor(
             output_tensor=all_gather_output,
             input_tensor=all_gather_input,
@@ -244,7 +264,7 @@ def foreach_all_gather_copy_out(
         all_gather_input_split_sizes,
     ) = all_gather_result
     if all_gather_event is not None:  # sync op
-        torch.cuda.current_stream().wait_event(all_gather_event)
+        get_current_stream().wait_event(all_gather_event)
     if isinstance(all_gather_work, dist.distributed_c10d.Work):  # async op
         all_gather_work.wait()
     world_size, device = group.size(), all_gather_output.device
@@ -282,16 +302,16 @@ def foreach_reduce(
     fsdp_params: List[FSDPParam],
     unsharded_grads: List[torch.Tensor],
     reduce_scatter_group: dist.ProcessGroup,
-    reduce_scatter_stream: torch.cuda.Stream,
+    reduce_scatter_stream: Stream,
     orig_dtype: torch.dtype,
     reduce_dtype: Optional[torch.dtype],
     device: torch.device,
     reduce_scatter_reduce_op: Optional[Union[dist.ReduceOp, dist.ReduceOp.RedOpType]],
     all_reduce_group: Optional[dist.ProcessGroup],  # not `None` iff HSDP
-    all_reduce_stream: torch.cuda.Stream,
+    all_reduce_stream: Stream,
     all_reduce_grads: bool,
     partial_reduce_output: Optional[torch.Tensor],  # only used for HSDP
-) -> Tuple[torch.Tensor, torch.cuda.Event, torch.cuda.Event, Optional[torch.Tensor]]:
+) -> Tuple[torch.Tensor, Event, Event, Optional[torch.Tensor]]:
     """
     ``unsharded_grads`` owns the references to the gradients computed by
     autograd, so clearing the list frees the gradients.
@@ -318,11 +338,23 @@ def foreach_reduce(
         (reduce_scatter_input_numel,), dtype=reduce_dtype, device=device
     )
     foreach_reduce_scatter_copy_in(unsharded_grads, reduce_scatter_input, world_size)
-    current_stream = torch.cuda.current_stream()
+    current_stream = get_current_stream()
     # Only after the copy-in finishes can we free the gradients
     unsharded_grads.clear()
     reduce_scatter_stream.wait_stream(current_stream)
-    with torch.cuda.stream(reduce_scatter_stream):
+    # with get_strem(reduce_scatter_stream):
+    if torch.cuda.is_available():
+        cm = torch.cuda.stream(reduce_scatter_stream)
+    elif torch.xpu.is_available():
+        # NOTE: The `predivide_factor` below is necessary since the `ReduceOp.AVG` is not supported on XPU.
+        # Explicitly, it crashes with:
+        #     RuntimeError: Cannot use ReduceOp.AVG with XPU
+        # TODO: Fix this / replace the predivide_factor by the size of the reduce_scatter_group
+        predivide_factor = reduce_scatter_group.size()
+        cm = torch.xpu.stream(reduce_scatter_stream)
+    else:
+        raise RuntimeError("No available device")
+    with cm:
         reduce_output = reduce_scatter_input.new_empty((reduce_scatter_output_numel,))
         _div_if_needed(reduce_scatter_input, predivide_factor)
         if reduce_scatter_reduce_op is None:
@@ -355,13 +387,27 @@ def foreach_reduce(
                 reduce_output += partial_reduce_output
             post_reduce_stream = all_reduce_stream
             all_reduce_stream.wait_stream(reduce_scatter_stream)
-            with torch.cuda.stream(all_reduce_stream):
+            if torch.cuda.is_available():
+                cm1 = torch.cuda.stream(all_reduce_stream)
+            elif torch.xpu.is_available():
+                cm1 = torch.xpu.stream(all_reduce_stream)
+            else:
+                raise RuntimeError("No available device")
+            # with get_stream(all_reduce_stream):
+            with cm1:
                 dist.all_reduce(
                     reduce_output,
                     group=all_reduce_group,
                     op=ReduceOp.AVG if predivide_factor is None else ReduceOp.SUM,
                 )
-    with torch.cuda.stream(post_reduce_stream):
+    # with get_stream(post_reduce_stream):
+    if torch.cuda.is_available():
+        cm2 = torch.cuda.stream(post_reduce_stream)
+    elif torch.xpu.is_available():
+        cm2 = torch.xpu.stream(post_reduce_stream)
+    else:
+        raise RuntimeError("No available device")
+    with cm2:
         _div_if_needed(reduce_output, postdivide_factor)
         reduce_output = _to_dtype_if_needed(reduce_output, orig_dtype)
         # View out and accumulate sharded gradients
```


### Diff of `_fsdp_param_group`


```diff
diff --git a/torch/distributed/_composable/fsdp/_fsdp_param_group.py b/torch/distributed/_composable/fsdp/_fsdp_param_group.py
index 3cb4a31c28d..351affaad98 100644
--- a/torch/distributed/_composable/fsdp/_fsdp_param_group.py
+++ b/torch/distributed/_composable/fsdp/_fsdp_param_group.py
@@ -1,7 +1,7 @@
 # mypy: allow-untyped-defs
 import contextlib
 import logging
-from typing import Any, cast, Dict, List, NamedTuple, Optional, Set, Tuple
+from typing import Any, Union, cast, Dict, List, NamedTuple, Optional, Set, Tuple
 
 import torch
 import torch._dynamo.compiled_autograd as ca
@@ -39,31 +39,90 @@ group free it after its copy-in. Finally, we have the last FSDP state flush the
 reference to avoid holding onto memory after forward.
 """
 
+Event = Union[torch.cuda.Event, torch.xpu.Event]
+Stream = Union[torch.cuda.Stream, torch.xpu.Stream]
+
+def get_current_stream() -> Stream:
+    if torch.cuda.is_available():
+        return torch.cuda.current_stream()
+    if torch.xpu.is_available():
+        return torch.xpu.current_stream()
+    raise RuntimeError("No CUDA or XPU device available")
+
+
+def get_stream(stream: Optional[Stream], priority: Optional[int] = None) -> Stream:
+    if torch.cuda.is_available():
+        if stream is None:
+            return torch.cuda.stream(priority)
+        return torch.cuda.stream(stream, priority)
+    if torch.xpu.is_available():
+        if stream is None:
+            return torch.xpu.stream(priority)
+        return torch.xpu.stream(stream, priority)
+    raise RuntimeError("No CUDA or XPU device available")
+
+
+def get_empty_stream() -> Stream:
+    if torch.cuda.is_available():
+        return torch.cuda.stream(None)
+    if torch.xpu.is_available():
+        return torch.xpu.stream(None)
+    raise RuntimeError("No CUDA or XPU device available")
+
+
+def get_event():
+    if torch.cuda.is_available():
+        return torch.cuda.Event()
+    if torch.xpu.is_available():
+        return torch.xpu.Event()
+    raise RuntimeError("No CUDA or XPU device available")
+
 
 class FSDPCommContext:
     """This has the communication state shared across FSDP states/parameter groups."""
 
     def lazy_init(self):
-        if not torch.cuda.is_available():
-            raise RuntimeError("FSDP requires CUDA for streams")
+        self.device_type = 'cuda'
+        if not torch.cuda.is_available() and torch.xpu.is_available():
+            logger.info(f'Using XPU for streams!!')
+            self.device_type = 'xpu'
+
+        # if not torch.cuda.is_available():
+        #     raise RuntimeError("FSDP requires CUDA for streams")
         # Setting the all-gather/reduce-scatter streams to be higher priority
         # can help avoid some issues where their copies in/out are delayed and
         # block computation (this is different from high-pri NCCL streams)
         high_priority = -1
-        # All-gather state and copy-in stream allow overlapping the next
-        # copy-in with the current all-gather in forward; copy-in overlaps with
-        # reduce-scatter in backward without the separate copy-in stream
-        self.all_gather_copy_in_stream = torch.cuda.Stream(priority=high_priority)
-        # All-gather stream allows overlapping next all-gather with current
-        # forward compute
-        self.all_gather_stream = torch.cuda.Stream(priority=high_priority)
-        # Reduce-scatter stream gives separate execution "thread" for post-
-        # backward logic like pre/post-gradient division and reduce-scatter
-        self.reduce_scatter_stream = torch.cuda.Stream(priority=high_priority)
-        # Run the HSDP all-reduces concurrently with all-gather/reduce-scatter
-        # since collectives use different network resources and can overlap
-        # in the typical intra-node sharding / inter-node replication case
-        self.all_reduce_stream = torch.cuda.Stream()
+        if self.device_type == 'cuda' and torch.cuda.is_available():
+            # All-gather state and copy-in stream allow overlapping the next
+            # copy-in with the current all-gather in forward; copy-in overlaps with
+            # reduce-scatter in backward without the separate copy-in stream
+            self.all_gather_copy_in_stream = torch.cuda.Stream(priority=high_priority)
+            # All-gather stream allows overlapping next all-gather with current
+            # forward compute
+            self.all_gather_stream = torch.cuda.Stream(priority=high_priority)
+            # Reduce-scatter stream gives separate execution "thread" for post-
+            # backward logic like pre/post-gradient division and reduce-scatter
+            self.reduce_scatter_stream = torch.cuda.Stream(priority=high_priority)
+            # Run the HSDP all-reduces concurrently with all-gather/reduce-scatter
+            # since collectives use different network resources and can overlap
+            # in the typical intra-node sharding / inter-node replication case
+            self.all_reduce_stream = torch.cuda.Stream()
+        elif self.device_type == 'xpu' and torch.xpu.is_available():
+            # All-gather state and copy-in stream allow overlapping the next
+            # copy-in with the current all-gather in forward; copy-in overlaps with
+            # reduce-scatter in backward without the separate copy-in stream
+            self.all_gather_copy_in_stream = torch.xpu.Stream(priority=high_priority)
+            # All-gather stream allows overlapping next all-gather with current
+            # forward compute
+            self.all_gather_stream = torch.xpu.Stream(priority=high_priority)
+            # Reduce-scatter stream gives separate execution "thread" for post-
+            # backward logic like pre/post-gradient division and reduce-scatter
+            self.reduce_scatter_stream = torch.xpu.Stream(priority=high_priority)
+            # Run the HSDP all-reduces concurrently with all-gather/reduce-scatter
+            # since collectives use different network resources and can overlap
+            # in the typical intra-node sharding / inter-node replication case
+            self.all_reduce_stream = torch.xpu.Stream()
         # All-gather/reduce-scatter states keep references to collective
         # tensors produced in one stream and used in another and accompanying
         # CUDA events for synchronization
@@ -78,19 +137,19 @@ class FSDPCommContext:
         if training_state in (TrainingState.FORWARD, TrainingState.PRE_BACKWARD):
             # Use separate streams for implicit prefetching
             return self.all_gather_copy_in_stream, self.all_gather_stream
-        current_stream = torch.cuda.current_stream()
+        current_stream = get_current_stream()
         return current_stream, current_stream
 
 
 # See [Note: Overlapping all-gather copy-in and all-gather]
 class AllGatherState(NamedTuple):
     all_gather_result: AllGatherResult
-    event: torch.cuda.Event  # all-gather copy-out
+    event: Event  # all-gather copy-out
 
 
 class ReduceScatterState(NamedTuple):
     reduce_scatter_input: torch.Tensor
-    event: torch.cuda.Event  # reduce-scatter event
+    event: Event  # reduce-scatter event
 
 
 class FSDPParamGroup:
@@ -162,10 +221,10 @@ class FSDPParamGroup:
         # Holds the reduce-scatter/all-reduce view-out CUDA event that marks the end of
         # the group's post-backward (e.g. reduce-scatter, all-reduce and div), which
         # should be waited on at the end of backward
-        self._post_reduce_event: Optional[torch.cuda.Event] = None
+        self._post_reduce_event: Optional[Event] = None
         # Holds the reshard-after-forward CUDA event when resharding to a
         # different world size, which should be waited on in the next unshard
-        self._reshard_after_forward_event: Optional[torch.cuda.Event] = None
+        self._reshard_after_forward_event: Optional[Event] = None
 
         # Only for HSDP, if accumulating gradients without all-reduce, save the
         # partial reduce output (only reduce-scattered but not all-reduced)
@@ -262,7 +321,7 @@ class FSDPParamGroup:
         for fsdp_param in self.fsdp_params:
             fsdp_param.init_unsharded_param()
         self._to_unsharded()
-        all_gather_copy_out_event = torch.cuda.Event()
+        all_gather_copy_out_event = get_event()
         all_gather_copy_out_event.record()
         if self._training_state == TrainingState.FORWARD:
             self.comm_ctx.all_gather_state = AllGatherState(
@@ -272,7 +331,7 @@ class FSDPParamGroup:
             self._wait_all_gather_streams_on_event(all_gather_copy_out_event)
         self._all_gather_result = None  # free unless saved in `all_gather_state`
 
-    def _wait_all_gather_streams_on_event(self, event: torch.cuda.Event):
+    def _wait_all_gather_streams_on_event(self, event: Event):
         # Calling `unshard` before lazy init means streams are not initialized
         if hasattr(self.comm_ctx, "all_gather_copy_in_stream"):
             self.comm_ctx.all_gather_copy_in_stream.wait_event(event)
@@ -285,7 +344,7 @@ class FSDPParamGroup:
                 return
             if self._use_post_forward_mesh:
                 self._to_sharded_post_forward()
-                self._reshard_after_forward_event = torch.cuda.Event()
+                self._reshard_after_forward_event = get_event()
                 self._reshard_after_forward_event.record()
                 return
         self._to_sharded()
@@ -365,7 +424,7 @@ class FSDPParamGroup:
             return
         with record_function(self._with_fqn("FSDP::post_backward_reduce")):
             if self.comm_ctx.reduce_scatter_state is not None:
-                torch.cuda.current_stream().wait_event(
+                get_current_stream().wait_event(
                     self.comm_ctx.reduce_scatter_state.event
                 )
                 self.comm_ctx.reduce_scatter_state = None
@@ -394,7 +453,7 @@ class FSDPParamGroup:
 
     def finalize_backward(self):
         if self._post_reduce_event is not None:
-            torch.cuda.current_stream().wait_event(self._post_reduce_event)
+            get_current_stream().wait_event(self._post_reduce_event)
             self._post_reduce_event = None
         for fsdp_param in self.fsdp_params:
             if fsdp_param.grad_offload_event is not None:
```

## ✅ Verify Fix


### 🦙 Llama-3.1-8B on Aurora

```bash
PYTORCH_ENABLE_XPU_FALLBACK=1 mpiexec -n 12 -np 12 python3 -m mmm.train --job.config_file train_configs/llama3_8b.toml | tee train-llama3-8b-$(tstamp).log
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


### 🦙 Llama-3.1-8B on Polaris

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
