"""
mmm/configs.py
"""

import argparse
from typing import Optional
import ezpz

import sys
import torch
from dataclasses import dataclass, field

from collections import defaultdict

try:
    import tomllib  # type:ignore
except ModuleNotFoundError:
    import tomli as tomllib


logger = ezpz.get_logger(__name__)

TORCH_DTYPES_MAP = {
    'bf16': torch.bfloat16,
    'bfloat16': torch.bfloat16,
    'fp16': torch.float16,
    'float16': torch.float16,
    'half': torch.float16,
    'fp32': torch.float32,
    'float32': torch.float32,
}


def string_list(raw_arg: str) -> list[str]:
    return raw_arg.split(',')


@dataclass
class SoftCapAttention:
    dim: int
    num_heads: int

    def __post_init__(self):
        self.head_dim = self.dim // self.num_heads
        self.scale = self.head_dim**-0.5

    def __call__(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        # apply soft-capping
        attn = 30 * torch.tanh(attn / 30)
        attn = attn.softmax(dim=-1)
        return attn @ v


@dataclass
class ViTConfig:
    img_size: int = 224
    batch_size: int = 128
    num_heads: int = 16
    head_dim: int = 64
    depth: int = 24
    patch_size: int = 16

    def __post_init__(self):
        self.seq_len = (self.img_size // self.patch_size) ** 2  # 196, default


@dataclass
class TrainArgs:
    img_size: int
    batch_size: int
    num_heads: int
    head_dim: int
    depth: int
    patch_size: int
    dtype: str
    compile: bool
    attn_type: str
    num_workers: int
    max_iters: int
    format: Optional[str] = field(default_factory=str)
    cuda_sdpa_backend: Optional[str] = field(default_factory=str)


@dataclass
class ProfilingConfig:
    enable_profiling: bool = False
    save_traces_folder: str = 'profile_traces'
    profile_freq: int = 10
    enable_memory_snapshot: bool = False
    save_memory_snapshot_folder: str = 'memory_snapshot'


@dataclass
class MetricsConfig:
    log_freq: int = 10
    enable_tensorboard: bool = False
    disable_color_printing: bool = False
    save_tb_folder: str = 'tb'
    rank_0_only: bool = True
    enable_wandb: bool = False


@dataclass
class ModelConfig:
    name: str = 'llama'
    flavor: str = 'debugmodel'
    norm_type: str = 'rmsnorm'
    tokenizer_path: str = 'data/tokenizer/tokenizer.model'


@dataclass
class OptimizerConfig:
    name: str = 'AdamW'
    lr: float = 8e-4
    fused: bool = False
    early_step_in_backward: bool = False


@dataclass
class TrainingConfig:
    dataset: str = 'c4_mini'
    dataset_path: Optional[str] = None
    batch_size: int = 8
    seq_len: int = 2048
    warmup_steps: int = 200
    max_norm: float = 1.0
    steps: int = 10000
    data_parallel_replicate_degree: int = 1
    data_parallel_shard_degree: int = -1
    enable_cpu_offload: bool = False
    tensor_parallel_degree: int = 1
    disable_loss_parallel: bool = False
    enable_async_tensor_parallel: bool = False
    pipeline_parallel_degree: int = 1
    pipeline_parallel_split_points: list[str] = field(default_factory=list)
    pipeline_parallel_schedule: str = '1F1B'
    pipeline_parallel_schedule_csv: str = ''
    pipeline_parallel_microbatches: Optional[int] = None
    enable_compiled_autograd: bool = False
    context_parallel_degree: int = 1
    context_parallel_rotate_method: str = 'allgather'
    mixed_precision_param: str = 'bfloat16'
    mixed_precision_reduce: str = 'float32'
    compile: bool = False
    gc_freq: int = 50
    seed: Optional[int] = None
    deterministic: bool = False


@dataclass
class CheckpointConfig:
    enable_checkpoint: bool = False
    folder: str = 'checkpoint'
    interval_type: str = 'steps'
    interval: int = 500
    model_weights_only: bool = False
    export_dtype: str = 'float32'
    create_seed_checkpoint: bool = False
    async_mode: str = 'disabled'
    keep_latest_k: int = 0
    load_step: int = -1


@dataclass
class Float8Config:
    enable_float8_linear: bool = False
    enable_fsdp_float8_all_gather: bool = False
    precompute_float8_dynamic_scale_for_fsdp: bool = False
    scaling_type_input: str = 'dynamic'
    scaling_type_weight: str = 'dynamic'
    scaling_type_grad_output: str = 'dynamic'


@dataclass
class CommConfig:
    init_timeout_seconds: int = 300
    train_timeout_seconds: int = 100
    trace_buf_size: int = 20000


@dataclass
class MemoryEstimationConfig:
    enabled: bool = False
    disable_fake_mode: bool = False


@dataclass
class JobConfig:
    """
    A helper class to manage the train configuration.

    Semantics:
    - Default config is loaded from a toml file. If not toml file is provided,
      then the default config is loaded from argparse defaults.
    - if toml file has missing keys, they are filled with argparse defaults
    - if additional explicit cmd args are provided in addition to the toml file,
      they will override the toml config and the argparse defaults.

    precedence order: cmdline > toml > argparse default

    Arg parsing semantics:

    Each argument starts with <prefix>_ which is the section name in the toml file
    followed by the name of the option in the toml file.
    For example:
    model.name translates to:
        [model]
        name
    in the toml file
    """

    def __init__(self):
        self.args_dict = None
        # main parser
        self.parser = argparse.ArgumentParser(description='mmm argp parser.')
        self.parser.add_argument(
            '--job.config_file', type=str, default=None, help='Job config file'
        )
        # job level configs
        self.parser.add_argument(
            '--job.dump_folder',
            type=str,
            default='outputs',
            help='Folder to dump job outputs',
        )
        self.parser.add_argument(
            '--job.description',
            type=str,
            default='default job',
            help='Description of the job',
        )
        self.parser.add_argument(
            '--job.use_for_integration_test',
            action='store_true',
            help='Add this config to the integration test suite',
        )
        self.parser.add_argument(
            '--job.print_args',
            action='store_true',
            help='Print the args to terminal',
        )

        # profiling configs
        self.parser.add_argument(
            '--profiling.enable_profiling',
            action='store_true',
            help='Whether to enable pytorch profiler',
        )
        self.parser.add_argument(
            '--profiling.save_traces_folder',
            type=str,
            default='profile_traces',
            help='Trace files location',
        )
        self.parser.add_argument(
            '--profiling.profile_freq',
            type=int,
            default=10,
            help='How often to collect profiler traces, in iterations',
        )
        self.parser.add_argument(
            '--profiling.enable_memory_snapshot',
            action='store_true',
            help='Whether to dump memory snapshot',
        )
        self.parser.add_argument(
            '--profiling.save_memory_snapshot_folder',
            type=str,
            default='memory_snapshot',
            help='Memeory snapshot files location',
        )

        # metrics configs
        self.parser.add_argument(
            '--metrics.log_freq',
            type=int,
            default=10,
            help='How often to log metrics to TensorBoard, in iterations',
        )
        self.parser.add_argument(
            '--metrics.enable_tensorboard',
            action='store_true',
            help='Whether to log metrics to TensorBoard',
        )
        self.parser.add_argument(
            '--metrics.disable_color_printing',
            action='store_true',
            help='Whether to disable color printing in logs',
        )
        self.parser.add_argument(
            '--metrics.save_tb_folder',
            type=str,
            default='tb',
            help='Folder to dump TensorBoard states',
        )
        # TODO: store_true & default=True make impossible for cmd to set it to False
        self.parser.add_argument(
            '--metrics.rank_0_only',
            action='store_true',
            default=True,
            help="""
                Whether to save TensorBoard metrics only for rank 0 or for all ranks.
                When pipeline_parallel_degree is > 1, this option uses the 0th rank of the last stage pipeline group,
                which is the only stage that computes loss metrics.
            """,
        )
        self.parser.add_argument(
            '--metrics.enable_wandb',
            action='store_true',
            help='Whether to log metrics to Weights & Biases',
        )

        # model configs
        self.parser.add_argument(
            '--model.name',
            type=str,
            default='llama',
            help='Which model to train',
        )
        self.parser.add_argument(
            '--model.flavor',
            type=str,
            default='debugmodel',
            help='Which model config to train',
        )
        self.parser.add_argument(
            '--model.norm_type',
            type=str,
            default='rmsnorm',
            help='Type of layer normalization to use [layernorm, np_layernorm, rmsnorm, fused_rmsnorm]',
        )
        self.parser.add_argument(
            '--model.tokenizer_path',
            type=str,
            default='data/tokenizer/tokenizer.model',
            help='Tokenizer path',
        )

        # optimizer configs
        self.parser.add_argument(
            '--optimizer.name',
            type=str,
            default='AdamW',
            help='Optimizer to use',
        )
        self.parser.add_argument(
            '--optimizer.lr',
            type=float,
            default=8e-4,
            help='Learning rate to use',
        )
        self.parser.add_argument(
            '--optimizer.fused',
            action='store_true',
            help='Whether the fused implementation(CUDA only) is used.',
        )
        self.parser.add_argument(
            '--optimizer.early_step_in_backward',
            action='store_true',
            help="""
            Whether to apply optimizer in the backward. Caution, optimizer_in_backward
            is not compatible with gradients clipping, users should not call
            register_post_accumulate_grad_hook after the optimizer is built.""",
        )

        # training configs
        self.parser.add_argument(
            '--training.dataset',
            type=str,
            default='c4_mini',
            help='Dataset to use',
        )
        self.parser.add_argument(
            '--training.dataset_path',
            type=str,
            help="""
                Path to the dataset in the file system. If provided, data will be
                loaded from this path instead of downloaded.""",
        )
        self.parser.add_argument(
            '--training.batch_size', type=int, default=8, help='Batch size'
        )
        self.parser.add_argument(
            '--training.seq_len',
            type=int,
            default=2048,
            help='Sequence length',
        )
        self.parser.add_argument(
            '--training.warmup_steps',
            type=int,
            default=200,
            help='Steps for lr scheduler warmup, normally 1/5 of --training.steps',
        )
        self.parser.add_argument(
            '--training.max_norm',
            type=float,
            # type=Union[float, int],
            default=1.0,
            help='Max norm for gradient clipping',
        )
        self.parser.add_argument(
            '--training.steps',
            type=int,
            default=10000,
            help='How many train steps to run',
        )
        self.parser.add_argument(
            '--training.data_parallel_replicate_degree',
            type=int,
            default=1,
            help="""
            The `data_parallel_replicate_degree` argument specifies the degree of
            data parallelism for weight replication. When this value is greater
            than 1, weights will be replicated across `data_parallel_replicate_degree`
            ranks. If `data_parallel_shard_degree` is also greater than 1, the parallelism
            method used is HSDP (Hybrid Sharded Data Parallelism). Otherwise, the
            parallelism method used is DDP (Distributed Data Parallelism).
            1 means disabled.""",
        )
        self.parser.add_argument(
            '--training.data_parallel_shard_degree',
            type=int,
            default=-1,
            help="""
            The `data_parallel_shard_degree` argument specifies the degree of data
            parallelism for weight sharding. When this value is greater than 1, weights
            will be sharded across `data_parallel_shard_degree` ranks. If
            `data_parallel_replicate_degree` is also greater than 1, the parallelism
            method used is HSDP (Hybrid Sharded Data Parallelism).  Otherwise, the
            parallelism method used is FSDP (Fully Sharded Data Parallelism).

            -1 means leftover ranks will be used (After DP_REPLICATE/SP/PP). Note that
            only `data_parallel_shard_degree` can be negative. 1 means disabled.""",
        )
        self.parser.add_argument(
            '--training.enable_cpu_offload',
            action='store_true',
            help="""
            Whether to apply CPU offloading of parameters, gradients, and optimizer states in FSDP""",
        )
        self.parser.add_argument(
            '--training.tensor_parallel_degree',
            type=int,
            default=1,
            help='Tensor Parallelism degree. 1 means disabled.',
        )
        self.parser.add_argument(
            '--training.disable_loss_parallel',
            action='store_true',
            help='Whether to apply loss parallel when sequence parallel is enabled',
        )
        self.parser.add_argument(
            '--experimental.enable_async_tensor_parallel',
            action='store_true',
            help='Whether to apply async tensor parallel (currently only effective when compile is enabled)',
        )
        self.parser.add_argument(
            '--experimental.pipeline_parallel_degree',
            type=int,
            default=1,
            help="""
                Pipeline Parallelism degree, or number of ranks. 1 means disabled.
                If using looped schedules, this still specifies the number of physical ranks, not the number
                of stages.  Stages per rank are inferred from split points degree, and schedule.""",
        )
        self.parser.add_argument(
            '--experimental.pipeline_parallel_split_points',
            type=string_list,
            nargs='+',
            default=[],
            help="""
                Specify comma-separated names of modules to use as the beginning of a split point.

                e.g. "layers.0,layers.2" will cause the model to be split into 3 stages,
                the first containing all the layers up to layers.0,
                the second containing layers.0 and up to layers.2,
                the third containing layers.2 and all the remaining layers.

                Note: fully-automated splitting may be enabled in the future,
                but currently the split points must be specified manually.""",
        )
        self.parser.add_argument(
            '--experimental.pipeline_parallel_schedule',
            type=str,
            default='1F1B',
            help="""
                Specify the Pipeline Parallel schedule to use. The supported schedules are:
                https://github.com/pytorch/pytorch/blob/de4c2a3b4e89d96334dc678d1c3f2ae51a6630a0/torch/distributed/pipelining/schedules.py#L2161.
                The schedule must be compatible with the split points and stages_per_rank.

                Looped schedules (e.g. Interleaved1F1B) require specifying pipeline_parallel_degree = number of ranks,
                and split_points = number of stages - 1
                """,
        )
        self.parser.add_argument(
            '--experimental.pipeline_parallel_schedule_csv',
            type=str,
            default='',
            help="""
                Specify the path to the pipeline parallel schedule csv file to use.
                The pipeline_parallel_schedule argument must be either
                PipelineScheduleSingle, PipelineScheduleMulti, or _PipelineScheduleRuntime.
            """,
        )

        self.parser.add_argument(
            '--experimental.pipeline_parallel_microbatches',
            type=int,
            default=None,
            help="""
                How many microbatches to split the global training batch into when using pipeline parallelism.

                The global training batch size must be evenly divisible by the number of microbatches.

                The default value will be the number of pipeline stages, if unspecified.
            """,
        )
        self.parser.add_argument(
            '--experimental.enable_compiled_autograd',
            action='store_true',
            help='Enable CompiledAutograd to compile the backward.',
        )
        self.parser.add_argument(
            '--experimental.context_parallel_degree',
            type=int,
            default=1,
            help='Context parallelism degree. 1 means disabled.',
        )
        self.parser.add_argument(
            '--experimental.context_parallel_rotate_method',
            type=str,
            default='allgather',
            help="""
                The collective to use in context parallel SDPA for kv shards exchange.

                'allgather' means to all-gather all kv shards on ranks after the first sub-SDPA computation,

                'alltoall' means to all-to-all shuffle the kv shards.

                The default value is 'allgather'.
            """,
        )
        self.parser.add_argument(
            '--training.mixed_precision_param',
            type=str,
            default='bfloat16',
            choices=['bfloat16', 'float32'],
            help="""
                torch dtype to use for parameters when applying mixed precision via FSDP.
                This feature only takes effect when data_parallel_shard_degree > 1
            """,
        )
        self.parser.add_argument(
            '--training.mixed_precision_reduce',
            type=str,
            default='float32',
            choices=['float32'],
            help="""
                torch dtype to use for reductions when applying mixed precision via FSDP.
                This feature only takes effect when data_parallel_shard_degree > 1
            """,
        )
        self.parser.add_argument(
            '--training.compile',
            action='store_true',
            help='Whether to compile the model',
        )
        self.parser.add_argument(
            '--training.gc_freq',
            type=int,
            default=50,
            help='Python garbage control scheduling interval, in steps',
        )
        self.parser.add_argument(
            '--training.seed',
            type=int,
            default=None,
            help='Choose the base RNG seed used for training',
        )
        self.parser.add_argument(
            '--training.deterministic',
            action='store_true',
            help='Use deterministic algorithms wherever possible, may be slower',
        )
        # checkpointing configs
        self.parser.add_argument(
            '--checkpoint.enable_checkpoint',
            action='store_true',
            help='Whether to enable checkpoint',
        )
        self.parser.add_argument(
            '--checkpoint.folder',
            type=str,
            default='checkpoint',
            help="""
                The folder to store the checkpoints.
                When enable_checkpoint is set to true, checkpoints will be in {--job.dump_folder}/{--checkpoint.folder}.
            """,
        )
        self.parser.add_argument(
            '--checkpoint.interval_type',
            type=str,
            default='steps',
            help="Checkpointing interval unit of measurement ['step', 'seconds']",
        )
        self.parser.add_argument(
            '--checkpoint.interval',
            type=int,
            default=500,
            help='Checkpointing interval, in steps or seconds depending on --checkpoint.interval_type',
        )
        self.parser.add_argument(
            '--checkpoint.model_weights_only',
            action='store_true',
            help="""
                When model_weights_only=True, only model weights will be saved at the end of training.
                With this, checkpoints can be loaded using `torch.load(..., weights_only=True)` after conversion.
                When model_weights_only=False, the full checkpoint will be saved.
                A full checkpoint includes model, optimizer and train_state, which can be used to resume training.
                The default value is false.
            """,
        )
        self.parser.add_argument(
            '--checkpoint.export_dtype',
            type=str,
            default='float32',
            choices=['float16', 'bfloat16', 'float32'],
            help="""
                Converts to the specified precision when training completes and model_weights_only=true.
                Currently supports float32, float16, and bfloat16.
                The default value is float32.
            """,
        )
        self.parser.add_argument(
            '--checkpoint.create_seed_checkpoint',
            action='store_true',
            help="""
                Initializes the full model without applying parallelisms, and then saves it as a seed checkpoint.
                Note: requires user to call train.py without specifying any parallelisms, e.g. NGPU=1.
                Could be implemented as a separate script, but this way shares more code.
            """,
        )
        self.parser.add_argument(
            '--checkpoint.async_mode',
            type=str,
            default='disabled',
            help="""
                Which async checkpoint mode to use. Currently there are 3 different modes.
                1. "disabled": synchronized checkpointing will be used.
                2. "async": torch.distributed.checkpoint.async_save will be used.
                3. "async_with_pinned_mem": this option utilizes a dedicated pinned memory
                   space and creates a separate process for faster GPU->CPU transfer
                   performance and eliminating GIL contention. The cost is increased CPU
                   memory usage. If insufficient CPU memory is available, performance may
                   degrade due to memory paging. For most users, "async" should suffice as
                   the performance overhead is typically small (on the order of tens of
                   seconds) compared to checkpointing frequency. This mode can be employed
                   to pursue near-zero checkpointing times (e.g., < 1 second) given
                   appropriate hardware support such as ample CPU memory and fast PCIe.

                "disabled" is the default mode.
            """,
        )
        self.parser.add_argument(
            '--checkpoint.keep_latest_k',
            type=int,
            default=0,
            help="""
                Keeps only the latest k checkpoints, and purging older ones. If 0, keep all checkpoints.
                0 is the default value.
            """,
        )
        self.parser.add_argument(
            '--checkpoint.load_step',
            type=int,
            default=-1,
            help='Load the checkpoint at the specified step. If -1, load the latest checkpoint.',
        )
        # activation checkpointing configs
        self.parser.add_argument(
            '--activation_checkpoint.mode',
            type=str,
            default='selective',
            help="Type of activation checkpointing to use ['none', 'full', 'selective']",
        )
        self.parser.add_argument(
            '--activation_checkpoint.selective_ac_option',
            type=str,
            default='2',  # 2 = checkpoint every other layer
            help="""
                Selective activation checkpointing options ['int', 'op'].
                'int' (e.g., 2) for every nth layer, or 'op' for op level ac.
            """,
        )

        # float8 configs
        self.parser.add_argument(
            '--float8.enable_float8_linear',
            action='store_true',
            help="""
                If true, swaps `torch.nn.Linear` with `Float8Linear`.
                This feature requires you to install 'torchao' which can be found
                here: https://github.com/pytorch/ao
            """,
        )
        self.parser.add_argument(
            '--float8.enable_fsdp_float8_all_gather',
            action='store_true',
            help='Whether enable float8 all-gather in FSDP',
        )
        self.parser.add_argument(
            '--float8.precompute_float8_dynamic_scale_for_fsdp',
            action='store_true',
            help='Whether precompute float8 scales dynamically for FSDP',
        )
        self.parser.add_argument(
            '--float8.scaling_type_input',
            type=str,
            default='dynamic',
            help='float8 scaling for input, dynamic (default) or delayed',
            choices=['dynamic', 'delayed'],
        )
        self.parser.add_argument(
            '--float8.scaling_type_weight',
            type=str,
            default='dynamic',
            help='float8 scaling for input, dynamic (default) or delayed',
        )
        self.parser.add_argument(
            '--float8.scaling_type_grad_output',
            type=str,
            default='dynamic',
            help='float8 scaling for input, dynamic (default) or delayed',
        )

        # communications library settings
        self.parser.add_argument(
            '--comm.init_timeout_seconds',
            type=int,
            default=300,
            help='Timeout for communication operations, during initialization and first train step.',
        )
        self.parser.add_argument(
            '--comm.train_timeout_seconds',
            type=int,
            default=100,
            help=(
                'Timeout for communication operations after the first train step -- '
                'usually a tighter bound than during initialization.'
            ),
        )
        self.parser.add_argument(
            '--comm.trace_buf_size',
            type=int,
            default=20000,
            help='Flight recorder ring buffer size, >0 means recording by default, 0 means disabled',
        )

        # memory estimation settings
        self.parser.add_argument(
            '--memory_estimation.enabled',
            help='Whether to estimate memory usage for FSDP',
            action='store_true',
        )

        self.parser.add_argument(
            '--memory_estimation.disable_fake_mode',
            help='Whether to estimate memory under FakeTensorMode',
            action='store_true',
        )

    def to_dict(self):
        return self.args_dict

    def parse_args(self, args_list: list = sys.argv[1:]):
        args, cmd_args = self.parse_args_from_command_line(args_list)
        config_file = getattr(args, 'job.config_file', None)
        # build up a two level dict
        args_dict = self._args_to_two_level_dict(args)
        if config_file is not None:
            try:
                with open(config_file, 'rb') as f:
                    for k, v in tomllib.load(f).items():
                        # to prevent overwrite of non-specified keys
                        args_dict[k] |= v
            except (FileNotFoundError, tomllib.TOMLDecodeError) as e:
                logger.exception(
                    f'Error while loading the configuration file: {config_file}'
                )
                logger.exception(f'Error details: {str(e)}')
                raise e

        # if split-points came from 'args' (from cmd line) it would have already been parsed into a list by that parser
        if (
            'experimental' in args_dict
            and 'pipeline_parallel_split_points' in args_dict['experimental']
            and isinstance(
                args_dict['experimental']['pipeline_parallel_split_points'],
                str,
            )
        ):
            exp = args_dict['experimental']
            exp['pipeline_parallel_split_points'] = string_list(
                exp['pipeline_parallel_split_points']
            )

        # override args dict with cmd_args
        cmd_args_dict = self._args_to_two_level_dict(cmd_args)
        for section, section_args in cmd_args_dict.items():
            for k, v in section_args.items():
                args_dict[section][k] = v

        self.args_dict = args_dict

        for k, v in args_dict.items():
            class_type = type(k.title(), (), v)
            setattr(self, k, class_type())
        self._validate_config()

    def _args_to_two_level_dict(self, args: argparse.Namespace) -> defaultdict:
        args_dict = defaultdict(defaultdict)
        for k, v in vars(args).items():
            first_level_key, second_level_key = k.split('.', 1)
            args_dict[first_level_key][second_level_key] = v
        return args_dict

    def _validate_config(self) -> None:
        # TODO: Add more mandatory validations
        assert self.model.name, 'Model name is mandatory'  # type:ignore
        assert self.model.flavor, 'Model flavor is mandatory'  # type:ignore
        assert self.model.tokenizer_path, 'Tokenizer path is mandatory'  # type:ignore

    def parse_args_from_command_line(
        self, args_list
    ) -> tuple[argparse.Namespace, argparse.Namespace]:
        """
        Parse command line arguments and return the parsed args and the command line only args
        """
        args = self.parser.parse_args(args_list)

        # aux parser to parse the command line only args, with no defaults from main parser
        aux_parser = argparse.ArgumentParser(
            argument_default=argparse.SUPPRESS
        )
        for arg, val in vars(args).items():
            if isinstance(val, bool):
                aux_parser.add_argument(
                    '--' + arg, action='store_true' if val else 'store_false'
                )
            elif arg == 'experimental.pipeline_parallel_split_points':
                # without this special case, type inference breaks here,
                # since the inferred type is just 'list' and it ends up flattening
                # e.g. from ["layers.0", "layers.1"] into ["l", "a", "y", "e", "r", "s", ".0", ...]
                aux_parser.add_argument('--' + arg, type=string_list)
            else:
                aux_parser.add_argument('--' + arg, type=type(val))

        cmd_args, _ = aux_parser.parse_known_args(args_list)

        return args, cmd_args
