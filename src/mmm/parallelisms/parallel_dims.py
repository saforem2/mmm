# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# This file applies the PT-D parallelisms (except pipeline parallelism) and various
# training techniques (e.g. activation checkpointing and compile) to the Llama model.
"""
mmm/parallelisms/parallel_dims.py


Modified from:
<https://github.com/pytorch/torchtitan/blob/2a4437014e66bcf88a3f0419b816266e6326d539/torchtitan/parallelisms/parallel_dims.py>
"""

from dataclasses import dataclass
from functools import cached_property

from torch.distributed.device_mesh import init_device_mesh
import ezpz


logger = ezpz.get_logger(__name__)


@dataclass
class ParallelDims:
    dp_replicate: int
    dp_shard: int
    cp: int
    tp: int
    pp: int
    world_size: int
    enable_loss_parallel: bool

    def __post_init__(self):
        self._validate()

    def _validate(self):
        dp_replicate, dp_shard, cp, tp, pp = (
            self.dp_replicate,
            self.dp_shard,
            self.cp,
            self.tp,
            self.pp,
        )
        for d in (dp_replicate, cp, tp, pp):
            assert d >= 1, (
                'Parallelism degree should be >= 1, except for dp_shard'
            )
        assert dp_shard == -1 or dp_shard >= 1, 'dp_shard should be -1 or >= 1'
        if dp_shard < 0:
            self.dp_shard = dp_shard = self.world_size // (
                dp_replicate * cp * tp * pp
            )
        assert dp_shard >= 1
        assert dp_replicate * dp_shard * cp * tp * pp == self.world_size, (
            ' '.join(
                [
                    'Invalid parallel dims:',
                    f'dp_replicate({dp_replicate})',
                    f'* dp_shard({dp_shard})',
                    f'* cp({cp}) * tp({tp}) * pp({pp})',
                    f'!= world_size({self.world_size})',
                ]
            )
        )

    def build_mesh(self, device_type):
        dims = []
        names = []
        for d, name in zip(
            [self.pp, self.dp_replicate, self.dp_shard, self.cp, self.tp],
            ['pp', 'dp_replicate', 'dp_shard', 'cp', 'tp'],
        ):
            if d > 1:
                dims.append(d)
                names.append(name)
        logger.info(f'Building {len(dims)}-D device mesh with {names}, {dims}')
        names = tuple(names)
        mesh = init_device_mesh(device_type, dims, mesh_dim_names=names)
        # Create all the submesh here to ensure all required process groups are initialized:
        # Mesh for data loading (no communication on this mesh)
        dp_mesh_dim_names = []
        # Mesh for param sharding
        dp_shard_cp_mesh_dim_names = []
        # Mesh for loss all-reduce
        dp_cp_mesh_dim_names = []
        if self.dp_replicate_enabled:
            dp_mesh_dim_names.append('dp_replicate')
            dp_cp_mesh_dim_names.append('dp_replicate')
        if self.dp_shard_enabled:
            dp_mesh_dim_names.append('dp_shard')
            dp_shard_cp_mesh_dim_names.append('dp_shard')
            dp_cp_mesh_dim_names.append('dp_shard')
        if self.cp_enabled:
            dp_shard_cp_mesh_dim_names.append('cp')
            dp_cp_mesh_dim_names.append('cp')

        if dp_mesh_dim_names != []:
            mesh[tuple(dp_mesh_dim_names)]._flatten(mesh_dim_name='dp')
        if dp_shard_cp_mesh_dim_names != []:
            mesh[tuple(dp_shard_cp_mesh_dim_names)]._flatten(
                mesh_dim_name='dp_shard_cp'
            )
        if dp_cp_mesh_dim_names != []:
            mesh[tuple(dp_cp_mesh_dim_names)]._flatten(mesh_dim_name='dp_cp')

        return mesh

    @property
    def dp_enabled(self) -> bool:
        return self.dp_replicate > 1 or self.dp_shard > 1

    @property
    def dp_replicate_enabled(self) -> bool:
        return self.dp_replicate > 1

    @property
    def dp_shard_enabled(self) -> bool:
        return self.dp_shard > 1

    @property
    def cp_enabled(self) -> bool:
        return self.cp > 1

    @property
    def tp_enabled(self) -> bool:
        return self.tp > 1

    @property
    def pp_enabled(self) -> bool:
        return self.pp > 1

    @property
    def loss_parallel_enabled(self) -> bool:
        return self.tp > 1 and self.enable_loss_parallel

    @cached_property
    def non_data_parallel_size(self) -> int:
        return self.cp * self.tp * self.pp


@dataclass
class ParallelDimsOld:
    dp_replicate: int
    dp_shard: int
    cp: int
    tp: int
    pp: int
    world_size: int
    enable_loss_parallel: bool
    # dp_replicate: int
    # dp_shard: int
    # cp: int
    # tp: int
    # pp: int
    # world_size: int
    # enable_loss_parallel: bool

    def __post_init__(self):
        self._validate()

    def _validate(self):
        dp_replicate, dp_shard, cp, tp, pp = (
            self.dp_replicate,
            self.dp_shard,
            self.cp,
            self.tp,
            self.pp,
        )
        for d in (dp_replicate, cp, tp, pp):
            assert d >= 1, (
                'Parallelism degree should be >= 1, except for dp_shard'
            )
        assert dp_shard == -1 or dp_shard >= 1, ' dp_shard must -1 or >=1.'

        dp = dp_replicate * dp_shard
        if dp < 0:
            dp = self.world_size // (tp * pp)
            self.dp_shard = dp_shard = dp // dp_replicate

        assert dp_replicate >= 1
        assert dp_shard >= 1
        assert tp >= 1, tp
        assert cp >= 1, cp
        assert pp >= 1, pp
        assert dp_replicate * dp_shard * cp * tp * pp == self.world_size, (
            f'Invalid parallel dims: dp_replicate({dp_replicate}) * dp_shard({dp_shard}) * '
            f'cp({cp}) * tp({tp}) * pp({pp}) != WORLD_SIZE({self.world_size})'
        )

    def build_mesh(self, device_type):
        dims = []
        names = []
        for d, name in zip(
            [self.pp, self.dp_replicate, self.dp_shard, self.cp, self.tp],
            ['pp', 'dp_replicate', 'dp_shard', 'cp', 'tp'],
        ):
            if d > 1:
                dims.append(d)
                if (name == 'dp_replicate' and self.dp_shard == 1) or (
                    name == 'dp_shard' and self.dp_replicate == 1
                ):
                    names.append('dp')
                else:
                    names.append(name)

        logger.info(f'Building {len(dims)}-D device mesh with {names}, {dims}')
        names = tuple(names)
        mesh = init_device_mesh(device_type, dims, mesh_dim_names=names)
        # Create all the submesh here to ensure all required process groups are
        # initialized
        if self.dp_replicate > 1 and self.dp_shard > 1:
            mesh['dp_replicate', 'dp_shard']._flatten(mesh_dim_name='dp')

        if self.cp > 1:
            if self.dp_replicate > 1 and self.dp_shard > 1:
                mesh['dp_replicate', 'dp_shard', 'cp']._flatten(
                    mesh_dim_name='dp_cp'
                )
            else:
                mesh['dp', 'cp']._flatten(mesh_dim_name='dp_cp')
        return mesh

    @property
    def dp_enabled(self):
        return self.dp_replicate > 1 or self.dp_shard > 1

    @property
    def dp_replicate_enabled(self):
        return self.dp_replicate > 1

    @property
    def dp_shard_enabled(self):
        return self.dp_shard > 1

    @property
    def cp_enabled(self):
        return self.cp > 1

    @property
    def tp_enabled(self):
        return self.tp > 1

    @property
    def pp_enabled(self):
        return self.pp > 1

    @property
    def loss_parallel_enabled(self):
        return self.tp > 1 and self.enable_loss_parallel

    @cached_property
    def non_data_parallel_size(self):
        return self.cp * self.tp * self.pp
