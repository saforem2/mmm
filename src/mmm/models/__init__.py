# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""
mmm/data/__init__.py
"""

import ezpz
from typing import Optional, Sequence

from mmm.models.llama import llama3_configs, Transformer
# from torchtitan.models.llama import llama3_configs, Transformer

models_config = {
    'llama3': llama3_configs,
}

model_name_to_cls = {'llama3': Transformer}

model_name_to_tokenizer = {
    'llama3': 'tiktoken',
}


logger = ezpz.get_logger(__name__)


def summarize_model(
    model, 
    verbose: bool = False,
    depth: int = 1,
    input_size: Optional[Sequence[int]] = None
):
    try:
        from torchinfo import summary

        summary_str = summary(
            model,
            input_size=input_size,
            depth=depth,
            verbose=verbose,
        )
        # logger.info(f'\n{summary_str}')
        return summary_str

    except (ImportError, ModuleNotFoundError):
        logger.warning(
            'torchinfo not installed, unable to print model summary!'
        )
