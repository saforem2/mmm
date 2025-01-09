# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import ezpz
from mmm.data.tokenizer.tiktoken import TikTokenizer
from mmm.data.tokenizer.tokenizer import Tokenizer
# from torchtitan.datasets.tokenizer.tiktoken import TikTokenizer
# from torchtitan.datasets.tokenizer.tokenizer import Tokenizer


logger = ezpz.get_logger(__name__)


def build_tokenizer(tokenizer_type: str, tokenizer_path: str) -> Tokenizer:
    logger.info(f"Building {tokenizer_type} tokenizer locally from {tokenizer_path}")
    if tokenizer_type == "tiktoken":
        return TikTokenizer(tokenizer_path)
    else:
        raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")
