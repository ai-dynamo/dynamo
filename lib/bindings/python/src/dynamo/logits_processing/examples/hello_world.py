# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Sequence

import torch
from transformers import PreTrainedTokenizerBase

from dynamo.logits_processing import BaseLogitsProcessor

RESPONSE = "Hello world!"


class HelloWorldLogitsProcessor(BaseLogitsProcessor):
    """
    Sample Logits Processor that always outputs a hardcoded
    response (`RESPONSE`), no matter the input
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        self.tokenizer = tokenizer
        self.token_ids = tokenizer.encode(RESPONSE, add_special_tokens=False)
        self.eos_id = tokenizer.eos_token_id
        if self.eos_id is None:
            raise ValueError(
                "Tokenizer has no eos_token_id; HelloWorldLogitsProcessor requires one."
            )
        self.state = 0

    def __call__(self, input_ids: Sequence[int], scores: torch.Tensor):
        mask = torch.full_like(scores, float("-inf"))

        if self.state < len(self.token_ids):
            token_idx = self.token_ids[self.state]
        else:
            token_idx = self.eos_id
        # Allow only a single token to be output
        mask[token_idx] = 0.0

        # The `scores` tensor *must* also be modified in-place
        scores.add_(mask)
        self.state += 1
