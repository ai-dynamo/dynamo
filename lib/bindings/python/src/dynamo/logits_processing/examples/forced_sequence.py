# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Forced-sequence logits processor.

Forces the model to emit a fixed sequence of token IDs, then EOS. Takes
the resolved token IDs directly (no tokenizer at construction time), so
it works behind any serialization boundary and is the target for
backend-neutral entry-based activation.
"""

from typing import Sequence

import torch

from dynamo.logits_processing import BaseLogitsProcessor


class ForcedSequenceLogitsProcessor(BaseLogitsProcessor):
    """Forces the next ``len(token_ids)`` outputs, then EOS thereafter.

    Stateful: keeps a position counter (`state`) advanced each step.
    Must be instantiated per request so concurrent streams don't mix
    positions.
    """

    def __init__(self, token_ids: Sequence[int], eos_token_id: int):
        if eos_token_id is None:
            raise ValueError("ForcedSequenceLogitsProcessor requires eos_token_id")
        self.token_ids = list(token_ids)
        self.eos_id = eos_token_id
        self.state = 0

    def __call__(self, input_ids: Sequence[int], scores: torch.Tensor) -> None:
        mask = torch.full_like(scores, float("-inf"))
        if self.state < len(self.token_ids):
            token_idx = self.token_ids[self.state]
        else:
            token_idx = self.eos_id
        mask[token_idx] = 0.0
        # In-place mutation per BaseLogitsProcessor contract.
        scores.add_(mask)
        self.state += 1
