# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Base logits processor protocol for Dynamo.

This module defines the core BaseLogitsProcessor interface that all
logits processors must implement.
"""

from typing import Protocol, Sequence

import torch


class BaseLogitsProcessor(Protocol):
    """
    Protocol for logits processors in Dynamo.

    All logits processors must implement this interface to be compatible
    with backend adapters (TRT-LLM, vLLM, SGLang).
    """

    def __call__(
        self,
        input_ids: Sequence[int],
        logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        Process the logits for the next token prediction.

        Args:
            input_ids: The input token IDs generated so far.
            logits: The raw logits for the next token. Shape: (vocab_size,)

        Returns:
            A tensor with the same shape, dtype, and device as `logits`.
        """
        ...
