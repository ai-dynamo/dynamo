"""
Base logits processor protocol for Dynamo.

This module defines the core BaseLogitsProcessor interface that all
logits processors must implement.
"""

from typing import List, Protocol

import torch


class BaseLogitsProcessor(Protocol):
    """
    Protocol for logits processors in Dynamo.

    All logits processors must implement this interface to be compatible
    with backend adapters (TRT-LLM, vLLM, SGLang).
    """

    def __call__(
        self,
        input_ids: List[int],
        logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        Process the logits for the next token prediction.

        Args:
            input_ids: The input token IDs generated so far.
            logits: The raw logits for the next token. Shape: (vocab_size,)

        Returns:
            The modified logits tensor with same shape as input.
        """
        raise NotImplementedError
