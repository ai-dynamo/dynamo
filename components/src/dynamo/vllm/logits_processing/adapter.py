# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Adapter that wraps Dynamo BaseLogitsProcessor instances to work with vLLM's
logits processor interface.

Similar to TrtllmDynamoLogitsAdapter but for vLLM.
"""

import logging
from typing import List, Sequence, Union

import torch

from dynamo.logits_processing import BaseLogitsProcessor

logger = logging.getLogger(__name__)


class VllmDynamoLogitsAdapter:
    """
    Adapter that wraps Dynamo BaseLogitsProcessor instances to work with
    vLLM's logits processor interface.

    vLLM logits processors are callables that receive token IDs and logits,
    and can modify the logits in-place or return modified logits.

    Args:
        processor: A Dynamo BaseLogitsProcessor instance

    Example:
        >>> from dynamo.logits_processing import ThinkingBudgetLogitsProcessor
        >>> processor = ThinkingBudgetLogitsProcessor(
        ...     thinking_start_token_id=128798,
        ...     thinking_end_token_id=128799,
        ...     newline_token_id=201,
        ...     max_thinking_tokens=100,
        ... )
        >>> adapter = VllmDynamoLogitsAdapter(processor)
        >>> # adapter can now be used with vLLM's sampling_params.logits_processors
    """

    def __init__(self, processor: BaseLogitsProcessor):
        self.processor = processor

    def __call__(
        self,
        input_ids: Union[List[int], torch.Tensor, Sequence[int]],
        logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        vLLM logits processor interface.

        Args:
            input_ids: Input token IDs generated so far. Can be a List[int],
                torch.Tensor, or any Sequence[int] depending on vLLM version.
            logits: Logits for next token. Shape typically (vocab_size,) or
                   (batch_size, vocab_size) depending on vLLM version.

        Returns:
            Modified logits tensor
        """
        try:
            # Convert input_ids to a sequence the processor can handle
            if isinstance(input_ids, torch.Tensor):
                input_ids_seq = input_ids.tolist()
            else:
                input_ids_seq = list(input_ids)

            # Handle both batched and unbatched cases
            if logits.dim() == 1:
                # Single sequence: shape (vocab_size,)
                self.processor(input_ids_seq, logits)
            elif logits.dim() == 2 and logits.shape[0] == 1:
                # Batched but single sequence: shape (1, vocab_size)
                self.processor(input_ids_seq, logits[0])
            else:
                # Multi-batch case - process each independently
                # Note: This may not be common in vLLM's per-request processors
                logger.warning(
                    f"VllmDynamoLogitsAdapter received unexpected logits shape: {logits.shape}. "
                    "Skipping logits modification."
                )
        except Exception as e:
            logger.error(f"Error in VllmDynamoLogitsAdapter: {e}")
            # Don't modify logits on error - return as-is

        return logits


def create_vllm_adapters(
    processors: List[BaseLogitsProcessor],
) -> List[VllmDynamoLogitsAdapter]:
    """
    Create vLLM compatible adapters from Dynamo logits processors.

    Args:
        processors: List of Dynamo BaseLogitsProcessor instances

    Returns:
        List of vLLM compatible logits processor adapters
    """
    return [VllmDynamoLogitsAdapter(p) for p in processors]
