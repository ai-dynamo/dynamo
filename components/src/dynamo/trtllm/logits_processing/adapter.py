# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import List, Optional

import torch
from tensorrt_llm.sampling_params import LogitsProcessor

from dynamo.logits_processing import BaseLogitsProcessor

logger = logging.getLogger(__name__)


class TrtllmDynamoLogitsAdapter(LogitsProcessor):
    """
    Adapter that wraps Dynamo BaseLogitsProcessor instances to work with TensorRT-LLM's logits processor interface.

    Inherits from tensorrt_llm.LogitsProcessor and implements the required interface:
    __call__(self, req_ids: int, logits: torch.Tensor, ids: List[List[int]], stream_ptr, client_id: Optional[int])

    This adapter maintains per-request state and converts between the interfaces.
    """

    def __init__(self, processor: BaseLogitsProcessor):
        super().__init__()
        self.processor = processor

    def __call__(
        self,
        req_ids: int,
        logits: torch.Tensor,
        ids: List[List[int]],
        stream_ptr,
        client_id: Optional[int] = None,
    ):
        """
        TensorRT-LLM logits processor interface.

        Args:
            req_ids: Request identifier
            logits: Logits tensor for current step
            ids: List of token sequences (batch of sequences)
            stream_ptr: CUDA stream pointer
            client_id: Optional client identifier

        Returns:
            Modified logits tensor (in-place modification expected by TRT-LLM)
        """
        stream = None if stream_ptr is None else torch.cuda.ExternalStream(stream_ptr)
        try:
            with torch.cuda.stream(stream):
                if logits.shape[0] != 1:
                    raise ValueError(
                        f"This logits adapter only supports per-request logits processing. "
                        f"Received logits with batch size {logits.shape[0]} expected 1"
                    )
                if logits.shape[1] != 1:
                    raise ValueError(
                        "Logits processing with beam width > 1 is not supported"
                    )
                # Call the processor which modifies the logits in-place
                self.processor(ids[0], logits[0, 0, :])

        except Exception as e:
            logger.error(f"Error in logits processor for request {req_ids}: {e}")
            # Don't modify logits on error

        # TRT-LLM expects void return (in-place modification)


def create_trtllm_adapters(
    processors: List[BaseLogitsProcessor],
) -> List[TrtllmDynamoLogitsAdapter]:
    """
    Create TensorRT-LLM compatible adapters from Dynamo logits processors.

    Args:
        processors: List of Dynamo BaseLogitsProcessor instances

    Returns:
        List of TensorRT-LLM compatible logits processor adapters
    """
    adapters = []
    for processor in processors:
        adapter = TrtllmDynamoLogitsAdapter(processor)
        adapters.append(adapter)
    return adapters
