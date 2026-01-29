# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
CUDA IPC utilities for multimodal embedding tensor extraction.

Extracts embedding tensors from CUDA IPC handles returned by TRT-LLM's
MultimodalEncoder and moves them to CPU for caching.

Usage:
    from dynamo.trtllm.multimodal.cuda_ipc import extract_embeddings_from_disaggregated_params

    tensors = extract_embeddings_from_disaggregated_params(ep_params)
    if tensors:
        for i, tensor in enumerate(tensors):
            cache.set(f"{mm_hash}_{i}", tensor)
"""

import logging
from typing import Any, List, Optional

import torch
from tensorrt_llm._torch.shared_tensor import SharedTensorContainer

logger = logging.getLogger(__name__)


def extract_embeddings_from_disaggregated_params(
    disaggregated_params: Any,
) -> Optional[List[torch.Tensor]]:
    """
    Extract all embedding tensors from DisaggregatedParams and move to CPU.

    Args:
        disaggregated_params: DisaggregatedParams from encoder response.

    Returns:
        List of embedding tensors on CPU, or None if no handles present.

    Raises:
        ValueError: If a handle is missing required fields.
        RuntimeError: If CUDA IPC reconstruction fails.

    Example:
        >>> tensors = extract_embeddings_from_disaggregated_params(ep_params)
        >>> if tensors:
        ...     embedding = tensors[0]  # First image embedding
    """
    if disaggregated_params is None:
        return None

    handles = getattr(disaggregated_params, "multimodal_embedding_handles", None)
    if not handles:
        return None

    tensors = []
    for i, handle_dict in enumerate(handles):
        try:
            container = SharedTensorContainer.from_dict(handle_dict)
            tensor = container.get_local_view().clone().cpu()
            tensors.append(tensor)
            logger.debug(
                f"Extracted embedding {i}: shape={tensor.shape}, dtype={tensor.dtype}"
            )
        except KeyError as e:
            raise ValueError(f"Invalid handle {i} - missing field: {e}")
        except Exception as e:
            logger.error(f"Failed to extract embedding {i}: {e}")
            raise RuntimeError(f"Failed to extract embedding {i}: {e}")

    return tensors
