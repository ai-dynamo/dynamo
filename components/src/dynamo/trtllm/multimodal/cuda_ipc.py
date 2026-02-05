# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
from typing import Any, Dict, List

import torch
from tensorrt_llm._torch.shared_tensor import SharedTensorContainer

logger = logging.getLogger(__name__)


async def extract_embeddings_from_handles(
    handles: List[Dict[str, Any]],
) -> List[torch.Tensor]:
    """
    Extract all embedding tensors from CUDA IPC handles and move to CPU.

    Runs extraction in a worker thread to avoid blocking the event loop
    during GPU→CPU transfers.

    WARNING: Do not reuse the given `handles` outside this function --
    https://github.com/pytorch/pytorch/issues/149187
    As of Jan 2026, it's safer to ensure one producer corresponds to one consumer so that
    the ref counter_value return to 0, allowing Encode Process to release GPU memory
    properly.

    Args:
        handles: List of CUDA IPC handle dictionaries from encoder response

    Returns:
        List of embedding tensors on CPU.

    Raises:
        ValueError: If a handle is missing required fields.
        RuntimeError: If CUDA IPC reconstruction fails.
    """
    # TODO(DIS-1398): expeiment
    # - pinned memory DMA
    # - parallelize GPU->CPU transfers in multiple threads
    # - combination fo both (i.e. `cpu(non_blocking=True)`)
    return await asyncio.to_thread(_extract_embeddings_sync, handles)


def _extract_embeddings_sync(handles: List[Dict[str, Any]]) -> List[torch.Tensor]:
    """Synchronously extract all embeddings from CUDA IPC handles."""
    tensors = []
    for i, handle_dict in enumerate(handles):
        try:
            container = SharedTensorContainer.from_dict(handle_dict)
            tensor = container.get_local_view().cpu()
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
