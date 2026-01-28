# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Common utilities shared across GMS integrations."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from gpu_memory_service.client.memory_manager import GMSClientMemoryManager

logger = logging.getLogger(__name__)


def setup_meta_tensor_workaround() -> None:
    """Enable workaround for meta tensor operations like torch.nonzero()."""
    try:
        import torch.fx.experimental._config as fx_config

        fx_config.meta_nonzero_assume_all_nonzero = True
    except (ImportError, AttributeError):
        pass


def finalize_gms_write(
    allocator: "GMSClientMemoryManager", model: torch.nn.Module
) -> int:
    """Finalize GMS write mode: register tensors, commit, switch to read.
    This is typically called when the (writing) model loader finishes, and
    is ready to commit the weights so that other engines can import these
    weights and read them.

    Args:
        allocator: The GMS client memory manager in write mode.
        model: The loaded model with weights to register.

    Returns:
        Total bytes committed.

    Raises:
        RuntimeError: If commit fails.
    """
    from gpu_memory_service.client.torch.module import register_module_tensors

    register_module_tensors(allocator, model)
    total_bytes = allocator.total_bytes

    # Wait for all writes to weights (from caller) to complete before mode switch
    torch.cuda.synchronize()

    if not allocator.commit():
        raise RuntimeError("GMS commit failed")

    allocator.switch_to_read()

    logger.info(
        "[GMS] Committed %.2f GiB, switched to read mode with %d mappings",
        total_bytes / (1 << 30),
        len(allocator._mappings),
    )

    return int(total_bytes)
