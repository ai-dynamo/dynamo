# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Common utilities shared across GMS integrations."""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import TYPE_CHECKING

import torch
from gpu_memory_service.client.torch.module import register_module_tensors
from gpu_memory_service.common.locks import RequestedLockType

if TYPE_CHECKING:
    from gpu_memory_service.client.memory_manager import GMSClientMemoryManager

logger = logging.getLogger(__name__)
GMS_TAGS = ("weights", "kv_cache")


def get_gms_lock_mode(extra_config: dict):
    """Resolve GMS lock mode from model_loader_extra_config.

    Returns RO if gms_read_only=True, otherwise RW_OR_RO (default).
    """
    if extra_config.get("gms_read_only", False):
        logger.info("[GMS] gms_read_only=True, forcing RO mode")
        return RequestedLockType.RO
    return RequestedLockType.RW_OR_RO


def strip_gms_model_loader_config(load_config, load_format: str):
    """Copy a loader config with GMS-only keys removed for backend loaders."""
    extra_config = getattr(load_config, "model_loader_extra_config", {}) or {}
    return replace(
        load_config,
        load_format=load_format,
        model_loader_extra_config={
            key: value
            for key, value in extra_config.items()
            if not key.startswith("gms_")
        },
    )


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
    """Finalize GMS write mode: register tensors, commit, reconnect in read mode.

    Flow: register tensors -> sync -> unmap + commit -> connect(RO) -> remap

    Args:
        allocator: The GMS client memory manager in write mode.
        model: The loaded model with weights to register.

    Returns:
        Total bytes committed.
    """
    register_module_tensors(allocator, model)
    total_bytes = allocator.total_bytes

    # Synchronize before commit — caller's writes must be visible
    torch.cuda.synchronize()

    allocator.commit()

    allocator.connect(RequestedLockType.RO)
    allocator.remap_all_vas()

    logger.info(
        "[GMS] Committed %.2f GiB, switched to read mode with %d mappings",
        total_bytes / (1 << 30),
        len(allocator._mappings),
    )

    return int(total_bytes)
