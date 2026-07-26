# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Common utilities shared across GMS integrations."""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import torch
from gpu_memory_service.client.torch.allocator import prune_allocations
from gpu_memory_service.client.torch.module import (
    rebind_nonparameter_tensors,
    register_module_tensors,
)
from gpu_memory_service.common.locks import RequestedLockType

if TYPE_CHECKING:
    from gpu_memory_service.client.memory_manager import GMSClientMemoryManager

logger = logging.getLogger(__name__)


def torch_device():
    """Return the torch device module (torch.cuda or torch.xpu) for the active VMM device."""
    from gpu_memory_service.common.vmm import VMMDeviceType, get_vmm_device_type

    device_type = get_vmm_device_type()
    if device_type == VMMDeviceType.CUDA:
        return torch.cuda
    if device_type == VMMDeviceType.XPU:
        return torch.xpu
    raise RuntimeError(f"Unsupported VMM device type: {device_type!r}")


@dataclass(frozen=True)
class GMSCommittedMemoryStats:
    committed_bytes: int
    pruned_bytes: int
    pruned_count: int = 0


def get_gms_lock_mode(extra_config: dict):
    """Resolve GMS lock mode from model_loader_extra_config.

    Returns RO if gms_read_only=True, otherwise RW_OR_RO (default).
    """
    if extra_config.get("gms_read_only", False):
        logger.info("[GMS] gms_read_only=True, forcing RO mode")
        return RequestedLockType.RO
    return RequestedLockType.RW_OR_RO


def get_gms_ro_connect_timeout_ms(extra_config: dict) -> int | None:
    """Weight RO reconnect timeout in ms. None waits indefinitely."""
    raw = extra_config.get("gms_ro_connect_timeout_ms")
    if raw is None:
        return None
    timeout_ms = int(raw)
    if timeout_ms < 0:
        raise ValueError(
            f"gms_ro_connect_timeout_ms must be non-negative, got {timeout_ms}"
        )
    return timeout_ms


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


def prepare_gms_write(
    allocator: "GMSClientMemoryManager",
    model: torch.nn.Module,
) -> GMSCommittedMemoryStats:
    """Register model tensors and prune unreferenced allocations.

    This is the first half of a GMS write: it does not commit. The allocator
    keeps its RW lease until :func:`publish_gms_write`, which lets a caller
    defer publication (e.g. until after vLLM memory profiling).

    Args:
        allocator: The GMS client memory manager in write mode.
        model: The loaded model with weights to register.

    Returns:
        Committed/pruned byte stats for the write awaiting publication.
    """
    referenced_allocation_ids = register_module_tensors(allocator, model)
    before_prune_bytes = allocator.total_bytes
    before_prune_count = len(allocator.mappings)

    # prune_allocations synchronizes allocator.device before destroying
    # unreferenced mappings. allocator.commit() performs the publish-barrier
    # sync before committing the remaining registered weights.
    prune_allocations(
        allocator,
        referenced_allocation_ids=referenced_allocation_ids,
    )
    total_bytes = allocator.total_bytes
    pruned_bytes = before_prune_bytes - total_bytes
    pruned_count = before_prune_count - len(allocator.mappings)

    return GMSCommittedMemoryStats(
        committed_bytes=int(total_bytes),
        pruned_bytes=int(pruned_bytes),
        pruned_count=pruned_count,
    )


def publish_gms_write(allocator: "GMSClientMemoryManager") -> None:
    """Publish a prepared write: commit, reconnect read-only, restore VAs."""
    allocator.commit()
    allocator.connect(RequestedLockType.RO)
    allocator.remap_all_vas()


def finalize_gms_write(
    allocator: "GMSClientMemoryManager",
    model: torch.nn.Module,
) -> GMSCommittedMemoryStats:
    """Finalize GMS write mode: register tensors, commit, reconnect in read mode.

    Flow: register tensors -> sync -> unmap + commit -> connect(RO) -> remap
    -> rebind non-parameter tensors to private clones

    The rebind mirrors the importer binding semantics from
    ``materialize_module_from_gms``: after publish, the read-only GMS
    mapping backs parameters only, while buffers and tensor attributes that
    may be written later (fp8 KV scale re-initialization on wake,
    quantization range updates) live in ordinary CUDA memory.

    Args:
        allocator: The GMS client memory manager in write mode.
        model: The loaded model with weights to register.

    Returns:
        Committed/pruned byte stats.
    """
    stats = prepare_gms_write(allocator, model)
    publish_gms_write(allocator)
    rebound_bytes = rebind_nonparameter_tensors(allocator, model)

    logger.info(
        "[GMS] Committed %.2f GiB, switched to read mode with %d mappings "
        "(pruned %d allocations / %.2f GiB before commit; rebound %.2f MiB "
        "of non-parameter tensors to private memory)",
        stats.committed_bytes / (1 << 30),
        len(allocator.mappings),
        stats.pruned_count,
        stats.pruned_bytes / (1 << 30),
        rebound_bytes / (1 << 20),
    )

    return stats
