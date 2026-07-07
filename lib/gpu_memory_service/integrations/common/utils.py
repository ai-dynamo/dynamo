# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Common utilities shared across GMS integrations."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
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
GMS_TAGS = ("weights", "kv_cache")


@dataclass(frozen=True)
class GMSCommittedMemoryStats:
    committed_bytes: int
    pruned_bytes: int


@dataclass
class PreparedGMSWrite:
    """A registered and pruned GMS write awaiting publication."""

    allocator: "GMSClientMemoryManager"
    model: torch.nn.Module
    stats: GMSCommittedMemoryStats
    pruned_count: int
    rebound_bytes: int = 0
    _state: str = field(default="prepared", init=False, repr=False)
    _nonparameters_rebound: bool = field(default=False, init=False, repr=False)
    _retained_gms_tensors: list[torch.Tensor] = field(
        default_factory=list, init=False, repr=False
    )

    def rebind_nonparameter_tensors(self) -> int:
        """Move mutable model tensors to private memory once."""
        if self._state in ("aborted", "failed"):
            raise RuntimeError(f"Cannot rebind a {self._state} GMS write")
        if self._nonparameters_rebound:
            return self.rebound_bytes

        retain_gms_tensors = (
            self._retained_gms_tensors if self._state == "prepared" else None
        )
        self.rebound_bytes = rebind_nonparameter_tensors(
            self.allocator,
            self.model,
            retain_gms_tensors=retain_gms_tensors,
        )
        self._nonparameters_rebound = True
        return self.rebound_bytes

    def publish(self) -> GMSCommittedMemoryStats:
        """Commit the write, reconnect read-only, and restore stable VAs."""
        if self._state == "published":
            return self.stats
        if self._state != "prepared":
            raise RuntimeError(f"Cannot publish a {self._state} GMS write")

        self._state = "publishing"
        try:
            self.allocator.commit()
            self.allocator.connect(RequestedLockType.RO)
            self.allocator.remap_all_vas()
        except BaseException:
            self._state = "failed"
            raise

        self._state = "published"
        self._retained_gms_tensors.clear()
        return self.stats

    def abort(self) -> None:
        """Release an unpublished writer without invoking CUDA cleanup."""
        if self._state == "aborted":
            return
        if self._state == "published":
            raise RuntimeError("Cannot abort a published GMS write")

        self._state = "aborted"
        try:
            # Profiling failures may leave CUDA in an error state where normal
            # close synchronizes and calls os._exit. Release the RPC lease
            # first and only clear local bookkeeping.
            self.allocator.close(best_effort=True)
        finally:
            self._retained_gms_tensors.clear()


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
) -> PreparedGMSWrite:
    """Register model tensors and prune unreferenced allocations.

    Args:
        allocator: The GMS client memory manager in write mode.
        model: The loaded model with weights to register.

    Returns:
        A prepared write that retains the RW lock until published or aborted.
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

    return PreparedGMSWrite(
        allocator=allocator,
        model=model,
        stats=GMSCommittedMemoryStats(
            committed_bytes=int(total_bytes),
            pruned_bytes=int(pruned_bytes),
        ),
        pruned_count=pruned_count,
    )


def finalize_gms_write(
    allocator: "GMSClientMemoryManager",
    model: torch.nn.Module,
) -> GMSCommittedMemoryStats:
    """Eagerly register, publish, and rebind a GMS-backed model.

    Existing eager integrations rely on this ordering:
    register -> prune -> commit -> reconnect RO -> remap -> rebind.
    """
    prepared = prepare_gms_write(allocator, model)
    stats = prepared.publish()
    rebound_bytes = prepared.rebind_nonparameter_tensors()

    logger.info(
        "[GMS] Committed %.2f GiB, switched to read mode with %d mappings "
        "(pruned %d allocations / %.2f GiB before commit; rebound %.2f MiB "
        "of non-parameter tensors to private memory)",
        stats.committed_bytes / (1 << 30),
        len(allocator.mappings),
        prepared.pruned_count,
        stats.pruned_bytes / (1 << 30),
        rebound_bytes / (1 << 20),
    )

    return stats
