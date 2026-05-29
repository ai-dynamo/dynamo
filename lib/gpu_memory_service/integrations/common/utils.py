# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Common utilities shared across GMS integrations."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

import torch
from gpu_memory_service.client.torch.module import (
    collect_module_tensor_names,
    register_module_tensors,
)
from gpu_memory_service.common.locks import RequestedLockType

if TYPE_CHECKING:
    from gpu_memory_service.client.memory_manager import GMSClientMemoryManager

logger = logging.getLogger(__name__)
GMS_TAGS = ("weights", "kv_cache")


@dataclass
class _StagedGMSWrite:
    allocator: "GMSClientMemoryManager"
    models: list[tuple[str | None, torch.nn.Module, frozenset[str]]] = field(
        default_factory=list
    )


_staged_gms_writes: dict[int, _StagedGMSWrite] = {}


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


def stage_gms_write_model(
    allocator: "GMSClientMemoryManager",
    model: torch.nn.Module,
    *,
    namespace: str | None = None,
) -> None:
    """Stage a model for a later GMS weight publish.

    The model object is intentionally stored by reference so engine-level
    post-load mutations, such as target/draft embedding sharing, are visible
    when metadata is registered just before the final commit.

    Args:
        allocator: The GMS client memory manager in write mode.
        model: The loaded model whose tensors should be registered later.
        namespace: Optional metadata namespace for this model.
    """
    staged = _staged_gms_writes.setdefault(
        id(allocator), _StagedGMSWrite(allocator=allocator)
    )
    for staged_namespace, _, _ in staged.models:
        if staged_namespace == namespace:
            raise RuntimeError(f"GMS write namespace {namespace!r} is already staged")
    tensor_names = collect_module_tensor_names(model)
    staged.models.append((namespace, model, tensor_names))
    logger.info(
        "[GMS] Staged model for write publish "
        "(namespace=%r, tensors=%d, staged=%d)",
        namespace,
        len(tensor_names),
        len(staged.models),
    )


def has_staged_gms_write(allocator: "GMSClientMemoryManager") -> bool:
    """Return whether there are models staged for this allocator."""
    staged = _staged_gms_writes.get(id(allocator))
    return bool(staged is not None and staged.models)


def finalize_staged_gms_write(allocator: "GMSClientMemoryManager") -> int:
    """Register all staged model tensors, then finalize one GMS write layout.

    Returns 0 when nothing is staged for the allocator.
    """
    staged = _staged_gms_writes.pop(id(allocator), None)
    if staged is None or not staged.models:
        return 0

    for namespace, model, tensor_names in staged.models:
        register_module_tensors(
            allocator,
            model,
            namespace=namespace,
            tensor_names=tensor_names,
        )

    return finalize_gms_write(allocator)


def finalize_gms_write(allocator: "GMSClientMemoryManager") -> int:
    """Commit the current GMS write layout, then reconnect and remap read-only.

    Flow: sync -> unmap + commit -> connect(RO) -> remap

    Args:
        allocator: The GMS client memory manager in write mode.

    Returns:
        Total bytes committed.
    """
    total_bytes = allocator.total_bytes

    # Synchronize before commit — caller's writes must be visible
    torch.cuda.synchronize()

    allocator.commit()

    allocator.connect(RequestedLockType.RO)
    allocator.remap_all_vas()

    logger.info(
        "[GMS] Committed %.2f GiB, switched to read mode with %d mappings",
        total_bytes / (1 << 30),
        len(allocator.mappings),
    )

    return int(total_bytes)
