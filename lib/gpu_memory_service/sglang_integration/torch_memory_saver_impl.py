# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU Memory Service hybrid mode for torch_memory_saver.

This is a HYBRID implementation that combines:
1. GPU Memory Service allocator for "weights" tag (VA-stable unmap/remap, shared across instances)
2. Torch mempool mode for other tags like "kv_cache" (CPU backup, per-instance)

Key features:
- Weights: VA-stable via GPU Memory Service, physical memory stays in allocation server for sharing
- KV cache: Managed by torch mempool allocator with CPU backup support
- Best of both worlds: shared weights + full KV cache sleep/wake
- No LD_PRELOAD required!

IMPORTANT: Memory behavior
--------------------------
- Weights (GPU Memory Service): Physical memory is NOT freed on pause - managed by allocation server
  for sharing between instances. VA mappings are unmapped/remapped.
- KV cache (torch): Physical memory IS freed on pause, optionally backed up to CPU.
  This is the standard torch_memory_saver behavior using PyTorch's CUDAPluggableAllocator.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import TYPE_CHECKING, Optional

import torch
from torch_memory_saver.hooks.base import HookUtilBase

if TYPE_CHECKING:
    from gpu_memory_service.client.memory_manager import GMSClientMemoryManager
    from torch.cuda.memory import MemPool
    from torch_memory_saver.entrypoint import _TorchMemorySaverImpl

logger = logging.getLogger(__name__)

# Module-level reference to the GPU Memory Service impl (one per process)
_gpu_memory_service_impl: Optional["GPUMemoryServiceMemorySaverImpl"] = None


def get_gpu_memory_service_impl() -> Optional["GPUMemoryServiceMemorySaverImpl"]:
    """Get the GPU Memory Service impl if installed."""
    return _gpu_memory_service_impl


def set_gpu_memory_service_impl(impl: "GPUMemoryServiceMemorySaverImpl") -> None:
    """Set the GPU Memory Service impl (called by patch)."""
    global _gpu_memory_service_impl
    _gpu_memory_service_impl = impl


class HookUtilModeGPUMemoryService(HookUtilBase):
    """GPU Memory Service hook utility - no binary needed, uses GPU Memory Service allocator directly."""

    def get_path_binary(self):
        # GPU Memory Service mode doesn't use a binary
        return None

    def get_allocator(self):
        # GPU Memory Service mode doesn't need a custom allocator for torch
        return None


class GPUMemoryServiceMemorySaverImpl:
    """Hybrid implementation: GPU Memory Service for weights, torch mempool for KV cache.

    This implementation routes operations based on tag:
    - "weights" or "model_weights": Handled by GPU Memory Service allocator (VA-stable)
    - Other tags (e.g., "kv_cache"): Delegated to torch mempool mode

    The impl OWNS the GPU Memory Service allocator and uses "auto" mode:
    - First process to connect gets RW lock and loads weights from disk
    - Subsequent processes get RO lock and import weights from metadata

    This enables automatic weight sharing without explicit configuration.
    The connection is established once and lives throughout the worker lifetime.

    Torch mempool mode is REQUIRED for this hybrid implementation.
    """

    def __init__(
        self,
        torch_impl: "_TorchMemorySaverImpl",
        socket_path: str,
        device_index: int,
    ):
        """Initialize impl and create allocator.

        Uses "auto" mode to connect to GMS:
        - First process to connect gets RW lock and loads from disk
        - Subsequent processes get RO lock and import from metadata

        Args:
            torch_impl: The underlying torch_memory_saver impl for non-weights tags.
            socket_path: Unix socket path for the GPU Memory Service allocation server.
            device_index: CUDA device index for this process.
        """
        self._torch_impl = torch_impl
        self._socket_path = socket_path
        self._device_index = device_index
        self._disabled = False

        # Track imported bytes for memory accounting
        self._imported_weights_bytes: int = 0

        # Initialize allocator with auto mode
        self._allocator: Optional["GMSClientMemoryManager"]
        self._mem_pool: Optional["MemPool"]
        self._mode: str
        self._allocator, self._mem_pool, self._mode = self._init_allocator()

        logger.info(
            "[GMS Hybrid] Initialized: weights=%s mode (device=%d, socket=%s), KV cache=torch mempool",
            self._mode.upper(),
            device_index,
            socket_path,
        )

    def _init_allocator(
        self,
    ) -> tuple[Optional["GMSClientMemoryManager"], Optional["MemPool"], str]:
        """Create allocator with automatic mode selection.

        Uses RW_OR_RO mode which tries RW first, falls back to RO if weights
        are already committed. This enables automatic weight sharing.

        Returns:
            Tuple of (allocator, mem_pool, actual_mode). mem_pool is None for READ mode.
        """
        from gpu_memory_service import get_or_create_gms_client_memory_manager
        from gpu_memory_service.common.types import GrantedLockType, RequestedLockType

        allocator, mem_pool = get_or_create_gms_client_memory_manager(
            self._socket_path,
            self._device_index,
            mode=RequestedLockType.RW_OR_RO,
            tag="weights",
        )
        granted_mode = allocator.mode
        if granted_mode == GrantedLockType.RW:
            allocator.clear_all()
            actual_mode = "write"
        else:
            actual_mode = "read"
        logger.info(
            "[GMS] Initialized in AUTO mode, granted=%s (device=%d)",
            actual_mode.upper(),
            self._device_index,
        )
        return allocator, mem_pool if granted_mode == GrantedLockType.RW else None, actual_mode

    def _is_weights_tag(self, tag: Optional[str]) -> bool:
        """Check if tag is for weights (handled by GPU Memory Service)."""
        return tag in ("weights", "model_weights")

    def get_mode(self) -> str:
        """Return the current mode ('write' or 'read')."""
        return self._mode

    def get_allocator(self) -> Optional["GMSClientMemoryManager"]:
        """Return the GMS allocator."""
        return self._allocator

    @contextmanager
    def region(self, tag: str, enable_cpu_backup: bool):
        """Mark allocation region with tag.

        - Weights (WRITE mode): Route allocations through GPU Memory Service mempool
        - Weights (READ mode): No-op (weights will be materialized from metadata)
        - Other tags: Delegate to torch mempool mode with synchronization
        """
        if not self._is_weights_tag(tag):
            # Delegate to torch mempool mode for KV cache etc.
            with self._torch_impl.region(tag=tag, enable_cpu_backup=enable_cpu_backup):
                yield
            torch.cuda.synchronize()
            logger.debug("[GMS Hybrid] Synchronized after region context (tag=%s)", tag)
            return

        # Weights handling
        if self._mode == "read":
            # READ mode: no mempool context needed, weights will be materialized
            logger.debug("[GMS Hybrid] region(weights) in READ mode - no-op")
            yield
            return

        # WRITE mode: route through GMS mempool
        if self._mem_pool is None:
            raise RuntimeError("GMS mempool is None in WRITE mode - this should not happen")

        target_device = torch.device("cuda", self._device_index)
        logger.debug("[GMS Hybrid] region(weights) WRITE mode - using GMS mempool")

        with torch.cuda.use_mem_pool(self._mem_pool, device=target_device):
            yield

        torch.cuda.synchronize()
        logger.debug("[GMS Hybrid] Synchronized after weights region context")

    def pause(self, tag: Optional[str] = None) -> None:
        """Pause memory for a tag (unmap weights / release KV cache)."""
        if self._disabled:
            return

        if tag is None or self._is_weights_tag(tag):
            self._pause_weights()

        if tag is None or not self._is_weights_tag(tag):
            self._torch_impl.pause(tag=tag)

    def resume(self, tag: Optional[str] = None) -> None:
        """Resume memory for a tag (remap weights / restore KV cache)."""
        if self._disabled:
            return

        if tag is None or self._is_weights_tag(tag):
            self._resume_weights()

        if tag is None or not self._is_weights_tag(tag):
            self._torch_impl.resume(tag=tag)

    def _pause_weights(self) -> None:
        """Unmap GMS weights (VA-stable)."""
        if self._allocator is None:
            logger.warning("[GMS Hybrid] No allocator for pause_weights")
            return

        if self._allocator.is_unmapped:
            logger.debug("[GMS Hybrid] Weights already unmapped")
            return

        logger.info("[GMS Hybrid] Unmapping weights (VA-stable)")
        self._allocator.unmap()

    def _resume_weights(self) -> None:
        """Remap GMS weights (VA-stable)."""
        if self._allocator is None:
            logger.warning("[GMS Hybrid] No allocator for resume_weights")
            return

        if not self._allocator.is_unmapped:
            logger.debug("[GMS Hybrid] Weights already mapped")
            return

        logger.info("[GMS Hybrid] Remapping weights (VA-stable)")
        self._allocator.remap()
        torch.cuda.synchronize()

    def finalize_write_mode(self, model: torch.nn.Module) -> None:
        """Finalize write mode: register tensors, commit, and switch to read.

        This should be called after the model has been loaded in write mode.
        """
        if self._mode != "write":
            logger.debug("[GMS Hybrid] finalize_write_mode called but mode is %s, skipping", self._mode)
            return

        if self._allocator is None:
            raise RuntimeError("Allocator is None in WRITE mode - this should not happen")

        from gpu_memory_service.client.torch.module import register_module_tensors

        # Register tensors in the GMS metadata store
        register_module_tensors(self._allocator, model)
        total_bytes = self._allocator.total_bytes
        self._imported_weights_bytes = int(total_bytes)

        # Commit and switch to read mode
        torch.cuda.synchronize()

        if not self._allocator.commit():
            raise RuntimeError("GPU Memory Service allocation server commit failed")

        self._allocator.switch_to_read()
        self._mode = "read"

        logger.info(
            "[GMS Hybrid] Committed %.2f GiB, switched to read mode with %d mappings",
            total_bytes / (1 << 30),
            len(self._allocator._mappings),
        )

    def set_imported_weights_bytes(self, bytes_count: int) -> None:
        """Set imported weights bytes (for import-only mode)."""
        self._imported_weights_bytes = bytes_count

    def get_imported_weights_bytes(self) -> int:
        """Get the total bytes of imported/committed weights."""
        return self._imported_weights_bytes

    def disable(self) -> None:
        """Disable the impl (no-op all operations)."""
        self._disabled = True

    def enable(self) -> None:
        """Enable the impl."""
        self._disabled = False
