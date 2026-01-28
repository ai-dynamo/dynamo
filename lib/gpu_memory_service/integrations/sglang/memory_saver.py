# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hybrid torch_memory_saver implementation for GPU Memory Service.

This module provides a hybrid implementation that combines:
1. GPU Memory Service allocator for "weights" tag (VA-stable unmap/remap, shared)
2. Torch mempool mode for other tags like "kv_cache" (CPU backup, per-instance)

The impl uses RW_OR_RO mode to connect to GMS:
- First process gets RW lock and loads weights from disk
- Subsequent processes get RO lock and import weights from metadata
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import TYPE_CHECKING, Optional

import torch

if TYPE_CHECKING:
    from gpu_memory_service.client.memory_manager import GMSClientMemoryManager
    from torch.cuda.memory import MemPool
    from torch_memory_saver.entrypoint import _TorchMemorySaverImpl

logger = logging.getLogger(__name__)


def get_gms_memory_saver_impl() -> Optional["GMSMemorySaverImpl"]:
    """Get the GMS memory saver impl from the torch_memory_saver singleton."""
    try:
        import torch_memory_saver

        return torch_memory_saver.torch_memory_saver.gms_impl
    except (ImportError, AttributeError):
        return None


class GMSMemorySaverImpl:
    """Hybrid implementation: GMS for weights, torch mempool for KV cache.

    Routes operations based on tag:
    - "weights" or "model_weights": Handled by GMS allocator (VA-stable)
    - Other tags (e.g., "kv_cache"): Delegated to torch mempool mode
    """

    def __init__(
        self,
        torch_impl: "_TorchMemorySaverImpl",
        socket_path: str,
        device_index: int,
    ):
        self._torch_impl = torch_impl
        self._socket_path = socket_path
        self._device_index = device_index
        self._disabled = False
        self._imported_weights_bytes: int = 0

        # Initialize allocator with auto mode
        self._allocator: Optional["GMSClientMemoryManager"]
        self._mem_pool: Optional["MemPool"]
        self._mode: str
        self._allocator, self._mem_pool, self._mode = self._init_allocator()

        logger.info(
            "[GMS] Initialized: weights=%s mode (device=%d, socket=%s)",
            self._mode.upper(),
            device_index,
            socket_path,
        )

    def _init_allocator(
        self,
    ) -> tuple[Optional["GMSClientMemoryManager"], Optional["MemPool"], str]:
        """Create allocator with automatic mode selection."""
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
        return (
            allocator,
            mem_pool if granted_mode == GrantedLockType.RW else None,
            actual_mode,
        )

    def _is_weights_tag(self, tag: Optional[str]) -> bool:
        return tag in ("weights", "model_weights")

    def get_mode(self) -> str:
        return self._mode

    def get_allocator(self) -> Optional["GMSClientMemoryManager"]:
        return self._allocator

    @contextmanager
    def region(self, tag: str, enable_cpu_backup: bool):
        """Mark allocation region with tag."""
        if not self._is_weights_tag(tag):
            with self._torch_impl.region(tag=tag, enable_cpu_backup=enable_cpu_backup):
                yield
            return

        if self._mode == "read":
            yield
            return

        if self._mem_pool is None:
            raise RuntimeError("GMS mempool is None in WRITE mode")

        target_device = torch.device("cuda", self._device_index)
        with torch.cuda.use_mem_pool(self._mem_pool, device=target_device):
            yield

    def pause(self, tag: Optional[str] = None) -> None:
        if self._disabled:
            return
        if tag is None or self._is_weights_tag(tag):
            self._pause_weights()
        if tag is None or not self._is_weights_tag(tag):
            self._torch_impl.pause(tag=tag)

    def resume(self, tag: Optional[str] = None) -> None:
        if self._disabled:
            return
        if tag is None or self._is_weights_tag(tag):
            self._resume_weights()
        if tag is None or not self._is_weights_tag(tag):
            self._torch_impl.resume(tag=tag)

    def _pause_weights(self) -> None:
        if self._allocator is None:
            return
        if self._allocator.is_unmapped:
            return
        logger.info("[GMS] Unmapping weights (VA-stable)")
        self._allocator.unmap()

    def _resume_weights(self) -> None:
        if self._allocator is None:
            return
        if not self._allocator.is_unmapped:
            return
        logger.info("[GMS] Remapping weights (VA-stable)")
        self._allocator.remap()
        torch.cuda.synchronize()

    def finalize_write_mode(self, model: torch.nn.Module) -> None:
        """Finalize write mode: register tensors, commit, and switch to read."""
        if self._mode != "write":
            return
        if self._allocator is None:
            raise RuntimeError("Allocator is None in WRITE mode")

        from gpu_memory_service.integrations.common.utils import finalize_gms_write

        self._imported_weights_bytes = finalize_gms_write(self._allocator, model)
        self._mode = "read"

    def set_imported_weights_bytes(self, bytes_count: int) -> None:
        self._imported_weights_bytes = bytes_count

    def get_imported_weights_bytes(self) -> int:
        return self._imported_weights_bytes

    def disable(self) -> None:
        self._disabled = True

    def enable(self) -> None:
        self._disabled = False
