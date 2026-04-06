# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hybrid torch_memory_saver implementation for GPU Memory Service.

This module uses:
1. GPU Memory Service for "weights" (shared RO/RW publish flow)
2. GPU Memory Service for "kv_cache" (RW-only failover flow)
3. torch_memory_saver for any remaining tags
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import TYPE_CHECKING, Optional

import torch
from gpu_memory_service import get_or_create_gms_client_memory_manager
from gpu_memory_service.client.torch.allocator import gms_use_mem_pool
from gpu_memory_service.common.types import GrantedLockType, RequestedLockType
from gpu_memory_service.common.utils import get_socket_path

if TYPE_CHECKING:
    from gpu_memory_service.client.memory_manager import GMSClientMemoryManager
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
    """Hybrid implementation: GMS for weights and KV cache."""

    def __init__(
        self,
        torch_impl: "_TorchMemorySaverImpl",
        device_index: int,
        mode=None,
    ):
        self._torch_impl = torch_impl
        self._device_index = device_index
        self._requested_mode = mode
        self._disabled = False
        self._imported_weights_bytes: int = 0

        self._weights_allocator: Optional["GMSClientMemoryManager"]
        self._kv_cache_allocator: "GMSClientMemoryManager"
        self._mode: str
        (
            self._weights_allocator,
            self._kv_cache_allocator,
            self._mode,
        ) = self._init_allocators()

        logger.info(
            "[GMS] Initialized weights=%s mode, kv_cache=RW (device=%d)",
            self._mode.upper(),
            device_index,
        )

    def _init_allocators(
        self,
    ) -> tuple[
        Optional["GMSClientMemoryManager"],
        "GMSClientMemoryManager",
        str,
    ]:
        """Create allocator with mode from config (default: RW_OR_RO)."""
        mode = self._requested_mode or RequestedLockType.RW_OR_RO
        weights_allocator = get_or_create_gms_client_memory_manager(
            get_socket_path(self._device_index, "weights"),
            self._device_index,
            mode=mode,
            tag="weights",
        )
        kv_cache_allocator = get_or_create_gms_client_memory_manager(
            get_socket_path(self._device_index, "kv_cache"),
            self._device_index,
            mode=RequestedLockType.RW,
            tag="kv_cache",
        )
        granted_mode = weights_allocator.granted_lock_type
        if granted_mode == GrantedLockType.RW:
            actual_mode = "write"
        else:
            actual_mode = "read"
        logger.info(
            "[GMS] Initialized in AUTO mode, granted=%s (device=%d)",
            actual_mode.upper(),
            self._device_index,
        )
        return weights_allocator, kv_cache_allocator, actual_mode

    def _is_weights_tag(self, tag: Optional[str]) -> bool:
        return tag in ("weights", "model_weights")

    def get_mode(self) -> str:
        return self._mode

    def get_allocator(self) -> Optional["GMSClientMemoryManager"]:
        return self._weights_allocator

    @contextmanager
    def region(self, tag: str, enable_cpu_backup: bool):
        """Mark allocation region with tag."""
        if self._is_weights_tag(tag):
            if self._mode == "read":
                yield
                return

            target_device = torch.device("cuda", self._device_index)
            with gms_use_mem_pool("weights", target_device):
                yield
            return

        if tag == "kv_cache":
            target_device = torch.device("cuda", self._device_index)
            with gms_use_mem_pool("kv_cache", target_device):
                yield
            return

        with self._torch_impl.region(tag=tag, enable_cpu_backup=enable_cpu_backup):
            yield

    def pause(self, tag: Optional[str] = None) -> None:
        if self._disabled:
            return
        if tag is None or self._is_weights_tag(tag):
            self._pause_weights()
        if tag is None or tag == "kv_cache":
            self._pause_kv_cache()
        if tag is None or (not self._is_weights_tag(tag) and tag != "kv_cache"):
            self._torch_impl.pause(tag=tag)

    def resume(self, tag: Optional[str] = None) -> None:
        if self._disabled:
            return
        if tag is None or self._is_weights_tag(tag):
            self._resume_weights()
        if tag is None or tag == "kv_cache":
            self._resume_kv_cache()
        if tag is None or (not self._is_weights_tag(tag) and tag != "kv_cache"):
            self._torch_impl.resume(tag=tag)

    def _pause_weights(self) -> None:
        if self._weights_allocator is None:
            return
        if self._weights_allocator.is_unmapped:
            return
        logger.info("[GMS] Unmapping weights (VA-stable)")
        self._weights_allocator.unmap_all_vas()
        self._weights_allocator.abort()

    def _resume_weights(self) -> None:
        if self._weights_allocator is None:
            return
        if not self._weights_allocator.is_unmapped:
            return
        logger.info("[GMS] Remapping weights (VA-stable)")
        self._weights_allocator.connect(RequestedLockType.RO)
        self._weights_allocator.remap_all_vas()

    def _pause_kv_cache(self) -> None:
        if self._kv_cache_allocator.is_unmapped:
            return
        logger.info("[GMS] Unmapping KV cache")
        self._kv_cache_allocator.unmap_all_vas()
        self._kv_cache_allocator.abort()

    def _resume_kv_cache(self) -> None:
        if not self._kv_cache_allocator.is_unmapped:
            return
        logger.info("[GMS] Remapping KV cache")
        self._kv_cache_allocator.connect(RequestedLockType.RW)
        self._kv_cache_allocator.reallocate_all_handles(tag="kv_cache")
        self._kv_cache_allocator.remap_all_vas()

    def finalize_write_mode(self, model: torch.nn.Module) -> None:
        """Finalize write mode: register tensors, commit, and switch to read."""
        if self._mode != "write":
            return
        if self._weights_allocator is None:
            raise RuntimeError("Allocator is None in WRITE mode")

        from gpu_memory_service.integrations.common.utils import finalize_gms_write

        self._imported_weights_bytes = finalize_gms_write(
            self._weights_allocator, model
        )
        self._mode = "read"

    def set_imported_weights_bytes(self, bytes_count: int) -> None:
        self._imported_weights_bytes = bytes_count

    def get_imported_weights_bytes(self) -> int:
        return self._imported_weights_bytes

    def disable(self) -> None:
        self._disabled = True

    def enable(self) -> None:
        self._disabled = False
