"""GPU Memory Service (GPU Memory Service) hybrid mode for torch_memory_saver.

This is a HYBRID implementation that combines:
1. GPU Memory Service allocator for "weights" tag (VA-stable sleep/wake, shared across instances)
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

Usage:
- Set GPU_MEMORY_SERVICE_SGLANG_AUTO_REGISTER=1 or GPU_MEMORY_SERVICE_VLLM_AUTO_REGISTER=1
- That's it! No LD_PRELOAD needed for KV cache support.
"""

from __future__ import annotations

import logging
import os
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


def is_gpu_memory_service_mode() -> bool:
    """Check if GPU Memory Service mode is enabled via environment variable."""
    return (
        os.environ.get("GPU_MEMORY_SERVICE_SGLANG_AUTO_REGISTER") == "1"
        or os.environ.get("GPU_MEMORY_SERVICE_VLLM_AUTO_REGISTER") == "1"
    )


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

    The impl OWNS the GPU Memory Service allocator. Mode must be specified explicitly:
    - "write": Cold start from disk, publish weights
    - "read": Import-only from existing committed weights

    Torch mempool mode is REQUIRED for this hybrid implementation.
    """

    def __init__(
        self,
        torch_impl: "_TorchMemorySaverImpl",
        socket_path: str,
        device_index: int,
        mode: str = "write",
    ):
        """Initialize impl and create allocator.

        The impl is created lazily via patched _ensure_initialized() when
        torch_memory_saver.region() is first called. At that point we're about
        to load model weights, so we need the allocator immediately.

        Args:
            torch_impl: The underlying torch_memory_saver impl for non-weights tags.
            socket_path: Unix socket path for the GPU Memory Service allocation server.
            device_index: CUDA device index for this process.
            mode: Explicit mode - "write" (cold start) or "read" (import only).
        """
        self._torch_impl = torch_impl
        self._socket_path = socket_path
        self._device_index = device_index
        self._disabled = False
        self._requested_mode = mode

        # Track imported bytes for memory accounting
        self._imported_weights_bytes: int = 0

        # Initialize allocator with explicit mode
        self._allocator: Optional["GMSClientMemoryManager"]
        self._mem_pool: Optional["MemPool"]
        self._mode: str
        self._allocator, self._mem_pool, self._mode = self._init_allocator(mode)

        logger.info(
            "[GPU Memory Service Hybrid] Initialized with torch mempool mode for KV cache support"
        )
        logger.info(
            "[GPU Memory Service Hybrid] Mode active: "
            "(1) Weights: GPU Memory Service %s mode (device=%d, socket=%s); "
            "(2) KV cache: Torch mempool mode with CPU backup support",
            self._mode.upper(),
            device_index,
            socket_path,
        )

    def _init_allocator(
        self,
        mode: str,
    ) -> tuple[Optional["GMSClientMemoryManager"], Optional["MemPool"], str]:
        """Create allocator with explicit mode.

        Args:
            mode: "write" for cold start from disk, "read" for import-only.

        Returns:
            Tuple of (allocator, mem_pool, mode). mem_pool is None for READ mode.
        """
        from gpu_memory_service import (
            GMSClientMemoryManager,
            get_or_create_allocator,
            register_allocator,
        )

        if mode == "read":
            # Import-only fast path - wait indefinitely for RO lock
            allocator = GMSClientMemoryManager(
                self._socket_path,
                mode="read",
                device=self._device_index,
            )
            # Register in lifecycle module for consumers (sleep/wake via get_allocator())
            register_allocator(allocator)
            logger.info(
                "[GPU Memory Service] Initialized in READ mode (import-only, device=%d)",
                self._device_index,
            )
            return allocator, None, "read"
        else:
            # WRITE mode - use lifecycle.py for full PyTorch integration
            allocator, mem_pool = get_or_create_allocator(
                self._socket_path,
                self._device_index,
                mode="write",
                tag="weights",
            )
            # Clear any stale state from previous runs
            allocator.clear_all()
            logger.info(
                "[GPU Memory Service] Initialized in WRITE mode (device=%d)",
                self._device_index,
            )
            return allocator, mem_pool, "write"

    def _is_weights_tag(self, tag: Optional[str]) -> bool:
        """Check if tag is for weights (handled by GPU Memory Service)."""
        return tag in ("weights", "model_weights")

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
            logger.debug(
                "[GPU Memory Service Hybrid] Synchronized after region context (tag=%s)",
                tag,
            )
            return

        # Weights region - use GPU Memory Service allocator
        if self._mode == "read":
            # Import-only mode: no allocations, weights will be materialized
            logger.debug(
                "[GPU Memory Service] Weights region (READ mode) - no allocations"
            )
            yield
        else:
            # Write mode: route allocations through mempool
            assert self._mem_pool is not None, "MemPool should exist in WRITE mode"
            target_device = torch.device(f"cuda:{self._device_index}")
            logger.debug(
                "[GPU Memory Service] Weights region (WRITE mode) - routing via mempool"
            )
            with torch.cuda.use_mem_pool(self._mem_pool, device=target_device):
                yield

    @contextmanager
    def cuda_graph(
        self,
        cuda_graph,
        pool,
        stream,
        capture_error_mode,
        tag: str,
        enable_cpu_backup: bool,
    ):
        """CUDA graph capture with memory tagging.

        - Weights: Standard torch.cuda.graph (no special handling)
        - Other tags: Delegate to torch mempool mode
        """
        if self._is_weights_tag(tag):
            # Weights don't need special CUDA graph handling
            with torch.cuda.graph(
                cuda_graph,
                pool=pool,
                stream=stream,
                capture_error_mode=capture_error_mode,
            ):
                yield
        else:
            # Delegate to torch mempool mode for pauseable CUDA graphs
            with self._torch_impl.cuda_graph(
                cuda_graph=cuda_graph,
                pool=pool,
                stream=stream,
                capture_error_mode=capture_error_mode,
                tag=tag,
                enable_cpu_backup=enable_cpu_backup,
            ):
                yield

    @contextmanager
    def disable(self):
        """Temporarily disable memory saving."""
        prev = self._disabled
        self._disabled = True
        try:
            with self._torch_impl.disable():
                yield
        finally:
            self._disabled = prev

    def pause(self, tag: Optional[str]):
        """Pause memory for the specified tag.

        - "weights"/"model_weights": GPU Memory Service VA-stable sleep
        - Other tags: Delegate to torch mempool mode
        - None (all): Pause both GPU Memory Service weights and torch allocations
        """
        if self._disabled:
            return

        # Handle weights via GPU Memory Service
        if tag is None or self._is_weights_tag(tag):
            if self._allocator is not None and not self._allocator.is_sleeping:
                self._allocator.sleep()
                logger.info(
                    "[GPU Memory Service Hybrid] Paused weights (VA-stable sleep)"
                )

        # Handle other tags via torch mempool mode
        if tag is None or not self._is_weights_tag(tag):
            torch_tag = None if tag is None else tag
            self._torch_impl.pause(torch_tag)
            if tag is not None:
                logger.info(
                    "[GPU Memory Service Hybrid] Paused via torch mempool mode (tag=%s)",
                    tag,
                )

        torch.cuda.synchronize()

    def resume(self, tag: Optional[str]):
        """Resume memory for the specified tag.

        - "weights"/"model_weights": GPU Memory Service VA-stable wake
        - Other tags: Delegate to torch mempool mode
        - None (all): Resume both GPU Memory Service weights and torch allocations
        """
        if self._disabled:
            return

        # Handle weights via GPU Memory Service
        if tag is None or self._is_weights_tag(tag):
            if self._allocator is not None and self._allocator.is_sleeping:
                self._allocator.wake()
                logger.info(
                    "[GPU Memory Service Hybrid] Resumed weights (VA-stable wake)"
                )

        # Handle other tags via torch mempool mode
        if tag is None or not self._is_weights_tag(tag):
            torch_tag = None if tag is None else tag
            self._torch_impl.resume(torch_tag)
            if tag is not None:
                logger.info(
                    "[GPU Memory Service Hybrid] Resumed via torch mempool mode (tag=%s)",
                    tag,
                )

        torch.cuda.synchronize()

    def get_cpu_backup(self, x):
        """Get CPU backup for a tensor.

        Only available for torch mempool allocations (not GPU Memory Service weights).
        """
        return self._torch_impl.get_cpu_backup(x)

    # --- Accessor methods for model loader ---

    def get_allocator(self) -> Optional["GMSClientMemoryManager"]:
        """Get the allocator (for model loader to use)."""
        return self._allocator

    def get_mode(self) -> str:
        """Get current mode ("read" or "write")."""
        return self._mode

    def get_device_index(self) -> int:
        """Get the CUDA device index."""
        return self._device_index

    def get_socket_path(self) -> str:
        """Get the socket path."""
        return self._socket_path

    def finalize_write_mode(self, model: torch.nn.Module, config_hash: str) -> None:
        """Called after model loading in WRITE mode to commit and switch to read.

        This registers all model tensors in the GMS metadata store,
        commits the allocations, and switches to read mode.

        Args:
            model: The loaded model with weights on GPU.
            config_hash: Hash prefix for metadata keys.
        """
        if self._mode != "write":
            logger.debug(
                "[GPU Memory Service] finalize_write_mode called but mode is %s, skipping",
                self._mode,
            )
            return

        if self._allocator is None:
            raise RuntimeError(
                "Allocator is None in WRITE mode - this should not happen"
            )

        from gpu_memory_service.client.torch.extensions import _allocator_ext as cumem
        from gpu_memory_service.client.torch.tensor import register_module_tensors

        # Register tensors in the GMS metadata store
        total_bytes = register_module_tensors(
            self._allocator, model, metadata_prefix=f"{config_hash}:"
        )
        self._imported_weights_bytes = int(total_bytes)

        # Commit and switch to read mode
        torch.cuda.synchronize()
        cumem.set_access_all(True)

        if not self._allocator.commit():
            raise RuntimeError("GPU Memory Service allocation server commit failed")

        self._allocator.switch_to_read()
        self._mode = "read"

        logger.info(
            "[GPU Memory Service] Committed %.2f GiB, switched to read mode with %d mappings",
            total_bytes / (1 << 30),
            len(self._allocator._mappings),
        )

    def set_imported_weights_bytes(self, bytes_count: int) -> None:
        """Set imported weights bytes (for import-only mode)."""
        self._imported_weights_bytes = bytes_count

    def get_imported_weights_bytes(self) -> int:
        """Get the total bytes of imported/committed weights."""
        return self._imported_weights_bytes
