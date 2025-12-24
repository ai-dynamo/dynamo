"""GPU Memory Service (GMS) hybrid mode for torch_memory_saver.

This is a HYBRID implementation that combines:
1. GMS allocator for "weights" tag (VA-stable sleep/wake, shared across instances)
2. Torch mempool mode for other tags like "kv_cache" (CPU backup, per-instance)

Key features:
- Weights: VA-stable via GMS, physical memory stays in allocation server for sharing
- KV cache: Managed by torch mempool allocator with CPU backup support
- Best of both worlds: shared weights + full KV cache sleep/wake
- No LD_PRELOAD required!

IMPORTANT: Memory behavior
--------------------------
- Weights (GMS): Physical memory is NOT freed on pause - managed by allocation server
  for sharing between instances. VA mappings are unmapped/remapped.
- KV cache (torch): Physical memory IS freed on pause, optionally backed up to CPU.
  This is the standard torch_memory_saver behavior using PyTorch's CUDAPluggableAllocator.

Usage:
- Set GMS_SGLANG_AUTO_REGISTER=1 or GMS_VLLM_AUTO_REGISTER=1
- That's it! No LD_PRELOAD needed for KV cache support.
"""

import logging
import os
from contextlib import contextmanager
from typing import Optional

from torch_memory_saver.hooks.base import HookUtilBase

logger = logging.getLogger(__name__)


def _get_gms_allocator():
    """Get the GMS allocator from the central registry."""
    try:
        from dynamo.gpu_memory_service.core import get_allocator

        return get_allocator()
    except ImportError:
        return None


def is_gms_mode() -> bool:
    """Check if GMS mode is enabled via environment variable."""
    return (
        os.environ.get("GMS_SGLANG_AUTO_REGISTER") == "1"
        or os.environ.get("GMS_VLLM_AUTO_REGISTER") == "1"
    )


class HookUtilModeGMS(HookUtilBase):
    """GMS hook utility - no binary needed, uses GMS allocator directly."""

    def get_path_binary(self):
        # GMS mode doesn't use a binary
        return None

    def get_allocator(self):
        # GMS mode doesn't need a custom allocator for torch
        return None


class GMSMemorySaverImpl:
    """Hybrid implementation: GMS for weights, torch mempool for KV cache.

    This implementation routes operations based on tag:
    - "weights" or "model_weights": Handled by GMS allocator (VA-stable)
    - Other tags (e.g., "kv_cache"): Delegated to torch mempool mode

    Torch mempool mode is REQUIRED for this hybrid implementation.
    """

    def __init__(self):
        self._disabled = False

        # Initialize torch mempool implementation for non-weights tags
        # This uses PyTorch's CUDAPluggableAllocator - no LD_PRELOAD needed!
        # Torch mempool mode is REQUIRED for hybrid GMS mode.
        from torch_memory_saver.entrypoint import _TorchMemorySaverImpl

        self._torch_impl = _TorchMemorySaverImpl(hook_mode="torch")
        logger.info(
            "[GMS Hybrid] Initialized with torch mempool mode for KV cache support"
        )
        logger.info(
            "[GMS Hybrid] Mode active: "
            "(1) Weights: GMS VA-stable sleep/wake (shared via allocation server); "
            "(2) KV cache: Torch mempool mode with CPU backup support (no LD_PRELOAD needed)"
        )

    def _is_weights_tag(self, tag: Optional[str]) -> bool:
        """Check if tag is for weights (handled by GMS)."""
        return tag in ("weights", "model_weights")

    @contextmanager
    def region(self, tag: str, enable_cpu_backup: bool):
        """Mark allocation region with tag.

        - Weights: No-op (handled by GPUServiceModelLoader)
        - Other tags: Delegate to torch mempool mode
        """
        if self._is_weights_tag(tag):
            # Weights allocation is handled by GPUServiceModelLoader
            yield
        else:
            # Delegate to torch mempool mode for KV cache etc.
            with self._torch_impl.region(tag=tag, enable_cpu_backup=enable_cpu_backup):
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
        import torch

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

        - "weights"/"model_weights": GMS VA-stable sleep
        - Other tags: Delegate to torch mempool mode
        - None (all): Pause both GMS weights and torch allocations
        """
        if self._disabled:
            return

        # Handle weights via GMS
        if tag is None or self._is_weights_tag(tag):
            allocator = _get_gms_allocator()
            if allocator is not None and not allocator.is_sleeping:
                allocator.sleep()
                logger.info("[GMS Hybrid] Paused weights (VA-stable sleep)")
            elif allocator is None and (tag is None or self._is_weights_tag(tag)):
                logger.debug(
                    "[GMS Hybrid] No GMS allocator available for weights pause"
                )

        # Handle other tags via torch mempool mode
        if not self._is_weights_tag(tag):
            torch_tag = None if tag is None else tag
            self._torch_impl.pause(torch_tag)
            logger.info(f"[GMS Hybrid] Paused via torch mempool mode (tag={tag})")

    def resume(self, tag: Optional[str]):
        """Resume memory for the specified tag.

        - "weights"/"model_weights": GMS VA-stable wake
        - Other tags: Delegate to torch mempool mode
        - None (all): Resume both GMS weights and torch allocations
        """
        if self._disabled:
            return

        # Handle weights via GMS
        if tag is None or self._is_weights_tag(tag):
            allocator = _get_gms_allocator()
            if allocator is not None and allocator.is_sleeping:
                allocator.wake()
                logger.info("[GMS Hybrid] Resumed weights (VA-stable wake)")
            elif allocator is None and (tag is None or self._is_weights_tag(tag)):
                logger.debug(
                    "[GMS Hybrid] No GMS allocator available for weights resume"
                )

        # Handle other tags via torch mempool mode
        if not self._is_weights_tag(tag):
            torch_tag = None if tag is None else tag
            self._torch_impl.resume(torch_tag)
            logger.info(f"[GMS Hybrid] Resumed via torch mempool mode (tag={tag})")

    def get_cpu_backup(self, x):
        """Get CPU backup for a tensor.

        Only available for torch mempool allocations (not GMS weights).
        """
        return self._torch_impl.get_cpu_backup(x)
