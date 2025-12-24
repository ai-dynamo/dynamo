"""SGLang integration for GPU Memory Service (Allocation Server + embedded registry).

This module provides a custom SGLang `load_format` class that can be passed as
`LoadConfig(load_format=...)`. It integrates with the GPUMemoryServiceMemorySaverImpl
which owns the GPU Memory Service allocator.

Flow:
1. torch_memory_saver patch creates GPUMemoryServiceMemorySaverImpl (owns allocator)
2. SGLang's ModelRunner calls region("weights") which sets up use_mem_pool() in WRITE mode
3. GPUServiceModelLoader.load_model() delegates to DefaultModelLoader (allocations routed via mempool)
4. After loading, GPUServiceModelLoader calls impl.finalize_write_mode() to commit and switch to read

For import-only mode (READ):
- Impl detects weights are already committed via timeout=0 probe
- GPUServiceModelLoader creates a meta model and materializes from registry

IMPORTANT: Sleep/Wake Memory Behavior
-------------------------------------
When using GPU Memory Service with SGLang's --enable-memory-saver, the sleep/wake functionality
does NOT actually free GPU memory. The physical memory for model weights remains
allocated by the Allocation Server. This is by design for weight sharing:

- The Allocation Server owns the physical memory for weights
- On sleep, the client unmaps its local VA mappings but the server keeps the memory
- On wake, the client remaps the same weights without reloading from disk

This enables fast context switching between inference instances.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import replace
from typing import Any, Optional

import torch

logger = logging.getLogger(__name__)


# =============================================================================
# CRITICAL: Patch torch.cuda.empty_cache at module import time.
#
# When weights are allocated through our VMM-based pluggable allocator, calling
# torch.cuda.empty_cache() causes segfaults because the native caching allocator
# tries to release blocks that were allocated through VMM APIs.
#
# This patch is applied when this module is imported (which happens in the
# subprocess where model loading occurs), ensuring it's active before any
# empty_cache calls.
# =============================================================================
_original_empty_cache = torch.cuda.empty_cache
_empty_cache_patched = False


def _safe_empty_cache() -> None:
    """Safe replacement for torch.cuda.empty_cache that skips when VMM allocations exist."""
    global _original_empty_cache
    # Check if we have GPU Memory Service VMM allocations
    try:
        from gpu_memory_service.extensions import _rpc_cumem_ext as cumem

        allocations = cumem.get_all_allocations()
        if allocations:
            # We have VMM allocations - skip empty_cache to prevent segfault
            logger.debug(
                "[GPU Memory Service] Skipping torch.cuda.empty_cache() - %d VMM allocations active",
                len(allocations),
            )
            return
    except Exception:
        pass
    # No GPU Memory Service allocations, safe to call original
    _original_empty_cache()


def _patch_empty_cache_if_needed() -> None:
    """Apply the empty_cache patch if not already applied."""
    global _empty_cache_patched
    if _empty_cache_patched:
        return
    torch.cuda.empty_cache = _safe_empty_cache
    _empty_cache_patched = True
    logger.info(
        "[GPU Memory Service] Patched torch.cuda.empty_cache for VMM allocation safety"
    )


# Apply patch immediately at module import
_patch_empty_cache_if_needed()


def _get_local_rank() -> int:
    """Get the local rank (GPU device index) for the current worker.

    Priority order:
    1. torch.cuda.current_device() if already set (SGLang sets this early)
    2. torch.distributed rank if initialized
    3. LOCAL_RANK environment variable
    4. Default to 0
    """
    # First check if CUDA device is already set (SGLang sets this in worker init)
    try:
        if torch.cuda.is_initialized():
            current_device = torch.cuda.current_device()
            if current_device != 0 or os.environ.get("LOCAL_RANK", "0") == "0":
                # Only trust current_device if it's non-zero OR LOCAL_RANK confirms 0
                return current_device
    except Exception:
        pass

    # Try torch.distributed if initialized
    try:
        import torch.distributed as dist

        if dist.is_initialized():
            return int(dist.get_rank() % torch.cuda.device_count())
    except Exception:
        pass

    # Fall back to environment variable
    return int(os.environ.get("LOCAL_RANK", "0"))


def _parse_extra_config(load_config: Any = None) -> dict:
    """Parse model_loader_extra_config from load_config.

    SGLang stores model_loader_extra_config as a string that needs to be parsed.
    """
    if load_config is None:
        return {}
    raw_config = getattr(load_config, "model_loader_extra_config", None)
    if not raw_config:
        return {}
    if isinstance(raw_config, str):
        return json.loads(raw_config)
    return raw_config


# Keys that GPU Memory Service adds to model_loader_extra_config - must be stripped before
# passing to DefaultModelLoader which validates unknown keys
GPU_MEMORY_SERVICE_EXTRA_CONFIG_KEYS = {
    "gpu_memory_service_socket_path",
    "gpu_memory_service_load_mode",
}


def _strip_gpu_memory_service_extra_config(load_config: Any) -> Any:
    """Return a copy of load_config with GPU Memory Service keys removed from model_loader_extra_config.

    SGLang's DefaultModelLoader validates model_loader_extra_config and rejects
    unknown keys. This strips GPU Memory Service-specific keys so we can delegate to DefaultModelLoader.
    """
    extra_config = _parse_extra_config(load_config)
    if not extra_config:
        return load_config

    # Remove GPU Memory Service keys
    cleaned = {
        k: v
        for k, v in extra_config.items()
        if k not in GPU_MEMORY_SERVICE_EXTRA_CONFIG_KEYS
    }

    # Create new load_config with cleaned extra_config
    # SGLang's DefaultModelLoader expects model_loader_extra_config as a dict (not None, not string)
    return replace(load_config, model_loader_extra_config=cleaned if cleaned else {})


def compute_sglang_config_hash(model_config: Any) -> str:
    """Best-effort stable hash for registry key prefixing."""
    payload = {
        "model_path": getattr(model_config, "model_path", None),
        "revision": getattr(model_config, "revision", None),
        "dtype": str(getattr(model_config, "dtype", None)),
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


class GPUServiceModelLoader:
    """Custom SGLang model loader using GPU Memory Service.

    This loader integrates with GPUMemoryServiceMemorySaverImpl which owns the allocator.
    The impl is created by the torch_memory_saver patch before model loading starts.

    In WRITE mode:
    - Allocations are routed through region("weights") -> use_mem_pool()
    - After loading, finalize_write_mode() commits and switches to read mode

    In READ mode (import-only):
    - Weights are materialized from the GPU Memory Service registry
    """

    # Exported for memory accounting patches
    _imported_weights_bytes: int = 0

    def __init__(self, load_config):
        self.load_config = load_config

    def download_model(self, model_config) -> None:
        try:
            from sglang.srt.model_loader.loader import DefaultModelLoader
        except ImportError as e:
            raise RuntimeError(f"SGLang not installed or incompatible: {e}")
        # Create a copy with valid load_format and stripped GPU Memory Service keys for DefaultModelLoader
        disk_load_config = _strip_gpu_memory_service_extra_config(self.load_config)
        disk_load_config = replace(disk_load_config, load_format="auto")
        DefaultModelLoader(disk_load_config).download_model(model_config)

    def load_model(self, *, model_config, device_config):
        """Load model using GPU Memory Service allocator.

        Gets the allocator from GPUMemoryServiceMemorySaverImpl (created by patch).
        Depending on mode:
        - READ: Materialize weights from registry
        - WRITE: Load from disk (allocations routed via region), then finalize
        """
        from dynamo.sglang.gpu_memory_service_adapters.torch_memory_saver_impl import (
            get_gpu_memory_service_impl,
        )

        impl = get_gpu_memory_service_impl()
        if impl is None:
            raise RuntimeError(
                "GPU Memory Service impl not initialized. "
                "Ensure torch_memory_saver patch is applied and "
                "GPU_MEMORY_SERVICE_SGLANG_AUTO_REGISTER=1 is set."
            )

        # Use device from impl (set at impl creation time from torch.cuda.current_device())
        device_index = (
            device_config.gpu_id if device_config.gpu_id >= 0 else impl.get_device_index()
        )
        config_hash = compute_sglang_config_hash(model_config)

        logger.info(
            "[GPU Memory Service] GPUServiceModelLoader.load_model: mode=%s, device=%d",
            impl.get_mode(),
            device_index,
        )

        if impl.get_mode() == "read":
            # Import-only mode: materialize from registry
            return self._load_import_only(impl, model_config, device_config, config_hash)
        else:
            # Write mode: load from disk
            # Allocations are routed via region("weights") -> use_mem_pool()
            # which is already set up by ModelRunner before calling load_model
            return self._load_write_mode(impl, model_config, device_config, config_hash)

    def _load_import_only(
        self, impl, model_config, device_config, config_hash: str
    ) -> torch.nn.Module:
        """Import weights from GPU Memory Service registry (READ mode)."""
        from gpu_memory_service.tensor import materialize_module_from_registry

        from dynamo.sglang.gpu_memory_service_adapters.import_only_loader import (
            ImportOnlyModelLoader,
        )

        allocator = impl.get_allocator()
        if allocator is None:
            raise RuntimeError("Allocator is None in READ mode - this should not happen")

        # Create meta model (model structure without weights)
        import_only_loader = ImportOnlyModelLoader(self.load_config)
        model = import_only_loader.load_model(
            model_config=model_config,
            device_config=device_config,
        )

        # Materialize weights from registry
        imported_bytes = materialize_module_from_registry(
            allocator,
            model,
            prefix=f"{config_hash}:",
            device_index=impl.get_device_index(),
            strict=True,
        )

        # Track bytes for memory accounting
        impl.set_imported_weights_bytes(int(imported_bytes))
        GPUServiceModelLoader._imported_weights_bytes = int(imported_bytes)

        logger.info(
            "[GPU Memory Service] Import-only loaded %.2f GiB from registry",
            imported_bytes / (1 << 30),
        )

        return model.eval()

    def _load_write_mode(
        self, impl, model_config, device_config, config_hash: str
    ) -> torch.nn.Module:
        """Load model from disk (WRITE mode).

        Allocations are routed through the mempool via region("weights") which
        is already active when this method is called. After loading, we call
        impl.finalize_write_mode() to register tensors, commit, and switch to read.
        """
        try:
            from sglang.srt.model_loader.loader import DefaultModelLoader
        except ImportError as e:
            raise RuntimeError(f"SGLang not installed or incompatible: {e}")

        allocator = impl.get_allocator()
        if allocator is None:
            raise RuntimeError("Allocator is None in WRITE mode - this should not happen")

        # Clear any stale registry entries for this config
        allocator.registry_delete_prefix(f"{config_hash}:")

        # Load model - allocations routed through region("weights") -> use_mem_pool()
        # The mempool context is already set up by ModelRunner.load_model() which wraps
        # the model loader call in region("weights")
        disk_load_config = _strip_gpu_memory_service_extra_config(self.load_config)
        disk_load_config = replace(disk_load_config, load_format="auto")

        logger.info("[GPU Memory Service] Loading model from disk via DefaultModelLoader")
        model = DefaultModelLoader(disk_load_config).load_model(
            model_config=model_config,
            device_config=device_config,
        )

        # Finalize: register tensors, commit, switch to read mode
        impl.finalize_write_mode(model, config_hash)
        GPUServiceModelLoader._imported_weights_bytes = impl.get_imported_weights_bytes()

        return model.eval()


def get_imported_weights_bytes() -> int:
    """Get the total bytes of imported/committed weights."""
    return int(GPUServiceModelLoader._imported_weights_bytes)
