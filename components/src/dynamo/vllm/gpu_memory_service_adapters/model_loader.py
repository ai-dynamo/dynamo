"""vLLM integration for GPU Memory Service (Allocation Server + embedded metadata store).

This module registers a vLLM `load_format` that loads model weights into
GPU Memory Service allocations (RW session), publishes via Commit(), and then
holds an RO lock for inference lifetime.

Configuration via model_loader_extra_config:
- gpu_memory_service_socket_path: Unix socket path for the Allocation Server (per GPU). You may
  include `{device}` which will be formatted with the GPU device index.
  Default: /tmp/gpu_memory_service_{device}.sock
- gpu_memory_service_load_mode: Load mode - "write" (cold start) or "read" (import only).
  Default: write

IMPORTANT: Sleep/Wake Memory Behavior
-------------------------------------
When using GPU Memory Service with vLLM's sleep/wake functionality, the sleep/wake does NOT
actually free GPU memory. The physical memory for model weights remains allocated
by the Allocation Server. This is by design for weight sharing:

- The Allocation Server owns the physical memory for weights
- On sleep, the client unmaps its local VA mappings but the server keeps the memory
- On wake, the client remaps the same weights without reloading from disk

This enables fast context switching between inference instances. If you need to
actually free GPU memory during sleep, use native vLLM sleep/wake (without GPU Memory Service).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import replace
from enum import Enum
from typing import Any, Optional

import torch

from dynamo.gpu_memory_service import GMSClientMemoryManager, get_or_create_allocator

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
        from dynamo.gpu_memory_service import _rpc_cumem_ext as cumem

        allocations = cumem.get_all_allocations()
        if allocations:
            # We have VMM allocations - skip empty_cache to prevent segfault
            import traceback

            logger.debug(
                "[GPU Memory Service PATCH] BLOCKING torch.cuda.empty_cache() - %d VMM allocations would be destroyed!",
                len(allocations),
            )
            logger.debug(
                "[GPU Memory Service PATCH] Call stack:\n%s",
                "".join(traceback.format_stack()[-6:]),
            )
            return
    except Exception as e:
        logger.warning("[GPU Memory Service PATCH] Error checking allocations: %s", e)
    # No GPU Memory Service allocations, safe to call original
    logger.debug("[GPU Memory Service PATCH] Allowing empty_cache (no VMM allocations)")
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

# Module-level storage for imported weights bytes (for memory accounting)
# This is set by GPUServiceModelLoader and read by get_imported_weights_bytes()
_gpu_memory_service_imported_weights_bytes: int = 0


def _get_local_rank() -> int:
    """Get the local rank (GPU device index) for the current worker.

    Priority order:
    1. torch.cuda.current_device() if already set (vLLM sets this early)
    2. vLLM's world group local_rank
    3. LOCAL_RANK environment variable
    4. Default to 0
    """
    # First check if CUDA device is already set (vLLM sets this in worker init)
    try:
        if torch.cuda.is_initialized():
            current_device = torch.cuda.current_device()
            if current_device != 0 or os.environ.get("LOCAL_RANK", "0") == "0":
                # Only trust current_device if it's non-zero OR LOCAL_RANK confirms 0
                return current_device
    except Exception:
        pass

    # Try vLLM's distributed group
    try:
        from vllm.distributed import get_world_group

        return int(get_world_group().local_rank)
    except Exception:
        pass

    # Fall back to environment variable
    return int(os.environ.get("LOCAL_RANK", "0"))


DEFAULT_GPU_MEMORY_SERVICE_SOCKET_PATH = "/tmp/gpu_memory_service_{device}.sock"


class GPUMemoryServiceLoadMode(Enum):
    """GPU Memory Service load modes."""

    WRITE = "write"  # Cold-start from disk and publish
    READ = "read"  # Import-only (fast restart; requires weights already committed)


# Keys that GPU Memory Service adds to model_loader_extra_config - must be stripped before
# passing to DefaultModelLoader which may validate unknown keys
GPU_MEMORY_SERVICE_EXTRA_CONFIG_KEYS = {
    "gpu_memory_service_socket_path",
    "gpu_memory_service_load_mode",
}


def _strip_gpu_memory_service_extra_config(load_config: Any) -> Any:
    """Return a copy of load_config with GPU Memory Service keys removed from model_loader_extra_config.

    vLLM's DefaultModelLoader may validate model_loader_extra_config and reject
    unknown keys. This strips GPU Memory Service-specific keys so we can delegate to DefaultModelLoader.
    """
    from dataclasses import replace

    if load_config is None:
        return load_config

    extra_config = getattr(load_config, "model_loader_extra_config", None) or {}
    if not extra_config:
        return load_config

    # Remove GPU Memory Service keys
    cleaned = {
        k: v
        for k, v in extra_config.items()
        if k not in GPU_MEMORY_SERVICE_EXTRA_CONFIG_KEYS
    }

    # Create new load_config with cleaned extra_config (empty dict if no keys remain)
    return replace(load_config, model_loader_extra_config=cleaned if cleaned else {})


def _resolve_socket_path(load_config: Any = None) -> str:
    """Get socket path from model_loader_extra_config (falls back to default).

    Args:
        load_config: vLLM LoadConfig object with model_loader_extra_config dict.

    Returns:
        Resolved socket path with {local_rank}/{device} placeholders expanded.
    """
    socket_path = None

    # Try model_loader_extra_config
    if load_config is not None:
        extra_config = getattr(load_config, "model_loader_extra_config", None) or {}
        socket_path = extra_config.get("gpu_memory_service_socket_path")

    # Fallback to default (matches GPU Memory Service server default)
    if not socket_path:
        socket_path = DEFAULT_GPU_MEMORY_SERVICE_SOCKET_PATH

    local_rank = _get_local_rank()
    # Support both {local_rank} and {device} placeholders for consistency with allocation server
    if "{local_rank}" in socket_path:
        socket_path = socket_path.format(local_rank=local_rank)
    if "{device}" in socket_path:
        socket_path = socket_path.format(device=local_rank)
    return socket_path


def _get_gpu_memory_service_load_mode(
    load_config: Any = None,
) -> GPUMemoryServiceLoadMode:
    """Return the GPU Memory Service load mode.

    Args:
        load_config: vLLM LoadConfig object with model_loader_extra_config dict.

    Returns:
        GPUMemoryServiceLoadMode enum value.
    """
    raw = None

    # Try model_loader_extra_config
    if load_config is not None:
        extra_config = getattr(load_config, "model_loader_extra_config", None) or {}
        raw = extra_config.get("gpu_memory_service_load_mode")

    # Default to write mode (cold start from disk)
    if not raw:
        return GPUMemoryServiceLoadMode.WRITE

    raw = raw.strip().lower()
    try:
        return GPUMemoryServiceLoadMode(raw)
    except ValueError:
        valid = [m.value for m in GPUMemoryServiceLoadMode]
        raise ValueError(
            f"Invalid gpu_memory_service_load_mode={raw!r}. Expected one of: {valid}"
        )


def compute_vllm_config_hash(vllm_config: Any) -> str:
    """Best-effort stable hash for metadata key prefixing."""
    # Avoid pickling vLLM internals; serialize a small stable subset.
    payload = {
        "model": getattr(getattr(vllm_config, "model_config", None), "model", None),
        "revision": getattr(
            getattr(vllm_config, "model_config", None), "revision", None
        ),
        "dtype": str(
            getattr(getattr(vllm_config, "model_config", None), "dtype", None)
        ),
        "tp": getattr(
            getattr(vllm_config, "parallel_config", None), "tensor_parallel_size", None
        ),
        "pp": getattr(
            getattr(vllm_config, "parallel_config", None),
            "pipeline_parallel_size",
            None,
        ),
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def register_gpu_memory_service_loader(load_format: str = "gpu_memory_service") -> None:
    """Register vLLM loader that allocates via GPU Memory Service."""
    try:
        from vllm.model_executor.model_loader import register_model_loader
        from vllm.model_executor.model_loader.base_loader import BaseModelLoader
        from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
        from vllm.model_executor.model_loader.utils import (
            initialize_model,
            process_weights_after_loading,
        )
        from vllm.utils.torch_utils import set_default_torch_dtype
    except ImportError as e:
        raise RuntimeError(f"vLLM not installed or incompatible: {e}")

    # Import the extension lazily so importing this module doesn't require it.
    try:
        from dynamo.gpu_memory_service import _rpc_cumem_ext as cumem
    except Exception as e:
        raise RuntimeError(
            "Missing CUDA VMM pluggable allocator extension. "
            "Build gpu_memory_service/core/csrc/rpc_cumem.cpp first."
        ) from e

    from torch.cuda.memory import use_mem_pool

    @register_model_loader(load_format)
    class GPUServiceModelLoader(BaseModelLoader):
        """vLLM model loader that publishes weights to GPU Memory Service."""

        # Exported for memory accounting patches.
        _imported_weights_bytes: int = 0

        def __init__(self, load_config):
            super().__init__(load_config)
            self._socket_path: Optional[str] = None

        def download_model(self, model_config) -> None:
            # Create a copy with valid load_format and stripped GPU Memory Service keys for DefaultModelLoader
            disk_load_config = _strip_gpu_memory_service_extra_config(self.load_config)
            disk_load_config = replace(disk_load_config, load_format="auto")
            DefaultModelLoader(disk_load_config).download_model(model_config)

        def load_weights(self, model: torch.nn.Module, model_config) -> None:
            # vLLM calls this from BaseModelLoader.load_model; we override load_model
            # to wrap allocations, so load_weights should not be called directly.
            disk_load_config = _strip_gpu_memory_service_extra_config(self.load_config)
            disk_load_config = replace(disk_load_config, load_format="auto")
            DefaultModelLoader(disk_load_config).load_weights(model, model_config)

        def load_model(self, vllm_config, model_config) -> torch.nn.Module:
            global _gpu_memory_service_imported_weights_bytes
            device_config = vllm_config.device_config
            load_config = vllm_config.load_config

            # Resolve socket path from config or env vars
            socket_path = _resolve_socket_path(load_config)
            self._socket_path = socket_path

            load_device = (
                device_config.device
                if load_config.device is None
                else load_config.device
            )
            target_device = torch.device(load_device)
            # Ensure device has an index for use_mem_pool
            if target_device.type == "cuda" and target_device.index is None:
                device_index = _get_local_rank()
                target_device = torch.device("cuda", device_index)
            else:
                device_index = (
                    target_device.index if target_device.index is not None else 0
                )

            # Acquire RW lock and prepare metadata namespace.
            config_hash = compute_vllm_config_hash(vllm_config)

            # Get load mode from config
            mode = _get_gpu_memory_service_load_mode(load_config)
            if mode == GPUMemoryServiceLoadMode.READ:
                ro_alloc: Optional[GMSClientMemoryManager] = None
                try:
                    from dynamo.gpu_memory_service import materialize_module_from_gms
                    from dynamo.vllm.gpu_memory_service_adapters.import_only_loader import (
                        ImportOnlyModelLoader as VLLMImportOnlyLoader,
                    )

                    # Pick a device index best-effort. In vLLM, local_rank typically maps 1:1 to GPU index.
                    device_index = _get_local_rank()
                    if torch.cuda.is_available():
                        torch.cuda.set_device(device_index)

                    # Read mode: wait indefinitely for RO lock (no timeout)
                    ro_alloc = GMSClientMemoryManager(
                        socket_path,
                        mode="read",
                        device=device_index,
                    )

                    # Create the model structure on meta device with post-processing.
                    # ImportOnlyModelLoader creates the model and runs quant post-processing
                    # to ensure the parameter structure matches what was registered.
                    import_only_loader = VLLMImportOnlyLoader(self.load_config)
                    model = import_only_loader.load_model(
                        vllm_config=vllm_config, model_config=model_config
                    )

                    imported_bytes = materialize_module_from_gms(
                        ro_alloc,
                        model,
                        prefix=f"{config_hash}:",
                        device_index=device_index,
                        strict=True,
                    )
                    GPUServiceModelLoader._imported_weights_bytes = int(imported_bytes)
                    _gpu_memory_service_imported_weights_bytes = int(imported_bytes)

                    # Success! Register the allocator in the client module.
                    # We do this after success to avoid polluting the metadata on failure.
                    from dynamo.gpu_memory_service import register_allocator

                    register_allocator(ro_alloc)

                    # Apply vLLM worker patches
                    from dynamo.vllm.gpu_memory_service_adapters.worker_extension import (
                        patch_worker_sleep_wake,
                    )

                    patch_worker_sleep_wake()

                    logger.info(
                        "[GPU Memory Service] Import-only loaded %.2f GiB from GPU memory service",
                        imported_bytes / (1 << 30),
                    )
                    return model.eval()
                except Exception:
                    if ro_alloc is not None:
                        ro_alloc.close()
                    raise

            # Get or create allocator in write mode with PyTorch MemPool.
            # The client module ensures only one allocator exists per process.
            allocator, pool = get_or_create_allocator(
                socket_path, device_index, mode="write", tag="weights"
            )

            # Start fresh (weights model load is authoritative).
            allocator.clear_all()
            allocator.metadata_delete_prefix(f"{config_hash}:")

            # Route allocations to the pool while initializing + loading weights.
            # Create a copy with valid load_format and stripped GPU Memory Service keys for DefaultModelLoader.
            disk_load_config = _strip_gpu_memory_service_extra_config(self.load_config)
            disk_load_config = replace(disk_load_config, load_format="auto")

            # TODO: Consider adding torch.cuda.empty_cache() here for consistency with SGLang.
            # PyTorch's caching allocator may have cached blocks from failed READ mode attempts
            # that could be reused instead of going through our custom allocator. The SGLang
            # model loader has this fix to address flaky allocation interception. vLLM may
            # benefit from the same fix for robustness.

            with set_default_torch_dtype(model_config.dtype):
                with use_mem_pool(pool, device=target_device):
                    with target_device:
                        model = initialize_model(
                            vllm_config=vllm_config, model_config=model_config
                        )

                    # Load weights from disk using vLLM default loader, but allocations
                    # are routed to GPU Memory Service via our pluggable allocator.
                    DefaultModelLoader(disk_load_config).load_weights(
                        model, model_config
                    )

                    process_weights_after_loading(model, model_config, target_device)

                    # IMPORTANT: Release any temporary cached blocks while still in the
                    # use_mem_pool context. This ensures they're freed through our custom
                    # allocator (my_free) rather than the default allocator which doesn't
                    # understand VMM memory.
                    torch.cuda.empty_cache()

            # Register all model tensors into the GMS metadata store
            from dynamo.gpu_memory_service import register_module_tensors

            total_bytes = register_module_tensors(
                allocator, model, metadata_prefix=f"{config_hash}:"
            )
            GPUServiceModelLoader._imported_weights_bytes = total_bytes
            _gpu_memory_service_imported_weights_bytes = total_bytes

            # Sync and flip access to read-only before publishing.
            torch.cuda.synchronize()
            cumem.set_access_all(True)

            # Commit and switch to read mode (same allocator instance!)
            ok = allocator.commit()
            if not ok:
                raise RuntimeError("Allocation Server commit failed")

            allocator.switch_to_read()

            # Allocator is already registered from get_or_create_allocator().
            # No need to register again.

            # Apply vLLM worker patches
            from dynamo.vllm.gpu_memory_service_adapters.worker_extension import (
                patch_worker_sleep_wake,
            )

            patch_worker_sleep_wake()

            logger.info(
                "[GPU Memory Service] Write mode published %.2f GiB, switched to read mode with %d mappings",
                total_bytes / (1 << 30),
                len(allocator._mappings),
            )

            # CRITICAL: Restore the correct CUDA device before returning.
            # The use_mem_pool context or other operations may have changed it.
            # If we don't restore it, subsequent vLLM operations (like profile_run)
            # will fail with "illegal memory access" because they expect the
            # correct device context.
            if target_device.index is not None:
                torch.cuda.set_device(target_device.index)

            return model.eval()

    # No return; registration happens via decorator.


def get_imported_weights_bytes() -> int:
    """Return last recorded weights bytes for vLLM memory accounting."""
    global _gpu_memory_service_imported_weights_bytes
    return _gpu_memory_service_imported_weights_bytes
