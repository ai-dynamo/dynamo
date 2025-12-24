"""SGLang integration for GPU Memory Service (Allocation Server + embedded registry).

This module provides a custom SGLang `load_format` class that can be passed as
`LoadConfig(load_format=...)`. It loads weights from disk while routing CUDA
allocations through the GPU Memory Service (RW session), publishes via Commit(),
and then holds an RO lock for inference lifetime.

Configuration via model_loader_extra_config:
- gms_socket_path: Unix socket path for the Allocation Server (per GPU). You may
  include `{device}` which will be formatted with the GPU device index.
  Default: /tmp/gms_{device}.sock
- gms_load_mode: Load mode - "write" (cold start), "read" (import only), or "auto".
  Default: auto

IMPORTANT: Sleep/Wake Memory Behavior
-------------------------------------
When using GMS with SGLang's --enable-memory-saver, the sleep/wake functionality
does NOT actually free GPU memory. The physical memory for model weights remains
allocated by the Allocation Server. This is by design for weight sharing:

- The Allocation Server owns the physical memory for weights
- On sleep, the client unmaps its local VA mappings but the server keeps the memory
- On wake, the client remaps the same weights without reloading from disk

This enables fast context switching between inference instances. If you need to
actually free GPU memory during sleep, use native torch_memory_saver (without GMS).
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

from dynamo.gpu_memory_service.core import RPCCumemAllocator, get_or_create_allocator

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
    # Check if we have GMS VMM allocations
    try:
        from dynamo.gpu_memory_service.core.csrc import _rpc_cumem_ext as cumem

        allocations = cumem.get_all_allocations()
        if allocations:
            # We have VMM allocations - skip empty_cache to prevent segfault
            logger.debug(
                "[GMS] Skipping torch.cuda.empty_cache() - %d VMM allocations active",
                len(allocations),
            )
            return
    except Exception:
        pass
    # No GMS allocations, safe to call original
    _original_empty_cache()


def _patch_empty_cache_if_needed() -> None:
    """Apply the empty_cache patch if not already applied."""
    global _empty_cache_patched
    if _empty_cache_patched:
        return
    torch.cuda.empty_cache = _safe_empty_cache
    _empty_cache_patched = True
    logger.info("[GMS] Patched torch.cuda.empty_cache for VMM allocation safety")


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


DEFAULT_GMS_SOCKET_PATH = "/tmp/gms_{device}.sock"


class GMSLoadMode(Enum):
    """GPU Memory Service load modes."""

    WRITE = "write"  # Cold-start from disk and publish
    READ = "read"  # Import-only (fast restart; requires weights already committed)
    AUTO = "auto"  # Try read first; if not READY, fall back to write


# Keys that GMS adds to model_loader_extra_config - must be stripped before
# passing to DefaultModelLoader which validates unknown keys
GMS_EXTRA_CONFIG_KEYS = {"gms_socket_path", "gms_load_mode"}


def _strip_gms_extra_config(load_config: Any) -> Any:
    """Return a copy of load_config with GMS keys removed from model_loader_extra_config.

    SGLang's DefaultModelLoader validates model_loader_extra_config and rejects
    unknown keys. This strips GMS-specific keys so we can delegate to DefaultModelLoader.
    """
    extra_config = _parse_extra_config(load_config)
    if not extra_config:
        return load_config

    # Remove GMS keys
    cleaned = {k: v for k, v in extra_config.items() if k not in GMS_EXTRA_CONFIG_KEYS}

    # Create new load_config with cleaned extra_config
    # SGLang's DefaultModelLoader expects model_loader_extra_config as a dict (not None, not string)
    return replace(load_config, model_loader_extra_config=cleaned if cleaned else {})


def _resolve_socket_path(
    device_index: Optional[int] = None, load_config: Any = None
) -> str:
    """Resolve socket path from model_loader_extra_config (falls back to default).

    Args:
        device_index: The device index to use for substitution. If None, will
            try to detect via _get_local_rank().
        load_config: SGLang LoadConfig object with model_loader_extra_config.

    Returns:
        Resolved socket path with {local_rank}/{device} placeholders expanded.
    """
    # Try model_loader_extra_config
    extra_config = _parse_extra_config(load_config)
    socket_path = extra_config.get("gms_socket_path")

    # Fallback to default (matches GMS server default)
    if not socket_path:
        socket_path = DEFAULT_GMS_SOCKET_PATH

    local_rank = device_index if device_index is not None else _get_local_rank()
    # Support both {local_rank} and {device} placeholders for consistency with allocation server
    if "{local_rank}" in socket_path:
        socket_path = socket_path.format(local_rank=local_rank)
    if "{device}" in socket_path:
        socket_path = socket_path.format(device=local_rank)
    return socket_path


def _get_gms_load_mode(load_config: Any = None) -> GMSLoadMode:
    """Return the GPU Memory Service load mode.

    Args:
        load_config: SGLang LoadConfig object with model_loader_extra_config.

    Returns:
        GMSLoadMode enum value.
    """
    # Try model_loader_extra_config
    extra_config = _parse_extra_config(load_config)
    raw = extra_config.get("gms_load_mode")

    # Default to auto mode (tries read first, falls back to write)
    if not raw:
        return GMSLoadMode.AUTO

    raw = raw.strip().lower()
    try:
        return GMSLoadMode(raw)
    except ValueError:
        valid = [m.value for m in GMSLoadMode]
        raise ValueError(f"Invalid gms_load_mode={raw!r}. Expected one of: {valid}")


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
    """Custom SGLang model loader (pass as LoadConfig(load_format=GPUServiceModelLoader))."""

    # Exported for memory accounting patches.
    _imported_weights_bytes: int = 0

    def __init__(self, load_config):
        self.load_config = load_config

    def download_model(self, model_config) -> None:
        try:
            from sglang.srt.model_loader.loader import DefaultModelLoader
        except ImportError as e:
            raise RuntimeError(f"SGLang not installed or incompatible: {e}")
        # Create a copy with valid load_format and stripped GMS keys for DefaultModelLoader
        disk_load_config = _strip_gms_extra_config(self.load_config)
        disk_load_config = replace(disk_load_config, load_format="auto")
        DefaultModelLoader(disk_load_config).download_model(model_config)

    def load_model(self, *, model_config, device_config):
        try:
            from sglang.srt.model_loader.loader import DefaultModelLoader
        except ImportError as e:
            raise RuntimeError(f"SGLang not installed or incompatible: {e}")

        # Import the extension lazily so importing this module doesn't require it.
        try:
            from dynamo.gpu_memory_service.core.csrc import (
                _rpc_cumem_ext as cumem,  # type: ignore
            )
        except Exception as e:
            raise RuntimeError(
                "Missing CUDA VMM pluggable allocator extension. "
                "Build gpu_memory_service/core/csrc/rpc_cumem.cpp first."
            ) from e

        from torch.cuda.memory import use_mem_pool

        # Use device_config.gpu_id which is explicitly passed by SGLang with the correct GPU ID
        # for this worker in tensor parallel setups. This is more reliable than _get_local_rank()
        # which tries to infer the device from various sources.
        device_index = (
            device_config.gpu_id if device_config.gpu_id >= 0 else _get_local_rank()
        )
        logger.info(
            "[GMS] Using device_index=%d (device_config.gpu_id=%d)",
            device_index,
            device_config.gpu_id,
        )

        # Resolve socket path and load mode from config or env vars
        socket_path = _resolve_socket_path(device_index, self.load_config)
        config_hash = compute_sglang_config_hash(model_config)
        mode = _get_gms_load_mode(self.load_config)

        if mode in (GMSLoadMode.READ, GMSLoadMode.AUTO):
            ro_alloc: Optional[RPCCumemAllocator] = None
            try:
                from dynamo.gpu_memory_service.tensor import (
                    materialize_module_from_registry,
                )
                from dynamo.sglang.gms_adapters.import_only_loader import (
                    ImportOnlyModelLoader,
                )

                if torch.cuda.is_available():
                    torch.cuda.set_device(device_index)

                # In auto mode, do an immediate readiness check (timeout=0).
                # For read mode, we create the allocator directly first to handle
                # timeout gracefully - if it times out, we can fall back to write mode.
                ro_timeout = 0 if mode == GMSLoadMode.AUTO else None
                ro_alloc = RPCCumemAllocator(
                    socket_path, mode="read", device=device_index, timeout_ms=ro_timeout
                )

                # Create a meta model using ImportOnlyModelLoader.
                # This creates the model structure without loading weights from disk.
                # Weights will be materialized from the GPU memory service registry.
                import_only_loader = ImportOnlyModelLoader(self.load_config)
                model = import_only_loader.load_model(
                    model_config=model_config,
                    device_config=device_config,
                )

                imported_bytes = materialize_module_from_registry(
                    ro_alloc,
                    model,
                    prefix=f"{config_hash}:",
                    device_index=device_index,
                    strict=True,
                )
                GPUServiceModelLoader._imported_weights_bytes = int(imported_bytes)

                # Success! Register the allocator in the client module.
                # We do this after success to avoid polluting the registry on failure.
                from dynamo.gpu_memory_service.core import register_allocator

                register_allocator(ro_alloc)
                logger.info(
                    "[GMS] Import-only loaded %.2f GiB from GPU memory service",
                    imported_bytes / (1 << 30),
                )
                return model.eval()
            except TimeoutError:
                if ro_alloc is not None:
                    ro_alloc.close()
                if mode == GMSLoadMode.READ:
                    raise
                logger.info(
                    "[GMS] Import-only fast path not READY; falling back to disk load+publish"
                )
            except Exception as e:
                if ro_alloc is not None:
                    ro_alloc.close()
                if mode == GMSLoadMode.READ:
                    raise
                logger.info(
                    f"[GMS] Import-only fast path failed; falling back to disk load+publish: {e}"
                )

        # Get or create allocator in write mode with PyTorch MemPool.
        # The client module ensures only one allocator exists per process.
        allocator, pool = get_or_create_allocator(
            socket_path, device_index, mode="write", tag="weights"
        )

        # Start fresh (weights model load is authoritative).
        allocator.clear_all()
        allocator.registry_delete_prefix(f"{config_hash}:")

        # Create a copy of load_config with a valid load_format for DefaultModelLoader.
        # When GPUServiceModelLoader is used as load_format (a class), we need to
        # replace it with "auto" so DefaultModelLoader can parse weight files correctly.
        # Also strip GMS keys from extra_config since DefaultModelLoader validates them.
        disk_load_config = _strip_gms_extra_config(self.load_config)
        disk_load_config = replace(disk_load_config, load_format="auto")

        # Route allocations to the pool while loading weights.
        # CRITICAL: Must specify device explicitly for multi-GPU setups.
        # Without the device parameter, use_mem_pool may not correctly associate
        # allocations with the right GPU, causing segfaults during subsequent
        # allocations (e.g., KV cache) on the default allocator.
        target_device = torch.device(f"cuda:{device_index}")
        with use_mem_pool(pool, device=target_device):
            model = DefaultModelLoader(disk_load_config).load_model(
                model_config=model_config,
                device_config=device_config,
            )
            # NOTE: We intentionally do NOT call empty_cache() here.
            # Due to PyTorch issue #145168, empty_cache() doesn't work properly
            # with CUDAPluggableAllocator + MemPool - the pool caches freed blocks
            # but never releases them through the custom allocator's free function.
            # Instead, we rely on the _safe_empty_cache patch to skip empty_cache()
            # calls when VMM allocations exist, preventing segfaults from the
            # caching allocator trying to cudaFree() our VMM memory.

        # Register all model tensors into the GMS registry
        from dynamo.gpu_memory_service.tensor import register_module_tensors

        total_bytes = register_module_tensors(
            allocator, model, registry_prefix=f"{config_hash}:"
        )
        GPUServiceModelLoader._imported_weights_bytes = total_bytes
        logger.info(
            "[GMS] Write mode registered %.2f GiB to GPU memory service",
            total_bytes / (1 << 30),
        )

        torch.cuda.synchronize()
        cumem.set_access_all(True)

        # Commit and switch to read mode (same allocator instance!)
        ok = allocator.commit()
        if not ok:
            raise RuntimeError("Allocation Server commit failed")

        allocator.switch_to_read()

        # Allocator is already in the registry from get_or_create_allocator().
        # No need to register again.

        logger.info(
            "[GMS] Write mode published %.2f GiB, switched to read mode with %d mappings",
            total_bytes / (1 << 30),
            len(allocator._mappings),
        )

        return model.eval()


def get_imported_weights_bytes() -> int:
    return int(GPUServiceModelLoader._imported_weights_bytes)
