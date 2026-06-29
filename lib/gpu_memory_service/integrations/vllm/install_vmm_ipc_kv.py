# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Install hook: route vLLM's KV-cache allocation through GMS-owned
VMM-IPC persistent allocations.

Engine code (typically the vLLM worker) imports this module BEFORE
the model is loaded. The installer monkey-patches the V1 KV cache
allocation path (``Worker._allocate_kv_cache_tensors`` /
``GPUWorker._initialize_kv_caches`` depending on vLLM version) so
the per-layer ``torch.empty`` / ``torch.zeros`` calls allocate from
a GMS persistent pool instead of the default ``cudaMalloc`` pool.

Effect:
  - daemon owns the KV pool's physical pages (cuMemCreate),
  - engine has its own VA into the same pages,
  - daemon can read/write directly via its va_daemon → no D2D copy
    on evict/restore,
  - engine restart with the same engine_id re-attaches to the SAME
    physical pages → KV survives without recompute.

Gates:
  GMS_VLLM_VMM_IPC_KV=0            optional test/debug disable
  GMS_VLLM_VMM_IPC_SOCKET=<path>   daemon UDS (default: derived from device)
  GMS_VLLM_VMM_IPC_ENGINE_ID=<id>  identifier for (engine_id, tag) keying
                                   (default: derived stable Dynamo id)
"""

from __future__ import annotations

import hashlib
import logging
import os
import sys
import time
from collections import Counter
from contextlib import contextmanager
from contextvars import ContextVar

from gpu_memory_service.integrations.common.utils import (
    env_enabled_by_default,
    get_gms_persistent_kv_socket,
)
from gpu_memory_service.integrations.vllm.kv_identity import (
    allocation_engine_id,
    allocation_shared,
    private_bootstrap_kv_enabled,
    use_existing_shared_geometry,
)

logger = logging.getLogger(__name__)

_INSTALLED = False
_LAZY_HOOK_INSTALLED = False
_GEOMETRY_PATCH_INSTALLED = False
_KV_CACHE_PROFILING: ContextVar[bool] = ContextVar(
    "gms_vllm_kv_cache_profiling", default=False
)


@contextmanager
def _private_bootstrap_kv_zeros_as_empty(enabled: bool):
    """Avoid touching reserve-only private-bootstrap KV pages.

    vLLM allocates KV buffers with ``torch.zeros``. In private-bootstrap mode
    the GMS allocator only reserves the final VAs until the shadow is promoted,
    so zero-filling those tensors launches a CUDA kernel against unbacked VAs
    and poisons the context. During this narrow allocation window, replace
    int8 zero allocations with ``torch.empty`` so vLLM can build tensor views
    without writing to KV. Scheduler-driven block zeroing runs later after
    promotion, when the shared persistent backing is attached.
    """
    if not enabled:
        yield
        return

    import torch

    original_zeros = torch.zeros
    replacements = 0

    def zeros_as_empty(*args, **kwargs):
        nonlocal replacements
        if kwargs.get("dtype") is torch.int8:
            replacements += 1
            return torch.empty(*args, **kwargs)
        return original_zeros(*args, **kwargs)

    torch.zeros = zeros_as_empty
    try:
        yield
    finally:
        torch.zeros = original_zeros
        if replacements:
            logger.info(
                "[GMS-VMM-IPC] allocated %d private-bootstrap KV tensors "
                "with torch.empty to avoid reserve-only zero-fill",
                replacements,
            )


def _is_enabled() -> bool:
    return env_enabled_by_default("GMS_VLLM_VMM_IPC_KV", default=True)


def _resolve_socket(device: int) -> str:
    return get_gms_persistent_kv_socket(device, "GMS_VLLM_VMM_IPC_SOCKET")


def _engine_id(device: int = 0) -> str:
    return allocation_engine_id(device)


def _current_cuda_device() -> int:
    import torch

    return int(torch.cuda.current_device())


def _device_index(device) -> int:
    if isinstance(device, int):
        return int(device)
    index = getattr(device, "index", None)
    if index is not None:
        return int(index)
    try:
        return _current_cuda_device()
    except Exception:
        logger.debug(
            "[GMS-VMM-IPC] Failed to resolve current CUDA device; using cuda:0",
            exc_info=True,
        )
        return 0


def _int_env_value(name: str, value: str | None, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        logger.warning("Ignoring invalid %s=%r for GMS KV geometry", name, value)
        return default


def _shared_kv_enabled() -> bool:
    return allocation_shared()


def _install_kv_leases() -> bool:
    try:
        from gpu_memory_service.integrations.vllm.install_kv_leases import (
            install as install_kv_leases,
        )
    except Exception:  # noqa: BLE001
        logger.debug("[GMS-VMM-IPC] vLLM KV lease installer unavailable", exc_info=True)
        return False
    try:
        return bool(install_kv_leases())
    except Exception:  # noqa: BLE001
        logger.exception("[GMS-VMM-IPC] vLLM KV lease install failed")
        raise


def _defer_physical_enabled() -> bool:
    return private_bootstrap_kv_enabled()


def _semantic_kv_tensor_tag(index: int, kv_cache_tensor) -> str:
    shared_by = tuple(
        sorted(str(layer) for layer in getattr(kv_cache_tensor, "shared_by", ()) or ())
    )
    if shared_by:
        key = "\0".join(shared_by)
    else:
        key = f"anonymous:{index}:{getattr(kv_cache_tensor, 'size', '')}"
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
    return f"kv_pool:v2:{digest}"


def _semantic_kv_tensor_tag_plan(kv_cache_config) -> list[str]:
    base_tags = [
        _semantic_kv_tensor_tag(index, kv_cache_tensor)
        for index, kv_cache_tensor in enumerate(
            getattr(kv_cache_config, "kv_cache_tensors", ()) or ()
        )
    ]
    counts = Counter(base_tags)
    seen: dict[str, int] = {}
    planned_tags: list[str] = []
    for base_tag in base_tags:
        if counts[base_tag] == 1:
            planned_tags.append(base_tag)
            continue
        duplicate_index = seen.get(base_tag, 0)
        seen[base_tag] = duplicate_index + 1
        planned_tags.append(f"{base_tag}:dup{duplicate_index}")
    return planned_tags


def _geometry_device() -> int:
    for name in ("GMS_VLLM_KV_LEASE_DEVICE", "LOCAL_RANK"):
        value = os.environ.get(name)
        if value is None:
            continue
        try:
            return int(value)
        except ValueError:
            logger.warning("Ignoring invalid %s=%r for GMS KV geometry", name, value)
    return 0


def _existing_shared_kv_blocks(*, wait_ms: int = 0) -> int | None:
    if not use_existing_shared_geometry():
        return None
    if env_enabled_by_default("GMS_KV_LEASE_SHM_RESET", default=False):
        return None

    from gpu_memory_service.integrations.common.kv_lease_client import (
        kv_leases_enabled,
        read_any_kv_lease_namespace_total_blocks,
        read_kv_lease_namespace_total_blocks,
    )

    if not kv_leases_enabled("vllm"):
        return None

    device = _geometry_device()
    deadline = time.monotonic() + max(0, wait_ms) / 1000.0
    logged_wait = False
    while True:
        namespace, total_blocks = read_kv_lease_namespace_total_blocks(
            "vllm",
            device,
            namespace_suffix="block-pool",
        )
        if total_blocks is None:
            namespace, total_blocks = read_any_kv_lease_namespace_total_blocks("vllm")
        if total_blocks is not None:
            break
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return None
        if not logged_wait:
            logger.info(
                "[GMS-VMM-IPC] Waiting up to %d ms for existing KV lease "
                "namespace geometry",
                wait_ms,
            )
            logged_wait = True
        time.sleep(min(0.05, remaining))

    logger.info(
        "[GMS-VMM-IPC] Reusing existing KV lease namespace geometry: "
        "namespace=%s blocks=%d",
        namespace,
        total_blocks,
    )
    return int(total_blocks)


def _available_memory_exhausted(available_memory) -> bool:
    try:
        if isinstance(available_memory, (int, float)):
            values = [available_memory]
        else:
            values = list(available_memory)
    except TypeError:
        return False
    if not values:
        return False
    try:
        return min(int(value) for value in values) <= 0
    except (TypeError, ValueError):
        return False


def _geometry_wait_ms(available_memory) -> int:
    private_bootstrap = private_bootstrap_kv_enabled()
    if not private_bootstrap and not _available_memory_exhausted(available_memory):
        return 0
    name = "GMS_VLLM_KV_GEOMETRY_WAIT_MS"
    value = os.environ.get(name)
    if value is not None:
        return max(0, _int_env_value(name, value, 300_000))

    if value is None:
        name = "GMS_KV_LEASE_GEOMETRY_WAIT_MS"
        value = os.environ.get(name)
    wait_ms = max(0, _int_env_value(name, value, 300_000))
    if private_bootstrap:
        # A private-bootstrap shadow must attach to the primary's published KV
        # geometry. Falling back to vLLM sizing before the primary has published
        # can observe negative available HBM and fail even though the shared KV
        # pool is about to become attachable.
        return max(wait_ms, 1_800_000)
    return wait_ms


def _wrap_get_kv_cache_configs(original):
    if getattr(original, "_gms_geometry_patched", False):
        return original

    def _patched_get_kv_cache_configs(vllm_config, kv_cache_specs, available_memory):
        existing_blocks = _existing_shared_kv_blocks(
            wait_ms=_geometry_wait_ms(available_memory)
        )
        if existing_blocks is None:
            return original(vllm_config, kv_cache_specs, available_memory)

        cache_config = getattr(vllm_config, "cache_config", None)
        if cache_config is None:
            return original(vllm_config, kv_cache_specs, available_memory)

        previous_override = getattr(cache_config, "num_gpu_blocks_override", None)
        if previous_override is not None and int(previous_override) != existing_blocks:
            logger.warning(
                "[GMS-VMM-IPC] Existing shared KV pool has %d blocks; "
                "temporarily replacing num_gpu_blocks_override=%s during attach",
                existing_blocks,
                previous_override,
            )

        cache_config.num_gpu_blocks_override = existing_blocks
        try:
            return original(vllm_config, kv_cache_specs, available_memory)
        finally:
            cache_config.num_gpu_blocks_override = previous_override

    _patched_get_kv_cache_configs._gms_geometry_patched = True
    _patched_get_kv_cache_configs._gms_geometry_original = original
    return _patched_get_kv_cache_configs


def install_geometry_patch() -> bool:
    """Patch vLLM KV sizing to reuse existing GMS shared-KV geometry.

    VMM-IPC reattach already works once vLLM reaches tensor allocation. The
    missing piece is earlier: vLLM profiles currently free HBM before it builds
    KVCacheConfig. A shadow/restarted engine can therefore fail or shrink its KV
    block count before it reaches the GMS persistent allocation path. When the
    primary has already initialized the shared lease namespace, its header is
    the authoritative logical block count for subsequent attachers.

    vLLM imports ``get_kv_cache_configs`` into ``vllm.v1.engine.core`` by value,
    so patching only ``kv_cache_utils`` is not enough if engine core is imported
    after the first GMS hook. Keep this function idempotent while still updating
    the late-bound engine-core alias whenever it becomes available.
    """
    global _GEOMETRY_PATCH_INSTALLED
    if not _is_enabled():
        return False

    try:
        from vllm.v1.core import kv_cache_utils
    except ImportError:
        logger.debug(
            "[GMS-VMM-IPC] vLLM KV cache utils unavailable; geometry patch skipped"
        )
        return False

    changed = False
    current = kv_cache_utils.get_kv_cache_configs
    if getattr(current, "_gms_geometry_patched", False):
        patched = current
    else:
        patched = _wrap_get_kv_cache_configs(current)
        kv_cache_utils.get_kv_cache_configs = patched
        _GEOMETRY_PATCH_INSTALLED = True
        changed = True

    engine_core = sys.modules.get("vllm.v1.engine.core")
    if engine_core is not None and hasattr(engine_core, "get_kv_cache_configs"):
        if getattr(engine_core, "get_kv_cache_configs") is not patched:
            engine_core.get_kv_cache_configs = patched
            changed = True

    if changed:
        logger.info("[GMS-VMM-IPC] patched vLLM KV cache geometry attach path")
    return changed


def install() -> bool:
    """Monkey-patch vLLM's KV-cache tensor allocation. Returns True iff
    a patch was actually installed. Safe to call multiple times; idempotent."""
    global _INSTALLED
    if not _is_enabled():
        logger.debug(
            "[GMS-VMM-IPC] GMS_VLLM_VMM_IPC_KV not set; skipping install",
        )
        return False
    install_geometry_patch()
    _install_kv_leases()
    if _INSTALLED:
        return False

    # vLLM has two model-runner code paths:
    #   V1: GPUModelRunner.initialize_kv_cache_tensors → _allocate_kv_cache_tensors
    #       (in vllm.v1.worker.gpu_model_runner)
    #   V2: gpu.attn_utils._allocate_kv_cache (module-level function called
    #       from vllm.v1.worker.gpu.model_runner.GPUModelRunner.initialize_kv_cache)
    # Patch BOTH so we don't depend on VLLM_USE_V2_MODEL_RUNNER.

    from gpu_memory_service.client.torch.allocator import (
        clear_persistent_allocator_tag_plan,
        get_or_create_persistent_allocator,
        gms_use_persistent_pool,
        set_persistent_allocator_tag_plan,
    )

    # ---- V2 patch: module-level _allocate_kv_cache ----
    try:
        from vllm.v1.worker.gpu import attn_utils as _gpu_attn_utils

        if hasattr(_gpu_attn_utils, "_allocate_kv_cache"):
            _orig_v2_alloc = _gpu_attn_utils._allocate_kv_cache
            if getattr(_orig_v2_alloc, "_gms_vmm_ipc_patched", False):
                _orig_v2_alloc = getattr(
                    _orig_v2_alloc, "_gms_original", _orig_v2_alloc
                )

            def _patched_v2_allocate_kv_cache(
                kv_cache_config, shared_layers=None, device=None, *args, **kwargs
            ):
                # Current vLLM passes (kv_cache_config, shared_layers, device).
                # Older snapshots passed (kv_cache_config, device). Keep both
                # shapes working so this integration can ride upstream churn.
                if device is None:
                    alloc_device = shared_layers
                    original_args = (kv_cache_config, shared_layers, *args)
                else:
                    alloc_device = device
                    original_args = (kv_cache_config, shared_layers, device, *args)

                if _KV_CACHE_PROFILING.get():
                    logger.debug(
                        "[GMS-VMM-IPC] bypassing persistent pool for vLLM "
                        "profiling KV allocation"
                    )
                    return _orig_v2_alloc(*original_args, **kwargs)

                dev_idx = _device_index(alloc_device)
                socket = _resolve_socket(dev_idx)
                engine_id = _engine_id(dev_idx)
                logger.debug(
                    "[GMS-VMM-IPC] V2 allocation pid=%d device=%d",
                    os.getpid(),
                    dev_idx,
                )
                try:
                    shared_kv = _shared_kv_enabled()
                    defer_physical = _defer_physical_enabled()
                    get_or_create_persistent_allocator(
                        socket,
                        dev_idx,
                        engine_id,
                        tag="kv_pool",
                        shared=shared_kv,
                        defer_physical=defer_physical,
                    )
                except Exception as exc:  # noqa: BLE001
                    raise RuntimeError(
                        "GMS vLLM persistent KV allocator registration failed"
                    ) from exc
                tag_plan = _semantic_kv_tensor_tag_plan(kv_cache_config)
                if tag_plan:
                    set_persistent_allocator_tag_plan("kv_pool", tag_plan)
                logger.debug(
                    "[GMS-VMM-IPC] V2 persistent pool engine_id=%s device=%d "
                    "defer_physical=%s semantic_tags=%d",
                    engine_id,
                    dev_idx,
                    defer_physical,
                    len(tag_plan),
                )
                try:
                    with gms_use_persistent_pool("kv_pool", dev_idx):
                        with _private_bootstrap_kv_zeros_as_empty(defer_physical):
                            return _orig_v2_alloc(*original_args, **kwargs)
                finally:
                    if tag_plan:
                        clear_persistent_allocator_tag_plan("kv_pool")

            _patched_v2_allocate_kv_cache._gms_vmm_ipc_patched = True
            _patched_v2_allocate_kv_cache._gms_original = _orig_v2_alloc
            _gpu_attn_utils._allocate_kv_cache = _patched_v2_allocate_kv_cache  # type: ignore[assignment]
            logger.debug(
                "[GMS-VMM-IPC] patched V2 allocation in pid=%d",
                os.getpid(),
            )
    except ImportError:
        pass

    # ---- V1 patch: GPUModelRunner.initialize_kv_cache_tensors ----
    try:
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    except ImportError:
        logger.warning(
            "[GMS-VMM-IPC] vLLM v1 GPUModelRunner not importable; V1 install skipped",
        )
        _INSTALLED = True  # V2 may have succeeded
        return True

    if not hasattr(GPUModelRunner, "initialize_kv_cache_tensors"):
        logger.warning(
            "[GMS-VMM-IPC] vLLM GPUModelRunner.initialize_kv_cache_tensors "
            "not found; V1 install skipped",
        )
        _INSTALLED = True
        return True

    if hasattr(GPUModelRunner, "initialize_kv_cache"):
        original_initialize_kv_cache = GPUModelRunner.initialize_kv_cache
        if not getattr(original_initialize_kv_cache, "_gms_profiling_guard", False):

            def _patched_initialize_kv_cache(self, *args, **kwargs):
                is_profiling = bool(kwargs.get("is_profiling", False))
                if len(args) >= 2:
                    is_profiling = bool(args[1])
                if not is_profiling:
                    return original_initialize_kv_cache(self, *args, **kwargs)
                token = _KV_CACHE_PROFILING.set(True)
                try:
                    return original_initialize_kv_cache(self, *args, **kwargs)
                finally:
                    _KV_CACHE_PROFILING.reset(token)

            _patched_initialize_kv_cache._gms_profiling_guard = True
            _patched_initialize_kv_cache._gms_original = original_initialize_kv_cache
            GPUModelRunner.initialize_kv_cache = _patched_initialize_kv_cache  # type: ignore[assignment]

    original = GPUModelRunner.initialize_kv_cache_tensors

    def _patched_initialize_kv_cache_tensors(self, *args, **kwargs):
        logger.debug(
            "[GMS-VMM-IPC] V1 allocation in pid=%d",
            os.getpid(),
        )
        if _KV_CACHE_PROFILING.get():
            logger.debug(
                "[GMS-VMM-IPC] bypassing persistent pool for vLLM profiling "
                "KV tensor initialization"
            )
            return original(self, *args, **kwargs)

        device = _device_index(getattr(self, "device", 0))
        socket = _resolve_socket(device)
        engine_id = _engine_id(device)
        try:
            shared_kv = _shared_kv_enabled()
            defer_physical = _defer_physical_enabled()
            get_or_create_persistent_allocator(
                socket,
                device,
                engine_id,
                tag="kv_pool",
                shared=shared_kv,
                defer_physical=defer_physical,
            )
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "GMS vLLM persistent KV allocator registration failed"
            ) from exc
        logger.info(
            "[GMS-VMM-IPC] Allocating vLLM KV-cache through GMS "
            "persistent pool (engine_id=%s, device=%d)",
            engine_id,
            device,
        )
        kv_cache_config = args[0] if args else kwargs.get("kv_cache_config")
        tag_plan = _semantic_kv_tensor_tag_plan(kv_cache_config)
        if tag_plan:
            set_persistent_allocator_tag_plan("kv_pool", tag_plan)
        logger.debug(
            "[GMS-VMM-IPC] V1 persistent pool engine_id=%s device=%d "
            "defer_physical=%s semantic_tags=%d",
            engine_id,
            device,
            defer_physical,
            len(tag_plan),
        )
        try:
            with gms_use_persistent_pool("kv_pool", device):
                with _private_bootstrap_kv_zeros_as_empty(defer_physical):
                    return original(self, *args, **kwargs)
        finally:
            if tag_plan:
                clear_persistent_allocator_tag_plan("kv_pool")

    GPUModelRunner.initialize_kv_cache_tensors = _patched_initialize_kv_cache_tensors  # type: ignore[assignment]
    _INSTALLED = True
    logger.info(
        "[GMS-VMM-IPC install] patched GPUModelRunner.initialize_kv_cache_tensors",
    )
    return True


def install_lazy() -> None:
    """Register a sys.meta_path finder that calls install() the first
    time `vllm.v1.worker.gpu_model_runner` is loaded. Designed for use
    from a .pth file (Python startup): avoids eagerly importing vLLM,
    which would interfere with pytest's assertion rewriter and other
    startup-sensitive consumers (e.g. anyio)."""
    global _LAZY_HOOK_INSTALLED
    if _LAZY_HOOK_INSTALLED:
        return
    if not _is_enabled():
        return

    targets = {
        "vllm.v1.core.block_pool",  # Scheduler-side KV lease publication
        "vllm.v1.engine.core",  # KV sizing call-site imports get_kv_cache_configs by value
        "vllm.v1.worker.gpu_model_runner",  # V1 path
        "vllm.v1.worker.gpu.attn_utils",  # V2 path
    }

    class _PatchAfterLoad:
        def __init__(self, real_loader):
            self._real = real_loader

        def create_module(self, spec):
            if hasattr(self._real, "create_module"):
                return self._real.create_module(spec)
            return None

        def exec_module(self, module):
            self._real.exec_module(module)
            try:
                install()
            except Exception:  # noqa: BLE001
                logger.exception("[GMS-VMM-IPC] post-load install raised")
                raise

    class _Finder:
        def find_spec(self, name, path=None, target_pkg=None):
            if name not in targets:
                return None
            for finder in sys.meta_path:
                if finder is self:
                    continue
                if hasattr(finder, "find_spec"):
                    spec = finder.find_spec(name, path, target_pkg)
                    if spec is not None and spec.loader is not None:
                        try:
                            sys.meta_path.remove(self)
                        except ValueError:
                            pass
                        spec.loader = _PatchAfterLoad(spec.loader)
                        return spec
            return None

    sys.meta_path.insert(0, _Finder())
    _LAZY_HOOK_INSTALLED = True
    logger.debug(
        "[GMS-VMM-IPC] lazy hook armed; will install on first import of %s",
        sorted(targets),
    )


# Eager install (when this module is imported AFTER vllm — e.g. tests
# that import vllm first then enable the hook). The .pth file calls
# install_lazy() instead to avoid eagerly importing vllm.
if _is_enabled() and "vllm.v1.worker.gpu_model_runner" in sys.modules:
    try:
        install()
    except Exception:  # noqa: BLE001
        logger.exception("[GMS-VMM-IPC] Auto-install raised")
        raise
