# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Route vLLM's native KV allocation context through GMS VMM-IPC.

Current vLLM passes ``Worker._maybe_get_memory_pool_context("kv_cache")``
to both GPU model runners. :class:`GMSWorker` implements that public worker
hook with :func:`persistent_kv_allocation_context`; this module retains only
the GMS allocation context and the scheduler-side geometry compatibility hook.

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
import inspect
import logging
import os
import sys
import time
from collections import Counter
from contextlib import contextmanager

from gpu_memory_service.integrations.common.utils import env_enabled_by_default
from gpu_memory_service.integrations.vllm.kv_identity import (
    use_existing_shared_geometry,
)

logger = logging.getLogger(__name__)

_LAZY_HOOK_INSTALLED = False
_GEOMETRY_PATCH_INSTALLED = False


@contextmanager
def _persistent_kv_zeros_as_empty(enabled: bool):
    """Avoid touching reserve-only or reattached persistent KV pages.

    vLLM allocates KV buffers with ``torch.zeros``. Private-bootstrap mode
    has reserve-only VAs, while a cold replacement maps the primary's existing
    physical pages. Zero-filling either would respectively poison the CUDA
    context or silently destroy the KV being recovered. During those allocation
    windows, replace only int8 zero allocations with ``torch.empty`` so vLLM
    can build tensor views without writing to KV. A genuinely new shared pool
    keeps vLLM's normal zero initialization.
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
                "[GMS-VMM-IPC] allocated %d persistent KV tensors with "
                "torch.empty to preserve existing or reserve-only pages",
                replacements,
            )


def _is_enabled() -> bool:
    return env_enabled_by_default("GMS_VLLM_VMM_IPC_KV", default=True)


def _int_env_value(name: str, value: str | None, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        logger.warning("Ignoring invalid %s=%r for GMS KV geometry", name, value)
        return default


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


def _kv_layout_fingerprint(kv_cache_config) -> str:
    """Stable digest of KV-layout-relevant parameters.

    The persistent identity is otherwise gated only by aligned-size equality on
    reattach, so a config change that preserves size but changes meaning -- most
    dangerously a KV dtype change (bf16 -> fp8 halves element size but a shared
    attacher may still map the larger old allocation) -- would silently reuse
    incompatible bytes as a different layout. Folding this fingerprint into the
    tag forces a distinct identity so reattach fails cleanly instead. The KV
    spec's per-rank head counts / page sizes already encode the TP degree.
    """
    parts: list[str] = []
    for group in getattr(kv_cache_config, "kv_cache_groups", ()) or ():
        spec = getattr(group, "kv_cache_spec", None)
        if spec is None:
            continue
        for attr in (
            "block_size",
            "num_kv_heads",
            "head_size",
            "dtype",
            "use_mla",
            "page_size_bytes",
        ):
            val = getattr(spec, attr, None)
            if val is not None:
                parts.append(f"{attr}={val}")
    if not parts:
        return "unknown"
    return hashlib.sha1("\0".join(parts).encode("utf-8")).hexdigest()[:12]


def _semantic_kv_tensor_tag(index: int, kv_cache_tensor, layout_fp: str) -> str:
    shared_by = tuple(
        sorted(str(layer) for layer in getattr(kv_cache_tensor, "shared_by", ()) or ())
    )
    if shared_by:
        key = "\0".join(shared_by)
    else:
        key = f"anonymous:{index}:{getattr(kv_cache_tensor, 'size', '')}"
    digest = hashlib.sha1((layout_fp + "\0" + key).encode("utf-8")).hexdigest()[:16]
    return f"kv_pool:v2:{digest}"


def _semantic_kv_tensor_tag_plan(kv_cache_config) -> list[str]:
    layout_fp = _kv_layout_fingerprint(kv_cache_config)
    base_tags = [
        _semantic_kv_tensor_tag(index, kv_cache_tensor, layout_fp)
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


def _persistent_tag_plan_reattaches(
    manager, engine_id: str, tag_plan: list[str]
) -> bool:
    """Return whether every semantic KV allocation already exists.

    A complete plan means this process is reattaching and must not zero the
    mapped pages. No matching tags means a new pool and retains normal vLLM
    initialization. A partial plan is unsafe: mixing preserved and new tensors
    would create a layout whose metadata cannot describe its contents.
    """
    if not tag_plan:
        return False
    existing = {
        str(getattr(allocation, "tag", ""))
        for allocation in manager.list_persistent(
            engine_id=engine_id, include_unclaimed=True
        )
    }
    planned = set(tag_plan)
    present = planned & existing
    if os.environ.get("GMS_KV_DIRECTORY_DIAGNOSTICS"):
        logger.warning(
            "[GMS-VMM-IPC] persistent plan engine_id=%s "
            "planned=%d existing=%d matching=%d",
            engine_id,
            len(planned),
            len(existing),
            len(present),
        )
    if not present:
        return False
    missing = planned - existing
    if missing:
        raise RuntimeError(
            "GMS persistent KV semantic tag plan is only partially present: "
            f"found={len(present)} missing={len(missing)}. Refusing to mix "
            "preserved and newly initialized KV tensors."
        )
    return True


@contextmanager
def persistent_kv_allocation_context(manager, engine_id: str, kv_cache_config, device):
    """Allocate vLLM KV through GMS with stable restart-safe identities.

    This is the supported integration point for vLLM versions that accept a
    ``kv_cache_allocation_context``. Keep semantic tags and zero suppression
    together so a native allocator hook cannot accidentally reattach the
    right pages and then overwrite them during tensor construction.
    """
    from gpu_memory_service.client.torch.allocator import (
        clear_persistent_allocator_tag_plan,
        gms_use_persistent_pool,
        set_persistent_allocator_tag_plan,
    )

    tag_plan = _semantic_kv_tensor_tag_plan(kv_cache_config)
    reattaching = _persistent_tag_plan_reattaches(manager, engine_id, tag_plan)
    if tag_plan:
        set_persistent_allocator_tag_plan("kv_pool", tag_plan)
    logger.debug(
        "[GMS-VMM-IPC] persistent pool engine_id=%s device=%s "
        "reattaching=%s semantic_tags=%d",
        engine_id,
        device,
        reattaching,
        len(tag_plan),
    )
    try:
        with gms_use_persistent_pool("kv_pool", device):
            with _persistent_kv_zeros_as_empty(reattaching):
                yield
    finally:
        if tag_plan:
            clear_persistent_allocator_tag_plan("kv_pool")


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
    if not _available_memory_exhausted(available_memory):
        return 0
    name = "GMS_VLLM_KV_GEOMETRY_WAIT_MS"
    value = os.environ.get(name)
    if value is not None:
        return max(0, _int_env_value(name, value, 300_000))

    if value is None:
        name = "GMS_KV_LEASE_GEOMETRY_WAIT_MS"
        value = os.environ.get(name)
    return max(0, _int_env_value(name, value, 300_000))


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


def geometry_hook_installed() -> bool:
    """Verify the live sizing helper and its imported EngineCore alias."""
    try:
        from vllm.v1.core import kv_cache_utils
    except Exception:  # noqa: BLE001
        return False
    patched = getattr(kv_cache_utils, "get_kv_cache_configs", None)
    if not getattr(patched, "_gms_geometry_patched", False):
        return False
    engine_core = sys.modules.get("vllm.v1.engine.core")
    if engine_core is None:
        return True
    return getattr(engine_core, "get_kv_cache_configs", None) is patched


def native_kv_allocation_hook_available() -> bool:
    """Check that every current GPU runner consumes vLLM's worker context.

    The worker forwarding check deliberately inspects bytecode names rather
    than a version string. This makes shared-KV startup fail closed if vLLM
    removes either the worker hook or the forwarding call while keeping a
    superficially compatible method signature.
    """
    try:
        from vllm.v1.worker.gpu import attn_utils
        from vllm.v1.worker.gpu.model_runner import GPUModelRunner as V2ModelRunner
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner as V1ModelRunner
        from vllm.v1.worker.gpu_worker import Worker
    except Exception:  # noqa: BLE001
        logger.debug("[GMS-VMM-IPC] native vLLM KV hook unavailable", exc_info=True)
        return False

    worker_hook = getattr(Worker, "_maybe_get_memory_pool_context", None)
    initialize = getattr(Worker, "initialize_from_config", None)
    if not callable(worker_hook) or not callable(initialize):
        return False
    code = getattr(inspect.unwrap(initialize), "__code__", None)
    forwarded_names = set(code.co_names) if code is not None else set()
    if not {"_maybe_get_memory_pool_context", "initialize_kv_cache"}.issubset(
        forwarded_names
    ):
        return False

    consumers = (
        V1ModelRunner.initialize_kv_cache,
        V1ModelRunner.initialize_kv_cache_tensors,
        V2ModelRunner.initialize_kv_cache,
        attn_utils.init_kv_cache,
    )
    try:
        return all(
            "kv_cache_allocation_context" in inspect.signature(consumer).parameters
            for consumer in consumers
        )
    except (TypeError, ValueError):
        logger.debug(
            "[GMS-VMM-IPC] could not inspect native vLLM KV hook", exc_info=True
        )
        return False


def install() -> bool:
    """Install scheduler hooks; KV allocation uses GMSWorker's native hook."""
    if not _is_enabled():
        logger.debug(
            "[GMS-VMM-IPC] GMS_VLLM_VMM_IPC_KV not set; skipping install",
        )
        return False
    geometry_changed = install_geometry_patch()
    leases_changed = _install_kv_leases()
    return geometry_changed or leases_changed


def persistent_kv_hooks_installed() -> bool:
    """Verify vLLM's native worker allocation path and GMS geometry hook."""
    return native_kv_allocation_hook_available() and geometry_hook_installed()


def install_lazy() -> None:
    """Register a sys.meta_path finder that calls install() the first
    time a scheduler-side patch target is loaded. This avoids eagerly importing
    vLLM from startup-sensitive consumers."""
    global _LAZY_HOOK_INSTALLED
    if _LAZY_HOOK_INSTALLED:
        return
    if not _is_enabled():
        return

    targets = {
        "vllm.v1.core.block_pool",  # Scheduler-side KV lease publication
        "vllm.v1.engine.core",  # KV sizing call-site imports get_kv_cache_configs by value
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
