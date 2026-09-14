# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Install hook: route SGLang's MHATokenToKVPool / MLATokenToKVPool
allocation through GMS-owned VMM-IPC persistent allocations.

SGLang allocates its KV pool inside ``MHATokenToKVPool.__init__`` (and
the MLA variant), which constructs ``self.kv_buffer`` as a list of
per-layer CUDA tensors. Current SGLang versions already wrap those
allocations with ``memory_saver_adapter.region("kv_cache")``. This hook
therefore only ensures the persistent allocator exists before vanilla init;
the memory saver adapter owns the single mempool allocation scope.

Gates:
  GMS_SGLANG_VMM_IPC_KV=0           optional test/debug disable
  GMS_SGLANG_VMM_IPC_SOCKET=<path>  daemon UDS (default: derived from device)
  GMS_SGLANG_VMM_IPC_ENGINE_ID=<id> identifier for (engine_id, tag) keying
                                    (default: derived stable Dynamo id)
"""

from __future__ import annotations

import logging
import os
from hashlib import sha256
from inspect import signature

from gpu_memory_service.integrations.common.kv_lease_client import resolve_lease_device
from gpu_memory_service.integrations.common.utils import (
    env_enabled_by_default,
    get_gms_persistent_kv_socket,
)
from gpu_memory_service.integrations.sglang.kv_identity import (
    allocation_engine_id,
    allocation_shared,
    allocator_tag,
)

logger = logging.getLogger(__name__)

_INSTALLED = False


def _is_enabled() -> bool:
    return env_enabled_by_default("GMS_SGLANG_VMM_IPC_KV", default=True)


def _resolve_socket(device: int) -> str:
    return get_gms_persistent_kv_socket(device, "GMS_SGLANG_VMM_IPC_SOCKET")


def _engine_id(device: int = 0) -> str:
    return allocation_engine_id(device)


def _device_index_from_value(value) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return int(value)
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None
        if ":" in raw:
            raw = raw.rsplit(":", 1)[1]
        try:
            return int(raw)
        except ValueError:
            return None
    index = getattr(value, "index", None)
    if index is not None and not callable(index):
        try:
            return int(index)
        except (TypeError, ValueError):
            return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _resolve_kv_pool_device(args, kwargs) -> int:
    candidates = [kwargs.get("device")]
    if len(args) > 6:
        candidates.append(args[6])
    for candidate in candidates:
        index = _device_index_from_value(candidate)
        if index is not None:
            return index
    return int(resolve_lease_device("GMS_SGLANG_KV_LEASE_DEVICE"))


def _constructor_values(original_init, instance, args, kwargs) -> dict[str, object]:
    bound = signature(original_init).bind(instance, *args, **kwargs)
    bound.apply_defaults()
    return dict(bound.arguments)


def _semantic_tag_plan(name: str, values: dict[str, object]) -> list[str]:
    """Describe every native tensor allocation without relying on its ordinal."""
    manifest = os.environ.get("GMS_KV_DIRECTORY_MANIFEST")
    if not manifest:
        from gpu_memory_service.integrations.sglang.kv_identity import shared_kv_enabled

        if shared_kv_enabled():
            raise RuntimeError(
                "Shared persistent SGLang KV requires "
                "GMS_KV_DIRECTORY_MANIFEST to identify the model deployment"
            )
        manifest = "process-local"

    layer_num = int(values["layer_num"])
    start_layer = int(values.get("start_layer") or 0)
    descriptors = [
        f"manifest={manifest}",
        f"pool={name}",
        *(
            f"{key}={values.get(key)}"
            for key in (
                "size",
                "page_size",
                "dtype",
                "head_num",
                "head_dim",
                "v_head_dim",
                "kv_lora_rank",
                "qk_rope_head_dim",
                "layer_num",
                "start_layer",
                "end_layer",
                "kv_cache_layout",
            )
            if key in values
        ),
        f"hnd={os.environ.get('SGLANG_USE_HND_KVCACHE', '0')}",
        f"aiter_layout={os.environ.get('SGLANG_AITER_KV_CACHE_LAYOUT', 'nhd')}",
    ]
    layout = sha256("\0".join(descriptors).encode()).hexdigest()[:16]
    layers = range(start_layer, start_layer + layer_num)
    kinds = ("kv",) if name == "MLATokenToKVPool" else ("k", "v")
    return [
        f"kv_pool:sglang:v1:{layout}:{kind}:layer{layer}"
        for kind in kinds
        for layer in layers
    ]


def _managed_tag(tag: str, base_tag: str) -> bool:
    return tag.startswith(("kv_pool:sglang:v", f"{base_tag}#"))


def _prepare_tag_plan(manager, engine_id: str, base_tag: str, plan: list[str]) -> bool:
    allocations = manager.list_persistent(engine_id=engine_id, include_unclaimed=True)
    planned = set(plan)
    incompatible = sorted(
        str(getattr(allocation, "tag", ""))
        for allocation in allocations
        if _managed_tag(str(getattr(allocation, "tag", "")), base_tag)
        and str(getattr(allocation, "tag", "")) not in planned
    )
    if incompatible:
        raise RuntimeError(
            "Incompatible persistent SGLang KV allocations remain for this engine: "
            f"{incompatible}. Reset the persistent KV pool and its directory/ring "
            "metadata explicitly before changing model or KV layout."
        )

    existing = {str(getattr(allocation, "tag", "")) for allocation in allocations}
    present = planned & existing
    if not present:
        return False
    missing = planned - existing
    if missing:
        raise RuntimeError(
            "Persistent SGLang KV tag plan is only partially present: "
            f"found={len(present)} missing={len(missing)}"
        )
    return True


def _release_new_plan(manager, engine_id: str, plan: list[str]) -> None:
    for tag in plan:
        try:
            manager.release_persistent(engine_id, tag)
        except Exception:  # noqa: BLE001
            logger.exception(
                "[GMS-VMM-IPC] failed to roll back SGLang KV allocation "
                "engine_id=%s tag=%s",
                engine_id,
                tag,
            )


def _persistent_init(original_init, name: str, instance, args, kwargs):
    values = _constructor_values(original_init, instance, args, kwargs)
    if values.get("post_capture_active"):
        raise RuntimeError(
            "GMS SGLang persistent KV does not support post-capture VMM pools; "
            "that path reserves virtual addresses outside the semantic "
            "persistent-allocation plan"
        )

    from gpu_memory_service.client.torch.allocator import (
        clear_persistent_allocator_tag_plan,
        get_or_create_persistent_allocator,
        set_persistent_allocator_tag_plan,
        validate_persistent_allocator_tag_plan_consumed,
    )
    from gpu_memory_service.integrations.sglang.memory_saver import (
        persistent_kv_pool_scope,
    )

    logger.debug("[GMS-VMM-IPC] %s init in pid=%d", name, os.getpid())
    device = _resolve_kv_pool_device(args, kwargs)
    socket = _resolve_socket(device)
    engine_id = _engine_id(device)
    try:
        manager = get_or_create_persistent_allocator(
            socket,
            device,
            engine_id,
            tag=allocator_tag(device),
            shared=allocation_shared(),
        )
    except Exception as exc:
        raise RuntimeError(
            f"GMS SGLang persistent KV allocator registration failed for {name}"
        ) from exc
    base_tag = allocator_tag(device)
    plan = _semantic_tag_plan(name, values)
    reattaching = _prepare_tag_plan(manager, engine_id, base_tag, plan)
    logger.info(
        "[GMS-VMM-IPC] %s persistent KV allocation engine_id=%s device=%d "
        "reattaching=%s semantic_tags=%d",
        name,
        engine_id,
        device,
        reattaching,
        len(plan),
    )
    set_persistent_allocator_tag_plan(base_tag, plan)
    try:
        with persistent_kv_pool_scope(reattaching=reattaching):
            result = original_init(instance, *args, **kwargs)
        validate_persistent_allocator_tag_plan_consumed(base_tag)
        # This marker is a correctness assertion consumed by the unified cache
        # adapter. Set it only after the native constructor actually consumed
        # the complete persistent allocation plan, never merely because the
        # wrapper class was selected.
        instance._gms_persistent_kv = True
        return result
    except BaseException:
        if not reattaching:
            _release_new_plan(manager, engine_id, plan)
        raise
    finally:
        clear_persistent_allocator_tag_plan(base_tag)


def _unsupported_pool_class(original, name: str):
    class UnsupportedPersistentPool(original):
        def __init__(self, *_args, **_kwargs):
            raise RuntimeError(
                f"GMS SGLang persistent KV does not support {name}; its memory "
                "layout is not represented by the current semantic allocation plan"
            )

    UnsupportedPersistentPool.__name__ = f"GMSUnsupported{name}"
    return UnsupportedPersistentPool


def install() -> bool:
    global _INSTALLED
    if _INSTALLED:
        return False
    if not _is_enabled():
        logger.debug(
            "[GMS-VMM-IPC] GMS_SGLANG_VMM_IPC_KV not set; skipping install",
        )
        return False

    try:
        from sglang.srt.mem_cache import kv_cache_configurator
        from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, MLATokenToKVPool
    except ImportError:
        logger.warning(
            "[GMS-VMM-IPC] sglang.srt.mem_cache.memory_pool not importable; "
            "cannot install VMM-IPC KV hook",
        )
        return False

    class GMSMHATokenToKVPool(MHATokenToKVPool):
        def __init__(self, *args, **kwargs):
            _persistent_init(
                MHATokenToKVPool.__init__,
                "MHATokenToKVPool",
                self,
                args,
                kwargs,
            )

    class GMSMLATokenToKVPool(MLATokenToKVPool):
        def __init__(self, *args, **kwargs):
            _persistent_init(
                MLATokenToKVPool.__init__,
                "MLATokenToKVPool",
                self,
                args,
                kwargs,
            )

    kv_cache_configurator.MHATokenToKVPool = GMSMHATokenToKVPool
    kv_cache_configurator.MLATokenToKVPool = GMSMLATokenToKVPool
    for class_name in (
        "MHATokenToKVPoolFP4",
        "MHATokenToKVPoolMXFP8",
        "MLATokenToKVPoolFP4",
        "DSATokenToKVPool",
    ):
        original = getattr(kv_cache_configurator, class_name, None)
        if original is not None:
            setattr(
                kv_cache_configurator,
                class_name,
                _unsupported_pool_class(original, class_name),
            )
    _INSTALLED = True
    logger.info(
        "[GMS-VMM-IPC install] registered persistent MHA/MLA KV pool subclasses",
    )
    return True


def install_lazy() -> None:
    """Compatibility alias; setup occurs before SGLang builds memory pools."""
    install()
