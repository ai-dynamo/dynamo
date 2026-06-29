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
import sys

from gpu_memory_service.integrations.common.kv_lease_client import resolve_lease_device
from gpu_memory_service.integrations.common.utils import (
    env_enabled_by_default,
    get_gms_persistent_kv_socket,
)
from gpu_memory_service.integrations.sglang.kv_identity import (
    allocation_engine_id,
    allocation_shared,
    allocator_tag,
    private_bootstrap_kv_enabled,
)

logger = logging.getLogger(__name__)

_INSTALLED = False
_LAZY_HOOK_INSTALLED = False


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


def _wrap_init(cls, name: str) -> None:
    """Wrap ``cls.__init__`` so it runs inside a persistent pool scope."""
    from gpu_memory_service.client.torch.allocator import (
        get_or_create_persistent_allocator,
    )

    original_init = cls.__init__

    def _patched_init(self, *args, **kwargs):
        logger.debug(
            "[GMS-VMM-IPC] %s init in pid=%d", name, os.getpid()
        )
        # The device is one of the constructor args on current SGLang,
        # but may be passed positionally as plain "cuda". In that case
        # SGLang has already set the rank-local current device.
        device = _resolve_kv_pool_device(args, kwargs)
        socket = _resolve_socket(device)
        engine_id = _engine_id(device)
        try:
            get_or_create_persistent_allocator(
                socket,
                device,
                engine_id,
                tag=allocator_tag(device),
                shared=allocation_shared(),
                defer_physical=private_bootstrap_kv_enabled(),
            )
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"GMS SGLang persistent KV allocator registration failed for {name}"
            ) from exc
        logger.info(
            "[GMS-VMM-IPC] %s using registered GMS persistent KV "
            "allocator (engine_id=%s, device=%d)",
            name,
            engine_id,
            device,
        )
        return original_init(self, *args, **kwargs)

    cls.__init__ = _patched_init  # type: ignore[method-assign]
    logger.debug(
        "[GMS-VMM-IPC] patched %s init in pid=%d", name, os.getpid()
    )


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
        from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, MLATokenToKVPool
    except ImportError:
        logger.warning(
            "[GMS-VMM-IPC] sglang.srt.mem_cache.memory_pool not importable; "
            "cannot install VMM-IPC KV hook",
        )
        return False

    _wrap_init(MHATokenToKVPool, "MHATokenToKVPool")
    _wrap_init(MLATokenToKVPool, "MLATokenToKVPool")
    _INSTALLED = True
    logger.info(
        "[GMS-VMM-IPC install] patched MHATokenToKVPool.__init__ and "
        "MLATokenToKVPool.__init__",
    )
    return True


def install_lazy() -> None:
    """Register a sys.meta_path finder that calls install() AFTER the
    target SGLang module's body has finished executing.

    We wrap the real Loader's exec_module so the patch lands once the
    target module has fully initialized — otherwise the module body's
    own ``__init__`` definition would overwrite our patch."""
    global _LAZY_HOOK_INSTALLED
    if _LAZY_HOOK_INSTALLED:
        return
    if not _is_enabled():
        return

    target_name = "sglang.srt.mem_cache.memory_pool"

    class _PatchAfterLoad:
        def __init__(self, real_loader):
            self._real = real_loader

        def create_module(self, spec):
            if hasattr(self._real, "create_module"):
                return self._real.create_module(spec)
            return None

        def exec_module(self, module):
            self._real.exec_module(module)
            # Module body has run; classes are defined. Now patch.
            try:
                install()
            except Exception:  # noqa: BLE001
                logger.exception("[GMS-VMM-IPC] SGLang post-load install raised")
                raise

    class _Finder:
        def find_spec(self, name, path=None, target_pkg=None):
            if name != target_name:
                return None
            # Find the real spec via the OTHER finders.
            for finder in sys.meta_path:
                if finder is self:
                    continue
                if hasattr(finder, "find_spec"):
                    spec = finder.find_spec(name, path, target_pkg)
                    if spec is not None and spec.loader is not None:
                        # Self-uninstall: only need to fire once.
                        try:
                            sys.meta_path.remove(self)
                        except ValueError:
                            pass
                        spec.loader = _PatchAfterLoad(spec.loader)
                        return spec
            return None

    sys.meta_path.insert(0, _Finder())
    _LAZY_HOOK_INSTALLED = True


# Eager install only if SGLang's memory_pool is already loaded.
if _is_enabled() and "sglang.srt.mem_cache.memory_pool" in sys.modules:
    try:
        install()
    except Exception:  # noqa: BLE001
        logger.exception("[GMS-VMM-IPC] SGLang auto-install raised")
        raise
