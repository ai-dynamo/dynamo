# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""torch_memory_saver implementation for GPU Memory Service.

SGLang with GMS owns exactly two memory classes:
1. "weights" via the shared RO/RW publish flow
2. "kv_cache" via the GMS-owned persistent KV pool

Unsupported release/resume tags stay no-ops with a warning so the generic
SGLang memory-control API can still pass broader tag sets without reintroducing
the old torch-memory-saver fallback. `cuda_graph` is a hard error because the
pauseable CUDA-graph path depends on the LD_PRELOAD torch allocator hooks that
GMS intentionally does not use.
"""

from __future__ import annotations

import gc
import logging
from contextlib import contextmanager
from contextvars import ContextVar

import torch
from gpu_memory_service.client.torch.allocator import (
    get_or_create_gms_client_memory_manager,
    get_or_create_persistent_allocator,
    gms_use_mem_pool,
    gms_use_persistent_pool,
)
from gpu_memory_service.common.locks import GrantedLockType, RequestedLockType
from gpu_memory_service.common.utils import get_socket_path
from gpu_memory_service.integrations.common.utils import (
    finalize_gms_write,
    get_gms_persistent_kv_socket,
)
from gpu_memory_service.integrations.sglang.kv_identity import (
    allocation_engine_id,
    allocation_shared,
    allocator_tag,
)

logger = logging.getLogger(__name__)

_PERSISTENT_KV_SCOPE: ContextVar[bool | None] = ContextVar(
    "gms_sglang_persistent_kv_scope", default=None
)
_PRESERVE_KV_ZERO_FILL: ContextVar[list[int] | None] = ContextVar(
    "gms_sglang_preserve_kv_zero_fill", default=None
)

# Published weights come back RO. KV cache remaps the same persistent
# (engine_id, allocator-tag) allocation; page leases arbitrate writes.
_MEMORY_SAVER_TAGS = ("weights", "kv_cache")
_TAG_LOCK_TYPES = {
    "weights": RequestedLockType.RO,
    "kv_pool": RequestedLockType.RW_PERSISTENT,
}


def _install_contextual_zeros_hook() -> None:
    current_zeros = torch.zeros
    if getattr(current_zeros, "_gms_sglang_contextual_zeros", False) is True:
        return

    def contextual_zeros(*args, **kwargs):
        replacements = _PRESERVE_KV_ZERO_FILL.get()
        if replacements is not None:
            replacements[0] += 1
            return torch.empty(*args, **kwargs)
        return current_zeros(*args, **kwargs)

    contextual_zeros._gms_sglang_contextual_zeros = True
    contextual_zeros._gms_original = current_zeros
    torch.zeros = contextual_zeros


@contextmanager
def persistent_kv_pool_scope(*, reattaching: bool):
    """Select GMS only for SGLang's physical KV-pool constructors.

    SGLang also labels request-to-token metadata as ``kv_cache``. That table is
    process-local scheduler state and must neither survive nor be shared with a
    standby. The pool-constructor hook establishes this narrow scope around the
    native MHA/MLA constructor; unscoped ``kv_cache`` regions use normal CUDA.
    """
    token = _PERSISTENT_KV_SCOPE.set(bool(reattaching))
    try:
        yield
    finally:
        _PERSISTENT_KV_SCOPE.reset(token)


@contextmanager
def _preserve_persistent_kv_zeros(enabled: bool):
    if not enabled:
        yield
        return

    _install_contextual_zeros_hook()
    replacements = [0]
    token = _PRESERVE_KV_ZERO_FILL.set(replacements)
    try:
        yield
    finally:
        _PRESERVE_KV_ZERO_FILL.reset(token)
        if replacements[0]:
            logger.info(
                "[GMS-VMM-IPC] allocated %d persistent SGLang KV tensors with "
                "torch.empty to preserve existing pages",
                replacements[0],
            )


def _pause_resume_tags(tag: str | None) -> tuple[str, ...]:
    if tag is None:
        return ("weights", "kv_pool")
    if tag == "kv_cache":
        return ("kv_pool",)
    if tag in _TAG_LOCK_TYPES:
        return (tag,)
    logger.warning(
        "[GMS] Ignoring unsupported torch_memory_saver tag %r; supported tags are %s",
        tag,
        list(_MEMORY_SAVER_TAGS),
    )
    return ()


def get_gms_memory_saver_impl() -> GMSMemorySaverImpl | None:
    """Get or lazily initialize the GMS impl from torch_memory_saver."""
    try:
        import torch_memory_saver

        singleton = torch_memory_saver.torch_memory_saver
        impl = getattr(singleton, "gms_impl", None)
        if impl is None and hasattr(singleton, "_ensure_initialized"):
            singleton._ensure_initialized()
            impl = getattr(singleton, "gms_impl", None)
        return impl
    except (ImportError, AttributeError):
        return None


class GMSMemorySaverImpl:
    """SGLang memory saver implementation backed only by GMS."""

    def __init__(
        self,
        device_index: int,
        mode=None,
        ro_connect_timeout_ms=None,
    ):
        self._device = torch.device("cuda", device_index)
        self._kv_engine_id = allocation_engine_id(device_index)
        self._kv_tag = allocator_tag(device_index)
        self._kv_shared = allocation_shared()
        self.imported_weights_bytes = 0
        self.preloaded_weights_bytes = 0
        self.ro_connect_timeout_ms = ro_connect_timeout_ms
        self._active_region_depth = 0
        self._pending_write_model: torch.nn.Module | None = None
        requested_mode = mode or RequestedLockType.RW_OR_RO
        self.allocators = {
            "weights": get_or_create_gms_client_memory_manager(
                get_socket_path(device_index, "weights"),
                device_index,
                mode=requested_mode,
                tag="weights",
            ),
            "kv_pool": get_or_create_persistent_allocator(
                get_gms_persistent_kv_socket(device_index, "GMS_SGLANG_VMM_IPC_SOCKET"),
                device_index,
                self._kv_engine_id,
                tag=self._kv_tag,
                shared=self._kv_shared,
            ),
        }

        logger.info(
            "[GMS] Initialized weights: requested=%s granted=%s (device=%d)",
            requested_mode.name,
            self.allocators["weights"].granted_lock_type.name,
            device_index,
        )

    @contextmanager
    def region(
        self,
        tag: str,
        enable_cpu_backup: bool,
        enable_disk_backup: bool = False,
        cpu_backup_backend: str | None = None,
    ):
        """Use the tag's RW pool and publish pending weights after clean exit."""
        if enable_cpu_backup:
            raise ValueError(
                "SGLang with GMS does not support CPU backup for allocations."
            )
        if enable_disk_backup:
            raise ValueError(
                "SGLang with GMS does not support disk backup for allocations."
            )
        if cpu_backup_backend is not None:
            raise ValueError(
                "SGLang with GMS does not support CPU backup backends for allocations."
            )

        if tag not in _MEMORY_SAVER_TAGS:
            logger.warning(
                "[GMS] Ignoring unsupported torch_memory_saver region tag %r; "
                "supported tags are %s",
                tag,
                list(_MEMORY_SAVER_TAGS),
            )
            yield
            return

        persistent_kv_reattach = (
            _PERSISTENT_KV_SCOPE.get() if tag == "kv_cache" else None
        )
        if tag == "kv_cache" and persistent_kv_reattach is None:
            yield
            return

        if (
            tag == "weights"
            and self.allocators["weights"].granted_lock_type == GrantedLockType.RO
        ):
            # Imported weights are already mapped and immutable in RO mode, so
            # there is no allocator swap to install for this region.
            yield
            return

        target_tag = "kv_pool" if tag == "kv_cache" else tag
        allocator = self.allocators[target_tag]
        required_lock = (
            GrantedLockType.RW_PERSISTENT
            if target_tag == "kv_pool"
            else GrantedLockType.RW
        )
        if allocator.granted_lock_type != required_lock:
            mode = (
                allocator.granted_lock_type.name
                if allocator.granted_lock_type is not None
                else "DISCONNECTED"
            )
            # The server would reject writes on a non-RW session too, but we
            # fail before entering the allocation path so SGLang never starts a
            # partial region with the wrong lock state.
            raise RuntimeError(
                f"SGLang with GMS requires {tag!r} to be {required_lock.name} "
                f"for allocations; got {mode}"
            )

        self._active_region_depth += 1
        clean_exit = False
        try:
            pool = (
                gms_use_persistent_pool(self._kv_tag, self._device)
                if tag == "kv_cache"
                else gms_use_mem_pool(tag, self._device)
            )
            with pool, _preserve_persistent_kv_zeros(bool(persistent_kv_reattach)):
                yield
            clean_exit = True
        finally:
            self._active_region_depth -= 1
            if not clean_exit:
                self._pending_write_model = None

        if self._active_region_depth == 0:
            self._finalize_pending_write()

    @contextmanager
    def cuda_graph(
        self,
        cuda_graph,
        pool,
        stream,
        capture_error_mode,
        tag: str,
        enable_cpu_backup: bool,
        cpu_backup_backend: str | None = None,
    ):
        # The old hybrid path could delegate this to torch_memory_saver, but
        # strict GMS mode has no compatible pauseable CUDA-graph allocator hook.
        raise RuntimeError(
            "SGLang with GMS does not support pauseable CUDA graphs. "
            "torch_memory_saver only supports cuda_graph in hook_mode=preload, "
            "and GMS does not use the LD_PRELOAD path."
        )

    def pause(self, tag: str | None = None) -> None:
        for target_tag in _pause_resume_tags(tag):
            if self.allocators[target_tag].is_unmapped:
                continue
            logger.info("[GMS] Unmapping %s", target_tag)
            self.allocators[target_tag].unmap_all_vas()
            # abort() drops the current session after unmapping while keeping
            # the VA reservation alive for the next resume().
            self.allocators[target_tag].abort()
        gc.collect()
        torch.cuda.empty_cache()

    def resume(self, tag: str | None = None) -> None:
        for target_tag in _pause_resume_tags(tag):
            if not self.allocators[target_tag].is_unmapped:
                continue

            logger.info("[GMS] Remapping %s", target_tag)
            timeout_ms = self.ro_connect_timeout_ms if target_tag == "weights" else None
            self.allocators[target_tag].connect(
                _TAG_LOCK_TYPES[target_tag], timeout_ms=timeout_ms
            )
            if target_tag == "kv_pool":
                self.allocators[target_tag].remap_persistent_vas(
                    self._kv_engine_id,
                    shared=self._kv_shared,
                    synchronize_per_mapping=False,
                )
            else:
                self.allocators[target_tag].remap_all_vas()

    def finalize_write_mode(self, model: torch.nn.Module) -> None:
        """Publish write-mode weights after all managed GMS regions exit."""
        if model is None:
            raise TypeError("GMS weight publication model must not be None")

        if self.allocators["weights"].granted_lock_type != GrantedLockType.RW:
            # Read-only import mode never republishes weights.
            self._pending_write_model = None
            return

        if self._pending_write_model is not None:
            raise RuntimeError("GMS weight publication is already pending")

        self._pending_write_model = model
        if self._active_region_depth == 0:
            self._finalize_pending_write()

    def _finalize_pending_write(self) -> None:
        """Consume and publish pending weights outside managed GMS regions."""
        model = self._pending_write_model
        self._pending_write_model = None
        if model is None:
            return

        stats = finalize_gms_write(self.allocators["weights"], model)
        self.imported_weights_bytes = stats.committed_bytes
        self.preloaded_weights_bytes = 0
