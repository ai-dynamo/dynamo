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
from typing import Optional

import torch
from gpu_memory_service.client.torch.allocator import (
    get_or_create_gms_client_memory_manager,
    get_or_create_persistent_allocator,
    gms_use_mem_pool,
    gms_use_persistent_pool,
    retarget_persistent_allocator,
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
    private_bootstrap_kv_enabled,
    promotion_engine_id,
    release_private_bootstrap_kv_pool,
    shared_kv_enabled,
)

logger = logging.getLogger(__name__)

# Published weights come back RO. KV cache is backed by persistent HBM and
# remaps by reclaiming the same (engine_id, allocation-tag) keys.
_MEMORY_SAVER_TAGS = ("weights", "kv_cache")
_TAG_LOCK_TYPES = {
    "weights": RequestedLockType.RO,
    "kv_pool": RequestedLockType.RW_PERSISTENT,
}


def _pause_resume_tags(tag: Optional[str]) -> tuple[str, ...]:
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


def get_gms_memory_saver_impl() -> Optional["GMSMemorySaverImpl"]:
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
        self._device_index = device_index
        self._kv_engine_id = allocation_engine_id(device_index)
        self._kv_tag = allocator_tag(device_index)
        self._kv_promote_engine_id = promotion_engine_id(device_index)
        self._kv_private_bootstrap = private_bootstrap_kv_enabled()
        self._kv_shared = allocation_shared()
        self.imported_weights_bytes = 0
        self.preloaded_weights_bytes = 0
        self.ro_connect_timeout_ms = ro_connect_timeout_ms
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
                defer_physical=self._kv_private_bootstrap,
            ),
        }

        logger.info(
            "[GMS] Initialized weights: requested=%s granted=%s (device=%d)",
            requested_mode.name,
            self.allocators["weights"].granted_lock_type.name,
            device_index,
        )

    @contextmanager
    def region(self, tag: str, enable_cpu_backup: bool):
        """Mark allocation region with tag."""
        if enable_cpu_backup:
            raise ValueError(
                "SGLang with GMS does not support CPU backup for allocations."
            )

        if tag == "kv_cache":
            with gms_use_persistent_pool(self._kv_tag, self._device):
                yield
            return

        if tag != "weights":
            logger.warning(
                "[GMS] Ignoring unsupported torch_memory_saver region tag %r; "
                "supported tags are %s",
                tag,
                list(_MEMORY_SAVER_TAGS),
            )
            yield
            return

        if self.allocators["weights"].granted_lock_type == GrantedLockType.RO:
            # Imported weights are already mapped and immutable in RO mode, so
            # there is no allocator swap to install for this region.
            yield
            return

        allocator = self.allocators[tag]
        if allocator.granted_lock_type != GrantedLockType.RW:
            mode = (
                allocator.granted_lock_type.name
                if allocator.granted_lock_type is not None
                else "DISCONNECTED"
            )
            # The server would reject writes on a non-RW session too, but we
            # fail before entering the allocation path so SGLang never starts a
            # partial region with the wrong lock state.
            raise RuntimeError(
                f"SGLang with GMS requires {tag!r} to be RW for allocations; got {mode}"
            )

        with gms_use_mem_pool(tag, self._device):
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
        # The old hybrid path could delegate this to torch_memory_saver, but
        # strict GMS mode has no compatible pauseable CUDA-graph allocator hook.
        raise RuntimeError(
            "SGLang with GMS does not support pauseable CUDA graphs. "
            "torch_memory_saver only supports cuda_graph in hook_mode=preload, "
            "and GMS does not use the LD_PRELOAD path."
        )

    def pause(self, tag: Optional[str] = None) -> None:
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

    def resume(self, tag: Optional[str] = None) -> None:
        for target_tag in _pause_resume_tags(tag):
            if not self.allocators[target_tag].is_unmapped:
                continue

            logger.info("[GMS] Remapping %s", target_tag)
            timeout_ms = self.ro_connect_timeout_ms if target_tag == "weights" else None
            self.allocators[target_tag].connect(
                _TAG_LOCK_TYPES[target_tag], timeout_ms=timeout_ms
            )
            if target_tag == "kv_pool":
                target_engine_id = self._kv_engine_id
                target_shared = self._kv_shared
                if self._kv_private_bootstrap:
                    target_engine_id = self._kv_promote_engine_id
                    target_shared = shared_kv_enabled()

                if self._kv_private_bootstrap:
                    self.allocators[target_tag].prepare_scratch_for_reallocation()

                self.allocators[target_tag].remap_persistent_vas(
                    target_engine_id,
                    shared=target_shared,
                )
                if target_engine_id != self._kv_engine_id:
                    retarget_persistent_allocator(
                        self._kv_tag, target_engine_id, shared=target_shared
                    )
                    release_private_bootstrap_kv_pool(
                        self.allocators[target_tag], self._kv_engine_id, logger=logger
                    )
                    logger.info(
                        "[GMS] Promoted SGLang KV namespace %s -> %s",
                        self._kv_engine_id,
                        target_engine_id,
                    )
                    self._kv_engine_id = target_engine_id
                    self._kv_shared = target_shared
                    self._kv_private_bootstrap = False
            else:
                self.allocators[target_tag].remap_all_vas()

    def finalize_write_mode(self, model: torch.nn.Module) -> None:
        """Finalize write mode: register tensors, commit, and switch to read."""
        if self.allocators["weights"].granted_lock_type != GrantedLockType.RW:
            # Read-only import mode never republishes weights.
            return

        stats = finalize_gms_write(self.allocators["weights"], model)
        self.imported_weights_bytes = stats.committed_bytes
        self.preloaded_weights_bytes = 0
