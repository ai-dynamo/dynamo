# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""torch_memory_saver implementation for GPU Memory Service.

SGLang with GMS owns exactly two memory classes:
1. "weights" via the shared RO/RW publish flow
2. "kv_cache" via the RW failover flow

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
    gms_use_mem_pool,
)
from gpu_memory_service.common.locks import GrantedLockType, RequestedLockType
from gpu_memory_service.common.utils import get_socket_path
from gpu_memory_service.integrations.common.utils import GMS_TAGS, finalize_gms_write

logger = logging.getLogger(__name__)

# Published weights must come back RO, while KV cache always resumes in a fresh
# RW epoch so the restored engine can rebuild mutable cache state.
_TAG_LOCK_TYPES = {"weights": RequestedLockType.RO, "kv_cache": RequestedLockType.RW}


def _pause_resume_tags(tag: Optional[str]) -> tuple[str, ...]:
    if tag is None:
        return GMS_TAGS
    if tag in _TAG_LOCK_TYPES:
        return (tag,)
    logger.warning(
        "[GMS] Ignoring unsupported torch_memory_saver tag %r; supported tags are %s",
        tag,
        list(GMS_TAGS),
    )
    return ()


def get_gms_memory_saver_impl() -> Optional["GMSMemorySaverImpl"]:
    """Get the GMS memory saver impl from the torch_memory_saver singleton."""
    try:
        import torch_memory_saver

        return torch_memory_saver.torch_memory_saver.gms_impl
    except (ImportError, AttributeError):
        return None


class GMSMemorySaverImpl:
    """SGLang memory saver implementation backed only by GMS."""

    def __init__(
        self,
        device_index: int,
        mode=None,
    ):
        self._device = torch.device("cuda", device_index)
        self.imported_weights_bytes = 0
        requested_mode = mode or RequestedLockType.RW_OR_RO
        self.allocators = {
            tag: get_or_create_gms_client_memory_manager(
                get_socket_path(device_index, tag),
                device_index,
                # weights follow the configured publish/import mode; kv_cache is
                # always mutable and therefore always needs an RW session.
                mode=requested_mode if tag == "weights" else RequestedLockType.RW,
                tag=tag,
            )
            for tag in GMS_TAGS
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

        if tag not in _TAG_LOCK_TYPES:
            logger.warning(
                "[GMS] Ignoring unsupported torch_memory_saver region tag %r; "
                "supported tags are %s",
                tag,
                list(GMS_TAGS),
            )
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
            self.allocators[target_tag].connect(_TAG_LOCK_TYPES[target_tag])
            if target_tag == "kv_cache":
                # KV cache resumes into a new RW layout epoch, so the handles
                # must be re-created before the VA range is mapped again.
                self.allocators[target_tag].reallocate_all_handles(tag=target_tag)
            self.allocators[target_tag].remap_all_vas()

    def finalize_write_mode(self, model: torch.nn.Module) -> None:
        """Finalize write mode: register tensors, commit, and switch to read."""
        if self.allocators["weights"].granted_lock_type != GrantedLockType.RW:
            # Read-only import mode never republishes weights.
            return

        self.imported_weights_bytes = finalize_gms_write(
            self.allocators["weights"], model
        )
