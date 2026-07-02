# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import gc
import logging
import os

import torch
from vllm.device_allocator.sleep_mode_backend import (
    CuMemBackend,
    SleepModeBackend,
    SleepModeBackendFactory,
)

logger = logging.getLogger(__name__)

BACKEND_NAME = "dynamo_snapshot"
GMS_BACKEND_NAME = "dynamo_gms_snapshot"
SNAPSHOT_CONTROL_DIR = "DYN_SNAPSHOT_CONTROL_DIR"


class DynamoSnapshotBackend(SleepModeBackend):
    """CuMem sleep with FlashInfer checkpoint lifecycle ordering."""

    def __init__(self) -> None:
        super().__init__()
        self._allocator = CuMemBackend()
        self._allocator_suspended = False

    def suspend(self, level: int = 1) -> None:
        from vllm.distributed.parallel_state import (
            checkpoint_prepare_distributed_state,
            checkpoint_restore_distributed_state,
        )

        try:
            checkpoint_prepare_distributed_state()
        except Exception:
            try:
                checkpoint_restore_distributed_state()
            except Exception:
                self._state = "SUSPENDED"
                logger.exception(
                    "Failed to roll back partial FlashInfer checkpoint preparation"
                )
            raise
        self._allocator_suspended = True
        try:
            self._allocator.suspend(level)
        except Exception:
            try:
                self._allocator.resume(None)
                self._allocator_suspended = False
                checkpoint_restore_distributed_state()
            except Exception:
                self._state = "SUSPENDED"
                logger.exception(
                    "Failed to roll back FlashInfer checkpoint preparation"
                )
            else:
                self._state = "RUNNING"
            raise
        self._state = "SUSPENDED"

    def resume(self, tags: list[str] | None = None) -> None:
        del tags
        if self._state == "RUNNING":
            return
        from vllm.distributed.parallel_state import (
            checkpoint_restore_distributed_state,
        )

        self._state = "RESUMING"
        if self._allocator_suspended:
            self._allocator.resume(None)
            self._allocator_suspended = False
        checkpoint_restore_distributed_state()
        self._state = "RUNNING"

    @classmethod
    def preserves_nccl(cls) -> bool:
        return True

    @classmethod
    def preserves_communicators(cls) -> bool:
        return cls.preserves_nccl()

    @classmethod
    def preserves_graphs_with_nccl(cls) -> bool:
        return True

    @classmethod
    def preserves_graphs_with_communicators(cls) -> bool:
        return cls.preserves_graphs_with_nccl()


class DynamoGMSSnapshotBackend(SleepModeBackend):
    """GMS unmap/remap with FlashInfer checkpoint lifecycle ordering.

    Selected instead of DynamoSnapshotBackend when weights load with
    --load-format gms. Weights and KV cache live in GMS-managed VMM memory,
    so suspend unmaps the client VAs and drops the sessions rather than
    offloading through CuMem; the GMS saver/loader sidecars move the weight
    bytes to and from durable storage. FlashInfer collectives are detached
    before the unmap and reattached after the remap, exactly as in the CuMem
    snapshot backend.
    """

    def suspend(self, level: int = 1) -> None:
        del level  # GMS has no CPU-backup levels; weights persist via GMS.
        from vllm.distributed.parallel_state import (
            checkpoint_prepare_distributed_state,
            checkpoint_restore_distributed_state,
        )

        try:
            checkpoint_prepare_distributed_state()
        except Exception:
            try:
                checkpoint_restore_distributed_state()
            except Exception:
                self._state = "SUSPENDED"
                logger.exception(
                    "Failed to roll back partial FlashInfer checkpoint preparation"
                )
            raise
        try:
            self._unmap_gms_tags()
        except Exception:
            try:
                self._resume_gms_tags()
                checkpoint_restore_distributed_state()
            except Exception:
                self._state = "SUSPENDED"
                logger.exception(
                    "Failed to roll back FlashInfer checkpoint preparation "
                    "after GMS unmap failure"
                )
            else:
                self._state = "RUNNING"
            raise
        gc.collect()
        torch.cuda.empty_cache()
        self._state = "SUSPENDED"

    def resume(self, tags: list[str] | None = None) -> None:
        if self._state == "RUNNING":
            return
        from vllm.distributed.parallel_state import (
            checkpoint_restore_distributed_state,
        )

        self._state = "RESUMING"
        self._resume_gms_tags(tags)
        checkpoint_restore_distributed_state()
        self._state = "RUNNING"

    def _unmap_gms_tags(self) -> None:
        from gpu_memory_service.client.torch.allocator import (
            get_gms_client_memory_manager,
        )
        from gpu_memory_service.integrations.common.utils import GMS_TAGS

        for tag in GMS_TAGS:
            manager = get_gms_client_memory_manager(tag)
            assert manager is not None, f"GMS {tag} client is not initialized"
            assert not manager.is_unmapped, f"GMS {tag} is already unmapped"
            manager.unmap_all_vas()
            manager.abort()

    def _resume_gms_tags(self, tags: list[str] | None = None) -> None:
        """Remap GMS memory for the given tags; already-mapped tags are kept.

        Weights reconnect read-only with an unbounded wait: on snapshot
        restore the GMS loader may still be streaming weights from durable
        storage, and readiness is gated downstream. KV cache reconnects
        read-write and gets fresh physical backing at the preserved VAs.
        """
        from gpu_memory_service.client.torch.allocator import (
            get_gms_client_memory_manager,
        )
        from gpu_memory_service.common.cuda_utils import cumem_set_access
        from gpu_memory_service.common.locks import GrantedLockType, RequestedLockType
        from gpu_memory_service.integrations.common.utils import GMS_TAGS

        if tags is None:
            tags = list(GMS_TAGS)
        for tag in tags:
            manager = get_gms_client_memory_manager(tag)
            assert manager is not None, f"GMS {tag} client is not initialized"
            if not manager.is_unmapped:
                continue
            if tag == "weights":
                manager.connect(RequestedLockType.RO, timeout_ms=None)
                manager.remap_all_vas()
                # The checkpointed engine published these allocations and kept
                # RW mappings, and some model paths write into weights-tag
                # buffers after load (observed as MMU FAULT_RO_VIOLATION
                # ACCESS_TYPE_VIRT_WRITE with NVFP4 MoE on B200). The CuMem
                # snapshot baseline restores all sleep-tag memory writable;
                # mirror that by upgrading the RO import mappings to
                # read-write device access.
                for mapping in manager.mappings.values():
                    if mapping.handle:
                        cumem_set_access(
                            mapping.va,
                            mapping.aligned_size,
                            manager.device,
                            GrantedLockType.RW,
                        )
            else:
                manager.connect(RequestedLockType.RW)
                manager.reallocate_all_handles(tag=tag)
                manager.remap_all_vas()

    @classmethod
    def preserves_nccl(cls) -> bool:
        return True

    @classmethod
    def preserves_communicators(cls) -> bool:
        return cls.preserves_nccl()

    @classmethod
    def preserves_graphs_with_nccl(cls) -> bool:
        return True

    @classmethod
    def preserves_graphs_with_communicators(cls) -> bool:
        return cls.preserves_graphs_with_nccl()


def register_dynamo_snapshot_backend() -> None:
    for name, class_name in (
        (BACKEND_NAME, "DynamoSnapshotBackend"),
        (GMS_BACKEND_NAME, "DynamoGMSSnapshotBackend"),
    ):
        if name not in SleepModeBackendFactory._registry:
            SleepModeBackendFactory.register_backend(
                name,
                "dynamo.vllm.snapshot_backend",
                class_name,
            )


def select_dynamo_snapshot_backend(vllm_config: object) -> None:
    if not os.getenv(SNAPSHOT_CONTROL_DIR):
        return
    load_format = getattr(
        getattr(vllm_config, "load_config", None), "load_format", None
    )
    vllm_config.model_config.sleep_mode_backend = (
        GMS_BACKEND_NAME if load_format == "gms" else BACKEND_NAME
    )
