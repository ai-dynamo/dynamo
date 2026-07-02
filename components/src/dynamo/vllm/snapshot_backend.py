# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import os

from vllm.device_allocator.sleep_mode_backend import (
    CuMemBackend,
    SleepModeBackend,
    SleepModeBackendFactory,
)

logger = logging.getLogger(__name__)

BACKEND_NAME = "dynamo_snapshot"
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


def register_dynamo_snapshot_backend() -> None:
    if BACKEND_NAME in SleepModeBackendFactory._registry:
        return
    SleepModeBackendFactory.register_backend(
        BACKEND_NAME,
        "dynamo.vllm.snapshot_backend",
        "DynamoSnapshotBackend",
    )


def select_dynamo_snapshot_backend(vllm_config: object) -> None:
    if os.getenv(SNAPSHOT_CONTROL_DIR):
        vllm_config.model_config.sleep_mode_backend = BACKEND_NAME
