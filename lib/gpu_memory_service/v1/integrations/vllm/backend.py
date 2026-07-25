# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Whole-engine snapshot lifecycle for the experimental GMS V1 worker."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
from gpu_memory_service.common.utils import get_socket_path
from gpu_memory_service.common.vmm import get_vmm
from gpu_memory_service.v1.memory_manager import SnapshotMemoryManager
from gpu_memory_service.v1.torch import SnapshotTorchPool
from vllm.device_allocator.sleep_mode_backend import (
    CuMemBackend,
    SleepModeBackendFactory,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from contextlib import AbstractContextManager

BACKEND_NAME = "gms-v1-snapshot"
logger = logging.getLogger(__name__)


class GMSV1SleepModeBackend(CuMemBackend):
    """Compose native KV-cache sleep with GMS V1 parameter sleep."""

    def __init__(self) -> None:
        super().__init__()
        device = torch.cuda.current_device()
        manager = SnapshotMemoryManager(
            get_socket_path(device, "snapshot-v1"),
            get_vmm(),
            device,
        )
        self._pool = SnapshotTorchPool(manager)

    def capture_weights(
        self, model: "Callable[[], object]"
    ) -> "AbstractContextManager[None]":
        return self._pool.capture_weights(model)

    def suspend(self, level: int = 1) -> None:
        if level != 1:
            raise ValueError("GMS V1 supports only whole-engine level 1 suspend")
        if self._state != "RUNNING":
            raise RuntimeError(f"cannot suspend GMS V1 from {self._state}")

        try:
            super().suspend(level)
            self._pool.sleep()
        except Exception as cause:
            logger.exception("GMS V1 suspend failed; terminating the worker process")
            raise SystemExit(1) from cause

    def resume(self, tags: list[str] | None = None) -> None:
        if tags is not None:
            raise ValueError("GMS V1 does not support partial-tag resume")
        if self._state != "SUSPENDED":
            raise RuntimeError(f"cannot resume GMS V1 from {self._state}")

        try:
            self._state = "RESUMING"
            self._pool.wake()
            super().resume(tags)
        except Exception as cause:
            logger.exception("GMS V1 resume failed; terminating the worker process")
            raise SystemExit(1) from cause


SleepModeBackendFactory.register_backend(
    BACKEND_NAME,
    "gpu_memory_service.v1.integrations.vllm.backend",
    "GMSV1SleepModeBackend",
)
