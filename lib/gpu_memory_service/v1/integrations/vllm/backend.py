# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GMS-owned vLLM sleep lifecycle for Parameters and KV cache."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
from gpu_memory_service.common.utils import get_socket_path
from gpu_memory_service.common.vmm import get_vmm
from gpu_memory_service.v1.memory_manager import (
    EphemeralKVCacheMemoryManager,
    PersistentParameterMemoryManager,
)
from gpu_memory_service.v1.torch import V1TorchPools
from vllm.device_allocator.sleep_mode_backend import (
    SleepModeBackend,
    SleepModeBackendFactory,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from contextlib import AbstractContextManager

BACKEND_NAME = "gms-v1"
logger = logging.getLogger(__name__)


class GMSV1SleepModeBackend(SleepModeBackend):
    """Own the rank-local GMS Parameter and ephemeral KV domains."""

    def __init__(self) -> None:
        super().__init__()
        device = torch.cuda.current_device()
        vmm = get_vmm()
        weights = PersistentParameterMemoryManager(
            get_socket_path(device, "weights"),
            vmm,
            device,
        )
        kv_cache = EphemeralKVCacheMemoryManager(
            get_socket_path(device, "kv_cache"),
            vmm,
            device,
        )
        self._pools = V1TorchPools(weights, kv_cache)

    def capture_weights(
        self, model: "Callable[[], object]"
    ) -> "AbstractContextManager[None]":
        return self._pools.capture_weights(model)

    def capture_kv_cache(self) -> "AbstractContextManager[None]":
        return self._pools.capture_kv_cache()

    def suspend(self, level: int = 1) -> None:
        if level != 1:
            raise ValueError("GMS V1 supports only whole-engine level 1 suspend")
        if self._state != "RUNNING":
            raise RuntimeError(f"cannot suspend GMS V1 from {self._state}")

        try:
            self._pools.raise_if_allocator_failed()
            self._pools.weights.sleep()
            self._pools.kv_cache.sleep()
            self._state = "SUSPENDED"
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
            self._pools.kv_cache.wake()
            self._pools.weights.wake()
            self._state = "RUNNING"
        except Exception as cause:
            logger.exception("GMS V1 resume failed; terminating the worker process")
            raise SystemExit(1) from cause

    @classmethod
    def preserves_communicators(cls) -> bool:
        return True


SleepModeBackendFactory.register_backend(
    BACKEND_NAME,
    "gpu_memory_service.v1.integrations.vllm.backend",
    "GMSV1SleepModeBackend",
)
