# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""vLLM sleep-mode integration for the GMS V1 Torch MemPool client."""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING

from gpu_memory_service.v1.client.mempool import TorchMempoolMemoryClient
from vllm.device_allocator.sleep_mode_backend import (
    SleepModeBackend,
    SleepModeBackendFactory,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

BACKEND_NAME = "gms-v1"


class GMSV1SleepModeBackend(SleepModeBackend):
    """Expose GMS V1 through vLLM's sleep-mode backend contract."""

    def __init__(self) -> None:
        super().__init__()
        self._client = TorchMempoolMemoryClient()

    @contextmanager
    def capture_weights(self, model: Callable[[], object]) -> Iterator[None]:
        with self._client.weight_region():
            yield
        self._client.publish_weights((model(),))

    def capture_kv_cache(self):
        return self._client.kv_cache_region()

    def suspend(self, level: int = 1) -> None:
        if level != 1:
            raise ValueError("GMS V1 supports only whole-engine level 1 suspend")
        self._client.suspend()
        self._state = "SUSPENDED"

    def resume(self, tags: list[str] | None = None) -> None:
        if tags is not None:
            raise ValueError("GMS V1 does not support partial-tag resume")
        self._client.resume()
        self._state = "RUNNING"

    @classmethod
    def preserves_communicators(cls) -> bool:
        return True


SleepModeBackendFactory.register_backend(
    BACKEND_NAME,
    "gpu_memory_service.v1.integrations.vllm.backend",
    "GMSV1SleepModeBackend",
)
