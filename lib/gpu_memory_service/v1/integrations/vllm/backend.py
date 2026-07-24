# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Whole-engine snapshot lifecycle for the experimental GMS V1 worker."""

from __future__ import annotations

from vllm.device_allocator.sleep_mode_backend import (
    SleepModeBackend,
    SleepModeBackendFactory,
)

from .runtime import current_runtime

BACKEND_NAME = "gms-v1-snapshot"


class GMSV1SleepModeBackend(SleepModeBackend):
    """Drive V1 pool sleep and wake for the dedicated worker."""

    def __init__(self) -> None:
        super().__init__()
        runtime = current_runtime()
        self._manager = runtime.manager
        self._pool = runtime.pool

    def suspend(self, level: int = 1) -> None:
        if self._state != "RUNNING":
            raise RuntimeError(f"cannot suspend GMS V1 from {self._state}")

        self._pool.prepare_snapshot()
        self._state = "SUSPENDED"

    def resume(self, tags: list[str] | None = None) -> None:
        if self._state != "SUSPENDED":
            raise RuntimeError(f"cannot resume GMS V1 from {self._state}")

        self._state = "RESUMING"
        self._manager.wake()
        self._state = "RUNNING"


SleepModeBackendFactory.register_backend(
    BACKEND_NAME,
    "gpu_memory_service.v1.integrations.vllm.backend",
    "GMSV1SleepModeBackend",
)
