# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Process-local ownership shared by the V1 vLLM worker and sleep backend."""

from __future__ import annotations

from dataclasses import dataclass

from ...client.memory_manager import SnapshotMemoryManager
from ...client.torch import SnapshotTorchPool


@dataclass(frozen=True)
class VllmSnapshotRuntime:
    manager: SnapshotMemoryManager
    pool: SnapshotTorchPool


_runtime: VllmSnapshotRuntime | None = None


def install_runtime(
    manager: SnapshotMemoryManager,
    pool: SnapshotTorchPool,
) -> VllmSnapshotRuntime:
    """Install the V1 state shared with the sleep backend."""
    global _runtime
    _runtime = VllmSnapshotRuntime(manager, pool)
    return _runtime


def current_runtime() -> VllmSnapshotRuntime:
    runtime = _runtime
    if runtime is None:
        raise RuntimeError("GMS V1 vLLM runtime is not initialized")
    return runtime
