# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""vLLM sleep-mode integration for the GMS V1 Torch MemPool client."""

from __future__ import annotations

import os
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

# Set by Dynamo Snapshot in the processes it checkpoints.
_SNAPSHOT_CONTROL_DIR_ENV = "DYN_SNAPSHOT_CONTROL_DIR"


class GMSV1SleepModeBackend(SleepModeBackend):
    """Expose GMS V1 through vLLM's sleep-mode backend contract."""

    # Whether the engine is suspended because vLLM discarded the KV cache.
    _kv_cache_discarded = False

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

    def discard(self, tags: tuple[str, ...]) -> None:
        """Release the KV cache for a Dynamo Snapshot process checkpoint.

        GMS V1 releases only the whole engine, so the weights are suspended
        too; a checkpoint must not hold their GMS mappings either. Waking the
        KV cache with ``resume(["kv_cache"])`` brings both back. Other callers,
        such as vLLM's ``release_kv_cache_memory()``, expect the weights to stay
        mapped, so outside a snapshot the discard is refused, as before. Inside a
        snapshot process such a call would suspend the weights too; Dynamo does
        not make it with GMS V1.
        """
        if tuple(tags) != ("kv_cache",):
            raise ValueError(f"GMS V1 can discard only the KV cache, not {tags}")
        # TODO(upstream-vllm): This env gate stands in for "this discard comes
        # from checkpoint prepare". In vLLM 0.34, checkpoint_prepare() goes
        # through release_kv_cache_memory(), so the backend sees the same
        # discard(("kv_cache",)) from both callers and cannot tell them apart.
        # The proper fix is in vLLM, e.g. a backend hook that checkpoint prepare
        # calls instead of the generic discard(). Once it exists, drop the gate.
        if not os.environ.get(_SNAPSHOT_CONTROL_DIR_ENV):
            raise RuntimeError(
                "GMS V1 cannot discard the KV cache without suspending the "
                "weights; it does so only for a Dynamo Snapshot checkpoint"
            )
        self.suspend(1)
        self._kv_cache_discarded = True

    def resume(self, tags: list[str] | None = None) -> None:
        # Only the KV cache wake after discard() names a tag; it resumes the
        # whole engine, as discard() suspended it.
        if tags is not None and not (
            self._kv_cache_discarded and list(tags) == ["kv_cache"]
        ):
            raise ValueError("GMS V1 does not support partial-tag resume")
        self._client.resume()
        self._kv_cache_discarded = False
        self._state = "RUNNING"

    @classmethod
    def preserves_communicators(cls) -> bool:
        return True


SleepModeBackendFactory.register_backend(
    BACKEND_NAME,
    "gpu_memory_service.v1.integrations.vllm.backend",
    "GMSV1SleepModeBackend",
)
