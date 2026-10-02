# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence
from typing import Protocol

from gpu_memory_service.common.persistent_pool import PersistentAllocation
from gpu_memory_service.kv_recovery.types import (
    KVBlockRecord,
    KVRecoveryManifest,
    RecoveryResult,
)


class RecoverySession(Protocol):
    result: RecoveryResult
    reason: str

    def blocks(self) -> Sequence[KVBlockRecord]: ...

    def invalidate_blocks(self, block_ids: Sequence[int]) -> None: ...

    def publish_blocks(self, blocks: Sequence[KVBlockRecord]) -> None: ...

    def clear(self) -> None: ...

    def close(self) -> None: ...


class KVRecoveryManager(Protocol):
    def recover(
        self,
        allocations: Sequence[PersistentAllocation],
        manifest: KVRecoveryManifest,
    ) -> RecoverySession:
        """Acquire metadata ownership after predecessor GPU writers are fenced."""
        ...
