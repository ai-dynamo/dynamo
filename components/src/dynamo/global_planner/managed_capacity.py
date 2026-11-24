# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Capacity contract for the opt-in, serialized allocation controller."""

from dataclasses import dataclass
from typing import Protocol


class ManagedCapacityError(RuntimeError):
    """Capacity or write ownership is uncertain; further mutations must stop."""


@dataclass(frozen=True)
class ManagedPoolSnapshot:
    pool_id: str
    identity: str
    target_replicas: int
    observed_replicas: int
    ready_replicas: int
    terminating_replicas: int
    settled: bool


class ManagedCapacity(Protocol):
    async def acquire(self, writer_id: str) -> None:
        """Claim all pool writer fences before reconstructing a baseline."""
        ...

    async def observe(self) -> dict[str, ManagedPoolSnapshot]:
        """Read every enrolled pool, rejecting unknown or changed ownership."""
        ...

    async def scale(
        self,
        pool_id: str,
        replicas: int,
        *,
        expected_identity: str,
        expected_target: int,
    ) -> None:
        """CAS-write an exact target using the current writer fence, without waiting."""
        ...
