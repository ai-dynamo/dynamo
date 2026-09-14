# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Backend-neutral ownership contract for persistent engine memory.

This contract covers allocation lifetime and attachment. It deliberately does
not grant permission to read or write KV blocks: engines coordinate block
access through the lease table and publish reusable contents through the KV
directory.
"""

from __future__ import annotations

import math
import os
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol, TypeVar


@dataclass(frozen=True, slots=True)
class PersistentPoolKey:
    """Stable identity of one allocation in a persistent engine pool."""

    engine_id: str
    tag: str

    def __post_init__(self) -> None:
        if not self.engine_id:
            raise ValueError("persistent pool engine_id must not be empty")
        if not self.tag:
            raise ValueError("persistent pool tag must not be empty")


@dataclass(frozen=True, slots=True)
class PersistentPoolAllocation:
    """Backend-neutral description returned by claim and inventory calls."""

    key: PersistentPoolKey
    allocation_id: str
    size: int
    aligned_size: int
    reattached: bool = False
    claimed: bool = True

    @property
    def engine_id(self) -> str:
        """Compatibility view for existing inventory consumers."""
        return self.key.engine_id

    @property
    def tag(self) -> str:
        """Compatibility view for existing inventory consumers."""
        return self.key.tag


class PersistentPoolBackend(Protocol):
    """Persistent allocation operations over one connected backend session.

    Closing the owning session drops its claims while retaining backing. Only
    :meth:`destroy` retires backing. File descriptors returned by :meth:`export`
    are owned by the caller and must be consumed or closed exactly once.

    Repeating a claim in the same session is idempotent, not a second claim.
    Changing its shared/exclusive mode requires unclaim first. Exclusive
    reattachment requires equal aligned size; shared reattachment accepts a
    smaller request but never grows backing. Shared claims require external
    block leases before writing; pool ownership is not a KV write lock.

    Refused operations raise RuntimeError (with backend-specific error details);
    allocation exhaustion raises MemoryError. Only ownership contention is
    retried, not incompatible geometry or claim-mode changes.
    """

    def claim(
        self,
        key: PersistentPoolKey,
        aligned_size: int,
        *,
        shared: bool = False,
    ) -> PersistentPoolAllocation:
        """Claim existing backing or create it and report which occurred."""
        ...

    def export(self, key: PersistentPoolKey) -> int:
        """Export claimed backing as a caller-owned file descriptor."""
        ...

    def unclaim(self, key: PersistentPoolKey) -> bool:
        """Drop this session's claim, retaining bytes; False if already absent."""
        ...

    def inventory(
        self,
        engine_id: str | None = None,
        *,
        include_unclaimed: bool = False,
    ) -> Sequence[PersistentPoolAllocation]:
        """Observe allocations; inventory alone does not establish a claim."""
        ...

    def destroy(self, key: PersistentPoolKey) -> bool:
        """Retire backing unless another session claims it; False if absent."""
        ...


_T = TypeVar("_T")


def retry_persistent_claim(
    operation: Callable[[], _T], is_busy: Callable[[RuntimeError], bool]
) -> _T:
    """Bound contention backoff, not transport RPC time, for either backend.

    GMS_PERSISTENT_CLAIM_RETRY_SECS is a finite nonnegative retry budget
    (default 2 seconds); zero makes a single attempt. RPC timeouts belong to
    the connected session and may exceed this backoff budget.
    """
    budget = float(os.environ.get("GMS_PERSISTENT_CLAIM_RETRY_SECS", "2.0"))
    if not math.isfinite(budget) or budget < 0:
        raise ValueError(
            "GMS_PERSISTENT_CLAIM_RETRY_SECS must be finite and nonnegative"
        )
    deadline = time.monotonic() + budget
    delay = 0.05
    while True:
        try:
            return operation()
        except RuntimeError as exc:
            remaining = deadline - time.monotonic()
            if not is_busy(exc) or remaining <= 0:
                raise
            time.sleep(min(delay, remaining))
            delay = min(delay * 2, 0.5)
