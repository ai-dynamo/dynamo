# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Backend-neutral ownership contract for persistent engine memory.

This contract covers allocation lifetime and attachment. It deliberately does
not grant permission to read or write KV blocks: engines coordinate block
access through the lease table and publish reusable contents through the KV
directory.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol


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

    def inventory(
        self,
        engine_id: str | None = None,
        *,
        include_unclaimed: bool = False,
    ) -> Sequence[PersistentPoolAllocation]:
        """Observe allocations; inventory alone does not establish a claim."""
        ...

    def destroy(self, key: PersistentPoolKey) -> bool:
        """Explicitly retire backing, subject to backend claimant checks."""
        ...
