# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Failover lock for GPU engine leader election.

In a failover deployment, there are multiple parallel engine processes ready to serve inference (warm standby engines).
The FailoverLock is used to determine which engine is active and serving inference while the others remain sleeping.

The lock couples two concerns:
  1. Leader election — which engine is active
  2. Resource safety — GPU memory is free for the new leader

Release happens on process death (implicit) or explicit call. By the
time a standby engine acquires, the old engine's GPU memory (KV cache,
CUDA contexts) has been reclaimed by the OS/driver.
"""

from abc import ABC, abstractmethod


class FailoverLock(ABC):
    @abstractmethod
    async def acquire(self, engine_id: str) -> None:
        """Block until this engine is granted the active role.

        Args:
            engine_id: Identity of the engine claiming the lock.
        """
        ...

    @abstractmethod
    async def release(self) -> None:
        """Release the lock (give up the active role).

        Called on graceful shutdown. Also happens implicitly on
        process death (e.g kernel closes the file descriptor, releasing
        the flock).
        """
        ...

    @abstractmethod
    async def owner(self) -> str | None:
        """Return the engine_id of the current lock holder, or None."""
        ...
