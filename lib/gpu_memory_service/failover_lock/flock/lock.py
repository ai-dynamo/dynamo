# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import fcntl
import logging
import os

from gpu_memory_service.failover_lock.interface import FailoverLock

logger = logging.getLogger(__name__)


class FlockFailoverLock(FailoverLock):
    """flock-based failover lock for single-node leader election.

    Uses POSIX flock(LOCK_EX) on a shared file as the lock primitive.
    The Linux kernel is the lock manager — no server process, no sidecar,
    no protocol. The lock is automatically released when the holding
    process dies (even via SIGKILL), because the kernel closes all file
    descriptors.

    Cross-container operation: A lock held by one container is visible to any other container accessing the same file
    on a shared emptyDir volume.
    """

    def __init__(self, lock_path: str):
        self._lock_path = lock_path
        self._fd: int | None = None
        self._engine_id: str | None = None

    async def acquire(self, engine_id: str, poll_interval: float = 0.1) -> None:
        """Acquire the exclusive flock via non-blocking poll loop.

        Uses LOCK_NB to avoid blocking the asyncio event loop. Polls
        every ``poll_interval`` seconds (default 500ms).
        Polling keeps us from blocking the event loop.
        """
        # O_CREAT: create the file if it doesn't exist
        # O_RDWR:  open for reading and writing (flock requires a valid fd,
        #          and we write our engine_id into the file after acquiring)
        self._fd = os.open(self._lock_path, os.O_CREAT | os.O_RDWR)

        while True:
            try:
                # LOCK_EX: exclusive lock — only one process can hold it
                # LOCK_NB: non-blocking — raises BlockingIOError instead of
                #          blocking the calling thread, so the asyncio event
                #          loop stays responsive between poll attempts
                fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                await asyncio.sleep(poll_interval)

        self._engine_id = engine_id

        # Write identity into the lock file for observability (owner() reads
        # this). We use raw fd ops because we must keep the fd open — closing
        # it would release the flock. ftruncate clears any stale content from
        # the previous holder, lseek rewinds, write stamps our id.
        os.ftruncate(self._fd, 0)
        os.lseek(self._fd, 0, os.SEEK_SET)
        os.write(self._fd, engine_id.encode())

        logger.info("Failover lock acquired: %s", engine_id)

    async def release(self) -> None:
        if self._fd is None:
            return

        # Guard: only the holder should release. If we don't hold the lock
        # (e.g. double-release or programming error), closing the fd is a
        # no-op for flock semantics — but log a warning for visibility.
        try:
            current = await self.owner()
            if current != self._engine_id:
                logger.warning(
                    "Releasing lock but owner is %r, expected %r",
                    current,
                    self._engine_id,
                )
        except Exception:
            pass

        logger.info("Failover lock released: %s", self._engine_id)
        os.close(self._fd)
        self._fd = None
        self._engine_id = None

    async def owner(self) -> str | None:
        try:
            with open(self._lock_path, "r") as f:
                content = f.read().strip()
                return content if content else None
        except FileNotFoundError:
            return None
