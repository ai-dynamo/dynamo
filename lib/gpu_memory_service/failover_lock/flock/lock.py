# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import fcntl
import logging
import os
import threading
import time

from gpu_memory_service.failover_lock.interface import (
    FailoverLock,
    FailoverLockContended,
    FailoverLockError,
)

logger = logging.getLogger(__name__)


class FlockFailoverLock(FailoverLock):
    """flock-based failover lock.

    Uses POSIX flock(LOCK_EX) on a shared file as the lock primitive.
    The Linux kernel is the lock manager — no server process, no sidecar,
    no protocol. The lock is automatically released when the holding
    process dies (even via SIGKILL), because the kernel closes all file
    descriptors. Containers sharing an emptyDir volume can contend for the
    same lock file. Concurrent acquisition through one instance is rejected.
    """

    def __init__(self, lock_path: str):
        self._lock_path = lock_path
        self._fd: int | None = None
        self._engine_id: str | None = None
        self._state_lock = threading.Lock()
        self._acquiring = False
        self._release_requested = False
        # True once acquire() had to wait for a predecessor to release the lock
        # (a real failover), False if it acquired immediately (initial bootup).
        self._was_contended: bool = False

    @property
    def was_contended(self) -> bool:
        """Whether the most recent acquire() blocked on a predecessor.

        True  -> a predecessor held the lock and released/died: a real failover.
        False -> the lock was free on the first try: an initial bootup, not a switch.
        """
        return self._was_contended

    async def acquire(
        self,
        engine_id: str,
        poll_interval: float = 0.1,
        timeout: float | None = None,
    ) -> None:
        """Acquire the exclusive flock without blocking the event loop."""
        with self._state_lock:
            if self._fd is not None or self._acquiring:
                raise FailoverLockError("failover lock acquisition already in progress")
            self._acquiring = True
            self._release_requested = False

        self._was_contended = False
        fd: int | None = None
        start = time.monotonic()
        try:
            fd = os.open(self._lock_path, os.O_CREAT | os.O_RDWR)
            while True:
                with self._state_lock:
                    if self._release_requested:
                        raise FailoverLockError("failover lock acquisition cancelled")
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    self._was_contended = True
                    if timeout is not None:
                        elapsed = time.monotonic() - start
                        if elapsed >= timeout:
                            raise FailoverLockContended(
                                f"Timed out acquiring flock at {self._lock_path} "
                                f"for engine {engine_id} after {elapsed:.1f}s"
                            )
                    await asyncio.sleep(poll_interval)

            os.ftruncate(fd, 0)
            os.lseek(fd, 0, os.SEEK_SET)
            os.write(fd, engine_id.encode())

            with self._state_lock:
                release_requested = self._release_requested
                self._release_requested = False
                self._acquiring = False
                if not release_requested:
                    self._fd = fd
                    self._engine_id = engine_id
                    fd = None

            if release_requested:
                raise FailoverLockError(
                    "failover lock released while acquisition was completing"
                )
        except BaseException as exc:
            with self._state_lock:
                self._acquiring = False
                self._release_requested = False
            if fd is not None:
                os.close(fd)
            if isinstance(exc, asyncio.CancelledError):
                raise
            logger.error(
                "Failed to acquire failover lock at %s for engine %s: %s",
                self._lock_path,
                engine_id,
                exc,
            )
            if isinstance(exc, FailoverLockError):
                raise
            raise FailoverLockError(
                f"Failed to acquire flock at {self._lock_path} for engine "
                f"{engine_id}: {exc}"
            ) from exc

        logger.info("Failover lock acquired: %s", engine_id)

    async def release(self) -> None:
        self.release_nowait()

    def release_nowait(self) -> bool:
        """Release from a liveness/watchdog thread without an event loop.

        Taking and clearing the fd under one lock makes concurrent graceful and
        crash-path releases idempotent. ``close(2)`` releases the flock.
        """
        with self._state_lock:
            fd = self._fd
            engine_id = self._engine_id
            if fd is None:
                if self._acquiring:
                    self._release_requested = True
                    return True
                return False
            self._fd = None
            self._engine_id = None

        logger.info("Failover lock released: %s", engine_id)
        os.close(fd)
        return True

    async def owner(self) -> str | None:
        try:
            with open(self._lock_path, "r") as f:
                content = f.read().strip()
                return content if content else None
        except FileNotFoundError:
            return None
