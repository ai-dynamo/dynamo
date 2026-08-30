# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""V1 server ownership of physical allocations by opaque ID."""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from collections.abc import Callable

from gpu_memory_service.common.vmm import VMMDevice

logger = logging.getLogger(__name__)

_ALLOCATION_RETRY_INTERVAL = 0.5
_RECLAIM_SHUTDOWN_TIMEOUT = 30.0


class GMSAllocationManager:
    """Own retained device allocation handles by opaque allocation ID.

    Clearing an epoch unlinks every allocation ID synchronously but releases the
    physical handles on one background thread. Release blocks on the driver
    tearing down the departed owner's context, which is unbounded and varies by
    more than an order of magnitude on identical inputs, and the caller clears
    while holding lock admission for the next writer.
    """

    def __init__(self, vmm: VMMDevice, device: int):
        self._vmm = vmm
        self._device = device
        self._vmm.ensure_initialized()
        self._granularity = int(self._vmm.get_allocation_granularity(device))
        if self._granularity <= 0:
            raise ValueError("allocation granularity must be positive")
        self._allocations: dict[str, int] = {}
        self._lock = threading.Lock()
        # Reclamation has its own lock because allocate() holds _lock across its
        # OOM retry sleeps, and those retries depend on the reclaimer to progress.
        self._reclaim = threading.Condition()
        self._pending: deque[int] = deque()
        self._releasing = 0
        self._reclaimer: threading.Thread | None = None
        self._stopped = False

    def allocate(
        self,
        allocation_id: str,
        aligned_size: int,
        is_connected: Callable[[], bool] | None = None,
    ) -> None:
        with self._lock:
            if not allocation_id:
                raise RuntimeError("allocation ID must not be empty")
            if aligned_size <= 0 or aligned_size % self._granularity:
                raise RuntimeError("allocation size is not aligned for this GPU")
            if allocation_id in self._allocations:
                raise RuntimeError("allocation ID already exists")
            while True:
                if is_connected is not None and not is_connected():
                    raise ConnectionAbortedError(
                        "RW client disconnected during allocation retry"
                    )
                allocated, handle = self._vmm.create_tolerate_oom(
                    aligned_size, self._device
                )
                if allocated:
                    break
                if is_connected is None:
                    raise MemoryError(f"cannot allocate {aligned_size} GPU bytes")
                logger.warning(
                    "cuMemCreate OOM for aligned_size=%d; retrying in %.3fs",
                    aligned_size,
                    _ALLOCATION_RETRY_INTERVAL,
                )
                time.sleep(_ALLOCATION_RETRY_INTERVAL)
            self._allocations[allocation_id] = int(handle)

    def export(self, allocation_id: str) -> int:
        with self._lock:
            handle = self._get(allocation_id)
            return int(self._vmm.export_to_shareable_handle(handle))

    def free(self, allocation_id: str) -> None:
        with self._lock:
            handle = self._get(allocation_id)
            self._vmm.release(handle)
            del self._allocations[allocation_id]

    def clear(self) -> int:
        """Unlink every allocation ID now; release the handles in the background."""
        with self._lock:
            handles = tuple(self._allocations.values())
            self._allocations.clear()
            # Enqueued under _lock so that on the queued path a handle moves
            # from allocation_snapshot() into reclaim_snapshot() without ever
            # being invisible to both. The inline fallback below cannot hold
            # that: it releases outside every lock, so both snapshots read
            # empty while the driver still owns the handles. Safe because the
            # only reader that must not miss them is the checkpoint fence,
            # which is also gated on writer_reserved (checkpoint.py
            # _require_quiesced) and that stays set across _clear_epoch().
            queued = self._enqueue_release(handles)
        if not queued:
            for handle in handles:
                self._release(handle)
        return len(handles)

    def drain(self, timeout: float | None = None) -> bool:
        """Wait until every queued handle release has completed."""
        with self._reclaim:
            deadline = None if timeout is None else time.monotonic() + timeout
            while self._pending or self._releasing:
                wait = None if deadline is None else deadline - time.monotonic()
                if wait is not None and wait <= 0:
                    return False
                self._reclaim.wait(wait)
            return True

    def shutdown(self) -> None:
        """Drain pending releases and retire the background reclaimer."""
        if not self.drain(_RECLAIM_SHUTDOWN_TIMEOUT):
            # The reclaimer is a daemon thread and context teardown frees the
            # device anyway, so a stalled driver must not hang server close.
            logger.warning("GPU allocation reclamation did not drain before shutdown")
        with self._reclaim:
            self._stopped = True
            reclaimer = self._reclaimer
            self._reclaimer = None
            self._reclaim.notify_all()
        if reclaimer is not None:
            reclaimer.join(timeout=_RECLAIM_SHUTDOWN_TIMEOUT)

    def reclaim_snapshot(self) -> tuple[int, int]:
        """Return the queued and in-flight background handle release counts."""
        with self._reclaim:
            return len(self._pending), self._releasing

    def _enqueue_release(self, handles: tuple[int, ...]) -> bool:
        """Queue handles for background release; False once shut down or unstartable."""
        if not handles:
            return True
        with self._reclaim:
            if self._stopped:
                return False
            if self._reclaimer is None:
                try:
                    # Published only once start() has succeeded: a dead thread in
                    # _reclaimer would strand every later handle in _pending.
                    self._reclaimer = self._start_reclaimer()
                except RuntimeError:
                    # Thread exhaustion. Leave _reclaimer unset so the next clear
                    # retries, and make the caller release inline: a slow clear
                    # beats leaking the departed owner's GPU memory forever.
                    logger.exception("Cannot start the GPU allocation reclaimer")
                    return False
            self._pending.extend(handles)
            self._reclaim.notify_all()
            return True

    def _start_reclaimer(self) -> threading.Thread:
        reclaimer = threading.Thread(
            target=self._reclaim_forever,
            name=f"gms-v1-reclaim-{self._device}",
            daemon=True,
        )
        reclaimer.start()
        return reclaimer

    def _reclaim_forever(self) -> None:
        while True:
            with self._reclaim:
                while not self._pending and not self._stopped:
                    self._reclaim.wait()
                if not self._pending:
                    return
                handle = self._pending.popleft()
                self._releasing += 1
            try:
                # Released outside every lock: cuMemRelease is slow and must not
                # serialize against allocate() or block a waiting writer.
                self._release(handle)
            finally:
                with self._reclaim:
                    self._releasing -= 1
                    self._reclaim.notify_all()

    def _release(self, handle: int) -> None:
        try:
            self._vmm.release(handle)
        except Exception:
            logger.exception("Failed to release GPU allocation handle %d", handle)

    def _get(self, allocation_id: str) -> int:
        if not allocation_id:
            raise RuntimeError("allocation ID must not be empty")
        try:
            return self._allocations[allocation_id]
        except KeyError:
            raise RuntimeError("unknown allocation ID") from None
