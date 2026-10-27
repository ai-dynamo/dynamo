# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SSD file for the prefix cache's SSD tier: fixed slots, and buffered reads
and writes of host KV rows to and from them on a small thread pool.

A slot holds one host row (one KV block, all layers packed). The data path is
always host row <-> file, never the GPU: a store waits on the CUDA event of the
GPU->host copy that filled the row, then writes it; a load waits on the event of
the copies queued before it that still use its rows. The I/O threads touch only
the file, the rows they were handed and their ticket; completed tickets are
queued and drained by :meth:`SSDPool.poll` on the engine thread, which owns all
cache state.

Imported lazily by rank workers: no ``dynamo._core``; torch only through the
rows and events it is handed (numpy rows for tests).
"""

from __future__ import annotations

import errno
import glob
import importlib
import logging
import os
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Slots are padded to this many bytes so O_DIRECT stays possible later.
ALIGN = 4096
DEFAULT_IO_THREADS = 4


def _round_up(n: int, align: int) -> int:
    return (n + align - 1) // align * align


def _row_view(row: Any) -> memoryview:
    """A flat byte view of a host row (a numpy array or a torch CPU tensor)."""
    if not isinstance(row, np.ndarray):
        # numpy has no bfloat16 or float8: read the bytes.
        row = row.view(importlib.import_module("torch").uint8).numpy()
    return memoryview(row.reshape(-1).view(np.uint8))


@dataclass
class Ticket:
    """One submitted store or load, handed back by :meth:`SSDPool.poll`."""

    kind: str  # "store" or "load"
    slots: list[int]
    # Free for the caller: which blocks and host rows the ticket is for.
    tag: Any = None
    error: Optional[BaseException] = None


class SSDPool:
    """A preallocated file of slots, a free list, and the I/O to it. The pool
    sizes itself: ``byte_budget`` is clamped to ``max_blocks`` and to this
    rank's ``share`` of what ``path``'s directory has free, less
    ``reserve_fraction``; ``capacity`` is 0 (and no file is made) when nothing
    is left."""

    def __init__(
        self,
        path: str,
        row_bytes: int,
        byte_budget: int,
        *,
        max_blocks: Optional[int] = None,
        reserve_fraction: float = 0.05,
        share: float = 1.0,
        max_inflight_writes: int = 64,
        io_threads: int = DEFAULT_IO_THREADS,
    ) -> None:
        self.path = path
        self.row_bytes = row_bytes
        self.stride = _round_up(row_bytes, ALIGN)
        self.max_inflight_writes = max_inflight_writes
        self._done: queue.SimpleQueue[Ticket] = queue.SimpleQueue()
        self._lock = threading.Lock()
        self._inflight_writes = 0
        self._executor = ThreadPoolExecutor(
            max_workers=io_threads, thread_name_prefix="deepinfer-ssd"
        )
        count = self._size(byte_budget, max_blocks, reserve_fraction, share)
        self.capacity = count
        self._free = list(range(count - 1, -1, -1))
        self._closed = count == 0
        if not count:
            self._fd = -1
            self._executor.shutdown()
            return
        self._fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            self._preallocate(count * self.stride)
        except BaseException:
            os.close(self._fd)
            os.unlink(path)
            raise

    def _size(
        self,
        byte_budget: int,
        max_blocks: Optional[int],
        reserve_fraction: float,
        share: float,
    ) -> int:
        """Slots to make: the budget, clamped to ``max_blocks`` and to ``share``
        of the free space of the directory. The space held by this node's other
        ``rank*.kv`` files counts as free (this path's old file is deleted first),
        so the result does not depend on the order the ranks start in."""
        want = count = max(0, byte_budget) // self.stride
        if max_blocks is not None:
            count = min(count, max_blocks)
        directory = os.path.dirname(self.path) or "."
        os.makedirs(directory, exist_ok=True)
        try:
            os.unlink(self.path)
        except FileNotFoundError:
            pass
        free = os.statvfs(directory)
        free_bytes = free.f_bavail * free.f_frsize
        for name in glob.glob(os.path.join(glob.escape(directory), "rank*.kv")):
            try:
                free_bytes += os.stat(name).st_blocks * 512
            except FileNotFoundError:
                pass
        usable = int(free_bytes * (1 - reserve_fraction) * share)
        by_space = usable // self.stride
        count = min(count, by_space)
        if count < want:
            if max_blocks is not None and count == max_blocks < min(want, by_space):
                why = f"the snapshots can serve only {max_blocks} blocks"
            else:
                why = (
                    f"{directory} has {free_bytes / 2**30:.1f} GiB free for this node "
                    f"(share {share:.0%}, {reserve_fraction:.0%} kept back)"
                )
            logger.warning(
                "SSD tier clamped to %d of %d blocks (%.1f of %.1f GiB): %s",
                count,
                want,
                count * self.stride / 2**30,
                want * self.stride / 2**30,
                why,
            )
        if count <= 0:
            if want == 0:
                reason = "the byte budget is below one block"
            elif max_blocks == 0:
                reason = "the snapshots cannot serve any block"
            else:
                reason = "no free space"
            logger.warning("No room for the prefix cache's SSD tier (%s): off", reason)
            return 0
        return count

    def _preallocate(self, size: int) -> None:
        if size == 0:
            return
        try:
            os.posix_fallocate(self._fd, 0, size)
        except (AttributeError, OSError) as e:
            if getattr(e, "errno", None) in (errno.ENOSPC, errno.EDQUOT):
                message = (
                    "could not preallocate the SSD file (%s): sparse, writes may fail"
                )
            else:
                message = "posix_fallocate unavailable (%s): sparse SSD file"
            logger.warning(message, e)
            os.ftruncate(self._fd, size)

    @property
    def nbytes(self) -> int:
        return self.capacity * self.stride

    @property
    def num_free(self) -> int:
        return len(self._free)

    @property
    def inflight_writes(self) -> int:
        return self._inflight_writes

    def alloc(self) -> Optional[int]:
        return self._free.pop() if self._free else None

    def release(self, slot: int) -> None:
        self._free.append(slot)

    def store(
        self,
        rows: list[Any],
        slots: list[int],
        after: Any = None,
        tag: Any = None,
    ) -> Ticket:
        """Host rows -> slots. ``after`` is the CUDA event of the GPU->host copy
        that filled the rows; the write waits for it. The rows must stay
        untouched until the ticket is polled."""
        ticket = Ticket("store", list(slots), tag)
        with self._lock:
            self._inflight_writes += 1
        self._submit(ticket, rows, after)
        return ticket

    def load(
        self,
        slots: list[int],
        rows: list[Any],
        after: Any = None,
        tag: Any = None,
    ) -> Ticket:
        """Slots -> host rows. ``after`` is a CUDA event recorded once the copies
        that read or wrote the rows before were queued; the read waits for it.
        The rows must stay untouched until the ticket is polled."""
        ticket = Ticket("load", list(slots), tag)
        self._submit(ticket, rows, after)
        return ticket

    def _submit(self, ticket: Ticket, rows: list[Any], after: Any) -> None:
        try:
            views = [_row_view(row) for row in rows]
            if len(views) != len(ticket.slots):
                raise ValueError("rows and slots differ in length")
            if any(len(v) != self.row_bytes for v in views):
                raise ValueError(f"every row must be {self.row_bytes} bytes")
            if self._closed:
                raise RuntimeError("SSD pool is closed")
            self._executor.submit(self._run, ticket, views, after)
        except BaseException as e:  # reported like an I/O error
            self._finish(ticket, e)
            return

    def _run(self, ticket: Ticket, views: list[memoryview], after: Any) -> None:
        error: Optional[BaseException] = None
        try:
            if after is not None:
                after.synchronize()
            write = ticket.kind == "store"
            for slot, view in zip(ticket.slots, views):
                self._transfer(slot, view, write=write)
        except BaseException as e:  # never raised from an I/O thread
            error = e
        self._finish(ticket, error)

    def _transfer(self, slot: int, view: memoryview, *, write: bool) -> None:
        """One row, looping over short reads and writes."""
        offset = slot * self.stride
        done = 0
        while done < self.row_bytes:
            rest = view[done:]
            if write:
                n = os.pwritev(self._fd, [rest], offset + done)
            else:
                n = os.preadv(self._fd, [rest], offset + done)
                if n == 0:
                    raise EOFError(f"short read of slot {slot} at byte {done}")
            done += n

    def _finish(self, ticket: Ticket, error: Optional[BaseException]) -> None:
        ticket.error = error
        if ticket.kind == "store":
            with self._lock:
                self._inflight_writes -= 1
        self._done.put(ticket)

    def poll(self) -> list[Ticket]:
        """The tickets completed since the last call (never blocks)."""
        out: list[Ticket] = []
        while True:
            try:
                out.append(self._done.get_nowait())
            except queue.Empty:
                return out

    def close(self) -> None:
        """Wait for the I/O in flight, then close and delete the file."""
        if self._closed:
            return
        self._closed = True
        self._executor.shutdown(wait=True)
        os.close(self._fd)
        try:
            os.unlink(self.path)
        except FileNotFoundError:
            pass
