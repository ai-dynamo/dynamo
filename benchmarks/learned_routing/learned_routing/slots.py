# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Machine-wide CPU slot pool: ``NUM_SLOTS`` ``flock`` lock files in ``CR/slots/``.

Every local replay holds one slot for its whole duration (CONTRACT "CPU discipline"). The pool is
shared by every ``lr-*`` process of every agent:

- A slot is an exclusive ``flock`` on ``slot-NN.lock``. ``flock`` locks belong to an open file
  description, so separate ``open()`` calls conflict even between threads of one process (POSIX
  ``fcntl``/``lockf`` record locks would not), and the kernel drops a lock when its holder dies, so
  a crashed process never leaks a slot.
- Acquisition tries the slots in a random order without blocking and polls with jitter, so no
  agent can starve another by queueing on one file.
- The holder writes ``pid host label time`` into its slot file for :func:`status`.
"""

from __future__ import annotations

import fcntl
import os
import random
import socket
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from learned_routing.paths import NUM_SLOTS


class SlotTimeout(TimeoutError):
    pass


@dataclass
class Slot:
    index: int
    path: Path
    fd: int

    def release(self) -> None:
        if self.fd < 0:
            return
        try:
            os.ftruncate(self.fd, 0)
        finally:
            fcntl.flock(self.fd, fcntl.LOCK_UN)
            os.close(self.fd)
            self.fd = -1


class SlotPool:
    def __init__(
        self, slots_dir: Path, num_slots: int = NUM_SLOTS, poll_s: float = 0.2
    ):
        if num_slots < 1:
            raise ValueError(f"num_slots must be >= 1, got {num_slots}")
        self.dir = Path(slots_dir)
        self.num_slots = num_slots
        self.poll_s = poll_s
        self.dir.mkdir(parents=True, exist_ok=True)

    def slot_path(self, index: int) -> Path:
        return self.dir / f"slot-{index:02d}.lock"

    def try_acquire(self, label: str = "") -> Slot | None:
        order = list(range(self.num_slots))
        random.shuffle(order)
        for index in order:
            path = self.slot_path(index)
            fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o666)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                os.close(fd)
                continue
            holder = f"{os.getpid()} {socket.gethostname()} {label} {time.time():.3f}\n"
            os.ftruncate(fd, 0)
            os.pwrite(fd, holder.encode(), 0)
            return Slot(index=index, path=path, fd=fd)
        return None

    def acquire(self, label: str = "", timeout_s: float | None = None) -> Slot:
        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        while True:
            slot = self.try_acquire(label)
            if slot is not None:
                return slot
            if deadline is not None and time.monotonic() >= deadline:
                raise SlotTimeout(f"no free slot in {self.dir} within {timeout_s} s")
            time.sleep(self.poll_s * (0.5 + random.random()))

    @contextmanager
    def slot(self, label: str = "", timeout_s: float | None = None) -> Iterator[Slot]:
        held = self.acquire(label, timeout_s)
        try:
            yield held
        finally:
            held.release()

    def status(self) -> list[dict]:
        """Which slots are held right now, and by whom (best effort)."""
        rows = []
        for index in range(self.num_slots):
            path = self.slot_path(index)
            fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o666)
            try:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    busy = False
                    fcntl.flock(fd, fcntl.LOCK_UN)
                except BlockingIOError:
                    busy = True
                holder = (
                    os.pread(fd, 512, 0).decode(errors="replace").strip()
                    if busy
                    else ""
                )
            finally:
                os.close(fd)
            rows.append({"slot": index, "busy": busy, "holder": holder})
        return rows
