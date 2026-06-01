# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Disk-backed disagg_machine_id slot allocator for TRT-LLM workers."""

from __future__ import annotations

import os
import threading
from pathlib import Path


class DisaggMachineIdAllocator:
    """Allocates unique 10-bit slots (1-1024) for disagg_machine_id.

    Uses O_CREAT | O_EXCL lock files under pool_dir to guarantee that no
    two live workers (in the same process or different processes) share a
    slot.

    The lock file is named ``slot_{N}.lock`` so that O_EXCL atomicity is
    scoped to the slot number, not the connection ID.  The connection ID is
    stored as the file *contents* to enable orphan recovery across process
    restarts.  Slot 0 is the sentinel (unset) value and is never allocated.
    """

    def __init__(self, pool_dir: str, max_slots: int = 1024) -> None:
        self._pool_dir = Path(pool_dir)
        self._max_slots = max_slots
        self._lock = threading.Lock()
        self._pool_dir.mkdir(parents=True, exist_ok=True)

    def _slot_lock_file(self, slot: int) -> Path:
        return self._pool_dir / f"slot_{slot}.lock"

    def allocate(self, connection_id: int) -> int:
        """Allocate a unique slot for connection_id.

        Recovers orphaned slots when this connection_id already owns one.
        Raises RuntimeError when the pool is exhausted (>1024 live workers).
        """
        with self._lock:
            # Recovery: check whether this connection_id already owns a slot.
            for f in self._pool_dir.iterdir():
                if not (f.name.startswith("slot_") and f.suffix == ".lock"):
                    continue
                try:
                    if f.read_text().strip() == str(connection_id):
                        return int(f.stem[5:])  # "slot_{N}" -> N
                except (ValueError, OSError):
                    continue

            # Claim the smallest free slot atomically.
            # O_EXCL on slot_{N}.lock ensures cross-process uniqueness:
            # at most one caller can successfully create a given slot file.
            for slot in range(1, self._max_slots + 1):
                lock_file = self._slot_lock_file(slot)
                try:
                    fd = os.open(
                        str(lock_file), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644
                    )
                    os.write(fd, str(connection_id).encode())
                    os.close(fd)
                    return slot
                except FileExistsError:
                    continue  # Slot claimed by another worker; try next.

            raise RuntimeError("disagg_machine_id pool exhausted (1024 workers live)")

    def free(self, connection_id: int) -> None:
        """Release the slot owned by connection_id."""
        with self._lock:
            for f in self._pool_dir.iterdir():
                if not (f.name.startswith("slot_") and f.suffix == ".lock"):
                    continue
                try:
                    if f.read_text().strip() == str(connection_id):
                        f.unlink()
                        return
                except (ValueError, OSError):
                    continue

    @property
    def num_allocated(self) -> int:
        with self._lock:
            return sum(
                1
                for f in self._pool_dir.iterdir()
                if f.name.startswith("slot_") and f.suffix == ".lock"
            )
