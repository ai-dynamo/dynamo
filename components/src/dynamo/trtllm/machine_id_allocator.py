"""Disk-backed disagg_machine_id slot allocator for TRT-LLM workers."""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Optional


class DisaggMachineIdAllocator:
    """Allocates unique 10-bit slots (1-1024) for disagg_machine_id.

    Uses O_CREAT | O_EXCL lock files under pool_dir to guarantee that no
    two live workers share a slot.  Slot 0 is the sentinel (unset) value
    and is never allocated.
    """

    def __init__(self, pool_dir: str, max_slots: int = 1024) -> None:
        self._pool_dir = Path(pool_dir)
        self._max_slots = max_slots
        self._lock = threading.Lock()
        self._pool_dir.mkdir(parents=True, exist_ok=True)

    def _slot_file(self, connection_id: int) -> Path:
        return self._pool_dir / f"{connection_id}.slot"

    def allocate(self, connection_id: int) -> int:
        """Allocate a unique slot for connection_id.

        Recovers orphaned slots (connection_id already has a file).
        Raises RuntimeError when the pool is exhausted (>1024 workers).
        Raises FileExistsError on collision (defensive; O_EXCL should prevent).
        """
        with self._lock:
            sf = self._slot_file(connection_id)
            # Recover orphaned slot (process died but file survived)
            if sf.exists():
                return int(sf.read_text().strip())

            # Find smallest free slot >= 1
            allocated: set[int] = set()
            for f in self._pool_dir.iterdir():
                if f.suffix == ".slot" and f.name != ".slot":
                    try:
                        allocated.add(int(f.read_text().strip()))
                    except (ValueError, OSError):
                        continue

            for slot in range(1, self._max_slots + 1):
                if slot not in allocated:
                    # Atomic slot file creation
                    try:
                        fd = os.open(str(sf), os.O_CREAT | os.O_EXCL, 0o644)
                        os.write(fd, str(slot).encode())
                        os.close(fd)
                    except FileExistsError:
                        # Another worker grabbed first; recover their slot
                        return int(sf.read_text().strip())
                    return slot

            raise RuntimeError("disagg_machine_id pool exhausted (1024 workers live)")

    def free(self, connection_id: int) -> None:
        """Remove the lock file for connection_id, freeing the slot."""
        with self._lock:
            sf = self._slot_file(connection_id)
            if sf.exists():
                sf.unlink()

    @property
    def num_allocated(self) -> int:
        with self._lock:
            return sum(
                1 for f in self._pool_dir.iterdir()
                if f.suffix == ".slot" and f.name != ".slot"
            )