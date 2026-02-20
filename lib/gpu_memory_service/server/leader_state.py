# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""File-backed shared state for leader-follower coordination.

The leader GMS (GPU 0) writes state transitions to a JSON file protected by
fcntl.flock. Follower GMS instances poll this file under a shared lock to
learn what the leader has granted, avoiding all two-phase locking deadlocks.
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

logger = logging.getLogger(__name__)


@dataclass
class LeaderState:
    """Snapshot of the leader's coordination state.

    Attributes:
        committed: Whether weights have been committed (valid for RO import).
        active_clients: List of currently connected clients with their modes.
            Each entry is {"client_id": str, "mode": "rw"|"ro"}.
    """

    committed: bool = False
    active_clients: list[dict] = field(default_factory=list)

    @property
    def derived_state(self) -> str:
        """Derive a human-readable state label from active clients."""
        has_rw = any(c["mode"] == "rw" for c in self.active_clients)
        has_ro = any(c["mode"] == "ro" for c in self.active_clients)
        if has_rw:
            return "rw"
        if has_ro:
            return "ro"
        if self.committed:
            return "committed"
        return "empty"

    def has_client(self, client_id: str) -> Optional[dict]:
        """Look up an active client entry by client_id, or None if absent."""
        for client in self.active_clients:
            if client["client_id"] == client_id:
                return client
        return None

    def to_dict(self) -> dict:
        return {"committed": self.committed, "active_clients": self.active_clients}

    @classmethod
    def from_dict(cls, data: dict) -> LeaderState:
        return cls(
            committed=data.get("committed", False),
            active_clients=data.get("active_clients", []),
        )


class LeaderStateFile:
    """File-backed shared state with flock-based atomicity.

    All mutations go through _transact(), which acquires an exclusive flock,
    reads the current state, applies a mutation, writes back, and releases.
    Followers read under a shared flock via read().
    """

    def __init__(self, path: str, lock_timeout_s: float = 5.0) -> None:
        self._path = path
        self._lock_timeout_s = lock_timeout_s

    @property
    def path(self) -> str:
        return self._path

    def _flock_with_timeout(self, fd: int, lock_type: int) -> None:
        """Acquire flock with non-blocking retry loop until timeout."""
        deadline = time.monotonic() + self._lock_timeout_s
        while True:
            try:
                fcntl.flock(fd, lock_type | fcntl.LOCK_NB)
                return
            except (OSError, BlockingIOError):
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"flock timed out after {self._lock_timeout_s}s on {self._path}"
                    )
                time.sleep(0.01)

    def _transact(self, mutate_fn: Callable[[LeaderState], None]) -> LeaderState:
        """Core read-modify-write primitive under exclusive flock.

        Opens the file (creating if needed), acquires LOCK_EX, reads current
        state, applies mutate_fn, writes back, fsyncs, and closes (releasing
        the lock).
        """
        fd = os.open(self._path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            self._flock_with_timeout(fd, fcntl.LOCK_EX)

            # Read current state
            raw = os.read(fd, 1 << 20)  # 1 MiB should be plenty
            if raw:
                state = LeaderState.from_dict(json.loads(raw.decode("utf-8")))
            else:
                state = LeaderState()

            # Apply mutation
            mutate_fn(state)

            # Write back
            data = json.dumps(state.to_dict()).encode("utf-8")
            os.lseek(fd, 0, os.SEEK_SET)
            os.ftruncate(fd, 0)
            os.write(fd, data)
            os.fsync(fd)

            return state
        finally:
            os.close(fd)

    def read(self) -> LeaderState:
        """Read state under shared flock (non-blocking for followers)."""
        try:
            fd = os.open(self._path, os.O_RDONLY)
        except FileNotFoundError:
            return LeaderState()

        try:
            self._flock_with_timeout(fd, fcntl.LOCK_SH)
            raw = os.read(fd, 1 << 20)
            if raw:
                return LeaderState.from_dict(json.loads(raw.decode("utf-8")))
            return LeaderState()
        finally:
            os.close(fd)

    def on_connect(self, client_id: str, mode: str) -> LeaderState:
        """Record a new client connection. Sets committed=False if RW."""

        def _mutate(state: LeaderState) -> None:
            state.active_clients.append({"client_id": client_id, "mode": mode})
            if mode == "rw":
                state.committed = False

        return self._transact(_mutate)

    def on_commit(self, client_id: str) -> LeaderState:
        """Record that a client committed weights. Removes RW entry, sets committed=True."""

        def _mutate(state: LeaderState) -> None:
            state.active_clients = [
                c
                for c in state.active_clients
                if not (c["client_id"] == client_id and c["mode"] == "rw")
            ]
            state.committed = True

        return self._transact(_mutate)

    def on_disconnect(self, client_id: str, mode: str, aborted: bool) -> LeaderState:
        """Record a client disconnection. Removes matching entry."""
        if aborted:
            logger.warning("Client aborted: client=%s mode=%s", client_id, mode)

        def _mutate(state: LeaderState) -> None:
            # Remove the first matching entry
            for i, c in enumerate(state.active_clients):
                if c["client_id"] == client_id and c["mode"] == mode:
                    state.active_clients.pop(i)
                    break

        return self._transact(_mutate)

    def reset_on_startup(self) -> LeaderState:
        """Clear stale active_clients on leader startup, preserving committed."""

        def _mutate(state: LeaderState) -> None:
            state.active_clients = []

        return self._transact(_mutate)
