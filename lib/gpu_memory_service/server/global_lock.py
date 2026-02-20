# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Leader-follower coordination for multi-GPU GMS weight sharing.

The leader (GPU 0) makes unilateral locking decisions and records them in a
shared state file. Followers poll the state file and only proceed when the
leader has already granted the corresponding engine. This eliminates all
two-phase locking deadlocks present in the old GlobalLock design.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from gpu_memory_service.common.types import GrantedLockType

from .leader_state import LeaderStateFile

logger = logging.getLogger(__name__)


class LeaderFollowerLock:
    """Coordinates multi-GPU lock acquisition via leader-follower pattern.

    Leader API: called after local FSM transitions to record state.
    Follower API: called before local FSM transitions to wait for leader grant.
    """

    def __init__(
        self,
        state_file: LeaderStateFile,
        *,
        is_leader: bool,
        poll_interval_ms: int = 100,
    ) -> None:
        self._state_file = state_file
        self._is_leader = is_leader
        self._poll_interval_s = poll_interval_ms / 1000.0

    @property
    def is_leader(self) -> bool:
        return self._is_leader

    # ── Leader API (called after local FSM transitions) ──

    async def on_connect(self, client_id: str, mode: str) -> None:
        """Record that the leader granted a connection."""
        await asyncio.to_thread(self._state_file.on_connect, client_id, mode)
        logger.info("Leader recorded connect: client=%s mode=%s", client_id, mode)

    async def on_commit(self, client_id: str) -> None:
        """Record that the leader committed weights."""
        await asyncio.to_thread(self._state_file.on_commit, client_id)
        logger.info("Leader recorded commit: client=%s", client_id)

    async def on_disconnect(self, client_id: str, mode: str, aborted: bool) -> None:
        """Record that a connection was cleaned up."""
        await asyncio.to_thread(
            self._state_file.on_disconnect, client_id, mode, aborted
        )
        logger.info(
            "Leader recorded disconnect: client=%s mode=%s aborted=%s",
            client_id,
            mode,
            aborted,
        )

    # ── Follower API (called before local FSM transitions) ──

    async def wait_for_leader_grant(
        self,
        client_id: str,
        requested_mode: str,
        timeout_ms: Optional[int],
    ) -> Optional[GrantedLockType]:
        """Poll the state file until client_id appears in active_clients.

        For RW_OR_RO, returns whatever mode the leader actually granted.
        Returns None on timeout.
        """
        deadline = (
            asyncio.get_running_loop().time() + (timeout_ms / 1000.0)
            if timeout_ms is not None
            else None
        )

        while True:
            state = await asyncio.to_thread(self._state_file.read)
            client = state.has_client(client_id)

            if client is not None:
                granted_mode = client["mode"]
                # Map the string mode to GrantedLockType
                if granted_mode == "rw":
                    result = GrantedLockType.RW
                else:
                    result = GrantedLockType.RO
                logger.info(
                    "Follower got leader grant: client=%s requested=%s granted=%s",
                    client_id,
                    requested_mode,
                    result.value,
                )
                return result

            # For RO requests, also check if leader has committed (weights available)
            # even if no active client entry — the leader's RW may have already committed
            # and disconnected, which means RO is available
            if requested_mode in ("ro", "rw_or_ro") and state.committed:
                # No active client entry, but weights are committed.
                # The follower can proceed with RO since committed weights exist.
                logger.info(
                    "Follower proceeding with RO: client=%s (leader committed, no active client)",
                    client_id,
                )
                return GrantedLockType.RO

            if deadline is not None and asyncio.get_running_loop().time() >= deadline:
                logger.warning(
                    "Follower timed out waiting for leader grant: client=%s mode=%s",
                    client_id,
                    requested_mode,
                )
                return None

            await asyncio.sleep(self._poll_interval_s)
