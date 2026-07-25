# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rank-local GMS identity, physical allocations, and session operations."""

from __future__ import annotations

from collections.abc import Callable
from uuid import uuid4

from gpu_memory_service.common.locks import RequestedLockType

from ..errors import GMSError
from .allocations import GMSAllocationManager
from .sessions import GMSSessionManager, ServerSession, SessionSnapshot


class GMS:
    """Own one rank-local physical allocation epoch."""

    def __init__(self, gpu_uuid: str, allocations: GMSAllocationManager):
        if not gpu_uuid:
            raise ValueError("GPU UUID must not be empty")
        self.server_nonce = str(uuid4())
        self.gpu_uuid = gpu_uuid
        self.allocations = allocations
        self.sessions = GMSSessionManager(
            lambda _reason, _replacing_committed: allocations.clear()
        )

    @property
    def identity(self) -> tuple[str, str]:
        return self.server_nonce, self.gpu_uuid

    def snapshot(self) -> SessionSnapshot:
        return self.sessions.snapshot()

    def acquire(
        self,
        requested: RequestedLockType,
        is_cancelled: Callable[[], bool] | None = None,
    ) -> ServerSession | None:
        session = self.sessions.acquire(requested, is_cancelled=is_cancelled)
        if is_cancelled is not None:
            return session
        if session is None:
            raise AssertionError("unbounded GMS session acquisition timed out")
        return session

    def close(self, session: ServerSession) -> None:
        self.sessions.close(session)

    def allocate(
        self,
        session: ServerSession,
        allocation_id: str,
        aligned_size: int,
    ) -> None:
        self._require_rw(session)
        self.allocations.allocate(allocation_id, aligned_size)

    def export(self, session: ServerSession, allocation_id: str) -> int:
        self._require_active(session)
        return self.allocations.export(allocation_id)

    def free(self, session: ServerSession, allocation_id: str) -> None:
        self._require_rw(session)
        self.allocations.free(allocation_id)

    def commit(self, session: ServerSession) -> None:
        self._require_rw(session)
        self.sessions.commit(session)

    def _require_rw(self, session: ServerSession) -> None:
        if session is not self.sessions.rw_session:
            raise GMSError("operation requires an RW session")

    def _require_active(self, session: ServerSession) -> None:
        if (
            session is not self.sessions.rw_session
            and session not in self.sessions.ro_sessions
        ):
            raise GMSError("operation requires an active GMS session")

    def dispatch(
        self,
        session: ServerSession,
        method: str,
        params: list[object],
    ) -> tuple[object, int]:
        if method == "allocate":
            self._expect(params, 2)
            allocation_id = self._string(params[0], "allocation ID")
            if type(params[1]) is not int:
                raise GMSError("allocation size must be an integer")
            self.allocate(session, allocation_id, params[1])
            return None, -1
        if method == "export":
            self._expect(params, 1)
            return None, self.export(
                session,
                self._string(params[0], "allocation ID"),
            )
        if method == "free":
            self._expect(params, 1)
            self.free(
                session,
                self._string(params[0], "allocation ID"),
            )
            return None, -1
        if method == "commit":
            self._expect(params, 0)
            self.commit(session)
            return None, -1
        raise GMSError(f"unknown GMS RPC method {method!r}")

    @staticmethod
    def _expect(params: list[object], count: int) -> None:
        if len(params) != count:
            raise GMSError("invalid GMS RPC parameters")

    @staticmethod
    def _string(value: object, name: str) -> str:
        if not isinstance(value, str) or not value:
            raise GMSError(f"{name} must be a non-empty string")
        return value
