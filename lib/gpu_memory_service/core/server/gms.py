# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deep server owner for one rank-local GMS allocation domain."""

from __future__ import annotations

from collections.abc import Callable
from uuid import uuid4

from gpu_memory_service.common.locks import RequestedLockType
from gpu_memory_service.common.vmm import VMMDevice

from ..protocol import (
    AllocateRequest,
    CommitRequest,
    ExportRequest,
    ExportResponse,
    FreeRequest,
    Request,
    SuccessResponse,
)
from .allocations import GMSAllocationManager
from .sessions import GMSSessionManager, ServerSession


class GMSServerMemoryManager:
    """Own identity, lock admission, and physical allocations for one socket."""

    def __init__(self, gpu_uuid: str, vmm: VMMDevice, device: int):
        if not gpu_uuid:
            raise ValueError("GPU UUID must not be empty")
        self._identity = (str(uuid4()), gpu_uuid)
        self._allocations = GMSAllocationManager(vmm, device)
        self._sessions = GMSSessionManager(self._allocations.clear)

    @property
    def identity(self) -> tuple[str, str]:
        return self._identity

    def acquire(
        self,
        requested: RequestedLockType,
        is_cancelled: Callable[[], bool] | None = None,
    ) -> ServerSession | None:
        return self._sessions.acquire(requested, is_cancelled=is_cancelled)

    def handle_request(
        self, session: ServerSession, request: Request
    ) -> tuple[SuccessResponse | ExportResponse, int]:
        if isinstance(request, AllocateRequest):
            self._require_rw(session)
            self._allocations.allocate(request.allocation_id, request.aligned_size)
            return SuccessResponse(), -1
        if isinstance(request, ExportRequest):
            self._require_active(session)
            return ExportResponse(), self._allocations.export(request.allocation_id)
        if isinstance(request, FreeRequest):
            self._require_rw(session)
            self._allocations.free(request.allocation_id)
            return SuccessResponse(), -1
        if isinstance(request, CommitRequest):
            self._require_rw(session)
            self._sessions.commit(session)
            return SuccessResponse(), -1
        raise RuntimeError(f"unsupported GMS request {type(request).__name__}")

    def close(self, session: ServerSession) -> None:
        self._sessions.close(session)

    def _require_rw(self, session: ServerSession) -> None:
        if not self._sessions.is_writer(session):
            raise RuntimeError("operation requires an RW session")

    def _require_active(self, session: ServerSession) -> None:
        if not self._sessions.is_active(session):
            raise RuntimeError("operation requires an active GMS session")
