# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Process-wide checkpoint lifecycle and control client for GMS V1."""

from __future__ import annotations

import logging
import os
import socket
import threading
from collections.abc import Mapping
from typing import Protocol
from uuid import uuid4

from gpu_memory_service.v1.protocol import (
    AbortCheckpointRequest,
    CheckpointControlRequest,
    CheckpointStateResponse,
    CompleteRestoreRequest,
    ErrorResponse,
    GetCheckpointStateRequest,
    PrepareCheckpointRequest,
    receive_message,
    send_message,
)
from gpu_memory_service.v1.server.rpc import SessionSnapshot

logger = logging.getLogger(__name__)

_SERVING = "serving"
_CHECKPOINT_READY = "checkpoint_ready"
_WEIGHTS_DOMAIN = "weights"
_KV_CACHE_DOMAIN = "kv_cache"


class _CheckpointDomainManager(Protocol):
    @property
    def checkpoint_lifecycle(self) -> GMSCheckpointLifecycle | None:
        ...

    def session_snapshot(self) -> SessionSnapshot:
        ...

    def allocation_snapshot(self) -> tuple[tuple[str, int], ...]:
        ...


class GMSCheckpointLifecycle:
    """Fence both V1 domains while an external controller snapshots the owner."""

    def __init__(self) -> None:
        self.condition = threading.Condition(threading.RLock())
        self._managers: Mapping[str, _CheckpointDomainManager] | None = None
        self._state = _SERVING
        self._generation = 0
        self._token: str | None = None
        self._last_resolution: tuple[str, str] | None = None

    def bind_domains(self, managers: Mapping[str, _CheckpointDomainManager]) -> None:
        """Bind exactly the weights and KV-cache managers sharing this lifecycle."""
        if set(managers) != {_WEIGHTS_DOMAIN, _KV_CACHE_DOMAIN}:
            raise ValueError("checkpoint lifecycle requires weights and kv_cache")
        if any(
            manager.checkpoint_lifecycle is not self for manager in managers.values()
        ):
            raise ValueError("checkpoint domains must share their checkpoint lifecycle")
        with self.condition:
            if self._managers is not None:
                raise RuntimeError("checkpoint lifecycle domains are already bound")
            self._managers = dict(managers)

    def admission_allowed(self) -> bool:
        """Return whether sessions may start, including re-entrant locked calls."""
        with self.condition:
            return self._state == _SERVING

    def handle(self, request: CheckpointControlRequest) -> CheckpointStateResponse:
        """Fence on prepare and apply tokened, retry-safe abort/complete requests."""
        with self.condition:
            if isinstance(request, PrepareCheckpointRequest):
                return self._prepare()
            if isinstance(request, AbortCheckpointRequest):
                return self._resolve(request.token, "abort")
            if isinstance(request, CompleteRestoreRequest):
                return self._resolve(request.token, "complete")
            if isinstance(request, GetCheckpointStateRequest):
                return self._response()
            raise RuntimeError(
                f"unsupported checkpoint request {type(request).__name__}"
            )

    def _prepare(self) -> CheckpointStateResponse:
        """Atomically require checkpoint-safe domains, then fence admission."""
        if self._state == _CHECKPOINT_READY:
            return self._response()

        managers = self._require_managers()
        weights = managers[_WEIGHTS_DOMAIN]
        kv_cache = managers[_KV_CACHE_DOMAIN]
        weights_sessions = weights.session_snapshot()
        kv_sessions = kv_cache.session_snapshot()
        self._require_quiesced(_WEIGHTS_DOMAIN, weights_sessions)
        self._require_quiesced(_KV_CACHE_DOMAIN, kv_sessions)
        if not weights_sessions.committed:
            raise RuntimeError("weights must be committed before checkpoint")
        if kv_sessions.committed:
            raise RuntimeError("kv_cache must not be committed before checkpoint")

        weight_allocations = weights.allocation_snapshot()
        kv_allocations = kv_cache.allocation_snapshot()
        if not weight_allocations:
            raise RuntimeError("weights must contain committed allocations")
        if kv_allocations:
            raise RuntimeError("kv_cache must be empty before checkpoint")

        self._generation += 1
        self._token = str(uuid4())
        self._state = _CHECKPOINT_READY
        self._last_resolution = None
        self.condition.notify_all()
        logger.info("GMS checkpoint generation %d is ready", self._generation)
        return self._response()

    def _resolve(self, token: str, resolution: str) -> CheckpointStateResponse:
        if not token:
            raise RuntimeError("checkpoint token must not be empty")
        if self._state == _SERVING:
            if self._last_resolution == (resolution, token):
                return self._response()
            raise RuntimeError("checkpoint token is stale or already resolved")
        if token != self._token:
            raise RuntimeError(
                "checkpoint token does not match the prepared checkpoint"
            )

        self._state = _SERVING
        self._token = None
        self._last_resolution = (resolution, token)
        self.condition.notify_all()
        logger.info(
            "GMS checkpoint generation %d resolved by %s",
            self._generation,
            resolution,
        )
        return self._response()

    @staticmethod
    def _require_quiesced(name: str, sessions: SessionSnapshot) -> None:
        if (
            sessions.rw_sessions
            or sessions.ro_sessions
            or sessions.waiting_writers
            or sessions.writer_reserved
        ):
            raise RuntimeError(f"{name} has active or waiting sessions")

    def _require_managers(self) -> Mapping[str, _CheckpointDomainManager]:
        if self._managers is None:
            raise RuntimeError("checkpoint lifecycle domains are not bound")
        return self._managers

    def _response(self) -> CheckpointStateResponse:
        return CheckpointStateResponse(self._state, self._token)


class GMSCheckpointClient:
    """Issue bounded one-shot checkpoint-control requests through a domain socket."""

    def __init__(self, path: str, *, timeout: float = 10.0):
        if timeout <= 0:
            raise ValueError("checkpoint control timeout must be positive")
        self._path = path
        self._timeout = timeout

    def prepare(self) -> CheckpointStateResponse:
        return self._call(PrepareCheckpointRequest())

    def abort(self, token: str) -> CheckpointStateResponse:
        return self._call(AbortCheckpointRequest(token))

    def complete(self, token: str) -> CheckpointStateResponse:
        return self._call(CompleteRestoreRequest(token))

    def state(self) -> CheckpointStateResponse:
        return self._call(GetCheckpointStateRequest())

    def _call(self, request: CheckpointControlRequest) -> CheckpointStateResponse:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as control_socket:
            control_socket.settimeout(self._timeout)
            control_socket.connect(self._path)
            send_message(control_socket, request)
            response, received_fd = receive_message(control_socket)
        if received_fd >= 0:
            os.close(received_fd)
            raise RuntimeError("checkpoint control returned an unexpected FD")
        if isinstance(response, ErrorResponse):
            raise RuntimeError(response.message)  # noqa: TRY004
        if not isinstance(response, CheckpointStateResponse):
            raise TypeError(f"checkpoint control returned {type(response).__name__}")
        return response
