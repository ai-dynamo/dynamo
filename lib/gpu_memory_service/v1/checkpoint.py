# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Process-wide checkpoint lifecycle and control client for GMS V1."""

from __future__ import annotations

import hashlib
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
    CheckpointDomainState,
    CheckpointStateResponse,
    CompleteRestoreRequest,
    ErrorResponse,
    GetCheckpointStateRequest,
    PrepareCheckpointRequest,
    receive_message,
    send_message,
)
from gpu_memory_service.v1.server import SessionSnapshot

logger = logging.getLogger(__name__)

_SERVING = "serving"
_CHECKPOINT_READY = "checkpoint_ready"
_WEIGHTS_DOMAIN = "weights"
_KV_CACHE_DOMAIN = "kv_cache"


class _CheckpointDomainManager(Protocol):
    @property
    def identity(self) -> tuple[str, str]:
        ...

    def session_snapshot(self) -> SessionSnapshot:
        ...

    def allocation_snapshot(self) -> tuple[tuple[str, int], ...]:
        ...


def _allocation_digest(allocations: tuple[tuple[str, int], ...]) -> str:
    digest = hashlib.sha256()
    for allocation_id, aligned_size in allocations:
        encoded_id = allocation_id.encode("utf-8")
        digest.update(len(encoded_id).to_bytes(8, "big"))
        digest.update(encoded_id)
        digest.update(aligned_size.to_bytes(8, "big"))
    return digest.hexdigest()


class GMSCheckpointLifecycle:
    """Fence both V1 domains while an external controller snapshots the owner."""

    def __init__(self) -> None:
        self.condition = threading.Condition(threading.RLock())
        self._managers: Mapping[str, _CheckpointDomainManager] | None = None
        self._state = _SERVING
        self._generation = 0
        self._token: str | None = None
        self._domains: tuple[CheckpointDomainState, ...] = ()
        self._last_resolution: tuple[str, str] | None = None

    def bind_domains(self, managers: Mapping[str, _CheckpointDomainManager]) -> None:
        if set(managers) != {_WEIGHTS_DOMAIN, _KV_CACHE_DOMAIN}:
            raise ValueError("checkpoint lifecycle requires weights and kv_cache")
        with self.condition:
            if self._managers is not None:
                raise RuntimeError("checkpoint lifecycle domains are already bound")
            self._managers = dict(managers)

    def admission_allowed(self) -> bool:
        with self.condition:
            return self._state == _SERVING

    def handle(self, request: CheckpointControlRequest) -> CheckpointStateResponse:
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
        self._domains = (
            self._domain_state(_WEIGHTS_DOMAIN, weights, weight_allocations),
            self._domain_state(_KV_CACHE_DOMAIN, kv_cache, kv_allocations),
        )
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
                "checkpoint token does not match the prepared generation"
            )
        if resolution == "complete":
            self._validate_domain_sessions()
            if self._current_domains() != self._domains:
                raise RuntimeError(
                    "restored GMS domain state does not match the prepared state"
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

    def _current_domains(self) -> tuple[CheckpointDomainState, ...]:
        managers = self._require_managers()
        return tuple(
            self._domain_state(name, manager, manager.allocation_snapshot())
            for name, manager in (
                (_WEIGHTS_DOMAIN, managers[_WEIGHTS_DOMAIN]),
                (_KV_CACHE_DOMAIN, managers[_KV_CACHE_DOMAIN]),
            )
        )

    def _validate_domain_sessions(self) -> None:
        managers = self._require_managers()
        weights = managers[_WEIGHTS_DOMAIN].session_snapshot()
        kv_cache = managers[_KV_CACHE_DOMAIN].session_snapshot()
        self._require_quiesced(_WEIGHTS_DOMAIN, weights)
        self._require_quiesced(_KV_CACHE_DOMAIN, kv_cache)
        if not weights.committed:
            raise RuntimeError("restored weights are not committed")
        if kv_cache.committed:
            raise RuntimeError("restored kv_cache is unexpectedly committed")

    @staticmethod
    def _require_quiesced(name: str, sessions: SessionSnapshot) -> None:
        if (
            sessions.rw_sessions
            or sessions.ro_sessions
            or sessions.waiting_writers
            or sessions.writer_reserved
        ):
            raise RuntimeError(f"{name} has active or waiting sessions")

    @staticmethod
    def _domain_state(
        name: str,
        manager: _CheckpointDomainManager,
        allocations: tuple[tuple[str, int], ...],
    ) -> CheckpointDomainState:
        server_nonce, gpu_uuid = manager.identity
        return CheckpointDomainState(
            name,
            server_nonce,
            gpu_uuid,
            len(allocations),
            _allocation_digest(allocations),
        )

    def _require_managers(self) -> Mapping[str, _CheckpointDomainManager]:
        if self._managers is None:
            raise RuntimeError("checkpoint lifecycle domains are not bound")
        return self._managers

    def _response(self) -> CheckpointStateResponse:
        return CheckpointStateResponse(
            self._state,
            self._generation,
            self._token,
            self._domains,
        )


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
            raise RuntimeError(response.message)
        if not isinstance(response, CheckpointStateResponse):
            raise TypeError(f"checkpoint control returned {type(response).__name__}")
        return response
