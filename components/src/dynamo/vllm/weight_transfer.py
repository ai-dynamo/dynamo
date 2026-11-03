# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fail-closed weight-install sessions, serialized by the worker's pause lock."""

import asyncio
import math
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from typing import Any, Literal


class WeightTransferValidationError(ValueError):
    """Reject a protocol request before invoking an engine operation."""


def _body(value: Any, keys: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise WeightTransferValidationError(
            "request body has missing or unsupported fields"
        )
    return value


def _identifier(value: Any) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 256:
        raise WeightTransferValidationError(
            "session and version identifiers must be nonempty strings (max 256)"
        )
    return value


@dataclass(frozen=True)
class WeightTransferStatus:
    """Versioned installation state shared by strict control acknowledgements."""

    version: str | None
    version_declared: bool
    session_id: str | None
    install_complete: bool
    pending_weight_version: str | None
    pending_session_id: str | None
    failed: bool
    next_seq: int
    protocol_version: int = 1


@dataclass(frozen=True)
class _State:
    paused: bool = False
    failed: bool = False
    phase: Literal[
        "uninitialized", "idle", "prepared", "started", "finished", "ready"
    ] = "uninitialized"
    pending_session_id: str | None = None
    pending_weight_version: str | None = None
    session_id: str | None = None
    version: str | None = None
    seq: int = 0
    used_sessions: frozenset[str] = frozenset()
    used_versions: frozenset[str] = frozenset()


class WeightTransferSession:
    """Enforce one initialization and unique, sequenced, complete installs."""

    def __init__(self, timeout_s: float = 120.0) -> None:
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("weight-transfer timeout must be finite and positive")
        self._state = _State()
        self._timeout_s = timeout_s

    def fail(self) -> None:
        """Permanently fence an engine whose mutation outcome is uncertain."""
        self._state = replace(self._state, failed=True)

    def _healthy(self) -> None:
        if self._state.failed:
            raise WeightTransferValidationError(
                "weight-transfer session failed; restart the worker"
            )

    def _require_paused(self) -> None:
        self._healthy()
        if not self._state.paused:
            raise WeightTransferValidationError(
                "engine must be explicitly paused with drain/abort semantics"
            )

    async def _mutate(self, operation: Callable[[], Awaitable[Any]]) -> Any:
        """Fence the session if an operation fails, times out, or is cancelled."""
        try:
            async with asyncio.timeout(self._timeout_s):
                return await operation()
        except BaseException:
            self.fail()
            raise

    async def pause(self, rpc: Callable, mode: str) -> None:
        """Drain or abort generation before changing a policy, even after failure."""
        if mode not in ("wait", "abort"):
            raise WeightTransferValidationError(
                "strict RL pause requires wait/abort semantics"
            )
        await self._mutate(lambda: rpc(mode=mode, clear_cache=False))
        self._state = replace(self._state, paused=True)

    async def resume(self, rpc: Callable) -> None:
        """Reopen generation only after a complete install and cache reset."""
        self._healthy()
        if self._state.phase != "ready":
            raise WeightTransferValidationError(
                "completed install and successful cache reset required before resume"
            )
        await self._mutate(rpc)
        self._state = replace(self._state, paused=False)

    async def initialize(self, body: Any, rpc: Callable) -> None:
        """Initialize the transfer backend while generation is paused."""
        self._require_paused()
        payload = _body(body, {"init_info"})
        if not isinstance(payload["init_info"], dict):
            raise WeightTransferValidationError("init_info must be an object")
        if self._state.phase != "uninitialized":
            raise WeightTransferValidationError(
                "weight transfer is already initialized"
            )
        await self._mutate(lambda: rpc(payload["init_info"]))
        self._state = replace(self._state, phase="idle")

    def begin(self, body: Any) -> None:
        """Reserve unique session and policy identities before receiving updates."""
        self._require_paused()
        payload = _body(body, {"session_id", "weight_version"})
        session_id = _identifier(payload["session_id"])
        version = _identifier(payload["weight_version"])
        state = self._state
        if state.phase not in ("idle", "ready"):
            raise WeightTransferValidationError(
                "initialize and complete the previous session before begin"
            )
        if session_id in state.used_sessions or version in state.used_versions:
            raise WeightTransferValidationError(
                "session and policy version identifiers must be unique"
            )
        self._state = replace(
            state,
            phase="prepared",
            pending_session_id=session_id,
            pending_weight_version=version,
            seq=0,
            used_sessions=state.used_sessions | {session_id},
            used_versions=state.used_versions | {version},
        )

    def _sequence(self, body: Any, *, phase: str, update: bool = False) -> dict:
        self._require_paused()
        payload = _body(
            body,
            {"session_id", "seq", "update_info"} if update else {"session_id", "seq"},
        )
        state = self._state
        if state.phase != phase or payload["session_id"] != state.pending_session_id:
            raise WeightTransferValidationError(
                "invalid weight-transfer session or lifecycle ordering"
            )
        seq = payload["seq"]
        if isinstance(seq, bool) or not isinstance(seq, int) or seq != state.seq:
            raise WeightTransferValidationError(
                "weight-transfer sequence must match the next expected integer"
            )
        return payload

    async def start(self, body: Any, rpc: Callable) -> None:
        """Start the prepared transfer at its first sequence number."""
        self._sequence(body, phase="prepared")
        await self._mutate(rpc)
        self._state = replace(self._state, phase="started", seq=1)

    async def update(self, body: Any, rpc: Callable, synchronize: Callable) -> None:
        """Synchronize an update before acknowledging its sequence number."""
        payload = self._sequence(body, phase="started", update=True)
        if not isinstance(payload["update_info"], dict):
            raise WeightTransferValidationError("update_info must be an object")
        await self._mutate(lambda: rpc(payload["update_info"]))
        await self._mutate(synchronize)
        self._state = replace(self._state, seq=self._state.seq + 1)

    async def finish(self, body: Any, rpc: Callable, synchronize: Callable) -> None:
        """Commit the policy identity after the final engine synchronization."""
        self._sequence(body, phase="started")
        state = self._state
        if state.seq == 1:
            raise WeightTransferValidationError(
                "at least one update chunk is required before finish"
            )
        await self._mutate(lambda: rpc(state.pending_weight_version))
        await self._mutate(synchronize)
        self._state = replace(
            self._state,
            phase="finished",
            session_id=state.pending_session_id,
            version=state.pending_weight_version,
            pending_session_id=None,
            pending_weight_version=None,
            seq=state.seq + 1,
        )

    async def flush(self, rpc: Callable) -> None:
        """Require a successful cache reset before reopening admission."""
        self._require_paused()
        if self._state.phase != "finished":
            raise WeightTransferValidationError(
                "cache reset requires a successfully finished install"
            )
        result = await self._mutate(rpc)
        if result is not True:
            self.fail()
            raise ValueError("engine cache reset did not report success")
        self._state = replace(self._state, phase="ready")

    def require_generation(self) -> str:
        """Return the installed policy identity only while admission is open."""
        self._healthy()
        if self._state.phase != "ready" or self._state.version is None:
            raise ValueError(
                "policy must be installed and cache reset before generation"
            )
        if self._state.paused:
            raise ValueError("generation is paused for weight transfer")
        return self._state.version

    def require_version(self, version: str) -> None:
        """Reject output that crossed an install or an uncertain engine mutation."""
        self._healthy()
        if self._state.version != version:
            raise ValueError("strict RL policy changed after request admission")

    def snapshot(self) -> WeightTransferStatus:
        """Return an immutable, typed snapshot of the installation state."""
        state = self._state
        return WeightTransferStatus(
            version=state.version,
            version_declared=state.version is not None,
            session_id=state.session_id,
            install_complete=state.phase in ("finished", "ready") and not state.failed,
            pending_weight_version=state.pending_weight_version,
            pending_session_id=state.pending_session_id,
            failed=state.failed,
            next_seq=state.seq,
        )
