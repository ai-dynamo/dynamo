# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Admission control for decode requests waiting on remote prefill KV."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Protocol


class DecodeRemotePrefillAdmissionMetrics(Protocol):
    """Metric callbacks for remote-prefill decode admission."""

    def set_state(
        self,
        dp_rank: int,
        *,
        limit: int,
        active: int,
        waiting: int,
    ) -> None:
        ...

    def observe_wait(
        self,
        dp_rank: int,
        *,
        wait_seconds: float,
        limit_hit: bool,
    ) -> None:
        ...

    def record_cancelled(self, dp_rank: int) -> None:
        ...

    def record_release(self, dp_rank: int, reason: str) -> None:
        ...


@dataclass(frozen=True)
class DecodeRemotePrefillAdmissionSnapshot:
    """Current and cumulative state for one data-parallel rank."""

    limit: int
    active: int
    waiting: int
    peak_active: int
    admissions: int
    limit_hits: int
    cancelled_waiters: int
    releases: int


class _Gate:
    def __init__(self, limit: int) -> None:
        self.semaphore = asyncio.Semaphore(limit)
        self.active = 0
        self.waiting = 0
        self.peak_active = 0
        self.admissions = 0
        self.limit_hits = 0
        self.cancelled_waiters = 0
        self.releases = 0


class DecodeRemotePrefillLease:
    """One admission permit held until the first decode output."""

    def __init__(
        self,
        owner: DecodeRemotePrefillAdmission,
        dp_rank: int,
        gate: _Gate,
    ) -> None:
        self._owner = owner
        self._dp_rank = dp_rank
        self._gate = gate
        self._released = False

    def release(self, reason: str) -> bool:
        """Release this permit once and record its terminal reason."""
        if self._released:
            return False

        self._released = True
        if self._gate.active <= 0:
            raise RuntimeError("remote-prefill admission gate underflow")

        self._gate.active -= 1
        self._gate.releases += 1
        self._gate.semaphore.release()
        self._owner._record_release(self._dp_rank, self._gate, reason)
        return True


class DecodeRemotePrefillAdmission:
    """Bound remote-prefill requests before their first decode output."""

    def __init__(
        self,
        limit: int,
        metrics: DecodeRemotePrefillAdmissionMetrics | None = None,
    ) -> None:
        if limit < 0:
            raise ValueError("remote-prefill admission limit must be non-negative")
        self.limit = limit
        self._metrics = metrics
        self._gates: dict[int, _Gate] = {}

    async def acquire(
        self,
        dp_rank: int,
    ) -> DecodeRemotePrefillLease | None:
        """Acquire one permit for a local DP rank.

        A zero limit disables admission and returns ``None``.
        """
        if self.limit == 0:
            return None

        gate = self._gates.get(dp_rank)
        if gate is None:
            gate = _Gate(self.limit)
            self._gates[dp_rank] = gate

        limit_hit = gate.semaphore.locked()
        if limit_hit:
            gate.limit_hits += 1

        gate.waiting += 1
        self._set_state(dp_rank, gate)
        wait_started_ns = time.monotonic_ns()
        try:
            await gate.semaphore.acquire()
        except asyncio.CancelledError:
            gate.waiting -= 1
            gate.cancelled_waiters += 1
            self._set_state(dp_rank, gate)
            if self._metrics is not None:
                self._metrics.record_cancelled(dp_rank)
            raise

        gate.waiting -= 1
        gate.active += 1
        gate.peak_active = max(gate.peak_active, gate.active)
        gate.admissions += 1
        acquired_ns = time.monotonic_ns()
        self._set_state(dp_rank, gate)
        if self._metrics is not None:
            self._metrics.observe_wait(
                dp_rank,
                wait_seconds=(acquired_ns - wait_started_ns) / 1_000_000_000,
                limit_hit=limit_hit,
            )
        return DecodeRemotePrefillLease(self, dp_rank, gate)

    def snapshot(
        self,
        dp_rank: int,
    ) -> DecodeRemotePrefillAdmissionSnapshot:
        """Return admission state for one local DP rank."""
        gate = self._gates.get(dp_rank)
        if gate is None:
            return DecodeRemotePrefillAdmissionSnapshot(
                limit=self.limit,
                active=0,
                waiting=0,
                peak_active=0,
                admissions=0,
                limit_hits=0,
                cancelled_waiters=0,
                releases=0,
            )
        return DecodeRemotePrefillAdmissionSnapshot(
            limit=self.limit,
            active=gate.active,
            waiting=gate.waiting,
            peak_active=gate.peak_active,
            admissions=gate.admissions,
            limit_hits=gate.limit_hits,
            cancelled_waiters=gate.cancelled_waiters,
            releases=gate.releases,
        )

    def _set_state(self, dp_rank: int, gate: _Gate) -> None:
        if self._metrics is not None:
            self._metrics.set_state(
                dp_rank,
                limit=self.limit,
                active=gate.active,
                waiting=gate.waiting,
            )

    def _record_release(
        self,
        dp_rank: int,
        gate: _Gate,
        reason: str,
    ) -> None:
        self._set_state(dp_rank, gate)
        if self._metrics is not None:
            self._metrics.record_release(dp_rank, reason)
