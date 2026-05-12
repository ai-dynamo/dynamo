# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-plugin circuit breaker.

State machine::

    CLOSED
      │  ``record_failure`` N times in a row
      ▼
    OPEN ──────── cooldown elapsed ────────► HALF_OPEN
      ▲                                           │
      │ ``record_failure`` (reset cooldown)       │ ``record_success``
      └───────────────────────────────────────────┘
                                                  ▼
                                                CLOSED

Defaults (v1): ``failure_threshold=5``, ``cooldown_seconds=30.0``. Tune
per-deployment via config; README recommends ``10 / 60s`` for production
to avoid amplifying transient network blips.

State is in-memory only — registry restart clears all circuits back to
CLOSED (v11 cache persistence table row 3).

Observers (the PluginScheduler) subscribe via ``on_open`` to receive
``plugin_id`` fan-out when a CLOSED → OPEN transition happens, so the
HOLD_LAST cache can be invalidated (v11 cache invalidation row 3).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable

from dynamo.planner.plugins.clock import Clock
from dynamo.planner.plugins.types import CircuitState


@dataclass
class _CircuitEntry:
    state: CircuitState = CircuitState.CLOSED
    consecutive_failures: int = 0
    opened_at: float = 0.0  # monotonic; meaningful when state == OPEN / HALF_OPEN


class CircuitBreaker:
    """Per-``plugin_id`` circuit breaker driven by a deterministic Clock.

    Instance methods are **sync** and must be called from the event loop
    main task (single-threaded asyncio invariant; see ``PluginScheduler``
    docstring). The internal map ``dict[plugin_id -> _CircuitEntry]`` has
    no locks.
    """

    def __init__(
        self,
        clock: Clock,
        failure_threshold: int = 5,
        cooldown_seconds: float = 30.0,
    ) -> None:
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be >= 1")
        if cooldown_seconds <= 0:
            raise ValueError("cooldown_seconds must be > 0")
        self._clock = clock
        self._failure_threshold = failure_threshold
        self._cooldown = cooldown_seconds
        self._entries: dict[str, _CircuitEntry] = {}
        self._open_callbacks: list[Callable[[str], None]] = []

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def state(self, plugin_id: str) -> CircuitState:
        """Return the current state, auto-transitioning OPEN → HALF_OPEN
        when the cooldown has elapsed. Unknown ``plugin_id`` returns
        ``CLOSED`` (implicit new entries)."""
        entry = self._entries.get(plugin_id)
        if entry is None:
            return CircuitState.CLOSED
        if (
            entry.state == CircuitState.OPEN
            and self._clock.monotonic() - entry.opened_at >= self._cooldown
        ):
            entry.state = CircuitState.HALF_OPEN
        return entry.state

    def can_call(self, plugin_id: str) -> bool:
        """``True`` if the orchestrator may attempt a plugin call now."""
        return self.state(plugin_id) != CircuitState.OPEN

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------

    def record_success(self, plugin_id: str) -> None:
        entry = self._entries.setdefault(plugin_id, _CircuitEntry())
        # Refresh implicit state transition first (OPEN → HALF_OPEN on cooldown).
        _ = self.state(plugin_id)
        entry.consecutive_failures = 0
        entry.state = CircuitState.CLOSED

    def record_failure(self, plugin_id: str) -> None:
        entry = self._entries.setdefault(plugin_id, _CircuitEntry())
        _ = self.state(plugin_id)
        if entry.state == CircuitState.HALF_OPEN:
            # HALF_OPEN probe failed → re-open + reset cooldown.
            entry.state = CircuitState.OPEN
            entry.opened_at = self._clock.monotonic()
            entry.consecutive_failures = self._failure_threshold  # already "at threshold"
            self._fan_out_open(plugin_id)
            return
        entry.consecutive_failures += 1
        if (
            entry.state == CircuitState.CLOSED
            and entry.consecutive_failures >= self._failure_threshold
        ):
            entry.state = CircuitState.OPEN
            entry.opened_at = self._clock.monotonic()
            self._fan_out_open(plugin_id)

    def reset(self, plugin_id: str) -> None:
        """Clear any state for a plugin — used on (un)register to stop
        state leaking across a plugin_id reuse (v11 Q6 clients are
        expected to Unregister + Register for version upgrades)."""
        self._entries.pop(plugin_id, None)

    # ------------------------------------------------------------------
    # Observers
    # ------------------------------------------------------------------

    def on_open(self, callback: Callable[[str], None]) -> None:
        """Register a callback invoked with ``plugin_id`` whenever a
        CLOSED / HALF_OPEN → OPEN transition occurs.

        Scheduler subscribes here during construction to invalidate
        HOLD_LAST cache entries for OPEN plugins (v11 cache invalidation
        row 3). Callbacks run synchronously on the event loop main task;
        they must not await.
        """
        self._open_callbacks.append(callback)

    # ------------------------------------------------------------------
    # Admin / debug
    # ------------------------------------------------------------------

    def tracked_plugin_ids(self) -> Iterable[str]:
        return self._entries.keys()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _fan_out_open(self, plugin_id: str) -> None:
        for cb in list(self._open_callbacks):
            cb(plugin_id)


__all__ = ["CircuitBreaker"]
