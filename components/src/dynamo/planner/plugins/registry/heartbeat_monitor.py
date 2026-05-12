# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Heartbeat-liveness background monitor.

Long-running asyncio coroutine that periodically scans the registry's
plugins and evicts any UDS / gRPC plugin whose last heartbeat is older
than ``timeout_seconds * missed_threshold``.

In-process plugins (``transport_type == "in_process"``, **regardless** of
``is_builtin``) are unconditionally skipped — they live inside the
planner process, so "heartbeat missed" isn't meaningful and evicting
them would kill correctly-registered user plugins that don't emit
heartbeats.

Eviction goes through the normal ``registry.unregister(plugin_id,
reason="heartbeat_missed")`` path so audit logs, circuit-breaker reset,
and scheduler cache invalidation all happen through a single code path.
"""

from __future__ import annotations

import logging
from typing import Optional

from dynamo.planner.plugins.clock import Clock
from dynamo.planner.plugins.registry.server import PluginRegistryServer

log = logging.getLogger(__name__)


class HeartbeatMonitor:
    """Periodic liveness checker; run as a background task."""

    def __init__(
        self,
        registry: PluginRegistryServer,
        clock: Clock,
        timeout_seconds: float = 15.0,
        missed_threshold: int = 2,
        check_interval_seconds: float = 5.0,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be > 0")
        if missed_threshold < 1:
            raise ValueError("missed_threshold must be >= 1")
        if check_interval_seconds <= 0:
            raise ValueError("check_interval_seconds must be > 0")
        self._registry = registry
        self._clock = clock
        self._timeout_seconds = timeout_seconds
        self._missed_threshold = missed_threshold
        self._check_interval = check_interval_seconds
        self._stopped = False

    @property
    def eviction_deadline_seconds(self) -> float:
        """The last-heartbeat staleness at which a plugin is evicted."""
        return self._timeout_seconds * self._missed_threshold

    def stop(self) -> None:
        """Request the ``run`` loop to exit at its next sleep boundary."""
        self._stopped = True

    async def run(self) -> None:
        """Long-running coroutine — orchestrator schedules at startup.

        Exits when ``stop()`` has been called **and** a pass has completed.
        """
        while not self._stopped:
            await self._check_once()
            if self._stopped:
                break
            await self._clock.sleep(self._check_interval)

    async def _check_once(self) -> None:
        """One sweep over all registered plugins. Safe to invoke directly
        from unit tests without spinning up the run loop."""
        now = self._clock.monotonic()
        deadline = self.eviction_deadline_seconds
        # Snapshot the list — `unregister` mutates `_plugins` mid-iteration.
        for plugin in self._registry.all_plugins():
            if plugin.transport_type == "in_process":
                # In-process plugins never miss heartbeats.
                continue
            elapsed = now - plugin.last_heartbeat_at
            if elapsed > deadline:
                log.info(
                    "heartbeat_monitor: evicting plugin_id=%s transport=%s "
                    "elapsed_since_last_heartbeat_seconds=%.1f deadline=%.1f",
                    plugin.plugin_id,
                    plugin.transport_type,
                    elapsed if elapsed != float("inf") else -1.0,
                    deadline,
                )
                await self._registry.unregister(
                    plugin.plugin_id, reason="heartbeat_missed"
                )


__all__ = ["HeartbeatMonitor"]
