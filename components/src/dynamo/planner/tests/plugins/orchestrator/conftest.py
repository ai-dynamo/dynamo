# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures and stub plugins for orchestrator tests."""

from __future__ import annotations

import asyncio
from typing import Any, Callable, Optional

import pytest

from dynamo.planner.plugins.clock import VirtualClock
from dynamo.planner.plugins.orchestrator.orchestrator import LocalPlannerOrchestrator
from dynamo.planner.plugins.registry.auth import AllowUnauthenticatedAuth
from dynamo.planner.plugins.registry.circuit_breaker import CircuitBreaker
from dynamo.planner.plugins.registry.server import PluginRegistryServer
from dynamo.planner.plugins.scheduler import PluginScheduler
from dynamo.planner.plugins.transport.config import TransportConfig, make_transport_for_endpoint


@pytest.fixture
def clock():
    return VirtualClock()


@pytest.fixture
def ctx_factory():
    """Build a fresh registry / scheduler / orchestrator triplet.

    Returned callable yields a dict with ``orchestrator`` / ``registry``
    / ``scheduler`` / ``circuit_breaker`` / ``clock`` so individual
    tests can interact with whichever layer is relevant.
    """

    def _make(
        *,
        failure_threshold: int = 3,
        cooldown_seconds: float = 30.0,
        tick_max_duration_seconds: float = 30.0,
    ):
        clk = VirtualClock()
        cb = CircuitBreaker(
            clk, failure_threshold=failure_threshold, cooldown_seconds=cooldown_seconds
        )
        transport_config = TransportConfig(request_timeout_seconds=1.0)

        def factory(plugin_id, endpoint, *, in_process_instance=None):
            return make_transport_for_endpoint(
                plugin_id,
                endpoint,
                transport_config,
                in_process_instance=in_process_instance,
            )

        server = PluginRegistryServer(
            clock=clk,
            auth=AllowUnauthenticatedAuth(),
            circuit_breaker=cb,
            transport_factory=factory,
        )
        scheduler = PluginScheduler(server, cb, clk)
        orchestrator = LocalPlannerOrchestrator(
            registry=server,
            scheduler=scheduler,
            circuit_breaker=cb,
            clock=clk,
            tick_max_duration_seconds=tick_max_duration_seconds,
        )
        return {
            "orchestrator": orchestrator,
            "registry": server,
            "scheduler": scheduler,
            "circuit_breaker": cb,
            "clock": clk,
        }

    return _make


class StubPlugin:
    """A plugin object with per-method response handlers.

    Pass one handler per stage as ``async def fn(request) -> Response``.
    Missing methods cause ``PluginUnknownMethodError`` from the
    InProcessTransport, which the pipeline treats as a plugin failure.
    """

    def __init__(
        self,
        *,
        predict: Optional[Callable[[Any], Any]] = None,
        propose: Optional[Callable[[Any], Any]] = None,
        reconcile: Optional[Callable[[Any], Any]] = None,
        constrain: Optional[Callable[[Any], Any]] = None,
    ) -> None:
        self._handlers = {
            "Predict": predict,
            "Propose": propose,
            "Reconcile": reconcile,
            "Constrain": constrain,
        }
        self.call_counts: dict[str, int] = {
            method: 0 for method in self._handlers
        }

    def __getattr__(self, name: str):
        handler = self._handlers.get(name)
        if handler is None:
            raise AttributeError(name)

        async def call(request):
            self.call_counts[name] += 1
            result = handler(request)
            if asyncio.iscoroutine(result):
                return await result
            return result

        return call
