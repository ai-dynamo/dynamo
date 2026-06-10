# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Phase alignment tests — `PluginRegistryServer._aligned_anchor`.

Plugins registered milliseconds apart but with the same
``execution_interval_seconds`` must fire on the same pipeline tick.
This is critical for ``requires_produced_fields`` to work — if plugin
B depends on plugin A's output, B's throttle must be in phase with A's
throttle, even if B registered slightly later than A during planner
startup.

The mechanism: at registration time, ``registered_at`` is snapped to
the nearest scale_interval boundary (``floor(now / scale_interval) *
scale_interval``) instead of using the raw monotonic clock value.

Disabled when ``scale_interval_seconds`` is 0 (default for the PSM
path; the orchestrator path constructs ``PluginRegistryServer`` with
the real value from ``SchedulingConfig.scale_interval_seconds``).
"""

from __future__ import annotations

import pytest

from dynamo.planner.plugins.clock import VirtualClock
from dynamo.planner.plugins.registry.auth import AllowUnauthenticatedAuth
from dynamo.planner.plugins.registry.circuit_breaker import CircuitBreaker
from dynamo.planner.plugins.registry.server import PluginRegistryServer
from dynamo.planner.plugins.transport.base import PluginTransport
from dynamo.planner.plugins.types import HoldPolicy, RegisterRequest

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]


class _StubTransport(PluginTransport):
    def __init__(self, plugin_id, endpoint, *, in_process_instance=None):
        self.plugin_id = plugin_id
        self.endpoint = endpoint
        self.timeout_seconds = 1.0

    async def call(self, method, request):
        return None

    async def close(self):
        pass


def _make_server(clock: VirtualClock, scale_interval_seconds: float):
    cb = CircuitBreaker(clock, failure_threshold=3, cooldown_seconds=30.0)

    def factory(plugin_id, endpoint, *, in_process_instance=None):
        return _StubTransport(plugin_id, endpoint)

    return PluginRegistryServer(
        clock=clock,
        auth=AllowUnauthenticatedAuth(),
        circuit_breaker=cb,
        transport_factory=factory,
        scale_interval_seconds=scale_interval_seconds,
    )


async def _register(server, plugin_id: str) -> None:
    resp = await server.register(
        RegisterRequest(
            plugin_id=plugin_id,
            plugin_type="propose",
            priority=10,
            endpoint="grpc://127.0.0.1:9000",
            protocol_version="1.0",
            execution_interval_seconds=180.0,
            hold_policy=HoldPolicy.ACCEPT_WHEN_IDLE,
        )
    )
    assert resp.accepted, resp.reject_reason


# ---------------------------------------------------------------------------
# Aligned anchor mechanics
# ---------------------------------------------------------------------------


def test_aligned_anchor_snaps_to_floor_boundary():
    """Direct unit test of ``_aligned_anchor`` arithmetic."""
    clock = VirtualClock()
    server = _make_server(clock, scale_interval_seconds=5.0)

    assert server._aligned_anchor(0.0) == 0.0  # boundary stays
    assert server._aligned_anchor(2.5) == 0.0  # below boundary -> floor 0
    assert server._aligned_anchor(4.999) == 0.0  # still in [0, 5)
    assert server._aligned_anchor(5.0) == 5.0  # boundary
    assert server._aligned_anchor(7.4) == 5.0  # snap down to 5
    assert server._aligned_anchor(180.6) == 180.0
    assert server._aligned_anchor(360.0) == 360.0


def test_disabled_when_scale_interval_zero():
    """``scale_interval_seconds=0.0`` (default) bypasses alignment —
    raw clock value comes through.  Preserves the legacy behaviour for
    the PSM path and for tests that don't explicitly enable alignment.
    """
    clock = VirtualClock()
    server = _make_server(clock, scale_interval_seconds=0.0)

    assert server._aligned_anchor(2.5) == 2.5
    assert server._aligned_anchor(180.6) == 180.6


# ---------------------------------------------------------------------------
# End-to-end: two plugins registered milliseconds apart fire in same tick
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_two_plugins_same_interval_phase_aligned_after_register_skew():
    """The bug scale_interval alignment fixes: plugin A registers at
    T=0, plugin B at T=0.003 (3ms later due to async startup order).
    Both declare ``execution_interval_seconds=180``.

    Without alignment, A's ``registered_at=0`` makes its first fire
    due at T=180, while B's ``registered_at=0.003`` shifts B's first
    fire to T=180.003 — and the pipeline tick at exactly T=180 sees
    A due but B not (179.997 < 180).  A and B drift apart forever.

    With alignment (``scale_interval=5``), both snap to
    ``registered_at=0``, and both fire at the same T=180 tick.
    """
    clock = VirtualClock()
    server = _make_server(clock, scale_interval_seconds=5.0)

    # Plugin A registers at clock=0.
    await _register(server, "plugin_a")

    # Plugin B registers 3ms later — typical async bootstrap order.
    clock.advance(0.003)
    await _register(server, "plugin_b")

    a = server.get_plugin("plugin_a")
    b = server.get_plugin("plugin_b")
    assert a is not None and b is not None
    # Both snap to the same scale_interval boundary (0.0).
    assert a.registered_at == 0.0
    assert b.registered_at == 0.0


@pytest.mark.asyncio
async def test_alignment_preserved_when_registration_crosses_boundary():
    """If two plugins register at T=4.9 and T=5.1, they land on
    different boundaries (0 and 5) — which is the correct semantic.
    The 0.2s of real elapsed time crosses a tick boundary, so they
    *should* be 5s out of phase.  Subsequent ticks at multiples of 5s
    bring them into the same fire moments once interval has elapsed.
    """
    clock = VirtualClock()
    server = _make_server(clock, scale_interval_seconds=5.0)

    clock.advance(4.9)
    await _register(server, "plugin_a")  # registered_at -> 0
    clock.advance(0.2)
    await _register(server, "plugin_b")  # registered_at -> 5

    a = server.get_plugin("plugin_a")
    b = server.get_plugin("plugin_b")
    assert a is not None and b is not None
    assert a.registered_at == 0.0
    assert b.registered_at == 5.0
