# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the startup entry points on
``LocalPlannerOrchestrator``:

- ``install_regressions(prefill=, decode=, agg=)`` — orchestrator-owned
  shared store (not a plugin concern)
- ``bootstrap_plugins(historical_traffic=)`` — plugin lifecycle: Python
  warm hook + Bootstrap RPC fan-out

The two are intentionally separate (Option A split from an earlier
combined form) so callers can audit which concern they're invoking.
"""

from __future__ import annotations

from typing import Any

import pytest

from dynamo.planner.config.planner_config import PlannerConfig
from dynamo.planner.core.types import (
    EngineCapabilities,
    TrafficObservation,
    WorkerCapabilities,
)
from dynamo.planner.plugins.builtins.base import BuiltinPluginBase
from dynamo.planner.plugins.builtins.load_predictor import BuiltinLoadPredictor
from dynamo.planner.plugins.types import (
    BootstrapRequest,
    BootstrapResponse,
    ResetRequest,
    ResetResponse,
)

from .conftest import StubPlugin

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]


def _caps():
    return WorkerCapabilities(
        decode=EngineCapabilities(num_gpu=1, max_num_batched_tokens=2048, max_kv_tokens=16384)
    )


def _sla_config():
    return PlannerConfig(
        mode="agg",
        enable_load_scaling=False,
        enable_throughput_scaling=True,
        optimization_target="sla",
        load_predictor="constant",
    )


# ---------------------------------------------------------------------------
# install_regressions — orchestrator state init (not plugin concern)
# ---------------------------------------------------------------------------


def test_install_regressions_populates_shared_store(ctx_factory):
    ctx = ctx_factory()
    orch = ctx["orchestrator"]

    prefill_reg = object()
    decode_reg = object()
    agg_reg = object()

    orch.install_regressions(
        prefill=prefill_reg, decode=decode_reg, agg=agg_reg
    )
    assert orch.get_regression("prefill") is prefill_reg
    assert orch.get_regression("decode") is decode_reg
    assert orch.get_regression("agg") is agg_reg


def test_install_regressions_none_slot_leaves_prior_value(ctx_factory):
    ctx = ctx_factory()
    orch = ctx["orchestrator"]
    prior = object()
    orch.update_regression("prefill", prior)

    orch.install_regressions()  # all None
    assert orch.get_regression("prefill") is prior


def test_install_regressions_partial_update(ctx_factory):
    ctx = ctx_factory()
    orch = ctx["orchestrator"]
    orch.install_regressions(prefill="p1", decode="d1", agg="a1")
    # Now update only decode — prefill/agg untouched.
    orch.install_regressions(decode="d2")
    assert orch.get_regression("prefill") == "p1"
    assert orch.get_regression("decode") == "d2"
    assert orch.get_regression("agg") == "a1"


def test_install_regressions_is_sync_no_await(ctx_factory):
    """``install_regressions`` is intentionally sync — it's a dict
    write. Keeping it sync keeps test harness + production callers
    from needing to await a trivial operation."""
    ctx = ctx_factory()
    orch = ctx["orchestrator"]
    # Just confirm it returns None (no coroutine).
    result = orch.install_regressions(prefill="p")
    assert result is None


# ---------------------------------------------------------------------------
# bootstrap_plugins — plugin lifecycle fan-out
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bootstrap_warms_load_predictor_from_history(ctx_factory):
    ctx = ctx_factory()
    orch = ctx["orchestrator"]
    predictor = BuiltinLoadPredictor(orch, _sla_config())
    orch.register_internal(
        plugin_id="builtin_load_predictor",
        plugin_type="predict",
        priority=1,
        instance=predictor,
    )

    history = [
        TrafficObservation(duration_s=1.0, num_req=5.0, isl=200.0, osl=50.0),
        TrafficObservation(duration_s=1.0, num_req=7.0, isl=250.0, osl=60.0),
    ]
    await orch.bootstrap_plugins(historical_traffic=history)

    # ConstantPredictor.predict_next returns the last data point after warm-up.
    assert predictor._num_req_predictor.predict_next() == 7.0
    assert predictor._isl_predictor.predict_next() == 250.0
    assert predictor._osl_predictor.predict_next() == 60.0


@pytest.mark.asyncio
async def test_bootstrap_skips_warm_for_plugins_without_helper(ctx_factory):
    """Plugins that don't expose ``warm_from_observations`` must not
    cause bootstrap to fail — we just skip them."""
    ctx = ctx_factory()
    orch = ctx["orchestrator"]
    # Register a stub plugin with no warm hook.
    orch.register_internal(
        plugin_id="stub",
        plugin_type="propose",
        priority=10,
        instance=StubPlugin(),
    )
    await orch.bootstrap_plugins(
        historical_traffic=[TrafficObservation(duration_s=1.0, num_req=1.0, isl=1.0, osl=1.0)]
    )
    # No exception raised, stub unchanged.


class _TrackingBuiltin(BuiltinPluginBase):
    def __init__(self, orch, config):
        super().__init__(orch, config)
        self.bootstrap_called = False
        self.reset_called = False

    async def Bootstrap(self, request: BootstrapRequest) -> BootstrapResponse:
        self.bootstrap_called = True
        return BootstrapResponse(ok=True)

    async def Reset(self, request: ResetRequest) -> ResetResponse:
        self.reset_called = True
        return ResetResponse(ok=True)


@pytest.mark.asyncio
async def test_bootstrap_dispatches_bootstrap_rpc_to_builtins(ctx_factory):
    ctx = ctx_factory()
    orch = ctx["orchestrator"]
    tracker = _TrackingBuiltin(orch, _sla_config())
    orch.register_internal(
        plugin_id="tracker",
        plugin_type="propose",
        priority=10,
        instance=tracker,
    )
    await orch.bootstrap_plugins()
    assert tracker.bootstrap_called is True
    assert tracker.reset_called is False


@pytest.mark.asyncio
async def test_bootstrap_continues_when_plugin_has_no_bootstrap(ctx_factory):
    """StubPlugin doesn't implement Bootstrap — InProcessTransport raises
    PluginUnknownMethodError; bootstrap_plugins must swallow that and
    keep going."""
    ctx = ctx_factory()
    orch = ctx["orchestrator"]
    tracker = _TrackingBuiltin(orch, _sla_config())
    orch.register_internal(
        plugin_id="tracker",
        plugin_type="propose",
        priority=1,
        instance=tracker,
    )
    orch.register_internal(
        plugin_id="no_bootstrap",
        plugin_type="propose",
        priority=10,
        instance=StubPlugin(),
    )
    await orch.bootstrap_plugins()
    # tracker still bootstrapped despite the other plugin missing the method.
    assert tracker.bootstrap_called is True


# ---------------------------------------------------------------------------
# Ordering: caller MUST install_regressions before bootstrap_plugins if
# plugin Bootstrap needs regression access.
# ---------------------------------------------------------------------------


class _RegressionReadingBuiltin(BuiltinPluginBase):
    """Reads regression during Bootstrap to prove ``install_regressions``
    happened first (caller-enforced ordering)."""

    def __init__(self, orch, config):
        super().__init__(orch, config)
        self.seen_regression: Any = None

    async def Bootstrap(self, request):
        self.seen_regression = self.get_regression("prefill")
        return BootstrapResponse(ok=True)

    async def Reset(self, request):
        return ResetResponse(ok=True)


@pytest.mark.asyncio
async def test_install_before_bootstrap_contract(ctx_factory):
    """When the caller installs regressions BEFORE calling
    bootstrap_plugins, plugin Bootstrap implementations see them via
    ``get_regression``. Conversely, if the caller reverses the order
    the plugin sees ``None`` — documenting the contract."""
    ctx = ctx_factory()
    orch = ctx["orchestrator"]
    reader = _RegressionReadingBuiltin(orch, _sla_config())
    orch.register_internal(
        plugin_id="reader", plugin_type="propose", priority=1, instance=reader
    )
    target = object()
    orch.install_regressions(prefill=target)
    await orch.bootstrap_plugins()
    assert reader.seen_regression is target


@pytest.mark.asyncio
async def test_bootstrap_before_install_leaves_plugin_unpopulated(ctx_factory):
    """Negative check: reverse the order and plugin reads ``None``.
    This test documents why callers must call install_regressions first."""
    ctx = ctx_factory()
    orch = ctx["orchestrator"]
    reader = _RegressionReadingBuiltin(orch, _sla_config())
    orch.register_internal(
        plugin_id="reader", plugin_type="propose", priority=1, instance=reader
    )
    await orch.bootstrap_plugins()  # plugin Bootstrap runs now
    orch.install_regressions(prefill=object())  # too late for this plugin
    assert reader.seen_regression is None
