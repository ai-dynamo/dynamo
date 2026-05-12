# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for OrchestratorEngineAdapter.

Scope:
- Protocol conformance: adapter satisfies ``EngineProtocol``.
- Lifecycle: construct → install_regressions → bootstrap_plugins →
  initial_tick → tick → shutdown runs without error.
- ``initial_tick`` matches PSM's scheduling semantics.
- G3 parity: for each scenario, drive the adapter through all ticks and
  compare ``scale_to`` to the golden fixture. This is the production
  version of the ``test_g3_real_parity.py`` coverage — identical
  validation but through the adapter's public API (no side-channel
  prime_tick / observe_fpm exposure).
"""

from __future__ import annotations

import pytest

from dynamo.planner.core.engine_protocol import EngineProtocol
from dynamo.planner.core.state_machine import PlannerStateMachine
from dynamo.planner.core.types import ScalingDecision
from dynamo.planner.plugins.orchestrator.engine_adapter import (
    OrchestratorEngineAdapter,
)
from dynamo.planner.tests.plugins.g3_fixtures.dump_tool import (
    DEFAULT_OUTPUT_DIR,
    _read_fixture,
    _tick_for,
)
from dynamo.planner.tests.plugins.g3_fixtures.scenarios import ALL_SCENARIOS

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_adapter_satisfies_engine_protocol():
    scenario = ALL_SCENARIOS[0]
    adapter = OrchestratorEngineAdapter(
        scenario.make_config(), scenario.caps_factory()
    )
    assert isinstance(adapter, EngineProtocol)


# ---------------------------------------------------------------------------
# initial_tick matches PSM
# ---------------------------------------------------------------------------


def test_initial_tick_matches_psm():
    scenario = ALL_SCENARIOS[0]  # baseline_disagg_throughput_only_sla
    config = scenario.make_config()
    caps = scenario.caps_factory()

    psm = PlannerStateMachine(config, caps)
    psm_tick = psm.initial_tick(scenario.initial_tick_at_s)

    adapter = OrchestratorEngineAdapter(config, caps)
    adapter_tick = adapter.initial_tick(scenario.initial_tick_at_s)

    assert adapter_tick.at_s == psm_tick.at_s
    assert adapter_tick.run_load_scaling == psm_tick.run_load_scaling
    assert adapter_tick.run_throughput_scaling == psm_tick.run_throughput_scaling
    assert adapter_tick.need_worker_states == psm_tick.need_worker_states
    assert adapter_tick.need_worker_fpm == psm_tick.need_worker_fpm
    assert adapter_tick.need_traffic_metrics == psm_tick.need_traffic_metrics


# ---------------------------------------------------------------------------
# G3 parity via adapter
# ---------------------------------------------------------------------------


def _psm_scale_to_from_fixture(record):
    s = record["planner_effects"]["scale_to"]
    if s is None:
        return None
    return ScalingDecision(
        num_prefill=s.get("num_prefill"), num_decode=s.get("num_decode")
    )


async def _run_scenario_through_adapter(scenario):
    """End-to-end driver: build adapter, bootstrap, iterate ticks
    through ``adapter.tick``, collect projected scale_to."""
    config = scenario.make_config()
    caps = scenario.caps_factory()

    # Build regressions via throwaway PSM (mirrors test_g3_real_parity).
    bootstrap_psm = PlannerStateMachine(config, caps)
    if scenario.bootstrap_fn is not None:
        scenario.bootstrap_fn(bootstrap_psm)

    adapter = OrchestratorEngineAdapter(config, caps)
    adapter.install_regressions(
        prefill=getattr(bootstrap_psm, "_prefill_regression", None),
        decode=getattr(bootstrap_psm, "_decode_regression", None),
        agg=getattr(bootstrap_psm, "_agg_regression", None),
    )
    await adapter.bootstrap_plugins()

    decisions = []
    for tick_input in scenario.ticks:
        scheduled = _tick_for(tick_input)
        effects = await adapter.tick(scheduled, tick_input)
        decisions.append(effects.scale_to)

    await adapter.shutdown()
    return decisions


@pytest.mark.asyncio
@pytest.mark.parametrize("scenario", ALL_SCENARIOS, ids=lambda s: s.name)
async def test_g3_parity_via_adapter(scenario):
    """For every G3 fixture scenario, driving the adapter should
    produce the same ``scale_to`` decision sequence as PSM.

    Any mismatch means the adapter diverged from PSM on a supported
    scenario — a regression that the ``NativePlannerBase`` wiring
    would inherit."""
    fixture_path = DEFAULT_OUTPUT_DIR / f"{scenario.name}.jsonl"
    assert fixture_path.exists(), f"missing fixture: {fixture_path}"
    fixture = _read_fixture(fixture_path)
    expected = [_psm_scale_to_from_fixture(rec) for rec in fixture[1:]]

    actual = await _run_scenario_through_adapter(scenario)

    assert len(expected) == len(actual)
    for i, (exp, act) in enumerate(zip(expected, actual)):
        assert exp == act, (
            f"scenario={scenario.name} tick={i}: expected={exp} actual={act}"
        )


# ---------------------------------------------------------------------------
# next_tick cadence advances as expected
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_next_tick_advances_after_throughput_tick():
    """After a throughput-scaling tick, ``next_tick.at_s`` should be
    ``now_s + throughput_adjustment_interval``."""
    scenario = next(
        s for s in ALL_SCENARIOS if s.name == "baseline_disagg_throughput_only_sla"
    )
    config = scenario.make_config()
    caps = scenario.caps_factory()

    bootstrap_psm = PlannerStateMachine(config, caps)
    scenario.bootstrap_fn(bootstrap_psm)

    adapter = OrchestratorEngineAdapter(config, caps)
    adapter.install_regressions(
        prefill=getattr(bootstrap_psm, "_prefill_regression", None),
        decode=getattr(bootstrap_psm, "_decode_regression", None),
    )
    await adapter.bootstrap_plugins()

    adapter.initial_tick(scenario.initial_tick_at_s)
    # Drive tick 1 (t=5, load tick).
    first = scenario.ticks[0]
    effects = await adapter.tick(_tick_for(first), first)
    # next_tick should be min(_next_load_s after load_adjustment_interval,
    # _next_throughput_s after throughput_adjustment_interval)
    assert effects.next_tick is not None
    assert effects.next_tick.at_s > first.now_s

    await adapter.shutdown()


# ---------------------------------------------------------------------------
# shutdown idempotent
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_shutdown_idempotent():
    scenario = ALL_SCENARIOS[0]
    adapter = OrchestratorEngineAdapter(
        scenario.make_config(), scenario.caps_factory()
    )
    await adapter.shutdown()
    await adapter.shutdown()  # second call survives
