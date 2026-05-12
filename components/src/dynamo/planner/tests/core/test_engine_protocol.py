# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for EngineProtocol + _PSMEngineAdapter.

Scope of this test file:
- Protocol conformance (runtime_checkable): both adapters pass
  ``isinstance(x, EngineProtocol)``.
- ``_PSMEngineAdapter`` forwards ``initial_tick`` and async-wraps
  ``on_tick`` to match the protocol's async ``tick``.
- ``_PSMEngineAdapter.shutdown`` is idempotent + a no-op.

Orchestrator-side adapter parity is out-of-scope for this file —
``test_engine_adapter.py`` exercises the full TickInput↔PipelineContext
bridge + ``OrchestratorEngineAdapter``.
"""

from __future__ import annotations

import pytest

from dynamo.planner.config.planner_config import PlannerConfig
from dynamo.planner.core.engine_protocol import EngineProtocol, _PSMEngineAdapter
from dynamo.planner.core.state_machine import PlannerStateMachine
from dynamo.planner.core.types import (
    EngineCapabilities,
    FpmObservations,
    ScheduledTick,
    TickInput,
    WorkerCapabilities,
    WorkerCounts,
)

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]


def _simple_caps() -> WorkerCapabilities:
    return WorkerCapabilities(
        decode=EngineCapabilities(num_gpu=1, max_num_batched_tokens=2048, max_kv_tokens=16384)
    )


def _easy_agg_config() -> PlannerConfig:
    # Easy mode avoids needing to seed regressions / predictors.
    return PlannerConfig(
        mode="agg",
        enable_load_scaling=True,
        enable_throughput_scaling=False,
        optimization_target="throughput",
    )


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_psm_adapter_satisfies_engine_protocol():
    psm = PlannerStateMachine(_easy_agg_config(), _simple_caps())
    adapter = _PSMEngineAdapter(psm)
    assert isinstance(adapter, EngineProtocol)


def test_plain_psm_is_not_engine_protocol():
    """PSM's ``on_tick`` is synchronous + named differently; it should
    NOT satisfy EngineProtocol directly. The adapter is required."""
    psm = PlannerStateMachine(_easy_agg_config(), _simple_caps())
    assert not isinstance(psm, EngineProtocol)


# ---------------------------------------------------------------------------
# _PSMEngineAdapter behaviour
# ---------------------------------------------------------------------------


def test_initial_tick_forwards_to_psm():
    psm = PlannerStateMachine(_easy_agg_config(), _simple_caps())
    adapter = _PSMEngineAdapter(psm)
    # PSM.initial_tick schedules first load/throughput tick.
    st = adapter.initial_tick(0.0)
    assert isinstance(st, ScheduledTick)
    # Either load or throughput scaling should be scheduled (easy agg + load on).
    assert st.run_load_scaling or st.run_throughput_scaling


@pytest.mark.asyncio
async def test_tick_async_wraps_psm_on_tick_identically():
    psm = PlannerStateMachine(_easy_agg_config(), _simple_caps())
    adapter = _PSMEngineAdapter(psm)

    # Baseline: call PSM directly.
    tick_input_a = TickInput(
        now_s=5.0,
        fpm_observations=FpmObservations(
            decode={("w1", 0): _make_fpm()},
        ),
        worker_counts=WorkerCounts(ready_num_decode=1),
    )
    scheduled = ScheduledTick(
        at_s=5.0, run_load_scaling=True, run_throughput_scaling=False
    )

    direct_effects = psm.on_tick(scheduled, tick_input_a)

    # Rebuild PSM for parity (on_tick mutates state).
    psm2 = PlannerStateMachine(_easy_agg_config(), _simple_caps())
    adapter2 = _PSMEngineAdapter(psm2)
    adapter_effects = await adapter2.tick(scheduled, tick_input_a)

    # Both produce PlannerEffects (same shape / equal fields here since
    # easy mode is deterministic given identical inputs).
    assert direct_effects.scale_to == adapter_effects.scale_to
    assert direct_effects.next_tick == adapter_effects.next_tick


@pytest.mark.asyncio
async def test_shutdown_is_noop_and_idempotent():
    psm = PlannerStateMachine(_easy_agg_config(), _simple_caps())
    adapter = _PSMEngineAdapter(psm)
    assert await adapter.shutdown() is None
    assert await adapter.shutdown() is None  # idempotent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fpm():
    from dynamo.common.forward_pass_metrics import (
        ForwardPassMetrics,
        QueuedRequestMetrics,
        ScheduledRequestMetrics,
    )

    return ForwardPassMetrics(
        worker_id="w1",
        dp_rank=0,
        wall_time=0.01,
        scheduled_requests=ScheduledRequestMetrics(
            sum_prefill_tokens=0,
            num_prefill_requests=0,
            sum_decode_kv_tokens=100,
            num_decode_requests=1,
        ),
        queued_requests=QueuedRequestMetrics(
            sum_prefill_tokens=0,
            sum_decode_kv_tokens=0,
        ),
    )
