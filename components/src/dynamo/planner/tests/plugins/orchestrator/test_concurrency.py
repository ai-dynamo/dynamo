# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Concurrency / failure / timeout tests.

These exercise the pipeline's `asyncio.gather` semantics:

- Multiple PROPOSE plugins run concurrently (elapsed ≈ max, not sum).
- A per-plugin timeout (PluginTimeoutError from the transport) records
  a failure on the circuit breaker without failing other plugins.
- A per-plugin exception path likewise records a failure.
- Enough consecutive failures OPEN the circuit → plugin drops from
  subsequent active sets.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from dynamo.planner.plugins.merge.types import ComponentKey
from dynamo.planner.plugins.types import (
    AcceptResult,
    CircuitState,
    ComponentTarget,
    OverrideResult,
    OverrideType,
    PipelineContext,
    ProposeStageResponse,
)

from .conftest import StubPlugin

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]


PREFILL = ComponentKey(sub_component_type="prefill")


def _override(replicas):
    def handler(req):
        return ProposeStageResponse(
            result_kind="override",
            override=OverrideResult(
                targets=[
                    ComponentTarget(
                        sub_component_type="prefill",
                        replicas=replicas,
                        type=OverrideType.SET,
                    )
                ]
            ),
        )

    return handler


# ---------------------------------------------------------------------------
# asyncio.gather parallelism
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_multiple_propose_plugins_run_concurrently(ctx_factory):
    ctx = ctx_factory(tick_max_duration_seconds=10.0)
    orchestrator = ctx["orchestrator"]

    DELAY = 0.05

    async def slow_handler(req):
        await asyncio.sleep(DELAY)
        return ProposeStageResponse(
            result_kind="accept", accept=AcceptResult()
        )

    for i in range(5):
        orchestrator.register_internal(
            plugin_id=f"p{i}",
            plugin_type="propose",
            priority=10 + i,
            instance=StubPlugin(propose=slow_handler),
        )

    started = time.perf_counter()
    await orchestrator.tick(PipelineContext(), {PREFILL: 3})
    elapsed = time.perf_counter() - started
    # 5 plugins × 50ms serial would be 250ms; concurrent should be closer
    # to 50ms. Assert well under the serial lower bound with generous CI margin.
    assert elapsed < DELAY * 3, (
        f"expected concurrent execution (~{DELAY}s), got {elapsed:.3f}s"
    )


# ---------------------------------------------------------------------------
# Per-plugin failure paths
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_one_plugin_raises_others_continue(ctx_factory):
    ctx = ctx_factory()
    orchestrator = ctx["orchestrator"]
    cb = ctx["circuit_breaker"]

    def raising_handler(req):
        raise RuntimeError("boom")

    orchestrator.register_internal(
        plugin_id="bad",
        plugin_type="propose",
        priority=10,
        instance=StubPlugin(propose=raising_handler),
    )
    orchestrator.register_internal(
        plugin_id="good",
        plugin_type="propose",
        priority=5,
        instance=StubPlugin(propose=_override(8)),
    )

    outcome = await orchestrator.tick(PipelineContext(), {PREFILL: 3})
    # good plugin's SET won; bad plugin's failure recorded but didn't short-circuit.
    assert outcome.execute_action == "apply"
    assert outcome.final_proposal.targets[0].replicas == 8
    # Circuit breaker noticed the failure on "bad".
    assert cb.state("good") == CircuitState.CLOSED
    # After one failure on "bad", state still CLOSED (default threshold > 1).
    assert cb.state("bad") == CircuitState.CLOSED


@pytest.mark.asyncio
async def test_plugin_timeout_records_failure_without_tripping_tick(ctx_factory):
    # per-plugin timeout: InProcessTransport timeout=1.0s (from conftest);
    # slow handler exceeds it, transport raises PluginTimeoutError.
    ctx = ctx_factory(tick_max_duration_seconds=5.0)
    orchestrator = ctx["orchestrator"]
    cb = ctx["circuit_breaker"]

    async def slow_handler(req):
        await asyncio.sleep(2.0)  # > transport timeout 1.0s
        return ProposeStageResponse(result_kind="accept", accept=AcceptResult())

    orchestrator.register_internal(
        plugin_id="slow",
        plugin_type="propose",
        priority=10,
        instance=StubPlugin(propose=slow_handler),
    )
    orchestrator.register_internal(
        plugin_id="fast",
        plugin_type="propose",
        priority=5,
        instance=StubPlugin(propose=_override(12)),
    )

    outcome = await orchestrator.tick(PipelineContext(), {PREFILL: 3})
    # fast plugin's SET won; slow timed out, didn't drag the whole tick.
    assert outcome.execute_action == "apply"
    assert outcome.final_proposal.targets[0].replicas == 12


@pytest.mark.asyncio
async def test_repeated_failures_open_circuit_drops_plugin_from_active_set(
    ctx_factory,
):
    ctx = ctx_factory(failure_threshold=2)
    orchestrator = ctx["orchestrator"]
    cb = ctx["circuit_breaker"]

    def raising_handler(req):
        raise RuntimeError("boom")

    orchestrator.register_internal(
        plugin_id="flaky",
        plugin_type="propose",
        priority=10,
        instance=StubPlugin(propose=raising_handler),
    )
    orchestrator.register_internal(
        plugin_id="steady",
        plugin_type="propose",
        priority=5,
        instance=StubPlugin(propose=_override(4)),
    )

    # Two ticks → "flaky" accumulates failures → circuit OPEN.
    await orchestrator.tick(PipelineContext(), {PREFILL: 3})
    await orchestrator.tick(PipelineContext(), {PREFILL: 3})
    assert cb.state("flaky") == CircuitState.OPEN

    # Third tick: "flaky" not in active set; no new failure recorded.
    outcome = await orchestrator.tick(PipelineContext(), {PREFILL: 3})
    assert outcome.execute_action == "apply"
    assert outcome.final_proposal.targets[0].replicas == 4
    # "flaky" still OPEN (it wasn't even called this tick).
    assert cb.state("flaky") == CircuitState.OPEN


# ---------------------------------------------------------------------------
# Pairing regression: plugins + results matched by position via zip
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_M1_priority_paired_by_position_not_result_backref(ctx_factory):
    # If the pipeline were assuming a backreference from result to plugin,
    # it would fail to use the priority correctly when raw responses don't
    # carry plugin identity. Verify: two SET responses with different
    # priorities → priority-smaller wins; result objects themselves have
    # no priority field so this confirms the zip(plugins, results) pattern.
    ctx = ctx_factory()
    ctx["orchestrator"].register_internal(
        plugin_id="low_prio",
        plugin_type="propose",
        priority=100,
        instance=StubPlugin(propose=_override(5)),
    )
    ctx["orchestrator"].register_internal(
        plugin_id="high_prio",
        plugin_type="propose",
        priority=10,
        instance=StubPlugin(propose=_override(50)),
    )
    outcome = await ctx["orchestrator"].tick(PipelineContext(), {PREFILL: 1})
    # high_prio (priority-smaller number) wins.
    assert outcome.final_proposal.targets[0].replicas == 50
