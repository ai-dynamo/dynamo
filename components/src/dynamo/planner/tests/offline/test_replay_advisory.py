# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402
# Optional-dependency preflight must run before replay imports.

"""Planner advisory mode in the replay host.

A scripted engine stands in for the planner, so these tests cover only what
``ReplayPlannerAdapter`` does with its effects: in advisory mode it hands Rust
no targets and no router parameters, keeps reporting the deployed counts as
expected, and records the recommendations instead of scaling events.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field

import pytest

pytest.importorskip(
    "aisimulate.replay",
    reason="AI Simulate is an optional Dynamo simulation dependency",
)

from dynamo.planner.config.planner_config import PlannerConfig
from dynamo.planner.core.types import (
    PlannerEffects,
    ScalingDecision,
    ScheduledTick,
    TickInput,
)
from dynamo.planner.offline.replay_adapter import ReplayPlannerAdapter
from dynamo.planner.tests.retune_stubs import ROUTER_MAP

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]

_QUIET = {"report_interval_hours": None, "live_dashboard_port": 0}
_HOLD = None  # a decision tick without scale_to


@dataclass(frozen=True)
class _Step:
    """One scripted tick: the decision flag and the planner's ``scale_to``."""

    scale_to: ScalingDecision | None
    decision_tick: bool = True


@dataclass
class _ScriptedEngine:
    """Returns one scripted ``PlannerEffects`` per tick; records the inputs."""

    steps: Sequence[_Step]
    inputs: list[TickInput] = field(default_factory=list)

    def initial_tick(self, now_s: float) -> ScheduledTick:
        return self._tick(now_s, 0)

    def _tick(self, at_s: float, index: int) -> ScheduledTick:
        decision = index < len(self.steps) and self.steps[index].decision_tick
        return ScheduledTick(
            at_s=at_s, run_load_scaling=decision, need_worker_states=True
        )

    async def tick(self, tick: ScheduledTick, tick_input: TickInput) -> PlannerEffects:
        index = len(self.inputs)
        self.inputs.append(tick_input)
        return PlannerEffects(
            scale_to=self.steps[index].scale_to,
            next_tick=self._tick(tick.at_s + 1.0, index + 1),
            router_parameters=ROUTER_MAP,
        )

    async def shutdown(self) -> None:
        return None


def _snapshot(ordinal: int, *, prefill: int = 0, decode: int = 1) -> dict:
    return {
        "tick_ordinal": ordinal,
        "active_prefill_count": prefill,
        "active_decode_count": decode,
        "active_prefill_ids": list(range(prefill)),
        "active_decode_ids": list(range(100, 100 + decode)),
    }


def _replay(config: PlannerConfig, steps: Sequence[_Step], **snapshot):
    engine = _ScriptedEngine(steps)
    with ReplayPlannerAdapter(planner_config=config, engine=engine) as adapter:
        adapter.start()
        # Rust keeps the deployment fixed: every snapshot reports one worker.
        decisions = [
            adapter.on_tick(_snapshot(ordinal, **snapshot))
            for ordinal in range(len(steps))
        ]
        details = adapter.finalize()
    return engine, decisions, details


def _events(events):
    return [(e.at_s, e.component, e.from_count, e.to_count) for e in events]


def _agg(num_decode):
    return ScalingDecision(num_decode=num_decode)


# Recommendations against a fixed deployment of one worker: a repeat is not a
# new recommendation, even across a tick that runs no scaling decision; a
# decision tick that holds resets it; and a recommendation equal to the
# deployed count records nothing.
_AGG_SCRIPT = (
    _Step(_agg(2)),
    _Step(_HOLD, decision_tick=False),
    _Step(_agg(2)),
    _Step(_HOLD),
    _Step(_agg(2)),
    _Step(_agg(3)),
    _Step(_agg(1)),
    _Step(_agg(3)),
)


def test_advisory_records_recommendations_and_keeps_the_deployment(caplog):
    config = PlannerConfig(mode="agg", advisory=True, **_QUIET)

    with caplog.at_level(logging.WARNING):
        engine, decisions, details = _replay(config, _AGG_SCRIPT)

    assert any(
        "[ADVISORY]" in record.getMessage() and "fixed" in record.getMessage()
        for record in caplog.records
        if record.levelno == logging.WARNING
    )
    # Rust gets nothing to apply: no targets and no router update.
    assert all(
        (d["target_prefill"], d["target_decode"], d["router_parameters"])
        == (None, None, None)
        for d in decisions
    )
    # The planner sees what a live advisory planner sees: the deployed count
    # as expected and no scaling in progress.
    for tick_input in engine.inputs:
        counts = tick_input.worker_counts
        assert (counts.ready_num_decode, counts.expected_num_decode) == (1, 1)
        assert not counts.decode_scaling_in_progress
    assert details.scaling_events == []
    assert _events(details.recommended_scaling_events) == [
        (0.0, "agg", 1, 2),
        (4.0, "agg", 1, 2),
        (5.0, "agg", 1, 3),
        (7.0, "agg", 1, 3),
    ]
    assert [e.reason for e in details.recommended_scaling_events] == ["scale_up"] * 4
    assert details.metadata["advisory"] is True
    payload = details.to_dict()
    assert payload["scaling_events"] == []
    assert payload["recommended_scaling_events"][0] == {
        "at_s": 0.0,
        "component": "agg",
        "from_count": 1,
        "to_count": 2,
        "reason": "scale_up",
    }
    # Per-tick recommendations stay in the tick capture.
    assert details.ticks[0]["effects"]["scale_to"] == {
        "num_prefill": None,
        "num_decode": 2,
    }
    assert details.ticks[0]["runtime_decision"]["router_parameters"] is None


def test_active_replay_applies_the_same_decisions():
    config = PlannerConfig(mode="agg", **_QUIET)

    engine, decisions, details = _replay(config, _AGG_SCRIPT[:2])

    assert decisions[0]["target_decode"] == 2
    assert decisions[0]["router_parameters"] == ROUTER_MAP
    # The target is pending in the next tick: expected 2, scaling in progress.
    counts = engine.inputs[1].worker_counts
    assert (counts.ready_num_decode, counts.expected_num_decode) == (1, 2)
    assert counts.decode_scaling_in_progress
    assert _events(details.scaling_events)[0] == (0.0, "agg", 1, 2)
    assert details.recommended_scaling_events is None
    assert details.metadata["advisory"] is False
    assert "recommended_scaling_events" not in details.to_dict()


def test_disagg_role_without_a_target_holds_that_role():
    config = PlannerConfig(mode="disagg", advisory=True, **_QUIET)
    steps = (
        _Step(ScalingDecision(num_prefill=2)),
        # No prefill target (the power-aware projection drops a role it
        # keeps): prefill holds, which resets its recommendation.
        _Step(ScalingDecision(num_decode=3)),
        _Step(ScalingDecision(num_prefill=2, num_decode=3)),
        _Step(ScalingDecision(num_prefill=1, num_decode=1)),
    )

    engine, decisions, details = _replay(config, steps, prefill=1, decode=1)

    assert all(
        (d["target_prefill"], d["target_decode"]) == (None, None) for d in decisions
    )
    counts = engine.inputs[-1].worker_counts
    assert (counts.expected_num_prefill, counts.expected_num_decode) == (1, 1)
    assert not (counts.prefill_scaling_in_progress or counts.decode_scaling_in_progress)
    assert _events(details.recommended_scaling_events) == [
        (0.0, "prefill", 1, 2),
        (1.0, "decode", 1, 3),
        (2.0, "prefill", 1, 2),
    ]


def test_advisory_without_recommendations_reports_an_empty_list():
    config = PlannerConfig(mode="agg", advisory=True, **_QUIET)

    _, _, details = _replay(config, (_Step(_HOLD), _Step(_agg(1))))

    assert details.recommended_scaling_events == []
    assert details.to_dict()["recommended_scaling_events"] == []
