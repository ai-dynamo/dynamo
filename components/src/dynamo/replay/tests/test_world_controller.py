# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for the multi-deployment Replay planner controller."""

import asyncio
from collections.abc import Iterator
from typing import Any

import pytest

from dynamo.global_planner.capacity_manager import PoolSpec
from dynamo.planner.offline.replay_adapter import PlannerTickResult, ScalingProposal
from dynamo.planner.plugins.clock import VirtualClock
from dynamo.replay.global_planner import ReplayParticipantSpec
from dynamo.replay.world import (
    ReplayGlobalPlannerConfig,
    ReplayWorldPlannerController,
)

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]


class _FakePlannerSession:
    def __init__(
        self,
        participant_id: str,
        results: list[PlannerTickResult],
        events: list[tuple[str, str]],
        *,
        fail_on_mediation: bool = False,
    ) -> None:
        self.participant_id = participant_id
        self._results = list(results)
        self._events = events
        self._fail_on_mediation = fail_on_mediation
        self.propose_ticks: list[dict[str, Any]] = []
        self.rejections: list[tuple[ScalingProposal, str | None]] = []
        self.mediations: list[tuple[ScalingProposal, str, str | None]] = []
        self.commits: list[dict[str, Any]] = []
        self.shutdown_calls = 0

    async def propose_tick_async(self, tick: dict[str, Any]) -> PlannerTickResult:
        self._events.append(("propose", self.participant_id))
        self.propose_ticks.append(tick)
        if not self._results:
            raise AssertionError(f"unexpected planner tick for {self.participant_id!r}")
        return self._results.pop(0)

    def observe_rejection(
        self,
        proposal: ScalingProposal,
        *,
        message: str | None = None,
    ) -> None:
        self._events.append(("reject", self.participant_id))
        self.rejections.append((proposal, message))

    def observe_mediation(
        self,
        proposal: ScalingProposal,
        *,
        status: str,
        message: str | None = None,
    ) -> None:
        self._events.append(("mediate", self.participant_id))
        self.mediations.append((proposal, status, message))
        if self._fail_on_mediation:
            raise RuntimeError(f"mediation callback failed for {self.participant_id}")

    def observe_committed_scale(self, **kwargs: Any) -> None:
        self._events.append(("commit", self.participant_id))
        self.commits.append(kwargs)

    async def shutdown_async(self) -> None:
        self.shutdown_calls += 1


@pytest.fixture
def controller_factory() -> Iterator[Any]:
    controllers: list[ReplayWorldPlannerController] = []

    def build(
        sessions: dict[str, _FakePlannerSession],
        *,
        initial_replicas: dict[str, int] | None = None,
        config: ReplayGlobalPlannerConfig = ReplayGlobalPlannerConfig(),
    ) -> ReplayWorldPlannerController:
        replicas = initial_replicas or {}
        participants = [
            ReplayParticipantSpec(
                participant_id=participant_id,
                pools=(
                    PoolSpec(
                        sub_type="decode",
                        current_replicas=replicas.get(participant_id, 1),
                        gpu_per_replica=1,
                    ),
                ),
            )
            for participant_id in sessions
        ]
        controller = ReplayWorldPlannerController(
            sessions,  # type: ignore[arg-type]
            participants,
            config=config,
            event_loop=asyncio.new_event_loop(),
            clock=VirtualClock(),
        )
        controllers.append(controller)
        return controller

    yield build

    for controller in reversed(controllers):
        controller.close()


def _proposal(
    *,
    at_s: float,
    current_decode: int,
    target_decode: int,
    next_tick_ms: float | None = None,
) -> PlannerTickResult:
    return PlannerTickResult(
        proposal=ScalingProposal(
            at_s=at_s,
            current_prefill=0,
            current_decode=current_decode,
            target_decode=target_decode,
        ),
        next_tick_ms=next_tick_ms,
    )


def _tick(participant_id: str, now_ms: float) -> dict[str, Any]:
    return {"deployment_id": participant_id, "now_ms": now_ms}


def test_all_due_proposals_run_before_any_commit_notification(
    controller_factory,
):
    events: list[tuple[str, str]] = []
    sessions = {
        "b": _FakePlannerSession(
            "b",
            [_proposal(at_s=1.0, current_decode=1, target_decode=2)],
            events,
        ),
        "a": _FakePlannerSession(
            "a",
            [_proposal(at_s=1.0, current_decode=1, target_decode=2)],
            events,
        ),
    }
    controller = controller_factory(sessions)

    result = controller.on_ticks([_tick("b", 1_000.0), _tick("a", 1_000.0)])

    assert result["actions"] == [
        {
            "deployment_id": "a",
            "target_prefill": None,
            "target_decode": 2,
        },
        {
            "deployment_id": "b",
            "target_prefill": None,
            "target_decode": 2,
        },
    ]
    assert events == [
        ("propose", "a"),
        ("propose", "b"),
        ("mediate", "a"),
        ("mediate", "b"),
        ("commit", "a"),
        ("commit", "b"),
    ]


def test_rejected_request_stages_no_action(controller_factory):
    events: list[tuple[str, str]] = []
    session = _FakePlannerSession(
        "deployment",
        [_proposal(at_s=1.0, current_decode=2, target_decode=3)],
        events,
    )
    controller = controller_factory(
        {"deployment": session},
        initial_replicas={"deployment": 2},
        config=ReplayGlobalPlannerConfig(max_total_gpus=2),
    )

    result = controller.on_ticks([_tick("deployment", 1_000.0)])

    assert result["actions"] == []
    assert len(session.rejections) == 1
    assert session.commits == []
    assert controller._capacity_manager.current_replicas("deployment") == {"decode": 2}


def test_fixed_budget_cached_partner_is_applied_down_then_up(
    controller_factory,
):
    events: list[tuple[str, str]] = []
    sessions = {
        "a": _FakePlannerSession(
            "a",
            [_proposal(at_s=1.0, current_decode=2, target_decode=1)],
            events,
        ),
        "b": _FakePlannerSession(
            "b",
            [_proposal(at_s=2.0, current_decode=2, target_decode=3)],
            events,
        ),
    }
    controller = controller_factory(
        sessions,
        initial_replicas={"a": 2, "b": 2},
        config=ReplayGlobalPlannerConfig(
            min_total_gpus=4,
            max_total_gpus=4,
        ),
    )

    first = controller.on_ticks([_tick("a", 1_000.0)])
    assert first["actions"] == []
    assert len(sessions["a"].rejections) == 1
    assert sessions["a"].commits == []

    events.clear()
    second = controller.on_ticks([_tick("b", 2_000.0)])

    assert second["actions"] == [
        {
            "deployment_id": "a",
            "target_prefill": None,
            "target_decode": 1,
        },
        {
            "deployment_id": "b",
            "target_prefill": None,
            "target_decode": 3,
        },
    ]
    assert events == [
        ("propose", "b"),
        ("mediate", "b"),
        ("commit", "a"),
        ("commit", "b"),
    ]
    assert len(sessions["a"].propose_ticks) == 1
    assert sessions["a"].commits == [
        {
            "at_s": 2.0,
            "target_prefill": None,
            "target_decode": 1,
            "reason": "global_planner",
        }
    ]


@pytest.mark.parametrize(
    ("ticks", "message"),
    [
        (
            [_tick("a", 1_000.0), _tick("a", 1_000.0)],
            "fired twice at one timestamp",
        ),
        (
            [_tick("a", 1_000.0), _tick("b", 1_001.0)],
            "exactly one timestamp",
        ),
    ],
)
def test_duplicate_or_mismatched_tick_barriers_are_rejected(
    controller_factory,
    ticks,
    message,
):
    events: list[tuple[str, str]] = []
    sessions = {
        "a": _FakePlannerSession("a", [], events),
        "b": _FakePlannerSession("b", [], events),
    }
    controller = controller_factory(sessions)

    with pytest.raises(ValueError, match=message):
        controller.on_ticks(ticks)

    assert events == []
    assert not controller._capacity_manager.batch_active


def test_callback_exception_aborts_capacity_batch(controller_factory):
    events: list[tuple[str, str]] = []
    sessions = {
        "a": _FakePlannerSession(
            "a",
            [_proposal(at_s=1.0, current_decode=1, target_decode=2)],
            events,
        ),
        "b": _FakePlannerSession(
            "b",
            [_proposal(at_s=1.0, current_decode=1, target_decode=2)],
            events,
            fail_on_mediation=True,
        ),
    }
    controller = controller_factory(sessions)
    manager = controller._capacity_manager

    with pytest.raises(RuntimeError, match="mediation callback failed for b"):
        controller.on_ticks([_tick("a", 1_000.0), _tick("b", 1_000.0)])

    assert not manager.batch_active
    assert manager.current_replicas("a") == {"decode": 1}
    assert manager.current_replicas("b") == {"decode": 1}
    assert sessions["a"].commits == []
    assert sessions["b"].commits == []
    manager.begin_batch()
    assert manager.finish_batch() == ()
