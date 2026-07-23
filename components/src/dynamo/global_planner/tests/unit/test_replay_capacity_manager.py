# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Replay's in-memory Global Planner capacity backend."""

from unittest.mock import AsyncMock, patch

import pytest

from dynamo.global_planner.capacity_manager import CapacityManager, PoolSpec
from dynamo.global_planner.orchestrator import Orchestrator
from dynamo.planner import SubComponentType, TargetReplica
from dynamo.planner.connectors.protocol import ScaleStatus
from dynamo.replay.global_planner import (
    ReplayCapacityManager,
    ReplayParticipantSpec,
    ReplayScaleAction,
    ReplayScaleTarget,
)

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]


def _participant(
    participant_id: str, *pools: tuple[str, int, int]
) -> ReplayParticipantSpec:
    return ReplayParticipantSpec(
        participant_id=participant_id,
        pools=tuple(
            PoolSpec(
                sub_type=role,
                current_replicas=replicas,
                gpu_per_replica=gpu,
                component_name=f"{participant_id}-{role}",
            )
            for role, replicas, gpu in pools
        ),
    )


def _target(role: SubComponentType, replicas: int) -> TargetReplica:
    return TargetReplica(sub_component_type=role, desired_replicas=replicas)


@pytest.mark.asyncio
async def test_capacity_manager_observe_async_offloads_sync_backend():
    manager = CapacityManager()
    expected = {"deployment": {}}

    with patch(
        "dynamo.global_planner.capacity_manager.asyncio.to_thread",
        new=AsyncMock(return_value=expected),
    ) as to_thread:
        result = await manager.observe_async(require_complete=True)

    assert result == expected
    to_thread.assert_awaited_once_with(manager.observe, True)


@pytest.mark.asyncio
async def test_snapshot_and_actions_are_deterministic_and_overlay_aware():
    manager = ReplayCapacityManager(
        [
            _participant("z-deployment", ("decode", 1, 2)),
            _participant(
                "a-deployment",
                ("prefill", 2, 1),
                ("decode", 3, 1),
            ),
        ]
    )

    snapshot = await manager.observe_async(require_complete=True)
    assert list(snapshot) == ["a-deployment", "z-deployment"]
    assert list(snapshot["a-deployment"]) == ["decode", "prefill"]

    manager.begin_batch()
    await manager.scale(
        "a-deployment",
        [
            _target(SubComponentType.PREFILL, 4),
            _target(SubComponentType.DECODE, 5),
        ],
        blocking=False,
    )

    assert manager.current_replicas("a-deployment") == {
        "decode": 5,
        "prefill": 4,
    }
    assert manager.observe()["a-deployment"]["decode"].current_replicas == 5

    actions = manager.finish_batch()
    assert actions == (
        ReplayScaleAction(
            participant_id="a-deployment",
            targets=(
                ReplayScaleTarget("prefill", 4),
                ReplayScaleTarget("decode", 5),
            ),
            blocking=False,
        ),
    )
    assert manager.current_replicas("a-deployment") == {
        "decode": 5,
        "prefill": 4,
    }


@pytest.mark.asyncio
async def test_abort_batch_discards_overlay_and_actions():
    manager = ReplayCapacityManager([_participant("deployment", ("decode", 2, 1))])

    manager.begin_batch()
    await manager.scale(
        "deployment",
        [_target(SubComponentType.DECODE, 7)],
        blocking=True,
    )
    assert manager.current_replicas("deployment") == {"decode": 7}

    manager.abort_batch()

    assert not manager.batch_active
    assert manager.current_replicas("deployment") == {"decode": 2}
    manager.begin_batch()
    assert manager.finish_batch() == ()


@pytest.mark.asyncio
async def test_invalid_action_is_atomic():
    manager = ReplayCapacityManager([_participant("deployment", ("decode", 2, 1))])
    manager.begin_batch()

    with pytest.raises(ValueError, match="unknown pool role 'prefill'"):
        await manager.scale(
            "deployment",
            [
                _target(SubComponentType.DECODE, 4),
                _target(SubComponentType.PREFILL, 1),
            ],
            blocking=False,
        )

    assert manager.current_replicas("deployment") == {"decode": 2}
    assert manager.finish_batch() == ()


@pytest.mark.asyncio
async def test_orchestrator_stages_cross_participant_pair_in_apply_order():
    manager = ReplayCapacityManager(
        [
            _participant("a", ("decode", 2, 1)),
            _participant("b", ("decode", 2, 1)),
        ]
    )
    orchestrator = Orchestrator(
        capacity_manager=manager,
        managed_deployments=None,
        min_total_gpus=4,
        max_total_gpus=4,
        now=lambda: 10.0,
        use_lock=False,
    )

    # A's standalone scale-down is denied but seeds the Global Planner intent.
    manager.begin_batch()
    first = await orchestrator.submit(
        "a",
        [_target(SubComponentType.DECODE, 1)],
        blocking=False,
        deployment_name="a",
        caller_name="a",
    )
    assert first.status == ScaleStatus.REJECTED
    assert manager.finish_batch() == ()

    # B's scale-up pairs with A's cached intent. Down is staged before up.
    manager.begin_batch()
    second = await orchestrator.submit(
        "b",
        [_target(SubComponentType.DECODE, 3)],
        blocking=False,
        deployment_name="b",
        caller_name="b",
    )
    actions = manager.finish_batch()

    assert second.status == ScaleStatus.SUCCESS
    assert second.current_replicas == {"decode": 3}
    assert [action.participant_id for action in actions] == ["a", "b"]
    assert [action.targets[0].desired_replicas for action in actions] == [1, 3]
    assert manager.current_replicas("a") == {"decode": 1}
    assert manager.current_replicas("b") == {"decode": 3}


@pytest.mark.parametrize(
    ("participants", "message"),
    [
        (
            [
                _participant("duplicate", ("decode", 1, 1)),
                _participant("duplicate", ("decode", 1, 1)),
            ],
            "duplicate replay participant",
        ),
        (
            [
                _participant(
                    "deployment",
                    ("decode", 1, 1),
                    ("decode", 2, 1),
                )
            ],
            "duplicate pool role",
        ),
        (
            [_participant("deployment", ("aggregate", 1, 1))],
            "pool role.*must be one of",
        ),
        (
            [_participant("deployment", ("decode", -1, 1))],
            "initial replicas.*must be non-negative",
        ),
        (
            [_participant("deployment", ("decode", 1, -1))],
            "gpu_per_replica.*must be non-negative",
        ),
    ],
)
def test_setup_validation(participants, message):
    with pytest.raises(ValueError, match=message):
        ReplayCapacityManager(participants)


@pytest.mark.asyncio
async def test_action_boundary_validation():
    manager = ReplayCapacityManager([_participant("deployment", ("decode", 2, 1))])

    with pytest.raises(RuntimeError, match="no replay capacity batch"):
        await manager.scale(
            "deployment",
            [_target(SubComponentType.DECODE, 3)],
            blocking=False,
        )

    manager.begin_batch()
    with pytest.raises(ValueError, match="desired replicas.*must be non-negative"):
        await manager.scale(
            "deployment",
            [_target(SubComponentType.DECODE, -1)],
            blocking=False,
        )
    with pytest.raises(ValueError, match="duplicate target role"):
        await manager.scale(
            "deployment",
            [
                _target(SubComponentType.DECODE, 3),
                _target(SubComponentType.DECODE, 4),
            ],
            blocking=False,
        )
    with pytest.raises(KeyError, match="was not pre-registered"):
        await manager.scale(
            "unknown",
            [_target(SubComponentType.DECODE, 3)],
            blocking=False,
        )
    assert manager.finish_batch() == ()


def test_batch_lifecycle_is_strict():
    manager = ReplayCapacityManager([_participant("deployment", ("decode", 2, 1))])

    with pytest.raises(RuntimeError, match="no replay capacity batch"):
        manager.finish_batch()
    with pytest.raises(RuntimeError, match="no replay capacity batch"):
        manager.abort_batch()

    manager.begin_batch()
    with pytest.raises(RuntimeError, match="already active"):
        manager.begin_batch()
    manager.abort_batch()
