# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Orchestrator — GPU-budget arbitration + observe/decide/actuate.

The decision tests exercise ``mediate`` directly (pure, no capacity manager
needed). The observe/decide/actuate tests drive ``submit`` over an in-memory
``FakeCapacityManager`` (subclassing the neutral ``CapacityManager`` base) — no
Kubernetes, no runtime — which previews how offline replay plugs a non-K8s
backend into the same orchestrator, including the ``use_lock`` toggle.
"""

import asyncio
from threading import Event

import pytest
from kubernetes import client

from dynamo.global_planner.capacity_manager import CapacityManager, PoolSpec
from dynamo.global_planner.orchestrator import Orchestrator, _ParticipantMutationBusy
from dynamo.planner import SubComponentType, TargetReplica
from dynamo.planner.connectors.base import ReplicaUpdateResult
from dynamo.planner.connectors.protocol import ScaleStatus
from dynamo.planner.errors import EmptyTargetReplicasError

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]


def _targets(prefill=None, decode=None):
    out = []
    if prefill is not None:
        out.append(
            TargetReplica(
                sub_component_type=SubComponentType.PREFILL, desired_replicas=prefill
            )
        )
    if decode is not None:
        out.append(
            TargetReplica(
                sub_component_type=SubComponentType.DECODE, desired_replicas=decode
            )
        )
    return out


# ---------------------------------------------------------------------------- #
# Decision layer (mediate) — pure, no capacity manager needed                  #
# ---------------------------------------------------------------------------- #


def _pools(**participants):
    """Build a snapshot: participant_id -> {sub_type: PoolSpec}.

    Each participant is ``key=(prefill, decode, gpu)`` where prefill/decode are
    current replica counts (None to omit the pool) and gpu is per-replica GPUs.
    ``__`` in the key becomes ``/``.
    """
    snapshot = {}
    for key, (prefill, decode, gpu) in participants.items():
        key = key.replace("__", "/")
        pools = {}
        if prefill is not None:
            pools["prefill"] = PoolSpec("prefill", prefill, gpu)
        if decode is not None:
            pools["decode"] = PoolSpec("decode", decode, gpu)
        snapshot[key] = pools
    return snapshot


def _decider(**kwargs):
    kwargs.setdefault("use_lock", False)
    return Orchestrator(capacity_manager=None, managed_deployments=None, **kwargs)


def _mediate(orch, key, targets, pools):
    return orch.mediate(
        key, targets, pools, log_deployment_name=key, log_caller_name=key
    )


def test_no_budget_approves_standalone():
    orch = _decider(max_total_gpus=-1, min_total_gpus=-1)
    pools = _pools(ns__a=(1, 1, 1))
    res = _mediate(orch, "ns/a", _targets(prefill=5), pools)
    assert res.approved
    assert res.selected_partners == []
    assert res.reject_message is None


def test_ceiling_allows_within_cap():
    orch = _decider(max_total_gpus=4)
    pools = _pools(ns__a=(1, 1, 1))  # total 2
    res = _mediate(orch, "ns/a", _targets(prefill=3), pools)  # -> total 4
    assert res.approved and res.selected_partners == []


def test_ceiling_rejects_breach():
    orch = _decider(max_total_gpus=2)
    pools = _pools(ns__a=(1, 1, 1))
    res = _mediate(orch, "ns/a", _targets(prefill=3), pools)  # -> total 4 > 2
    assert not res.approved
    assert "budget breach" in (res.reject_message or "").lower()
    assert "ceiling" in (res.reject_message or "").lower()


def test_floor_rejects_unpaired_scale_down():
    orch = _decider(max_total_gpus=6, min_total_gpus=6)
    pools = _pools(ns__a=(3, 3, 1))  # total 6
    res = _mediate(orch, "ns/a", _targets(prefill=2), pools)  # -> 5 < floor, no pair
    assert not res.approved
    assert "budget breach" in (res.reject_message or "").lower()


def test_floor_pairs_scale_down_across_participants():
    orch = _decider(max_total_gpus=4, min_total_gpus=4)
    pools = _pools(nsA__a=(None, 2, 1), nsB__b=(None, 2, 1))  # fixed total 4

    # A wants down to 1 (total 3 < floor): rejected standalone, intent cached.
    res_a = _mediate(orch, "nsA/a", _targets(decode=1), pools)
    assert not res_a.approved

    # B wants up to 3: pairs with A's pending scale-down. Combined 1+3 = 4.
    res_b = _mediate(orch, "nsB/b", _targets(decode=3), pools)
    assert res_b.approved
    assert len(res_b.selected_partners) == 1
    partner = res_b.selected_partners[0]
    assert (partner.participant_id, partner.sub_type, partner.applied_desired) == (
        "nsA/a",
        "decode",
        1,
    )


def test_intent_cache_ttl_expiry_blocks_pairing():
    clock = {"t": 1000.0}
    orch = _decider(
        max_total_gpus=4,
        min_total_gpus=4,
        intent_cache_ttl_seconds=360.0,
        now=lambda: clock["t"],
    )
    pools = _pools(nsA__a=(None, 2, 1), nsB__b=(None, 2, 1))

    _mediate(orch, "nsA/a", _targets(decode=1), pools)  # seed A's intent at t=1000

    clock["t"] = 1000.0 + 361.0  # A's intent now stale
    res_b = _mediate(orch, "nsB/b", _targets(decode=3), pools)
    assert not res_b.approved
    assert res_b.selected_partners == []


def test_total_gpus():
    orch = _decider()
    pools = _pools(ns__a=(2, 3, 2))  # (2+3) replicas * 2 GPU = 10
    assert orch.total_gpus(pools) == 10
    assert orch.total_gpus(pools, {("ns/a", "prefill"): 0}) == 6


def test_update_intent_cache_skips_unknown_pool():
    orch = _decider(max_total_gpus=4)
    pools = _pools(ns__a=(1, None, 1))  # decode pool absent
    orch.update_intent_cache("ns/a", _targets(decode=2), pools["ns/a"])
    assert "ns/a/decode" not in orch._intent_cache


# ---------------------------------------------------------------------------- #
# observe -> decide -> actuate (submit) over an in-memory capacity manager     #
# ---------------------------------------------------------------------------- #


class FakeCapacityManager(CapacityManager):
    """In-memory backend: pool state is a mutable dict, actuation mutates it."""

    def __init__(self):
        # participant_id -> {sub_type: [current_replicas, gpu_per_replica]}
        self.pools: dict[str, dict[str, list]] = {}
        self.refresh_calls: list[str] = []

    def add(self, key, prefill=None, decode=None, gpu=1):
        entry = {}
        if prefill is not None:
            entry["prefill"] = [prefill, gpu]
        if decode is not None:
            entry["decode"] = [decode, gpu]
        self.pools[key] = entry

    def ensure_participant(
        self, participant_id, caller_name, namespace, deployment_name
    ):
        self.pools.setdefault(participant_id, {})

    def participant_exists(self, participant_id):
        return participant_id in self.pools

    def observe(self, require_complete=False):
        return {
            key: {st: PoolSpec(st, r, g) for st, (r, g) in pools.items()}
            for key, pools in self.pools.items()
        }

    def refresh_participant(self, participant_id):
        self.refresh_calls.append(participant_id)

    async def scale(self, participant_id, targets, blocking):
        for t in targets:
            st = t.sub_component_type.value
            if st in self.pools[participant_id]:
                self.pools[participant_id][st][0] = t.desired_replicas
        return ReplicaUpdateResult(accepted=True, applied=True)

    async def wait_for_ready(self, participant_id):
        return None

    def supports_split_readiness(self):
        return True

    def current_replicas(self, participant_id):
        return {st: r for st, (r, _g) in self.pools[participant_id].items()}


def _orch(capacity_manager, *, max_gpus=-1, min_gpus=-1, managed=None, use_lock=True):
    return Orchestrator(
        capacity_manager=capacity_manager,
        managed_deployments=managed,
        max_total_gpus=max_gpus,
        min_total_gpus=min_gpus,
        use_lock=use_lock,
    )


async def _submit(orch, key, targets):
    return await orch.submit(
        key, targets, blocking=False, deployment_name=key, caller_name=key
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("use_lock", [True, False])
async def test_submit_approves_and_actuates(use_lock):
    cm = FakeCapacityManager()
    cm.add("default/a", prefill=1, decode=1)
    orch = _orch(cm, use_lock=use_lock)

    out = await _submit(orch, "default/a", _targets(prefill=3))

    assert out.status == ScaleStatus.SUCCESS
    assert cm.pools["default/a"]["prefill"][0] == 3  # actuated
    assert out.current_replicas == {"prefill": 3, "decode": 1}
    assert cm.refresh_calls == ["default/a"]


@pytest.mark.asyncio
@pytest.mark.parametrize("max_gpus", [-1, 2])
async def test_submit_rejects_empty_targets_before_backend_dispatch(max_gpus):
    cm = FakeCapacityManager()
    cm.add("default/a", prefill=1, decode=1)
    orch = _orch(cm, max_gpus=max_gpus)

    with pytest.raises(EmptyTargetReplicasError):
        await _submit(orch, "default/a", [])


@pytest.mark.asyncio
async def test_budget_submit_rejects_duplicate_resolved_pool_targets():
    cm = FakeCapacityManager()
    cm.add("default/a", decode=2)
    orch = _orch(cm, max_gpus=2)
    targets = [
        TargetReplica(sub_component_type=SubComponentType.DECODE, desired_replicas=10),
        TargetReplica(sub_component_type=SubComponentType.DECODE, desired_replicas=2),
    ]

    with pytest.raises(ValueError, match="duplicate replica target"):
        await _submit(orch, "default/a", targets)

    assert cm.pools["default/a"]["decode"][0] == 2


@pytest.mark.asyncio
async def test_budget_submit_settles_equal_target_before_increase():
    class RecordingCapacityManager(FakeCapacityManager):
        def __init__(self):
            super().__init__()
            self.scale_targets = []
            self.waits = []

        async def scale(self, participant_id, targets, blocking):
            self.scale_targets.append(
                [
                    (target.sub_component_type.value, target.desired_replicas)
                    for target in targets
                ]
            )
            return await super().scale(participant_id, targets, blocking)

        async def wait_for_ready(self, participant_id):
            self.waits.append(participant_id)

    cm = RecordingCapacityManager()
    cm.add("default/a", prefill=1, decode=1)
    orch = _orch(cm, max_gpus=3)

    result = await _submit(orch, "default/a", _targets(prefill=2, decode=1))

    assert result.status == ScaleStatus.SUCCESS
    assert cm.scale_targets == [[("decode", 1)], [("prefill", 2)]]
    assert cm.waits == ["default/a", "default/a"]


@pytest.mark.asyncio
async def test_cancelled_request_waiting_for_participant_reservation_never_scales():
    cm = FakeCapacityManager()
    cm.add("default/a", prefill=1, decode=1)
    orch = _orch(cm, max_gpus=10)
    reservation = orch._participant_mutation_locks.setdefault(
        "default/a", asyncio.Lock()
    )
    await reservation.acquire()

    scaling = asyncio.create_task(_submit(orch, "default/a", _targets(prefill=2)))
    try:
        intent = None
        for _ in range(100):
            intent = orch._intent_cache.get("default/a/prefill")
            if intent is not None:
                break
            await asyncio.sleep(0)
        assert intent is not None

        scaling.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(scaling, 1)
    finally:
        reservation.release()

    assert cm.pools["default/a"]["prefill"][0] == 1
    assert "default/a/prefill" not in orch._intent_cache


@pytest.mark.asyncio
async def test_submit_preserves_legacy_blocking_scale_contract():
    class LegacyCapacityManager(FakeCapacityManager):
        def __init__(self):
            super().__init__()
            self.blocking = None

        async def scale(self, participant_id, targets, blocking):
            self.blocking = blocking
            await super().scale(participant_id, targets, blocking)
            return None

        def supports_split_readiness(self):
            return False

        async def wait_for_ready(self, participant_id):
            raise AssertionError("legacy backend owns its blocking wait")

    cm = LegacyCapacityManager()
    cm.add("default/a", prefill=1, decode=1)
    outcome = await _orch(cm).submit(
        "default/a",
        _targets(prefill=2),
        blocking=True,
        deployment_name="default/a",
        caller_name="default/a",
    )

    assert outcome.status == ScaleStatus.SUCCESS
    assert cm.blocking is True


@pytest.mark.asyncio
async def test_submit_preserves_duck_typed_capacity_manager_contract():
    class DuckCapacityManager:
        def __init__(self):
            self.replicas = 1

        def remember_roles(self, participant_id, targets):
            return None

        def participant_exists(self, participant_id):
            return participant_id == "default/a"

        async def scale(self, participant_id, targets, blocking):
            self.replicas = targets[0].desired_replicas
            return None

        def current_replicas(self, participant_id):
            return {"decode": self.replicas}

    cm = DuckCapacityManager()
    result = await _submit(_orch(cm), "default/a", _targets(decode=2))

    assert result.status == ScaleStatus.SUCCESS
    assert result.current_replicas == {"decode": 2}


@pytest.mark.asyncio
async def test_conflict_retry_does_not_restart_after_participant_lock_wait():
    class ConflictError(Exception):
        pass

    class RetryingCapacityManager(FakeCapacityManager):
        async def run_with_conflict_retry(self, operation):
            try:
                return await operation()
            except ConflictError:
                return await operation()

    cm = RetryingCapacityManager()
    orch = _orch(cm)
    participant_lock = asyncio.Lock()
    await participant_lock.acquire()
    attempts = 0

    async def operation(*_args, **_kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ConflictError()
        raise _ParticipantMutationBusy(participant_lock)

    orch._observe_decide_apply = operation
    try:
        with pytest.raises(RuntimeError, match="participant set changed"):
            await _submit(orch, "default/a", _targets(decode=2))
    finally:
        participant_lock.release()

    assert attempts == 2


@pytest.mark.asyncio
async def test_conflict_retry_reobserves_before_reapplying_budget_decision():
    class ConflictError(Exception):
        pass

    class RetryingCapacityManager(FakeCapacityManager):
        def __init__(self):
            super().__init__()
            self.observe_calls = 0
            self.scale_calls = 0

        def observe(self, require_complete=False):
            self.observe_calls += 1
            return super().observe(require_complete)

        async def scale(self, participant_id, targets, blocking):
            self.scale_calls += 1
            if self.scale_calls == 1:
                self.pools["default/a"]["decode"][0] = 9
                raise ConflictError()
            return await super().scale(participant_id, targets, blocking)

        async def run_with_conflict_retry(self, operation):
            try:
                return await operation()
            except ConflictError:
                return await operation()

    cm = RetryingCapacityManager()
    cm.add("default/a", decode=2)
    cm.add("default/b", decode=2)
    orch = _orch(cm, max_gpus=10)

    result = await _submit(orch, "default/b", _targets(decode=3))

    assert result.status == ScaleStatus.REJECTED
    assert cm.observe_calls == 2
    assert cm.scale_calls == 1
    assert cm.pools["default/b"]["decode"][0] == 2


@pytest.mark.asyncio
async def test_noop_before_conflict_retries_complete_budget_transaction():
    class ConflictError(Exception):
        pass

    class NoopThenConflictCapacityManager(FakeCapacityManager):
        def __init__(self):
            super().__init__()
            self.observe_calls = 0
            self.scale_attempts = []

        def observe(self, require_complete=False):
            self.observe_calls += 1
            return super().observe(require_complete)

        async def scale(self, participant_id, targets, blocking):
            self.scale_attempts.append(participant_id)
            if participant_id == "default/a":
                self.pools[participant_id]["decode"][0] = 1
                return ReplicaUpdateResult(accepted=True, applied=False)
            if self.scale_attempts.count("default/b") == 1:
                raise ConflictError()
            return await super().scale(participant_id, targets, blocking)

        async def run_with_conflict_retry(self, operation):
            try:
                return await operation()
            except ConflictError:
                return await operation()

    cm = NoopThenConflictCapacityManager()
    cm.add("default/a", decode=2)
    cm.add("default/b", decode=2)
    orch = _orch(cm, max_gpus=4, min_gpus=4)

    rejected = await _submit(orch, "default/a", _targets(decode=1))
    assert rejected.status == ScaleStatus.REJECTED

    result = await _submit(orch, "default/b", _targets(decode=3))

    assert result.status == ScaleStatus.SUCCESS
    assert cm.observe_calls == 3
    assert cm.scale_attempts == ["default/a", "default/b", "default/b"]
    assert cm.pools["default/b"]["decode"][0] == 3


@pytest.mark.asyncio
async def test_applied_partner_before_conflict_retries_complete_budget_transaction():
    class ConflictError(Exception):
        pass

    class AppliedThenConflictCapacityManager(FakeCapacityManager):
        def __init__(self):
            super().__init__()
            self.observe_calls = 0
            self.scale_attempts = []

        def observe(self, require_complete=False):
            self.observe_calls += 1
            return super().observe(require_complete)

        async def scale(self, participant_id, targets, blocking):
            self.scale_attempts.append(participant_id)
            if (
                participant_id == "default/b"
                and self.scale_attempts.count("default/b") == 1
            ):
                raise ConflictError()
            return await super().scale(participant_id, targets, blocking)

        def is_conflict(self, error):
            return isinstance(error, ConflictError)

        async def run_with_conflict_retry(self, operation):
            try:
                return await operation()
            except ConflictError:
                return await operation()

    cm = AppliedThenConflictCapacityManager()
    cm.add("default/a", decode=2)
    cm.add("default/b", decode=2)
    orch = _orch(cm, max_gpus=4, min_gpus=4)

    rejected = await _submit(orch, "default/a", _targets(decode=1))
    assert rejected.status == ScaleStatus.REJECTED

    result = await _submit(orch, "default/b", _targets(decode=3))

    assert result.status == ScaleStatus.SUCCESS
    assert cm.observe_calls == 3
    assert cm.scale_attempts == ["default/a", "default/b", "default/b"]
    assert cm.pools["default/a"]["decode"][0] == 1
    assert cm.pools["default/b"]["decode"][0] == 3


@pytest.mark.asyncio
async def test_retry_rejection_reports_write_committed_before_conflict():
    class ConflictError(Exception):
        pass

    class AppliedThenRejectedCapacityManager(FakeCapacityManager):
        def __init__(self):
            super().__init__()
            self.scale_attempts = []

        async def scale(self, participant_id, targets, blocking):
            self.scale_attempts.append(participant_id)
            if participant_id == "default/b":
                if self.scale_attempts.count("default/b") == 1:
                    raise ConflictError()
                return ReplicaUpdateResult(accepted=False, applied=False)
            return await super().scale(participant_id, targets, blocking)

        def is_conflict(self, error):
            return isinstance(error, ConflictError)

        async def run_with_conflict_retry(self, operation):
            try:
                return await operation()
            except ConflictError:
                return await operation()

    cm = AppliedThenRejectedCapacityManager()
    cm.add("default/a", decode=2)
    cm.add("default/b", decode=2)
    orch = _orch(cm, max_gpus=4, min_gpus=4)

    rejected = await _submit(orch, "default/a", _targets(decode=1))
    assert rejected.status == ScaleStatus.REJECTED

    result = await _submit(orch, "default/b", _targets(decode=3))

    assert result.status == ScaleStatus.ERROR
    assert "after applying ['default/a']" in result.message
    assert cm.scale_attempts == ["default/a", "default/b", "default/b"]
    assert cm.pools["default/a"]["decode"][0] == 1
    assert cm.pools["default/b"]["decode"][0] == 2


@pytest.mark.asyncio
async def test_submit_returns_post_write_readback():
    cm = FakeCapacityManager()
    cm.add("default/a", prefill=1, decode=1)
    orch = _orch(cm)

    result = await _submit(orch, "default/a", _targets(prefill=2))

    assert result.current_replicas == {"prefill": 2, "decode": 1}


@pytest.mark.asyncio
async def test_cancelled_admitted_conflict_finishes_retry_before_propagating():
    class ConflictError(Exception):
        pass

    class BlockingConflictCapacityManager(FakeCapacityManager):
        def __init__(self):
            super().__init__()
            self.observe_calls = 0
            self.scale_calls = 0
            self.scale_started = asyncio.Event()
            self.release_scale = asyncio.Event()

        def observe(self, require_complete=False):
            self.observe_calls += 1
            return super().observe(require_complete)

        async def scale(self, participant_id, targets, blocking):
            self.scale_calls += 1
            if self.scale_calls == 1:
                self.scale_started.set()
                await self.release_scale.wait()
                raise ConflictError()
            return await super().scale(participant_id, targets, blocking)

        async def run_with_conflict_retry(self, operation):
            try:
                return await operation()
            except ConflictError:
                return await operation()

    cm = BlockingConflictCapacityManager()
    cm.add("default/a", decode=1)
    orch = _orch(cm)

    scaling = asyncio.create_task(_submit(orch, "default/a", _targets(decode=2)))
    await cm.scale_started.wait()
    scaling.cancel()
    await asyncio.sleep(0)
    assert not scaling.done()

    cm.release_scale.set()
    with pytest.raises(asyncio.CancelledError):
        await scaling
    assert cm.observe_calls == 0
    assert cm.scale_calls == 2
    assert cm.pools["default/a"]["decode"][0] == 2


@pytest.mark.asyncio
async def test_cancelled_cross_participant_transfer_finishes_admitted_plan():
    class BlockingFirstScaleCapacityManager(FakeCapacityManager):
        def __init__(self):
            super().__init__()
            self.first_scale_started = asyncio.Event()
            self.release_first_scale = asyncio.Event()

        async def scale(self, participant_id, targets, blocking):
            if participant_id == "default/a":
                self.first_scale_started.set()
                await self.release_first_scale.wait()
            return await super().scale(participant_id, targets, blocking)

    cm = BlockingFirstScaleCapacityManager()
    cm.add("default/a", decode=2)
    cm.add("default/b", decode=2)
    orch = _orch(cm, max_gpus=4, min_gpus=4)

    rejected = await _submit(orch, "default/a", _targets(decode=1))
    assert rejected.status == ScaleStatus.REJECTED

    transfer = asyncio.create_task(
        orch.submit(
            "default/b",
            _targets(decode=3),
            blocking=True,
            deployment_name="default/b",
            caller_name="default/b",
        )
    )
    await cm.first_scale_started.wait()
    transfer.cancel()
    await asyncio.sleep(0)
    assert not transfer.done()

    cm.release_first_scale.set()
    with pytest.raises(asyncio.CancelledError):
        await transfer
    assert cm.pools["default/a"]["decode"][0] == 1
    assert cm.pools["default/b"]["decode"][0] == 3


@pytest.mark.asyncio
async def test_budget_transfer_settles_downscale_before_nonblocking_upscale():
    class BlockingDownscaleSettlementCapacityManager(FakeCapacityManager):
        def __init__(self):
            super().__init__()
            self.downscale_wait_started = asyncio.Event()
            self.release_downscale = asyncio.Event()

        async def wait_for_ready(self, participant_id):
            if participant_id == "default/a":
                self.downscale_wait_started.set()
                await self.release_downscale.wait()

    cm = BlockingDownscaleSettlementCapacityManager()
    cm.add("default/a", decode=2)
    cm.add("default/b", decode=2)
    orch = _orch(cm, max_gpus=4, min_gpus=4)

    rejected = await _submit(orch, "default/a", _targets(decode=1))
    assert rejected.status == ScaleStatus.REJECTED

    transfer = asyncio.create_task(
        orch.submit(
            "default/b",
            _targets(decode=3),
            blocking=False,
            deployment_name="default/b",
            caller_name="default/b",
        )
    )
    await cm.downscale_wait_started.wait()
    assert cm.pools["default/a"]["decode"][0] == 1
    assert cm.pools["default/b"]["decode"][0] == 2

    cm.release_downscale.set()
    result = await transfer
    assert result.status == ScaleStatus.SUCCESS
    assert cm.pools["default/b"]["decode"][0] == 3


@pytest.mark.asyncio
async def test_budget_same_participant_settles_reduction_before_increase():
    class BlockingReductionCapacityManager(FakeCapacityManager):
        def __init__(self):
            super().__init__()
            self.reduction_wait_started = asyncio.Event()
            self.release_reduction = asyncio.Event()
            self.wait_calls = 0

        async def wait_for_ready(self, participant_id):
            self.wait_calls += 1
            if self.wait_calls == 1:
                self.reduction_wait_started.set()
                await self.release_reduction.wait()

    cm = BlockingReductionCapacityManager()
    cm.add("default/a", prefill=1, decode=3)
    orch = _orch(cm, max_gpus=4, min_gpus=4)

    transfer = asyncio.create_task(
        orch.submit(
            "default/a",
            _targets(prefill=2, decode=2),
            blocking=False,
            deployment_name="default/a",
            caller_name="default/a",
        )
    )
    await cm.reduction_wait_started.wait()
    assert cm.pools["default/a"]["decode"][0] == 2
    assert cm.pools["default/a"]["prefill"][0] == 1

    cm.release_reduction.set()
    result = await transfer
    assert result.status == ScaleStatus.SUCCESS
    assert cm.pools["default/a"]["prefill"][0] == 2
    assert cm.wait_calls == 2


@pytest.mark.asyncio
async def test_blocking_transfer_settles_accepted_noop_before_upscale():
    class NoopDownscaleCapacityManager(FakeCapacityManager):
        def __init__(self):
            super().__init__()
            self.downscale_wait_started = asyncio.Event()
            self.release_downscale = asyncio.Event()

        async def scale(self, participant_id, targets, blocking):
            if participant_id == "default/a":
                self.pools[participant_id]["decode"][0] = 1
                return ReplicaUpdateResult(accepted=True, applied=False)
            return await super().scale(participant_id, targets, blocking)

        async def wait_for_ready(self, participant_id):
            if participant_id == "default/a":
                self.downscale_wait_started.set()
                await self.release_downscale.wait()

    cm = NoopDownscaleCapacityManager()
    cm.add("default/a", decode=2)
    cm.add("default/b", decode=2)
    orch = _orch(cm, max_gpus=4, min_gpus=4)

    rejected = await _submit(orch, "default/a", _targets(decode=1))
    assert rejected.status == ScaleStatus.REJECTED

    transfer = asyncio.create_task(
        orch.submit(
            "default/b",
            _targets(decode=3),
            blocking=True,
            deployment_name="default/b",
            caller_name="default/b",
        )
    )
    await cm.downscale_wait_started.wait()
    assert cm.pools["default/b"]["decode"][0] == 2

    cm.release_downscale.set()
    result = await transfer
    assert result.status == ScaleStatus.SUCCESS
    assert cm.pools["default/b"]["decode"][0] == 3


@pytest.mark.asyncio
async def test_blocking_transfer_reports_partial_error_when_downscale_does_not_settle():
    class FailingDownscaleSettlementCapacityManager(FakeCapacityManager):
        async def wait_for_ready(self, participant_id):
            if participant_id == "default/a":
                raise RuntimeError("downscale did not settle")

    cm = FailingDownscaleSettlementCapacityManager()
    cm.add("default/a", decode=2)
    cm.add("default/b", decode=2)
    orch = _orch(cm, max_gpus=4, min_gpus=4)

    rejected = await _submit(orch, "default/a", _targets(decode=1))
    assert rejected.status == ScaleStatus.REJECTED

    result = await orch.submit(
        "default/b",
        _targets(decode=3),
        blocking=True,
        deployment_name="default/b",
        caller_name="default/b",
    )

    assert result.status == ScaleStatus.ERROR
    assert "after applying ['default/a']" in result.message
    assert "downscale did not settle" in result.message
    assert cm.pools["default/a"]["decode"][0] == 1
    assert cm.pools["default/b"]["decode"][0] == 2


@pytest.mark.asyncio
async def test_readiness_conflict_after_write_does_not_replay_transaction():
    class ReadinessConflictCapacityManager(FakeCapacityManager):
        def __init__(self):
            super().__init__()
            self.operation_attempts = 0
            self.scale_calls = 0

        async def scale(self, participant_id, targets, blocking):
            self.scale_calls += 1
            return await super().scale(participant_id, targets, blocking)

        async def wait_for_ready(self, participant_id):
            raise client.ApiException(status=409)

        async def run_with_conflict_retry(self, operation):
            self.operation_attempts += 1
            try:
                return await operation()
            except client.ApiException as error:
                if error.status != 409:
                    raise
                self.operation_attempts += 1
                return await operation()

    cm = ReadinessConflictCapacityManager()
    cm.add("default/a", decode=1)
    orch = _orch(cm)

    result = await orch.submit(
        "default/a",
        _targets(decode=2),
        blocking=True,
        deployment_name="default/a",
        caller_name="default/a",
    )

    assert result.status == ScaleStatus.ERROR
    assert "readiness wait failed" in result.message
    assert cm.operation_attempts == 1
    assert cm.scale_calls == 1


@pytest.mark.asyncio
async def test_skipped_scale_is_not_reported_or_waited_as_applied():
    class SkippingCapacityManager(FakeCapacityManager):
        async def scale(self, participant_id, targets, blocking):
            return ReplicaUpdateResult(accepted=False, applied=False)

        async def wait_for_ready(self, participant_id):
            raise AssertionError("skipped scale must not wait for readiness")

    cm = SkippingCapacityManager()
    cm.add("default/a", prefill=1, decode=1)
    orch = _orch(cm)

    result = await orch.submit(
        "default/a",
        _targets(prefill=2),
        blocking=True,
        deployment_name="default/a",
        caller_name="default/a",
    )

    assert result.status == ScaleStatus.REJECTED
    assert result.current_replicas == {}


@pytest.mark.asyncio
async def test_skipped_downscale_stops_dependent_upscale():
    class SkippingDownscaleCapacityManager(FakeCapacityManager):
        def __init__(self):
            super().__init__()
            self.scale_attempts = []

        async def scale(self, participant_id, targets, blocking):
            self.scale_attempts.append(participant_id)
            if participant_id == "default/a":
                return ReplicaUpdateResult(accepted=False, applied=False)
            return await super().scale(participant_id, targets, blocking)

    cm = SkippingDownscaleCapacityManager()
    cm.add("default/a", decode=2)
    cm.add("default/b", decode=2)
    orch = _orch(cm, max_gpus=4, min_gpus=4)

    rejected = await _submit(orch, "default/a", _targets(decode=1))
    assert rejected.status == ScaleStatus.REJECTED

    result = await orch.submit(
        "default/b",
        _targets(decode=3),
        blocking=True,
        deployment_name="default/b",
        caller_name="default/b",
    )

    assert result.status == ScaleStatus.REJECTED
    assert cm.scale_attempts == ["default/a"]
    assert cm.pools["default/a"]["decode"][0] == 2
    assert cm.pools["default/b"]["decode"][0] == 2


@pytest.mark.asyncio
async def test_cancelled_admitted_readiness_wait_keeps_scale_lock_until_settled():
    class BlockingReadyCapacityManager(FakeCapacityManager):
        def __init__(self):
            super().__init__()
            self.ready_wait_started = asyncio.Event()
            self.release_ready_wait = asyncio.Event()

        async def wait_for_ready(self, participant_id):
            self.ready_wait_started.set()
            await self.release_ready_wait.wait()

    cm = BlockingReadyCapacityManager()
    cm.add("default/a", prefill=1, decode=1)
    orch = _orch(cm)
    first = asyncio.create_task(
        orch.submit(
            "default/a",
            _targets(prefill=2),
            blocking=True,
            deployment_name="default/a",
            caller_name="default/a",
        )
    )
    await cm.ready_wait_started.wait()
    assert cm.pools["default/a"]["prefill"][0] == 2

    first.cancel()
    second = asyncio.create_task(_submit(orch, "default/a", _targets(prefill=3)))
    await asyncio.sleep(0)
    assert not first.done()
    assert not second.done()

    cm.release_ready_wait.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(first, 1)
    result = await asyncio.wait_for(second, 1)
    assert result.status == ScaleStatus.SUCCESS
    assert cm.pools["default/a"]["prefill"][0] == 3


@pytest.mark.asyncio
async def test_cancelled_observation_joins_worker_before_releasing_scale_lock():
    class BlockingFirstObserveCapacityManager(FakeCapacityManager):
        def __init__(self):
            super().__init__()
            self.observe_started = Event()
            self.release_observe = Event()
            self.observe_calls = 0

        def observe(self, require_complete=False):
            self.observe_calls += 1
            if self.observe_calls == 1:
                self.observe_started.set()
                self.release_observe.wait()
            return super().observe(require_complete)

    cm = BlockingFirstObserveCapacityManager()
    cm.add("default/a", prefill=1, decode=1)
    orch = _orch(cm, max_gpus=4)
    first = asyncio.create_task(_submit(orch, "default/a", _targets(prefill=2)))
    assert await asyncio.to_thread(cm.observe_started.wait, 1)

    first.cancel()
    second = asyncio.create_task(_submit(orch, "default/a", _targets(prefill=3)))
    try:
        await asyncio.sleep(0)
        assert not first.done()
        assert not second.done()
    finally:
        cm.release_observe.set()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(first, 1)
    result = await asyncio.wait_for(second, 1)
    assert result.status == ScaleStatus.SUCCESS


@pytest.mark.asyncio
async def test_post_write_readback_releases_scale_lock():
    class BlockingFirstReadbackCapacityManager(FakeCapacityManager):
        def __init__(self):
            super().__init__()
            self.readback_started = Event()
            self.release_readback = Event()
            self.readback_calls = 0
            self.scale_calls = 0

        async def scale(self, participant_id, targets, blocking):
            self.scale_calls += 1
            return await super().scale(participant_id, targets, blocking)

        def current_replicas(self, participant_id):
            self.readback_calls += 1
            if self.readback_calls == 1:
                self.readback_started.set()
                self.release_readback.wait()
            return super().current_replicas(participant_id)

    cm = BlockingFirstReadbackCapacityManager()
    cm.add("default/a", prefill=1, decode=1)
    cm.add("default/b", prefill=1, decode=1)
    orch = _orch(cm)
    first = asyncio.create_task(_submit(orch, "default/a", _targets(prefill=2)))
    assert await asyncio.to_thread(cm.readback_started.wait, 1)

    second = asyncio.create_task(_submit(orch, "default/b", _targets(prefill=3)))
    try:
        for _ in range(100):
            if cm.scale_calls == 2:
                break
            await asyncio.sleep(0)
        assert cm.scale_calls == 2
        assert (await asyncio.wait_for(second, 1)).status == ScaleStatus.SUCCESS
        assert not first.done()
    finally:
        cm.release_readback.set()

    assert (await asyncio.wait_for(first, 1)).status == ScaleStatus.SUCCESS
    assert cm.scale_calls == 2


@pytest.mark.asyncio
async def test_post_write_readback_releases_settled_partner():
    class BlockingRequesterReadbackCapacityManager(FakeCapacityManager):
        def __init__(self):
            super().__init__()
            self.readback_started = Event()
            self.release_readback = Event()
            self.blocked_requester = False

        def current_replicas(self, participant_id):
            if participant_id == "default/b" and not self.blocked_requester:
                self.blocked_requester = True
                self.readback_started.set()
                self.release_readback.wait()
            return super().current_replicas(participant_id)

    cm = BlockingRequesterReadbackCapacityManager()
    cm.add("default/a", decode=2)
    cm.add("default/b", decode=2)
    orch = _orch(cm, max_gpus=4, min_gpus=4)
    rejected = await _submit(orch, "default/a", _targets(decode=1))
    assert rejected.status == ScaleStatus.REJECTED

    paired = asyncio.create_task(_submit(orch, "default/b", _targets(decode=3)))
    assert await asyncio.to_thread(cm.readback_started.wait, 1)

    partner = asyncio.create_task(_submit(orch, "default/a", _targets(decode=1)))
    try:
        partner_result = await asyncio.wait_for(partner, 1)
        assert partner_result.status == ScaleStatus.SUCCESS
        assert not paired.done()
    finally:
        cm.release_readback.set()

    paired_result = await asyncio.wait_for(paired, 1)
    assert paired_result.status == ScaleStatus.SUCCESS


@pytest.mark.asyncio
async def test_post_write_readback_serializes_same_participant():
    class BlockingFirstReadbackCapacityManager(FakeCapacityManager):
        def __init__(self):
            super().__init__()
            self.readback_started = Event()
            self.release_readback = Event()
            self.readback_calls = 0
            self.scale_calls = 0
            self.observe_calls = 0

        def observe(self, require_complete=False):
            self.observe_calls += 1
            return super().observe(require_complete)

        async def scale(self, participant_id, targets, blocking):
            self.scale_calls += 1
            return await super().scale(participant_id, targets, blocking)

        def current_replicas(self, participant_id):
            self.readback_calls += 1
            if self.readback_calls == 1:
                self.readback_started.set()
                self.release_readback.wait()
            return super().current_replicas(participant_id)

    cm = BlockingFirstReadbackCapacityManager()
    cm.add("default/a", prefill=1, decode=1)
    orch = _orch(cm, max_gpus=10)
    first = asyncio.create_task(_submit(orch, "default/a", _targets(prefill=2)))
    assert await asyncio.to_thread(cm.readback_started.wait, 1)

    second = asyncio.create_task(_submit(orch, "default/a", _targets(prefill=3)))
    try:
        for _ in range(100):
            if cm.observe_calls == 2:
                break
            await asyncio.sleep(0)
        assert cm.observe_calls == 2
        assert cm.scale_calls == 1
        assert not second.done()
    finally:
        cm.release_readback.set()

    first_result = await asyncio.wait_for(first, 1)
    second_result = await asyncio.wait_for(second, 1)
    assert first_result.current_replicas["prefill"] == 2
    assert second_result.current_replicas["prefill"] == 3


@pytest.mark.asyncio
async def test_cancelled_pre_admission_request_stops_at_participant_wait():
    class BlockingFirstReadbackCapacityManager(FakeCapacityManager):
        def __init__(self):
            super().__init__()
            self.readback_started = Event()
            self.release_readback = Event()
            self.readback_calls = 0
            self.scale_calls = 0

        async def scale(self, participant_id, targets, blocking):
            self.scale_calls += 1
            return await super().scale(participant_id, targets, blocking)

        def current_replicas(self, participant_id):
            self.readback_calls += 1
            if self.readback_calls == 1:
                self.readback_started.set()
                self.release_readback.wait()
            return super().current_replicas(participant_id)

    cm = BlockingFirstReadbackCapacityManager()
    cm.add("default/a", prefill=1, decode=1)
    orch = _orch(cm, max_gpus=10)
    first = asyncio.create_task(_submit(orch, "default/a", _targets(prefill=2)))
    assert await asyncio.to_thread(cm.readback_started.wait, 1)

    second = asyncio.create_task(_submit(orch, "default/a", _targets(prefill=3)))
    try:
        for _ in range(100):
            intent = orch._intent_cache.get("default/a/prefill")
            if intent is not None and intent.last_desired == 3:
                break
            await asyncio.sleep(0)
        assert intent is not None and intent.last_desired == 3
        assert cm.scale_calls == 1

        second.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(second, 1)
    finally:
        cm.release_readback.set()

    assert (await asyncio.wait_for(first, 1)).status == ScaleStatus.SUCCESS
    assert cm.scale_calls == 1
    assert cm.pools["default/a"]["prefill"][0] == 2
    assert orch._intent_cache["default/a/prefill"].last_desired == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("use_lock", [True, False])
async def test_submit_ceiling_rejects_without_mutating(use_lock):
    cm = FakeCapacityManager()
    cm.add("default/a", prefill=1, decode=1, gpu=1)  # total 2 GPUs
    orch = _orch(cm, max_gpus=2, use_lock=use_lock)

    out = await _submit(orch, "default/a", _targets(prefill=3))  # would be total 4

    assert out.status == ScaleStatus.REJECTED
    assert "budget breach" in out.message.lower()
    assert cm.pools["default/a"]["prefill"][0] == 1  # NOT actuated


@pytest.mark.asyncio
async def test_submit_scale_down_always_allowed_under_ceiling():
    cm = FakeCapacityManager()
    cm.add("default/a", prefill=3, decode=3, gpu=1)
    orch = _orch(cm, max_gpus=6)

    out = await _submit(orch, "default/a", _targets(prefill=1))
    assert out.status == ScaleStatus.SUCCESS
    assert cm.pools["default/a"]["prefill"][0] == 1


def test_is_authorized_explicit_and_implicit():
    explicit = _orch(FakeCapacityManager(), managed=["default-a"])
    assert explicit.is_authorized("default-a")
    assert not explicit.is_authorized("default-b")

    implicit = _orch(FakeCapacityManager(), managed=None)
    assert implicit.is_authorized("anything")
