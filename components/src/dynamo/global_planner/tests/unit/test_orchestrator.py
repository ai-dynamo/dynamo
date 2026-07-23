# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Orchestrator — GPU-budget arbitration + observe/decide/actuate.

The decision tests exercise ``mediate`` directly (pure, no capacity manager
needed). The observe/decide/actuate tests drive ``submit`` over an in-memory
``FakeCapacityManager`` (subclassing the neutral ``CapacityManager`` base) — no
Kubernetes, no runtime — which previews how offline replay plugs a non-K8s
backend into the same orchestrator, including the ``use_lock`` toggle.
"""

import pytest

from dynamo.global_planner.capacity_manager import CapacityManager, PoolSpec
from dynamo.global_planner.orchestrator import Orchestrator
from dynamo.planner import SubComponentType, TargetReplica
from dynamo.planner.connectors.protocol import ScaleStatus

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

    async def scale(self, participant_id, targets, blocking):
        for t in targets:
            st = t.sub_component_type.value
            if st in self.pools[participant_id]:
                self.pools[participant_id][st][0] = t.desired_replicas

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
