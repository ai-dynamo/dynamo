# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the live retune state snapshot, write rules and Lease leadership."""

from __future__ import annotations

import dataclasses
import json
import math
from collections.abc import Callable
from typing import Optional

import pytest

from dynamo.planner.config.planner_config import (
    LIVE_RETUNE_K8S_TIMEOUT_SECONDS,
    PlannerConfig,
    live_retune_io_timeout_seconds,
    live_retune_lease_margin_seconds,
    live_retune_min_lease_seconds,
)
from dynamo.planner.retune.live.contracts import (
    LABEL_DGD,
    LABEL_DGD_UID,
    DgdRef,
    FrontendStartup,
    KubeConflict,
    KubeError,
)
from dynamo.planner.retune.live.state import (
    HISTORY_ANNOTATION,
    HISTORY_LIMIT,
    STATE_ANNOTATION,
    STATE_FORMAT_VERSION,
    LeaseLeadership,
    RetuneState,
    StateFormatError,
    StateSnapshot,
    StateStore,
    StateVersionError,
    append_history,
    claim,
    deploy,
    deployed_off,
    find_retune_leases,
    fingerprint,
    has_decision,
    leader_term,
    leadership_lease_name,
    maps_equal,
    merge_initials,
    micro_time,
    owner_reference,
    record_decision,
    reset,
    rollback_complete,
    rollback_start,
    state_lease_name,
)
from dynamo.planner.tests.live_retune_fakes import FakeRetuneKube

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]

DGD = DgdRef(name="graph", namespace="test-ns", uid="uid-1", dynamo_namespace="dyn")
STARTUP = {"aggregated": {"overlap_score_credit": 1.0, "router_temperature": 0.0}}
DEPLOYED = {"aggregated": {"overlap_score_credit": 0.5, "router_temperature": 0.2}}


class FakeClock:
    """Shared wall and monotonic clocks a test advances by hand."""

    def __init__(self, wall: float = 1_800_000_000.0, mono: float = 1000.0) -> None:
        self.wall = wall
        self.mono = mono

    def time(self) -> float:
        return self.wall

    def monotonic(self) -> float:
        return self.mono

    def advance(self, seconds: float) -> None:
        self.wall += seconds
        self.mono += seconds


def live_config(**overrides) -> PlannerConfig:
    """A planner config that passes the live retune configuration premises."""

    retune = {
        "interval_seconds": 21600,
        "grace_seconds": 120,
        "timeout_seconds": 20400,
        "router": {
            "overlap_score_credit": {"choices": [0.5, 1.0]},
            "prefill_load_scale": 1.0,
            "temperature": {"choices": [0.0, 0.2]},
        },
        "executor": "dgdr",
        "dgdr": {
            "claim_name": "request-traces",
            "model": "Qwen/Qwen3-0.6B",
            "sku": "h200_sxm",
            "engine": {
                "mode": "aggregated",
                "backend": "vllm",
                "workers": {
                    "aggregated": {
                        "parallelism": {
                            "preset": [
                                {
                                    "tensor": 1,
                                    "pipeline": 1,
                                    "attention_data": 1,
                                    "moe_tensor": 1,
                                    "moe_expert": 1,
                                }
                            ]
                        },
                        "scheduler": {"max_batched_tokens": 8192, "max_sequences": 256},
                    },
                    "prefill": None,
                    "decode": None,
                },
            },
        },
    }
    retune.update(overrides.pop("retune", {}))
    return PlannerConfig(
        mode="agg",
        environment="kubernetes",
        enable_load_scaling=False,
        enable_throughput_scaling=False,
        retune=retune,
        **overrides,
    )


def startup(source: str = "a") -> FrontendStartup:
    return FrontendStartup(
        router_mode="kv_router",
        router_config={},
        startup_maps=STARTUP,
        frontend_component="Frontend",
        replicas=1,
        replica_sync=False,
        fingerprint_source=source,
    )


def deployed_state(**changes) -> RetuneState:
    state = RetuneState(
        dgd_uid="uid-1",
        leader_term="1/me/1",
        fingerprint="fp",
        mode="active",
        deploy_seq=3,
        flat={"overlap_score_credit": 0.5},
        by_wt=DEPLOYED,
        initials={"r1": dict(STARTUP["aggregated"])},
        last_window_end_ms=1_000,
    )
    return dataclasses.replace(state, **changes)


# --------------------------------------------------------------- helpers


def test_names_term_owner_and_time() -> None:
    assert leadership_lease_name("graph") == "graph-planner-retune"
    assert state_lease_name("graph") == "graph-planner-retune-state"
    assert leader_term(2, "pod/abc", 7) == "2/pod/abc/7"
    assert owner_reference(DGD) == {
        "apiVersion": "nvidia.com/v1beta1",
        "kind": "DynamoGraphDeployment",
        "name": "graph",
        "uid": "uid-1",
        "blockOwnerDeletion": False,
    }
    assert micro_time(0.5) == "1970-01-01T00:00:00.500000Z"


def test_maps_equal() -> None:
    assert maps_equal({"a": 1, "b": {"c": 0.1 + 0.2}}, {"a": 1.0, "b": {"c": 0.3}})
    assert not maps_equal({"a": 1}, {"a": 1, "b": 2})
    assert not maps_equal({"a": {"c": 1.0}}, {"a": 1.0})
    assert not maps_equal({"a": 1.0}, {"a": 1.1})
    assert not maps_equal(None, {})
    assert maps_equal(None, None)


def test_fingerprint_tracks_what_the_point_was_chosen_under() -> None:
    config = live_config()
    base = fingerprint(config, startup("a"))
    assert base == fingerprint(live_config(), startup("a"))
    assert len(base) == 16
    # The parsed frontend startup is covered.
    assert fingerprint(config, startup("b")) != base
    assert fingerprint(config, None) != base
    # So are the objective and the scoring SLA ...
    assert (
        fingerprint(live_config(retune={"objective": "Throughput"}), startup()) != base
    )
    assert fingerprint(live_config(retune={"sla_ttft_ms": 99.0}), startup()) != base
    # ... but not advisory, which the state's mode records.
    assert fingerprint(live_config(advisory=True), startup()) == base


# ----------------------------------------------------------------- state


def test_state_round_trips_through_annotations() -> None:
    state = append_history(deployed_state(rollback_pending=True), {"decision_id": "d1"})
    annotations = state.to_annotations()
    assert set(annotations) == {STATE_ANNOTATION, HISTORY_ANNOTATION}
    body = json.loads(annotations[STATE_ANNOTATION])
    assert body["leader_term"] == "1/me/1" and body["deploy_seq"] == 3
    assert RetuneState.from_annotations(annotations) == state


def _body(**fields) -> str:
    return json.dumps({"version": STATE_FORMAT_VERSION, **fields})


def test_state_parsing_edge_cases() -> None:
    assert RetuneState.from_annotations(None) is None
    assert RetuneState.from_annotations({"other": "x"}) is None
    claimed = RetuneState.from_annotations(
        {STATE_ANNOTATION: _body(dgd_uid="u", leader_term="t")}
    )
    assert claimed == RetuneState(dgd_uid="u", leader_term="t")
    assert not claimed.has_deployment
    for bad in (
        "{",
        "[]",
        json.dumps({"dgd_uid": "u"}),  # no version
        _body(leader_term="t"),
        _body(dgd_uid="u", mode="bogus"),
        _body(dgd_uid="u", flat={"a": "x"}),
        _body(dgd_uid="u", deploy_seq="3"),
        # json.loads accepts NaN and Infinity; they are not numbers here.
        _body(dgd_uid="u", flat={"a": math.nan}),
        _body(dgd_uid="u", last_window_end_ms=math.nan),
        _body(dgd_uid="u", last_window_end_ms=math.inf),
        _body(dgd_uid="u", updated_ms="x"),
        _body(dgd_uid="u", updated_ms=[1]),
        _body(dgd_uid="u", fingerprint=math.nan),
    ):
        with pytest.raises(StateFormatError):
            RetuneState.from_annotations({STATE_ANNOTATION: bad})
    # A damaged history never blocks the state.
    parsed = RetuneState.from_annotations(
        {STATE_ANNOTATION: _body(dgd_uid="u"), HISTORY_ANNOTATION: "not json"}
    )
    assert parsed is not None and parsed.history == ()


def test_unwritable_history_entries_are_dropped_on_read() -> None:
    # json.loads accepts NaN and Infinity, canonical_json does not: one such
    # entry kept would make every later state write raise.
    history = [
        {"decision_id": "ok", "best": 1.5},
        {"decision_id": "nan", "best": math.nan},
        {"decision_id": "inf", "by_wt": {"aggregated": {"t": math.inf}}},
        "not an object",
    ]
    parsed = RetuneState.from_annotations(
        {
            STATE_ANNOTATION: _body(dgd_uid="u"),
            HISTORY_ANNOTATION: json.dumps(history),
        }
    )
    assert parsed is not None
    assert [entry["decision_id"] for entry in parsed.history] == ["ok"]
    written = append_history(parsed, {"decision_id": "next"}).to_annotations()
    assert json.loads(written[HISTORY_ANNOTATION])[0] == history[0]


def test_another_format_version_is_rejected_not_corrupt() -> None:
    # Checked first: a newer planner's state is never judged by this
    # planner's field rules (which would make it corrupt, then overwritten).
    newer = json.dumps(
        {"version": STATE_FORMAT_VERSION + 1, "dgd_uid": "u", "flat": {"a": "x"}}
    )
    with pytest.raises(StateVersionError):
        RetuneState.from_annotations({STATE_ANNOTATION: newer})
    assert not issubclass(StateVersionError, StateFormatError)

    kube = FakeRetuneKube()
    store = StateStore(kube, DGD)
    kube.create_lease(
        {
            "metadata": {"name": store.name, "annotations": {STATE_ANNOTATION: newer}},
            "spec": {},
        }
    )
    with pytest.raises(StateVersionError):
        store.read()


def test_find_retune_leases() -> None:
    kube = FakeRetuneKube()
    assert find_retune_leases(kube, "graph", timeout_s=1.0) == ()
    kube.create_lease({"metadata": {"name": state_lease_name("graph")}, "spec": {}})
    assert find_retune_leases(kube, "graph", timeout_s=1.0) == (
        state_lease_name("graph"),
    )


def test_deployed_off() -> None:
    assert not deployed_off(None, STARTUP)
    assert not deployed_off(RetuneState(dgd_uid="u"), STARTUP)
    assert deployed_off(deployed_state(), STARTUP)
    assert not deployed_off(deployed_state(by_wt=STARTUP), STARTUP)
    assert deployed_off(deployed_state(by_wt=STARTUP, rollback_pending=True), STARTUP)
    # Unknown startup: an active record may have deployed something.
    assert deployed_off(deployed_state(by_wt=STARTUP), None)
    # An advisory record only observed: its by_wt (an older startup) is no
    # deployment, unless a rollback is still pending.
    assert not deployed_off(deployed_state(mode="advisory"), STARTUP)
    assert not deployed_off(deployed_state(mode="advisory", by_wt=STARTUP), None)
    assert deployed_off(deployed_state(mode="advisory", rollback_pending=True), STARTUP)


# ----------------------------------------------------------- write rules


def test_claim_changes_only_the_term() -> None:
    fresh = claim(None, dgd_uid="uid-1", term="t2")
    assert fresh == RetuneState(dgd_uid="uid-1", leader_term="t2")
    state = deployed_state()
    claimed = claim(state, dgd_uid="ignored", term="t9")
    assert claimed == dataclasses.replace(state, leader_term="t9")


def test_deploy_rollback_and_reset_replace_the_deployment() -> None:
    state = deployed_state(rollback_pending=True)
    applied = deploy(
        state,
        fingerprint="fp2",
        mode="active",
        flat={"k": 1.0},
        by_wt={"w": {"k": 1.0}},
    )
    assert (applied.fingerprint, applied.mode, applied.flat, applied.by_wt) == (
        "fp2",
        "active",
        {"k": 1.0},
        {"w": {"k": 1.0}},
    )
    assert applied.rollback_pending is False and applied.deploy_seq == 4
    assert applied.initials == state.initials and applied.history == state.history

    started = rollback_start(
        deployed_state(), fingerprint="fp", mode="advisory", flat={}, by_wt=STARTUP
    )
    assert started.rollback_pending and started.deploy_seq == 4
    assert started.by_wt == STARTUP and started.mode == "advisory"

    for pending in (True, False):
        reset_state = reset(
            deployed_state(),
            fingerprint="fp3",
            mode="active",
            flat={},
            by_wt=STARTUP,
            rollback_pending=pending,
        )
        assert reset_state.rollback_pending is pending
        assert reset_state.deploy_seq == 4 and reset_state.fingerprint == "fp3"


def test_rollback_complete_is_conditional_on_deploy_seq() -> None:
    pending = deployed_state(rollback_pending=True, by_wt=STARTUP)
    entry = {"decision_id": "rolled_back-3", "outcome": "rolled_back"}
    done = rollback_complete(pending, expected_deploy_seq=3, entry=entry)
    assert done is not None
    # Only the flag flips (plus the history record).
    assert dataclasses.replace(done, history=pending.history) == dataclasses.replace(
        pending, rollback_pending=False
    )
    assert done.history[-1] == entry
    # An applied (deploy_seq moved) or an earlier completion: discard.
    assert rollback_complete(pending, expected_deploy_seq=2) is None
    assert rollback_complete(done, expected_deploy_seq=3) is None


def test_merge_initials_merges_and_prunes() -> None:
    state = deployed_state(initials={"r1": {"a": 1.0}, "r2": {"a": 2.0}})
    merged = merge_initials(state, {"r3": {"a": 3}}, keep={"r1", "r3"})
    assert merged.initials == {"r1": {"a": 1.0}, "r3": {"a": 3.0}}
    assert merge_initials(state, {}).initials == state.initials


def test_history_is_deduplicated_and_bounded() -> None:
    state = deployed_state(last_window_end_ms=None)
    for index in range(HISTORY_LIMIT + 5):
        state = record_decision(
            state, {"decision_id": f"d{index}", "outcome": "kept"}, 1000 + index
        )
    assert len(state.history) == HISTORY_LIMIT
    assert state.history[-1]["decision_id"] == f"d{HISTORY_LIMIT + 4}"
    assert state.last_window_end_ms == 1000 + HISTORY_LIMIT + 4
    # Same id again: replaced in place at the end; the window never moves back.
    again = record_decision(state, {"decision_id": "d10", "outcome": "failed"}, 5)
    assert again.last_window_end_ms == state.last_window_end_ms
    assert [item["decision_id"] for item in again.history].count("d10") == 1
    assert again.history[-1] == {"decision_id": "d10", "outcome": "failed"}
    assert has_decision(again, "d10") and not has_decision(again, "nope")
    assert record_decision(state, {"decision_id": "x"}, None).last_window_end_ms == (
        state.last_window_end_ms
    )


# ----------------------------------------------------------------- store


def test_store_creates_reads_and_conditionally_replaces() -> None:
    clock = FakeClock()
    kube = FakeRetuneKube(now=clock.time)
    store = StateStore(kube, DGD, clock=clock.time)
    empty = store.read()
    assert empty == StateSnapshot(state=None, resource_version=None, lease=None)

    created = store.write(RetuneState(dgd_uid="uid-1", leader_term="t1"), empty)
    lease = kube.leases[store.name]
    assert lease["metadata"]["ownerReferences"] == [owner_reference(DGD)]
    assert lease["metadata"]["labels"] == {LABEL_DGD: "graph", LABEL_DGD_UID: "uid-1"}
    assert created.state.updated_ms == int(clock.wall * 1000)

    read = store.read()
    assert (
        read.state == created.state
        and read.resource_version == created.resource_version
    )

    updated = store.write(dataclasses.replace(read.state, leader_term="t2"), read)
    assert store.read().state.leader_term == "t2"
    # The first snapshot is stale now.
    with pytest.raises(KubeConflict):
        store.write(dataclasses.replace(read.state, leader_term="t3"), read)
    # A create that loses the race is a conflict too.
    with pytest.raises(KubeConflict):
        store.write(RetuneState(dgd_uid="uid-1"), empty)
    assert store.read().state.leader_term == "t2"
    assert updated.resource_version != read.resource_version

    # The delete is conditional too: a stale snapshot leaves the Lease.
    with pytest.raises(KubeConflict):
        store.delete(read)
    store.delete(store.read())
    assert store.read().lease is None


def test_store_reports_a_corrupt_snapshot() -> None:
    kube = FakeRetuneKube()
    store = StateStore(kube, DGD)
    kube.create_lease(
        {
            "metadata": {
                "name": store.name,
                "annotations": {STATE_ANNOTATION: "{broken"},
            },
            "spec": {},
        }
    )
    snapshot = store.read()
    assert snapshot.corrupt and snapshot.state is None and snapshot.lease is not None
    # The claim overwrites it, conditional on its resourceVersion.
    written = store.write(RetuneState(dgd_uid="uid-1", leader_term="t"), snapshot)
    assert written.state.leader_term == "t" and not store.read().corrupt


# ------------------------------------------------------------- leadership


def leadership(kube, clock, identity="a/1", **kwargs) -> LeaseLeadership:
    return LeaseLeadership(
        kube,
        DGD,
        identity=identity,
        lease_duration_s=60.0,
        margin_s=15.0,
        monotonic=clock.monotonic,
        clock=clock.time,
        **kwargs,
    )


def holder(kube) -> str | None:
    return kube.leases[leadership_lease_name("graph")]["spec"].get("holderIdentity")


def test_leadership_create_renew_and_expire_locally() -> None:
    clock = FakeClock()
    kube = FakeRetuneKube()
    a = leadership(kube, clock)
    assert not a.is_leader()
    assert a.step() == "acquired"
    assert a.is_leader() and a.transitions == 0
    lease = kube.leases[a.name]
    assert lease["spec"]["leaseDurationSeconds"] == 60
    assert lease["metadata"]["ownerReferences"] == [owner_reference(DGD)]
    assert a.leader_until == clock.mono + 45.0

    clock.advance(20)
    assert a.step() == "renewed"
    assert a.leader_until == clock.mono + 45.0
    # What a conditional delete of the Lease needs.
    assert a.resource_version == kube.leases[a.name]["metadata"]["resourceVersion"]
    # Renewal failures keep leadership only until leader_until.
    kube.fail("get_lease", times=10)
    clock.advance(20)
    assert a.step() == "error" and a.is_leader()
    clock.advance(25)
    assert a.step() == "error" and not a.is_leader()
    kube.clear_faults()
    # Its own lease again after losing track: acquired (a new term).
    assert a.step() == "acquired" and a.is_leader()


def test_leadership_takes_over_only_after_an_unchanged_lease_duration() -> None:
    clock = FakeClock()
    kube = FakeRetuneKube()
    a = leadership(kube, clock, identity="a/1")
    b = leadership(kube, clock, identity="b/1")
    assert a.step() == "acquired"
    assert b.step() == "standby"
    # a keeps renewing: b never takes over.
    for _ in range(6):
        clock.advance(20)
        assert a.step() == "renewed"
        assert b.step() == "standby"
    # a dies (SIGKILL) after its last renewal, which b saw: b takes over
    # once it has watched the lease unchanged for 60s.
    clock.advance(30)
    assert b.step() == "standby"
    clock.advance(29)
    assert b.step() == "standby"
    assert not a.is_leader()  # a's local leadership expired long before
    clock.advance(1)
    assert b.step() == "acquired"
    assert b.is_leader() and b.transitions == 1 and holder(kube) == "b/1"
    # a comes back: it has lost, then stands by.
    assert a.step() == "lost" and not a.is_leader()
    assert a.step() == "standby"


def test_leadership_never_overlaps() -> None:
    """a stops renewing at an arbitrary point; b never leads while a does."""

    for stop_after in range(0, 60, 7):
        clock = FakeClock()
        kube = FakeRetuneKube()
        a = leadership(kube, clock, identity="a/1")
        b = leadership(kube, clock, identity="b/1")
        a.step()
        b.step()
        for second in range(0, 400):
            clock.advance(1)
            if second < stop_after and second % 20 == 0:
                a.step()
            if second % 7 == 0:
                b.step()
            assert not (a.is_leader() and b.is_leader())
        assert b.is_leader()


def test_leadership_lost_to_another_holder() -> None:
    clock = FakeClock()
    kube = FakeRetuneKube()
    a = leadership(kube, clock, identity="a/1")
    assert a.step() == "acquired"
    lease = kube.leases[a.name]
    lease["spec"]["holderIdentity"] = "intruder"
    kube.bump(lease)
    assert a.step() == "lost"
    assert not a.is_leader()


def test_empty_holder_is_taken_over_at_once_and_release_is_conditional() -> None:
    clock = FakeClock()
    kube = FakeRetuneKube()
    a = leadership(kube, clock, identity="a/1")
    b = leadership(kube, clock, identity="b/1")
    assert a.step() == "acquired"
    assert b.step() == "standby"
    assert a.release()
    assert holder(kube) is None and not a.is_leader()
    # Released: b takes over without waiting a lease duration.
    assert b.step() == "acquired" and b.transitions == 1
    # a no longer holds it: release is a no-op that never clears b.
    assert not a.release()
    assert holder(kube) == "b/1"


def test_a_step_after_release_never_leads_again() -> None:
    clock = FakeClock()
    kube = FakeRetuneKube()
    a = leadership(kube, clock, identity="a/1")
    assert a.step() == "acquired"
    assert a.release() and holder(kube) is None
    # A step still running when the Lease was released stands by instead of
    # taking the empty holder over, or creating the Lease again.
    assert a.step() == "standby"
    assert holder(kube) is None and not a.is_leader()
    del kube.leases[a.name]
    assert a.step() == "standby"
    assert a.name not in kube.leases and not a.is_leader()


def test_leadership_conflicts_and_errors() -> None:
    clock = FakeClock()
    kube = FakeRetuneKube()
    a = leadership(kube, clock, identity="a/1")
    # Creating the lease races another creator.
    kube.fail("create_lease", KubeConflict("raced"))
    assert a.step() == "standby"
    kube.fail("create_lease", KubeError("timeout"))
    assert a.step() == "error"
    assert a.step() == "acquired"
    clock.advance(20)
    kube.fail("replace_lease", KubeConflict("stale"))
    assert a.step() == "error" and a.is_leader()
    kube.fail("replace_lease", KubeError("timeout"))
    assert a.step() == "error" and a.is_leader()
    assert a.step() == "renewed"
    # Releasing tolerates API errors.
    kube.fail("get_lease", KubeError("down"))
    assert not a.release() and not a.is_leader()


def test_leadership_validates_arguments() -> None:
    kube = FakeRetuneKube()
    clock = FakeClock()
    with pytest.raises(ValueError):
        LeaseLeadership(kube, DGD, identity="")
    with pytest.raises(ValueError):
        LeaseLeadership(kube, DGD, identity="a", lease_duration_s=0)
    # No time left to lead in; the default margin is three request timeouts.
    for margin_s in (None, 15.0, 100.0, -1.0, math.nan):
        with pytest.raises(ValueError, match="margin_s"):
            LeaseLeadership(
                kube, DGD, identity="a", lease_duration_s=15.0, margin_s=margin_s
            )
    assert leadership(kube, clock).margin_s == 15.0
    short = LeaseLeadership(
        kube, DGD, identity="a", lease_duration_s=15.0, margin_s=10.0
    )
    assert short.margin_s == 10.0
    short.set_owner(dataclasses.replace(DGD, uid="uid-2"))
    assert short.step() == "acquired"
    owner = kube.leases[short.name]["metadata"]["ownerReferences"][0]
    assert owner["uid"] == "uid-2"


# The shortest lease the live supervisor accepts, with its request timeout.
K8S_TIMEOUT = LIVE_RETUNE_K8S_TIMEOUT_SECONDS
MIN_LEASE = live_retune_min_lease_seconds(K8S_TIMEOUT)


def minimum_leadership(kube, clock) -> LeaseLeadership:
    return LeaseLeadership(
        kube,
        DGD,
        identity="a/1",
        lease_duration_s=MIN_LEASE,
        margin_s=live_retune_lease_margin_seconds(K8S_TIMEOUT),
        timeout_s=K8S_TIMEOUT,
        monotonic=clock.monotonic,
        clock=clock.time,
    )


INTERVAL = MIN_LEASE / 3.0
BUDGET = live_retune_io_timeout_seconds(K8S_TIMEOUT)


def renewal_wait(a: LeaseLeadership) -> float:
    """The supervisor's lease loop wait after a successful renewal."""

    return a.renewal_wait_s(interval_s=INTERVAL, budget_s=BUDGET)


def test_leadership_renews_the_minimum_lease_in_one_term_until_it_lapses() -> None:
    clock = FakeClock()
    kube = FakeRetuneKube()
    a = minimum_leadership(kube, clock)
    assert renewal_wait(a) == INTERVAL
    assert a.step() == "acquired" and a.acquisitions == 1
    acquired = kube.leases[a.name]["spec"]["acquireTime"]
    # One I/O budget before leader_until, sooner than the renew interval.
    margin = live_retune_lease_margin_seconds(K8S_TIMEOUT)
    assert renewal_wait(a) == MIN_LEASE - margin - BUDGET < INTERVAL
    for _ in range(10):
        clock.advance(renewal_wait(a))
        assert a.leads_since(1)
        assert a.step() == "renewed"
    spec = kube.leases[a.name]["spec"]
    assert a.transitions == 0 and spec["leaseTransitions"] == 0
    assert spec["acquireTime"] == acquired
    # A failed attempt's floor wins over the cap, even past leader_until.
    clock.advance(a.leader_until - clock.mono - 1.0)
    assert a.renewal_wait_s(interval_s=INTERVAL, budget_s=BUDGET, floor_s=5.0) == 5.0
    clock.advance(5.0)
    assert not a.is_leader() and renewal_wait(a) == INTERVAL
    assert a.step() == "acquired" and a.acquisitions == 2
    assert a.is_leader() and not a.leads_since(1)


class SlowLeaseKube(FakeRetuneKube):
    """Every Lease call takes ``latency_s`` of ``clock``. A replace calls
    ``probe`` when its response is due, before the caller gets it."""

    def __init__(self, clock: FakeClock, latency_s: float) -> None:
        super().__init__(now=clock.time)
        self.clock = clock
        self.latency_s = latency_s
        self.probe: Optional[Callable[[], bool]] = None
        self.probes: list[bool] = []

    def get_lease(self, name: str, *, timeout_s: float = 10.0) -> Optional[dict]:
        self.clock.advance(self.latency_s)
        return super().get_lease(name, timeout_s=timeout_s)

    def create_lease(self, body: dict, *, timeout_s: float = 10.0) -> dict:
        self.clock.advance(self.latency_s)
        return super().create_lease(body, timeout_s=timeout_s)

    def replace_lease(self, name: str, body: dict, *, timeout_s: float = 10.0) -> dict:
        self.clock.advance(self.latency_s)
        if self.probe is not None:
            self.probes.append(self.probe())
        return super().replace_lease(name, body, timeout_s=timeout_s)


def test_leadership_never_lapses_while_slow_renewals_are_in_flight() -> None:
    # The GET and the replace together take just under the I/O budget that
    # bounds one renewal (the supervisor's backstop around step()); the
    # client's (connect, read) timeout alone does not cap a call at K8S_TIMEOUT.
    clock = FakeClock()
    kube = SlowLeaseKube(clock, latency_s=BUDGET / 2 - 0.01)
    a = minimum_leadership(kube, clock)
    kube.probe = a.is_leader
    assert a.step() == "acquired"
    for _ in range(10):
        clock.advance(renewal_wait(a))
        assert a.step() == "renewed"
    assert kube.probes == [True] * 10 and a.transitions == 0
