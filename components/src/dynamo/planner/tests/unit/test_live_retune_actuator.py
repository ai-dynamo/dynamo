# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the live retune router actuator and its HTTP client.

``FakePolicyClient`` only records the calls and answers them: a successful
write sets the router to what the actuator asked for (the PATCHed keys, or
its initial after a reset) and bumps the revision. Error statuses are
injected by the tests; the server's own rules (``router_policy.rs``) are not
modelled here.
"""

from __future__ import annotations

import json
import re
import socket
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Optional

import pytest
from aiohttp import test_utils, web

from dynamo.planner.retune.live.actuator import (
    DEFAULT_CLIENT_TIMEOUT_S,
    ROUTER_CONTROL_TIMEOUT_S,
    AiohttpRouterPolicyClient,
    RouterActuator,
)
from dynamo.planner.retune.live.contracts import (
    POD_LABEL_COMPONENT_TYPE,
    POD_LABEL_DGD_NAME,
    ActuationTarget,
    DgdRef,
    KubeError,
    PodInfo,
    PolicyCallResult,
    RouterInfo,
)

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]

DGD = DgdRef(name="dgd", namespace="ns", uid="dgd-uid", dynamo_namespace="dyn-ns")

# KvRouterConfig defaults resolved by policy_parameters("aggregated").
STARTUP = {
    "decode_active_request_weight": 0.0,
    "disk_cache_hit_weight": 0.25,
    "host_cache_hit_weight": 0.75,
    "overlap_score_credit": 1.0,
    "overlap_score_credit_decay": 0.0,
    "prefill_load_scale": 1.0,
    "router_temperature": 0.0,
    "shared_cache_multiplier": 0.0,
}
CHOSEN = {
    **STARTUP,
    "overlap_score_credit": 0.5,
    "prefill_load_scale": 2.0,
    "router_temperature": 0.2,
}
# policy_parameters("decode") of the defaults: plain decode has no cache input.
DECODE_STARTUP = {
    **STARTUP,
    "overlap_score_credit": 0.0,
    "disk_cache_hit_weight": 0.0,
    "host_cache_hit_weight": 0.0,
}


# ------------------------------------------------------------------ fakes


@dataclass
class FakeRouter:
    router_id: str
    initial: dict[str, float]
    worker_type: str = "aggregated"
    namespace: str = DGD.dynamo_namespace
    revision: int = 0
    current: Optional[dict[str, float]] = None
    runtime_config: bool = True  # False: a custom policy, configuration null

    def __post_init__(self) -> None:
        if self.current is None:
            self.current = dict(self.initial)

    def info(self) -> RouterInfo:
        return RouterInfo(
            router_id=self.router_id,
            namespace=self.namespace,
            worker_type=self.worker_type,
            revision=self.revision if self.runtime_config else 0,
            configuration=dict(self.current) if self.runtime_config else None,
            model="model",
        )


@dataclass
class FakeFrontend:
    routers: dict[str, FakeRouter] = field(default_factory=dict)
    # Statuses returned by the next GETs before the real answer.
    get_failures: list[int] = field(default_factory=list)
    # Routers GET leaves out (their control call timed out).
    timed_out: set[str] = field(default_factory=set)

    def add(self, router: FakeRouter) -> FakeRouter:
        self.routers[router.router_id] = router
        return router


class FakePolicyClient:
    """In-memory frontends keyed by pod name; records every call."""

    def __init__(self) -> None:
        self.frontends: dict[str, FakeFrontend] = {}
        self.calls: list[tuple[Any, ...]] = []
        # Statuses returned by the next writes (any router) instead of applying.
        self.write_failures: list[int] = []
        # Called before a write is answered: (op, pod name, router_id).
        self.before_write: Optional[Callable[[str, str, str], None]] = None

    def frontend(self, pod: str) -> FakeFrontend:
        return self.frontends.setdefault(pod, FakeFrontend())

    def writes(self) -> list[tuple[Any, ...]]:
        return [call for call in self.calls if call[0] in ("patch", "reset")]

    async def get(self, pod: PodInfo) -> PolicyCallResult:
        self.calls.append(("get", pod.name))
        frontend = self.frontends.get(pod.name)
        if frontend is None:
            return PolicyCallResult(status=0, error="connection refused")
        if frontend.get_failures:
            return PolicyCallResult(status=frontend.get_failures.pop(0))
        return PolicyCallResult(
            status=200,
            routers=tuple(
                router.info()
                for router in frontend.routers.values()
                if router.router_id not in frontend.timed_out
            ),
        )

    async def patch(
        self,
        pod: PodInfo,
        router_id: str,
        expected_revision: int,
        parameters: Mapping[str, float],
    ) -> PolicyCallResult:
        self.calls.append(
            ("patch", pod.name, router_id, expected_revision, dict(parameters))
        )
        return self._answer("patch", pod, router_id, parameters)

    async def reset(
        self, pod: PodInfo, router_id: str, expected_revision: int
    ) -> PolicyCallResult:
        self.calls.append(("reset", pod.name, router_id, expected_revision))
        return self._answer("reset", pod, router_id, None)

    def _answer(
        self,
        op: str,
        pod: PodInfo,
        router_id: str,
        parameters: Optional[Mapping[str, float]],
    ) -> PolicyCallResult:
        if self.before_write is not None:
            self.before_write(op, pod.name, router_id)
        if self.write_failures:
            return PolicyCallResult(status=self.write_failures.pop(0), error="injected")
        router = self.frontend(pod.name).routers.get(router_id)
        if router is None:
            return PolicyCallResult(status=404, error="router instance not found")
        if parameters is None:
            router.current = dict(router.initial)
        else:
            router.current = {**(router.current or {}), **parameters}
        router.revision += 1
        return PolicyCallResult(status=200, router=router.info())


class FakeKube:
    namespace = DGD.namespace

    def __init__(self, pods: Optional[list[PodInfo]] = None) -> None:
        self.pods: list[PodInfo] = list(pods or [])
        self.selectors: list[str] = []
        self.fail: Optional[Exception] = None

    def list_pods(
        self, label_selector: str, *, timeout_s: float = 10.0
    ) -> list[PodInfo]:
        self.selectors.append(label_selector)
        if self.fail is not None:
            raise self.fail
        return list(self.pods)


class RecordingMetrics:
    def __init__(self) -> None:
        self.converged: list[bool] = []
        self.patch_errors: list[str] = []

    def router_converged(self, converged: bool) -> None:
        self.converged.append(converged)

    def patch_error(self, code: str) -> None:
        self.patch_errors.append(code)

    def __getattr__(self, name: str) -> Callable[..., None]:
        return lambda *args, **kwargs: None


def pod(
    name: str,
    *,
    ready: bool = True,
    terminating: bool = False,
    ip: Optional[str] = "10.0.0.1",
    phase: Optional[str] = None,
    restarts: int = 0,
) -> PodInfo:
    return PodInfo(
        name=name,
        uid=f"{name}-uid",
        ip=ip,
        ready=ready,
        terminating=terminating,
        labels={POD_LABEL_DGD_NAME: DGD.name, POD_LABEL_COMPONENT_TYPE: "frontend"},
        admin_port=8002,
        phase=phase,
        restarts=restarts,
    )


def make_actuator(
    pods: list[PodInfo],
    *,
    startup_maps: Optional[dict[str, dict[str, float]]] = None,
) -> tuple[RouterActuator, FakeKube, FakePolicyClient, RecordingMetrics]:
    kube = FakeKube(pods)
    client = FakePolicyClient()
    metrics = RecordingMetrics()
    actuator = RouterActuator(
        kube, client, DGD, startup_maps or {"aggregated": dict(STARTUP)}, metrics
    )
    return actuator, kube, client, metrics


def apply(desired: Mapping[str, float], wt: str = "aggregated") -> ActuationTarget:
    return ActuationTarget(mode="apply", desired_by_wt={wt: dict(desired)})


OBSERVE = ActuationTarget(mode="observe")
ROLLBACK = ActuationTarget(mode="rollback", include_not_ready=True)


# ---------------------------------------------------------------- latching


async def test_observe_latches_initials_and_never_writes() -> None:
    actuator, kube, client, metrics = make_actuator([pod("fe-0")])
    client.frontend("fe-0").add(FakeRouter("r1", STARTUP))

    report = await actuator.reconcile(OBSERVE, initials={}, expected_frontends=1)

    assert kube.selectors == [
        f"{POD_LABEL_DGD_NAME}=dgd,{POD_LABEL_COMPONENT_TYPE}=frontend"
    ]
    assert client.writes() == []
    assert report.converged
    assert report.new_initials == {"r1": STARTUP}
    assert report.seen_router_ids == frozenset({"r1"})
    assert report.parity_ok is True
    assert report.parity_mismatches == ()
    assert metrics.converged == [True]

    # Persisted initials are no longer reported as new.
    report = await actuator.reconcile(
        OBSERVE, initials={"r1": STARTUP}, expected_frontends=1
    )
    assert report.new_initials == {}
    assert report.converged


async def test_a_pod_without_an_admin_port_is_admin_disabled_at_once() -> None:
    # Its env starts no admin listener the planner can read: P12 names the
    # admin listener in the first round instead of after 404 rounds.
    actuator, _, client, _ = make_actuator([replace(pod("fe-0"), admin_port=None)])
    client.frontend("fe-0").add(FakeRouter("r1", STARTUP))
    report = await actuator.reconcile(OBSERVE, initials={}, expected_frontends=1)
    assert report.admin_api_disabled


async def test_a_terminating_pod_stops_counting_as_admin_disabled() -> None:
    # A pod without an admin port that starts terminating gets no further
    # outcome. While it is still listed, its count must not keep blocking P12
    # once a replacement answers /router_policy.
    old = replace(pod("fe-0"), admin_port=None)
    actuator, kube, client, _ = make_actuator([old])
    report = await actuator.reconcile(OBSERVE, initials={}, expected_frontends=1)
    assert report.admin_api_disabled

    kube.pods = [replace(old, terminating=True), pod("fe-1", ip="10.0.0.2")]
    client.frontend("fe-1").add(FakeRouter("r1", STARTUP))
    report = await actuator.reconcile(OBSERVE, initials={}, expected_frontends=1)
    assert not report.admin_api_disabled
    assert report.routers_present


# ------------------------------------------------------------------- apply


async def test_apply_holds_a_router_until_its_initial_is_persisted() -> None:
    # A process stopped between a PATCH and the state write of the router's
    # initial would leave a router no rollback can restore: an advisory or
    # clean-up planner never resets a router whose initial it does not know.
    actuator, _, client, _ = make_actuator([pod("fe-0")])
    router = client.frontend("fe-0").add(FakeRouter("r1", STARTUP))

    for _ in range(2):  # the state write has not landed yet
        report = await actuator.reconcile(
            apply(CHOSEN), initials={}, expected_frontends=1
        )
        assert client.writes() == []
        assert report.new_initials == {"r1": STARTUP}
        assert not report.converged

    report = await actuator.reconcile(
        apply(CHOSEN), initials={"r1": STARTUP}, expected_frontends=1
    )
    assert [call[0] for call in client.writes()] == ["patch"]
    assert router.current == CHOSEN
    assert report.converged and report.new_initials == {}


async def test_apply_patches_only_differing_keys_with_explicit_values() -> None:
    actuator, _, client, _ = make_actuator([pod("fe-0")])
    router = client.frontend("fe-0").add(FakeRouter("r1", STARTUP))
    initials = {"r1": dict(STARTUP)}

    report = await actuator.reconcile(
        apply(CHOSEN), initials=initials, expected_frontends=1
    )

    assert client.writes() == [
        (
            "patch",
            "fe-0",
            "r1",
            0,
            {
                "overlap_score_credit": 0.5,
                "prefill_load_scale": 2.0,
                "router_temperature": 0.2,
            },
        )
    ]
    assert router.current == CHOSEN
    assert report.converged
    assert report.parity_ok is True

    # Converged: the next rounds write nothing.
    for _ in range(2):
        report = await actuator.reconcile(
            apply(CHOSEN), initials=initials, expected_frontends=1
        )
        assert report.converged
    assert len(client.writes()) == 1


async def test_apply_learns_an_unknown_initial_with_a_reset_first() -> None:
    actuator, _, client, _ = make_actuator([pod("fe-0")])
    tampered = {**STARTUP, "router_temperature": 0.9}
    router = client.frontend("fe-0").add(
        FakeRouter("r1", STARTUP, revision=2, current=tampered)
    )

    # A reset restores the router's initial and reports it; the PATCH waits
    # for the state to record it.
    report = await actuator.reconcile(apply(CHOSEN), initials={}, expected_frontends=1)
    assert client.writes() == [("reset", "fe-0", "r1", 2)]
    assert router.current == STARTUP
    assert report.new_initials == {"r1": STARTUP}
    assert report.parity_ok is True
    assert not report.converged

    report = await actuator.reconcile(
        apply(CHOSEN), initials=report.new_initials, expected_frontends=1
    )
    assert [call[0] for call in client.writes()] == ["reset", "patch"]
    assert router.current == CHOSEN
    assert report.converged


async def test_parity_mismatch_blocks_apply_and_never_loops() -> None:
    actuator, _, client, _ = make_actuator([pod("fe-0"), pod("fe-1", ip="10.0.0.2")])
    odd = {**STARTUP, "overlap_score_credit_decay": 0.5}
    client.frontend("fe-0").add(FakeRouter("good", STARTUP))
    client.frontend("fe-1").add(FakeRouter("odd", odd))
    initials = {"good": dict(STARTUP), "odd": dict(odd)}

    for _ in range(3):
        report = await actuator.reconcile(
            apply(CHOSEN), initials=initials, expected_frontends=2
        )
        assert report.parity_ok is False
        assert len(report.parity_mismatches) == 1
        assert report.parity_mismatches[0].startswith("fe-1/odd/aggregated:")
        assert "overlap_score_credit_decay initial=0.5 startup=0.0" in (
            report.parity_mismatches[0]
        )
        assert not report.converged
    # Only the router whose initial matches startup was written, once.
    assert [call[2] for call in client.writes()] == ["good"]
    assert client.frontend("fe-1").routers["odd"].current == odd


async def test_disagg_needs_both_worker_types() -> None:
    maps = {"prefill": dict(STARTUP), "decode": dict(DECODE_STARTUP)}
    actuator, _, client, _ = make_actuator([pod("fe-0")], startup_maps=maps)
    frontend = client.frontend("fe-0")
    prefill = frontend.add(FakeRouter("p", STARTUP, worker_type="prefill"))
    target = ActuationTarget(
        mode="apply",
        desired_by_wt={
            "prefill": dict(CHOSEN),
            "decode": {**DECODE_STARTUP, "router_temperature": 0.2},
        },
    )
    initials = {"p": dict(STARTUP), "d": dict(DECODE_STARTUP)}

    report = await actuator.reconcile(target, initials=initials, expected_frontends=1)
    assert not report.converged  # no decode router yet
    assert prefill.current == CHOSEN

    decode = frontend.add(FakeRouter("d", DECODE_STARTUP, worker_type="decode"))
    report = await actuator.reconcile(target, initials=initials, expected_frontends=1)
    assert decode.current == {**DECODE_STARTUP, "router_temperature": 0.2}
    assert report.converged
    assert report.parity_ok is True


async def test_only_the_dgd_namespace_and_planner_worker_types_are_managed() -> None:
    actuator, _, client, _ = make_actuator([pod("fe-0")])
    frontend = client.frontend("fe-0")
    ours = frontend.add(FakeRouter("ours", STARTUP))
    foreign = frontend.add(FakeRouter("foreign", STARTUP, namespace="other-ns"))
    encode = frontend.add(FakeRouter("encode", STARTUP, worker_type="encode"))

    report = await actuator.reconcile(
        apply(CHOSEN), initials={"ours": dict(STARTUP)}, expected_frontends=1
    )

    assert {call[2] for call in client.writes()} == {"ours"}
    assert ours.current == CHOSEN
    assert foreign.current == STARTUP and encode.current == STARTUP
    assert report.seen_router_ids == frozenset({"ours"})
    assert report.converged


# ---------------------------------------------------------------- rollback


async def test_rollback_resets_only_deviating_routers_and_never_by_revision() -> None:
    actuator, _, client, _ = make_actuator([pod("fe-0")])
    frontend = client.frontend("fe-0")
    # At its initial although its revision is not 0 (patched, then reset).
    at_initial = frontend.add(FakeRouter("a", STARTUP, revision=3))
    deviating = frontend.add(FakeRouter("b", STARTUP, revision=1, current=dict(CHOSEN)))
    initials = {"a": dict(STARTUP), "b": dict(STARTUP)}

    report = await actuator.reconcile(ROLLBACK, initials=initials, expected_frontends=1)

    assert client.writes() == [("reset", "fe-0", "b", 1)]
    assert deviating.current == STARTUP
    assert at_initial.revision == 3
    assert report.converged

    # The reset bumped b's revision; comparing configurations, not
    # revisions, means no further resets.
    for _ in range(3):
        report = await actuator.reconcile(
            ROLLBACK, initials=initials, expected_frontends=1
        )
        assert report.converged
    assert len(client.writes()) == 1


async def test_rollback_without_learning_leaves_unknown_routers_alone() -> None:
    actuator, _, client, _ = make_actuator([pod("fe-0")])
    frontend = client.frontend("fe-0")
    # Changed by someone else before any planner latched it.
    foreign = frontend.add(
        FakeRouter("foreign", STARTUP, revision=4, current=dict(CHOSEN))
    )
    ours = frontend.add(FakeRouter("ours", STARTUP, revision=1, current=dict(CHOSEN)))
    target = ActuationTarget(
        mode="rollback", include_not_ready=True, learn_unknown=False
    )

    for _ in range(2):
        report = await actuator.reconcile(
            target, initials={"ours": STARTUP}, expected_frontends=1
        )
        assert report.converged
    assert client.writes() == [("reset", "fe-0", "ours", 1)]
    assert ours.current == STARTUP
    assert foreign.current == CHOSEN and foreign.revision == 4
    assert report.new_initials == {}
    assert report.parity_ok is None  # the foreign router's initial is unknown

    # The default (active) target learns it with one reset.
    report = await actuator.reconcile(
        ROLLBACK, initials={"ours": STARTUP}, expected_frontends=1
    )
    assert client.writes()[-1] == ("reset", "fe-0", "foreign", 4)
    assert foreign.current == STARTUP
    assert report.converged


async def test_rollback_targets_not_ready_pods_and_waits_only_for_ready_ones() -> None:
    pods = [
        pod("ready", ip="10.0.0.1"),
        pod("not-ready", ready=False, ip="10.0.0.2"),
        pod("crashing", ready=False, ip="10.0.0.3"),
        pod("pending", ready=False, ip=None),
        pod("leaving", terminating=True, ip="10.0.0.4"),
    ]
    actuator, kube, client, _ = make_actuator(pods)
    initials = {}
    for name in ("ready", "not-ready", "leaving"):
        client.frontend(name).add(
            FakeRouter(f"r-{name}", STARTUP, revision=1, current=dict(CHOSEN))
        )
        initials[f"r-{name}"] = dict(STARTUP)
    # CrashLoopBackOff: an unreachable NotReady pod never blocks convergence.
    client.frontend("crashing").get_failures = [0]

    report = await actuator.reconcile(ROLLBACK, initials=initials, expected_frontends=1)
    assert sorted(call[1] for call in client.writes()) == ["not-ready", "ready"]
    assert ("get", "leaving") not in client.calls
    assert ("get", "pending") not in client.calls
    assert report.converged and report.errors == {"get:network": 1}

    # An unreachable Ready pod does block it; apply never targets NotReady pods.
    kube.pods = [pod("ready"), pod("crashing", ip="10.0.0.3")]
    client.frontend("crashing").get_failures = [0]
    client.calls.clear()
    report = await actuator.reconcile(
        apply(CHOSEN), initials=initials, expected_frontends=1
    )
    assert not report.converged
    kube.pods = pods
    client.calls.clear()
    await actuator.reconcile(apply(CHOSEN), initials=initials, expected_frontends=1)
    assert {call[1] for call in client.calls} == {"ready"}


# ----------------------------------------------------------- status codes


async def test_conflict_regets_and_retries_once() -> None:
    actuator, _, client, metrics = make_actuator([pod("fe-0")])
    router = client.frontend("fe-0").add(FakeRouter("r1", STARTUP))
    client.write_failures = [409]

    def someone_else_writes(op: str, pod_name: str, router_id: str) -> None:
        if client.write_failures:  # the write that answers 409
            router.current = {**STARTUP, "router_temperature": 0.7}
            router.revision += 1

    client.before_write = someone_else_writes

    report = await actuator.reconcile(
        apply(CHOSEN), initials={"r1": STARTUP}, expected_frontends=1
    )

    patches = [call for call in client.writes() if call[0] == "patch"]
    assert [call[3] for call in patches] == [0, 1]
    # The retry diff comes from the fresh GET (temperature 0.7 -> 0.2).
    assert patches[1][4]["router_temperature"] == 0.2
    assert router.current == CHOSEN
    assert report.errors == {"patch:409": 1}
    assert metrics.patch_errors == ["409"]
    assert report.converged


# ------------------------------------------------------------- convergence


async def test_convergence_and_presence_need_the_expected_frontends() -> None:
    actuator, _, client, _ = make_actuator([pod("fe-0")])
    frontend = client.frontend("fe-0")
    frontend.add(FakeRouter("custom", STARTUP, runtime_config=False))

    # No configured router: neither converged nor present (P12).
    report = await actuator.reconcile(apply(CHOSEN), initials={}, expected_frontends=1)
    assert not report.converged and report.routers_present is False
    assert report.parity_ok is None and client.writes() == []

    # Present even though the router does not match the apply target.
    frontend.add(FakeRouter("r1", {**STARTUP, "router_temperature": 0.9}))
    report = await actuator.reconcile(OBSERVE, initials={}, expected_frontends=1)
    assert report.converged and report.routers_present is True
    report = await actuator.reconcile(OBSERVE, initials={}, expected_frontends=2)
    assert not report.converged and report.routers_present is False


# ------------------------------------------------------------ gone routers


async def test_seen_router_ids_drop_only_routers_that_are_provably_gone() -> None:
    pods = [pod("fe-0"), pod("fe-1", ip="10.0.0.2")]
    actuator, kube, client, _ = make_actuator(pods)
    frontend0 = client.frontend("fe-0")
    frontend0.add(FakeRouter("a", STARTUP))
    client.frontend("fe-1").add(FakeRouter("b", STARTUP))
    report = await actuator.reconcile(OBSERVE, initials={}, expected_frontends=2)
    initials = dict(report.new_initials)
    assert report.seen_router_ids == frozenset({"a", "b"})

    # GET leaves out a router whose control call timed out, and a pod that
    # does not answer shows none: both are kept, as on a list failure.
    frontend0.timed_out.add("a")
    client.frontend("fe-1").get_failures = [0]
    report = await actuator.reconcile(OBSERVE, initials=initials, expected_frontends=2)
    assert report.seen_router_ids == frozenset({"a", "b"})
    kube.fail = KubeError("apiserver down")
    report = await actuator.reconcile(OBSERVE, initials=initials, expected_frontends=2)
    assert report.errors == {"list_pods": 1} and actuator.last_pods is None
    assert report.seen_router_ids == frozenset({"a", "b"})
    kube.fail = None

    # fe-0's frontend restarted with a new router; fe-1 was evicted (the pod
    # object stays, with its IP, in a finished phase and is never asked).
    del frontend0.routers["a"]
    frontend0.timed_out.clear()
    frontend0.add(FakeRouter("a2", STARTUP))
    kube.pods = [
        pod("fe-0", restarts=1),
        pod("fe-1", ready=False, ip="10.0.0.2", phase="Failed"),
    ]
    client.calls.clear()
    report = await actuator.reconcile(OBSERVE, initials=initials, expected_frontends=1)
    assert ("get", "fe-1") not in client.calls
    assert report.seen_router_ids == frozenset({"a2"})


async def test_a_router_missing_from_get_is_still_rolled_back() -> None:
    # A clean-up (advisory, learn_unknown off) planner restarts with the
    # initial of a retuned router persisted. Its first GET leaves the router
    # out (control call timed out): the initial must survive, or the router
    # comes back at revision > 0, is left alone, and stays retuned.
    target = ActuationTarget(
        mode="rollback", include_not_ready=True, learn_unknown=False
    )
    actuator, kube, client, _ = make_actuator([pod("fe-0")])
    frontend = client.frontend("fe-0")
    router = frontend.add(FakeRouter("r1", STARTUP, revision=1, current=dict(CHOSEN)))
    frontend.timed_out.add("r1")
    initials = {"r1": dict(STARTUP)}

    report = await actuator.reconcile(target, initials=initials, expected_frontends=1)
    assert report.seen_router_ids == frozenset({"r1"})
    assert not report.converged and client.writes() == []

    frontend.timed_out.clear()
    initials = {k: v for k, v in initials.items() if k in report.seen_router_ids}
    report = await actuator.reconcile(target, initials=initials, expected_frontends=1)
    assert client.writes() == [("reset", "fe-0", "r1", 1)]
    assert router.current == STARTUP and report.converged

    # Once every frontend listed at the first miss is gone, so is the router.
    actuator, kube, client, _ = make_actuator([pod("fe-0")])
    client.frontend("fe-0").add(FakeRouter("r2", STARTUP))
    report = await actuator.reconcile(target, initials=initials, expected_frontends=1)
    assert report.seen_router_ids == frozenset({"r1", "r2"})
    kube.pods = [pod("fe-new", ip="10.0.0.3")]
    report = await actuator.reconcile(target, initials=initials, expected_frontends=1)
    assert report.seen_router_ids == frozenset()


# ------------------------------------------------------------- HTTP client


class _FrontendApp:
    """A tiny aiohttp server speaking the /router_policy wire format."""

    def __init__(self) -> None:
        self.requests: list[tuple[str, Any]] = []
        self.patch_status = 200
        self.list_body: Any = {
            "routers": [
                {
                    "router_id": "6f1c",
                    "model": "Qwen/Qwen3-0.6B",
                    "namespace": "dyn-ns",
                    "component": "backend",
                    "endpoint": "generate",
                    "worker_type": "aggregated",
                    "configuration": {"revision": 3, "parameters": dict(STARTUP)},
                },
                {
                    "router_id": "custom",
                    "model": "Qwen/Qwen3-0.6B",
                    "namespace": "dyn-ns",
                    "component": "backend",
                    "endpoint": "generate",
                    "worker_type": "decode",
                    "configuration": None,
                },
            ]
        }

    def app(self) -> web.Application:
        app = web.Application()
        app.router.add_get("/router_policy", self.list)
        app.router.add_patch("/router_policy", self.update)
        return app

    async def list(self, request: web.Request) -> web.Response:
        self.requests.append(("GET", dict(request.query)))
        if isinstance(self.list_body, str):
            return web.Response(text=self.list_body, content_type="application/json")
        return web.json_response(self.list_body)

    async def update(self, request: web.Request) -> web.Response:
        body = await request.json()
        self.requests.append(("PATCH", body))
        if self.patch_status != 200:
            return web.json_response(
                {"error": "policy revision conflict: expected 1, current 2"},
                status=self.patch_status,
            )
        parameters = dict(STARTUP)
        parameters.update(body.get("parameters") or {})
        return web.json_response(
            {
                "router_id": body["router_id"],
                "model": "Qwen/Qwen3-0.6B",
                "namespace": "dyn-ns",
                "component": "backend",
                "endpoint": "generate",
                "worker_type": "aggregated",
                "configuration": {
                    "revision": body["expected_revision"] + 1,
                    "parameters": parameters,
                },
            }
        )


@pytest.fixture
async def frontend_server():
    frontend = _FrontendApp()
    server = test_utils.TestServer(frontend.app(), host="127.0.0.1")
    await server.start_server()
    client = AiohttpRouterPolicyClient(timeout_s=2.0)
    target = PodInfo(
        name="fe-0",
        uid="u",
        ip="127.0.0.1",
        ready=True,
        terminating=False,
        admin_port=server.port,
    )
    try:
        yield frontend, client, target
    finally:
        await client.close()
        await server.close()


async def test_client_speaks_the_router_policy_wire_format(frontend_server) -> None:
    frontend, client, target = frontend_server

    result = await client.get(target)
    assert result.status == 200 and result.error is None
    first, custom = result.routers
    assert first == RouterInfo(
        router_id="6f1c",
        namespace="dyn-ns",
        worker_type="aggregated",
        revision=3,
        configuration=STARTUP,
        model="Qwen/Qwen3-0.6B",
    )
    assert custom.configuration is None and custom.revision == 0

    patched = await client.patch(target, "6f1c", 3, {"overlap_score_credit": 0})
    reset = await client.reset(target, "6f1c", 7)
    assert frontend.requests == [
        ("GET", {}),  # no ?model=
        (
            "PATCH",
            {
                "router_id": "6f1c",
                "expected_revision": 3,
                "parameters": {"overlap_score_credit": 0.0},
            },
        ),
        ("PATCH", {"router_id": "6f1c", "expected_revision": 7, "reset": True}),
    ]
    assert patched.router is not None and patched.router.revision == 4
    assert patched.router.configuration["overlap_score_credit"] == 0.0
    assert reset.status == 200 and reset.router.revision == 8


async def test_client_errors_are_statuses(frontend_server) -> None:
    frontend, client, target = frontend_server
    frontend.patch_status = 409
    result = await client.patch(target, "6f1c", 1, {"router_temperature": 0.1})
    assert (result.status, result.router) == (409, None)
    assert result.error == "policy revision conflict: expected 1, current 2"

    for body in ("not json", json.dumps({"routers": "x"}), json.dumps({"other": []})):
        frontend.list_body = body
        result = await client.get(target)
        assert result.status == 0
        assert result.error and result.error.startswith("malformed GET response")

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        closed_port = sock.getsockname()[1]
    result = await client.get(replace(target, admin_port=closed_port))
    assert result.status == 0 and result.error

    # A pod whose env enables no admin listener is never called.
    calls = len(frontend.requests)
    result = await client.get(replace(target, admin_port=None))
    assert result.status == 0
    assert result.error and "DYN_ENABLE_FRONTEND_ADMIN_API" in result.error
    assert len(frontend.requests) == calls


def test_client_targets_the_admin_listener() -> None:
    client = AiohttpRouterPolicyClient()
    pod = PodInfo(
        name="fe",
        uid="u",
        ip="10.0.0.5",
        ready=True,
        terminating=False,
        admin_port=8002,
    )
    assert client.url(pod) == "http://10.0.0.5:8002/router_policy"
    assert (
        client.url(replace(pod, ip="fd00::5")) == "http://[fd00::5]:8002/router_policy"
    )
    assert client.url(replace(pod, admin_port=None)) is None
    assert client.url(replace(pod, ip=None)) is None


def test_client_timeout_outlasts_the_server_control_timeout() -> None:
    # A stuck router makes the server wait CONTROL_TIMEOUT before answering
    # GET 200 without it (or PATCH 503); the client must still be listening.
    assert DEFAULT_CLIENT_TIMEOUT_S == 8.0
    assert DEFAULT_CLIENT_TIMEOUT_S > ROUTER_CONTROL_TIMEOUT_S
    assert AiohttpRouterPolicyClient()._timeout.total == DEFAULT_CLIENT_TIMEOUT_S

    source = next(
        (
            parent / "lib/llm/src/http/service/router_policy.rs"
            for parent in Path(__file__).resolve().parents
            if (parent / "lib/llm/src/http/service/router_policy.rs").is_file()
        ),
        None,
    )
    if source is None:
        pytest.skip("router_policy.rs is not in this checkout")
    match = re.search(
        r"CONTROL_TIMEOUT: Duration = Duration::from_secs\((\d+)\)",
        source.read_text(),
    )
    assert match is not None
    assert float(match.group(1)) == ROUTER_CONTROL_TIMEOUT_S
