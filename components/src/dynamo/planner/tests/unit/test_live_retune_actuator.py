# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the live retune router actuator and its HTTP client.

``FakePolicyClient`` implements the ``/router_policy`` semantics of
``lib/kv-router/src/plugins/worker_selection/runtime.rs`` and
``lib/llm/src/http/service/router_policy.rs``: CAS on ``expected_revision``,
per-key PATCH merge, ``reset`` restores the startup values, and every
successful write bumps the revision.
"""

from __future__ import annotations

import asyncio
import json
import re
import socket
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
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
    POLICY_PARAMETER_KEYS,
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
# overlap_score_credit = 0 also zeroes the inherited cache weights.
ZERO_CREDIT = {
    **STARTUP,
    "overlap_score_credit": 0.0,
    "disk_cache_hit_weight": 0.0,
    "host_cache_hit_weight": 0.0,
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
_CACHE_KEYS = (
    "overlap_score_credit",
    "host_cache_hit_weight",
    "disk_cache_hit_weight",
    "shared_cache_multiplier",
)


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
    plain_decode: bool = False  # rejects nonzero cache inputs with 422

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

    def add(self, router: FakeRouter) -> FakeRouter:
        self.routers[router.router_id] = router
        return router


class FakePolicyClient:
    """In-memory frontends keyed by pod name."""

    def __init__(self) -> None:
        self.frontends: dict[str, FakeFrontend] = {}
        self.calls: list[tuple[Any, ...]] = []
        # Statuses returned by the next writes (any router) before applying.
        self.write_failures: list[int] = []
        # Called before a write is applied: (op, pod name, router_id).
        self.before_write: Optional[Callable[[str, str, str], None]] = None
        self.raise_on_get: Optional[BaseException] = None
        self.active = 0
        self.max_active = 0
        self.get_delay_s = 0.0

    def frontend(self, pod: str) -> FakeFrontend:
        return self.frontends.setdefault(pod, FakeFrontend())

    def writes(self) -> list[tuple[Any, ...]]:
        return [call for call in self.calls if call[0] in ("patch", "reset")]

    async def get(self, pod: PodInfo) -> PolicyCallResult:
        self.calls.append(("get", pod.name))
        if self.raise_on_get is not None:
            raise self.raise_on_get
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            if self.get_delay_s:
                await asyncio.sleep(self.get_delay_s)
            frontend = self.frontends.get(pod.name)
            if frontend is None:
                return PolicyCallResult(status=0, error="connection refused")
            if frontend.get_failures:
                return PolicyCallResult(status=frontend.get_failures.pop(0))
            return PolicyCallResult(
                status=200,
                routers=tuple(router.info() for router in frontend.routers.values()),
            )
        finally:
            self.active -= 1

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
        return self._write("patch", pod, router_id, expected_revision, parameters)

    async def reset(
        self, pod: PodInfo, router_id: str, expected_revision: int
    ) -> PolicyCallResult:
        self.calls.append(("reset", pod.name, router_id, expected_revision))
        return self._write("reset", pod, router_id, expected_revision, None)

    def _write(
        self,
        op: str,
        pod: PodInfo,
        router_id: str,
        expected_revision: int,
        parameters: Optional[Mapping[str, float]],
    ) -> PolicyCallResult:
        if self.before_write is not None:
            self.before_write(op, pod.name, router_id)
        if self.write_failures:
            return PolicyCallResult(status=self.write_failures.pop(0), error="injected")
        router = self.frontend(pod.name).routers.get(router_id)
        if router is None:
            return PolicyCallResult(status=404, error="router instance not found")
        if not router.runtime_config:
            return PolicyCallResult(status=422, error="unsupported")
        if expected_revision != router.revision:
            return PolicyCallResult(status=409, error="policy revision conflict")
        if parameters is None:
            updated = dict(router.initial)
        else:
            if not parameters:
                return PolicyCallResult(status=400, error="empty")
            updated = dict(router.current or {})
            for key, value in parameters.items():
                if key not in router.initial:
                    return PolicyCallResult(status=422, error=f"unknown {key}")
                updated[key] = router.initial[key] if value is None else float(value)
        if router.plain_decode and any(updated[key] != 0.0 for key in _CACHE_KEYS):
            return PolicyCallResult(status=422, error="decode has no cache inputs")
        router.current = updated
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
) -> PodInfo:
    return PodInfo(
        name=name,
        uid=f"{name}-uid",
        ip=ip,
        ready=ready,
        terminating=terminating,
        labels={POD_LABEL_DGD_NAME: DGD.name, POD_LABEL_COMPONENT_TYPE: "frontend"},
        http_port=8000,
        phase=phase,
    )


def make_actuator(
    pods: list[PodInfo],
    *,
    startup_maps: Optional[dict[str, dict[str, float]]] = None,
    max_concurrency: int = 8,
) -> tuple[RouterActuator, FakeKube, FakePolicyClient, RecordingMetrics]:
    kube = FakeKube(pods)
    client = FakePolicyClient()
    metrics = RecordingMetrics()
    actuator = RouterActuator(
        kube,
        client,
        DGD,
        startup_maps or {"aggregated": dict(STARTUP)},
        metrics,
        max_concurrency=max_concurrency,
    )
    return actuator, kube, client, metrics


def apply(desired: Mapping[str, float], wt: str = "aggregated") -> ActuationTarget:
    return ActuationTarget(mode="apply", desired_by_wt={wt: dict(desired)})


OBSERVE = ActuationTarget(mode="observe")
ROLLBACK = ActuationTarget(mode="rollback", include_not_ready=True)


def diff_keys(left: Mapping[str, float], right: Mapping[str, float]) -> set[str]:
    return {key for key in POLICY_PARAMETER_KEYS if left[key] != right[key]}


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


async def test_new_initials_reported_until_persisted() -> None:
    actuator, _, client, _ = make_actuator([pod("fe-0")])
    client.frontend("fe-0").add(FakeRouter("r1", STARTUP))

    await actuator.reconcile(apply(CHOSEN), initials={}, expected_frontends=1)
    # The state write has not landed: the router is at revision 1 now but
    # its initial stays known locally (no learning reset) and is reported.
    report = await actuator.reconcile(apply(CHOSEN), initials={}, expected_frontends=1)
    assert [call[0] for call in client.writes()] == ["patch"]
    assert report.new_initials == {"r1": STARTUP}
    assert report.converged


async def test_observe_with_unknown_initial_writes_nothing() -> None:
    actuator, _, client, _ = make_actuator([pod("fe-0")])
    client.frontend("fe-0").add(
        FakeRouter("r1", STARTUP, revision=4, current=dict(CHOSEN))
    )

    for _ in range(3):
        report = await actuator.reconcile(OBSERVE, initials={}, expected_frontends=1)
        assert not report.converged
        assert report.parity_ok is None
        assert report.new_initials == {}
    assert client.writes() == []


# ------------------------------------------------------------------- apply


async def test_apply_patches_only_differing_keys_with_explicit_values() -> None:
    actuator, _, client, _ = make_actuator([pod("fe-0")])
    router = client.frontend("fe-0").add(FakeRouter("r1", STARTUP))

    report = await actuator.reconcile(apply(CHOSEN), initials={}, expected_frontends=1)

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
            apply(CHOSEN), initials={"r1": STARTUP}, expected_frontends=1
        )
        assert report.converged
    assert len(client.writes()) == 1


async def test_zero_credit_round_trip_restores_cache_weights() -> None:
    actuator, _, client, _ = make_actuator([pod("fe-0")])
    router = client.frontend("fe-0").add(FakeRouter("r1", STARTUP))
    initials: dict[str, dict[str, float]] = {}

    report = await actuator.reconcile(
        apply(ZERO_CREDIT), initials=initials, expected_frontends=1
    )
    initials.update(report.new_initials)
    assert set(client.writes()[-1][4]) == diff_keys(STARTUP, ZERO_CREDIT)
    assert router.current == ZERO_CREDIT

    report = await actuator.reconcile(
        apply(CHOSEN), initials=initials, expected_frontends=1
    )
    # host/disk come back with the credit, as explicit values.
    assert client.writes()[-1][4] == {
        key: CHOSEN[key] for key in diff_keys(ZERO_CREDIT, CHOSEN)
    }
    assert router.current == CHOSEN
    assert report.converged

    await actuator.reconcile(
        apply(ZERO_CREDIT), initials=initials, expected_frontends=1
    )
    report = await actuator.reconcile(ROLLBACK, initials=initials, expected_frontends=1)
    assert client.writes()[-1] == ("reset", "fe-0", "r1", 3)
    assert router.current == STARTUP
    assert router.current["host_cache_hit_weight"] == 0.75
    assert router.current["disk_cache_hit_weight"] == 0.25
    assert report.converged


async def test_apply_learns_unknown_initial_then_patches_in_one_round() -> None:
    actuator, _, client, _ = make_actuator([pod("fe-0")])
    tampered = {**STARTUP, "router_temperature": 0.9}
    router = client.frontend("fe-0").add(
        FakeRouter("r1", STARTUP, revision=2, current=tampered)
    )

    report = await actuator.reconcile(apply(CHOSEN), initials={}, expected_frontends=1)

    assert [call[0] for call in client.writes()] == ["reset", "patch"]
    assert client.writes()[0] == ("reset", "fe-0", "r1", 2)
    assert client.writes()[1][3] == 3
    assert router.current == CHOSEN
    assert report.new_initials == {"r1": STARTUP}
    assert report.parity_ok is True
    assert report.converged

    await actuator.reconcile(apply(CHOSEN), initials={}, expected_frontends=1)
    assert len(client.writes()) == 2


async def test_parity_mismatch_blocks_apply_and_never_loops() -> None:
    actuator, _, client, _ = make_actuator([pod("fe-0"), pod("fe-1", ip="10.0.0.2")])
    odd = {**STARTUP, "overlap_score_credit_decay": 0.5}
    client.frontend("fe-0").add(FakeRouter("good", STARTUP))
    client.frontend("fe-1").add(FakeRouter("odd", odd))

    for _ in range(3):
        report = await actuator.reconcile(
            apply(CHOSEN), initials={}, expected_frontends=2
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


async def test_rollback_with_parity_mismatch_resets_to_each_routers_initial() -> None:
    actuator, _, client, _ = make_actuator([pod("fe-0")])
    odd = {**STARTUP, "overlap_score_credit_decay": 0.5}
    router = client.frontend("fe-0").add(FakeRouter("odd", odd))
    await actuator.reconcile(OBSERVE, initials={}, expected_frontends=1)
    router.current = {**odd, "router_temperature": 1.0}
    router.revision = 1

    report = await actuator.reconcile(ROLLBACK, initials={}, expected_frontends=1)

    assert client.writes() == [("reset", "fe-0", "odd", 1)]
    assert router.current == odd
    assert report.converged  # rollback converges to the latched initial
    assert report.parity_ok is False


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
    assert deviating.revision == 2
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


async def test_rollback_learns_unknown_initial_with_a_single_reset() -> None:
    actuator, _, client, _ = make_actuator([pod("fe-0")])
    router = client.frontend("fe-0").add(
        FakeRouter("r1", STARTUP, revision=5, current=dict(CHOSEN))
    )

    report = await actuator.reconcile(ROLLBACK, initials={}, expected_frontends=1)
    assert client.writes() == [("reset", "fe-0", "r1", 5)]
    assert report.new_initials == {"r1": STARTUP}
    assert router.current == STARTUP
    assert report.converged

    for _ in range(2):
        report = await actuator.reconcile(ROLLBACK, initials={}, expected_frontends=1)
        assert report.converged
    assert len(client.writes()) == 1


async def test_rollback_targets_not_ready_pods_but_never_terminating_ones() -> None:
    pods = [
        pod("ready", ip="10.0.0.1"),
        pod("not-ready", ready=False, ip="10.0.0.2"),
        pod("pending", ready=False, ip=None),
        pod("leaving", terminating=True, ip="10.0.0.4"),
    ]
    actuator, _, client, _ = make_actuator(pods)
    initials = {}
    for name in ("ready", "not-ready", "leaving"):
        client.frontend(name).add(
            FakeRouter(f"r-{name}", STARTUP, revision=1, current=dict(CHOSEN))
        )
        initials[f"r-{name}"] = dict(STARTUP)

    report = await actuator.reconcile(ROLLBACK, initials=initials, expected_frontends=1)

    assert sorted(call[1] for call in client.writes()) == ["not-ready", "ready"]
    assert ("get", "leaving") not in client.calls
    assert ("get", "pending") not in client.calls
    assert report.converged

    # apply never targets NotReady pods.
    client.calls.clear()
    await actuator.reconcile(apply(CHOSEN), initials=initials, expected_frontends=1)
    assert {call[1] for call in client.calls} == {"ready"}


async def test_rollback_converges_without_an_unreachable_not_ready_pod() -> None:
    actuator, _, client, _ = make_actuator(
        [pod("ready"), pod("crashing", ready=False, ip="10.0.0.2")]
    )
    router = client.frontend("ready").add(
        FakeRouter("r1", STARTUP, revision=1, current=dict(CHOSEN))
    )
    # CrashLoopBackOff: the GET is refused on every round.
    client.frontend("crashing").get_failures = [0, 0, 0]

    for _ in range(3):
        report = await actuator.reconcile(
            ROLLBACK, initials={"r1": STARTUP}, expected_frontends=1
        )
        assert report.converged
        assert report.errors == {"get:network": 1}
    assert router.current == STARTUP
    assert client.writes() == [("reset", "ready", "r1", 1)]


async def test_an_unreachable_ready_pod_blocks_convergence() -> None:
    actuator, _, client, _ = make_actuator([pod("fe-0"), pod("fe-1", ip="10.0.0.2")])
    client.frontend("fe-0").add(FakeRouter("r0", STARTUP))
    client.frontend("fe-1").add(FakeRouter("r1", STARTUP))
    client.frontend("fe-1").get_failures = [0]

    report = await actuator.reconcile(ROLLBACK, initials={}, expected_frontends=1)
    assert not report.converged

    report = await actuator.reconcile(ROLLBACK, initials={}, expected_frontends=1)
    assert report.converged


async def test_an_answering_not_ready_pod_is_still_checked() -> None:
    actuator, _, client, _ = make_actuator(
        [pod("ready"), pod("starting", ready=False, ip="10.0.0.2")]
    )
    client.frontend("ready").add(FakeRouter("r0", STARTUP))
    stuck = client.frontend("starting").add(
        FakeRouter("r1", STARTUP, revision=2, current=dict(CHOSEN))
    )
    initials = {"r0": dict(STARTUP), "r1": dict(STARTUP)}
    client.write_failures = [500]

    report = await actuator.reconcile(ROLLBACK, initials=initials, expected_frontends=1)
    assert not report.converged  # it answered, and it is not at its initial
    assert stuck.current == CHOSEN

    report = await actuator.reconcile(ROLLBACK, initials=initials, expected_frontends=1)
    assert report.converged
    assert stuck.current == STARTUP


@pytest.mark.parametrize("phase", ["Failed", "Succeeded"])
async def test_finished_pods_are_never_targeted_and_their_routers_are_gone(
    phase: str,
) -> None:
    actuator, kube, client, _ = make_actuator(
        [pod("fe-0"), pod("evicted", ip="10.0.0.2")]
    )
    client.frontend("fe-0").add(FakeRouter("r0", STARTUP))
    client.frontend("evicted").add(FakeRouter("r-ev", STARTUP))
    report = await actuator.reconcile(OBSERVE, initials={}, expected_frontends=1)
    initials = dict(report.new_initials)
    assert set(initials) == {"r0", "r-ev"}

    # Evicted: the pod object stays, with its IP, in a finished phase.
    kube.pods = [pod("fe-0"), pod("evicted", ready=False, ip="10.0.0.2", phase=phase)]
    client.frontend("evicted").get_failures = [0]
    client.calls.clear()

    report = await actuator.reconcile(ROLLBACK, initials=initials, expected_frontends=1)

    assert ("get", "evicted") not in client.calls
    assert report.converged
    assert report.errors == {}
    assert report.seen_router_ids == frozenset({"r0"})


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


async def test_apply_without_learning_never_resets_an_unknown_router() -> None:
    actuator, _, client, _ = make_actuator([pod("fe-0")])
    client.frontend("fe-0").add(
        FakeRouter("r1", STARTUP, revision=2, current=dict(CHOSEN))
    )
    target = ActuationTarget(
        mode="apply",
        desired_by_wt={"aggregated": dict(ZERO_CREDIT)},
        learn_unknown=False,
    )

    report = await actuator.reconcile(target, initials={}, expected_frontends=1)

    assert client.writes() == []
    assert not report.converged


# ----------------------------------------------------------- status codes


async def test_conflict_regets_and_retries_once() -> None:
    actuator, _, client, metrics = make_actuator([pod("fe-0")])
    router = client.frontend("fe-0").add(FakeRouter("r1", STARTUP))
    interfered = []

    def someone_else_writes(op: str, pod_name: str, router_id: str) -> None:
        if not interfered:
            interfered.append(True)
            router.current = {**STARTUP, "router_temperature": 0.7}
            router.revision += 1

    client.before_write = someone_else_writes

    report = await actuator.reconcile(apply(CHOSEN), initials={}, expected_frontends=1)

    patches = [call for call in client.writes() if call[0] == "patch"]
    assert [call[3] for call in patches] == [0, 1]
    # The retry diff comes from the fresh GET (temperature 0.7 -> 0.2).
    assert patches[1][4]["router_temperature"] == 0.2
    assert router.current == CHOSEN
    assert report.errors == {"patch:409": 1}
    assert metrics.patch_errors == ["409"]
    assert report.converged


async def test_conflict_twice_gives_up_until_next_round() -> None:
    actuator, _, client, _ = make_actuator([pod("fe-0")])
    router = client.frontend("fe-0").add(FakeRouter("r1", STARTUP))
    client.write_failures = [409, 409]

    report = await actuator.reconcile(apply(CHOSEN), initials={}, expected_frontends=1)
    assert len(client.writes()) == 2
    assert not report.converged

    report = await actuator.reconcile(apply(CHOSEN), initials={}, expected_frontends=1)
    assert router.current == CHOSEN
    assert report.converged


async def test_patch_404_regets_and_patches_the_rebuilt_router() -> None:
    actuator, _, client, _ = make_actuator([pod("fe-0")])
    frontend = client.frontend("fe-0")
    frontend.add(FakeRouter("old", STARTUP))

    def rebuild(op: str, pod_name: str, router_id: str) -> None:
        if router_id == "old":
            del frontend.routers["old"]
            frontend.add(FakeRouter("new", STARTUP))

    client.before_write = rebuild

    report = await actuator.reconcile(apply(CHOSEN), initials={}, expected_frontends=1)

    assert [(call[0], call[2]) for call in client.writes()] == [
        ("patch", "old"),
        ("patch", "new"),
    ]
    assert frontend.routers["new"].current == CHOSEN
    assert report.errors == {"patch:404": 1}
    assert report.new_initials == {"new": STARTUP}
    assert report.seen_router_ids == frozenset({"new"})
    assert report.converged


async def test_503_regets_then_retries() -> None:
    actuator, _, client, metrics = make_actuator([pod("fe-0")])
    router = client.frontend("fe-0").add(FakeRouter("r1", STARTUP))
    client.write_failures = [503]

    report = await actuator.reconcile(apply(CHOSEN), initials={}, expected_frontends=1)

    assert client.calls[:3] == [
        ("get", "fe-0"),
        ("patch", "fe-0", "r1", 0, client.writes()[0][4]),
        ("get", "fe-0"),
    ]
    assert router.current == CHOSEN
    assert metrics.patch_errors == ["503"]
    assert report.converged


async def test_422_is_not_retried_until_the_target_changes() -> None:
    actuator, _, client, metrics = make_actuator([pod("fe-0")])
    client.frontend("fe-0").add(FakeRouter("r1", STARTUP))
    client.write_failures = [422]

    for _ in range(3):
        report = await actuator.reconcile(
            apply(CHOSEN), initials={}, expected_frontends=1
        )
        assert not report.converged
    assert len(client.writes()) == 1
    assert metrics.patch_errors == ["422"]

    other = {**CHOSEN, "router_temperature": 0.4}
    report = await actuator.reconcile(apply(other), initials={}, expected_frontends=1)
    assert len(client.writes()) == 2
    assert report.converged


async def test_get_404_for_three_rounds_reports_admin_api_disabled() -> None:
    actuator, _, client, _ = make_actuator([pod("fe-0")])
    frontend = client.frontend("fe-0")
    frontend.add(FakeRouter("r1", STARTUP))
    frontend.get_failures = [404, 404, 404]

    flags = []
    for _ in range(3):
        report = await actuator.reconcile(OBSERVE, initials={}, expected_frontends=1)
        flags.append(report.admin_api_disabled)
        assert report.errors == {"get:404": 1}
    assert flags == [False, False, True]

    report = await actuator.reconcile(OBSERVE, initials={}, expected_frontends=1)
    assert not report.admin_api_disabled
    assert report.converged


async def test_get_404_counter_needs_consecutive_rounds() -> None:
    actuator, _, client, _ = make_actuator([pod("fe-0")])
    frontend = client.frontend("fe-0")
    frontend.add(FakeRouter("r1", STARTUP))
    frontend.get_failures = [404, 404]
    await actuator.reconcile(OBSERVE, initials={}, expected_frontends=1)
    await actuator.reconcile(OBSERVE, initials={}, expected_frontends=1)
    await actuator.reconcile(OBSERVE, initials={}, expected_frontends=1)  # 200
    frontend.get_failures = [404]
    report = await actuator.reconcile(OBSERVE, initials={}, expected_frontends=1)
    assert not report.admin_api_disabled


# ------------------------------------------------------------- convergence


async def test_not_converged_below_the_expected_ready_frontends() -> None:
    actuator, _, client, _ = make_actuator([pod("fe-0")])
    client.frontend("fe-0").add(FakeRouter("r1", STARTUP))

    report = await actuator.reconcile(OBSERVE, initials={}, expected_frontends=2)
    assert not report.converged
    report = await actuator.reconcile(OBSERVE, initials={}, expected_frontends=0)
    assert report.converged  # max(1, 0)


async def test_not_converged_without_any_router() -> None:
    actuator, _, client, _ = make_actuator([pod("fe-0")])
    client.frontend("fe-0")

    report = await actuator.reconcile(OBSERVE, initials={}, expected_frontends=1)
    assert not report.converged
    assert report.parity_ok is None


async def test_routers_present_does_not_need_the_target_values() -> None:
    actuator, _, client, _ = make_actuator([pod("fe-0")])
    client.frontend("fe-0").add(FakeRouter("r1", STARTUP))

    # Present (P12) even though the routers do not match the apply target.
    report = await actuator.reconcile(
        ActuationTarget(mode="observe"), initials={}, expected_frontends=1
    )
    assert report.routers_present is True
    report = await actuator.reconcile(
        ActuationTarget(mode="observe"), initials={}, expected_frontends=2
    )
    assert report.routers_present is False


async def test_routers_present_needs_a_configured_router() -> None:
    actuator, _, client, _ = make_actuator([pod("fe-0")])
    client.frontend("fe-0").add(FakeRouter("custom", STARTUP, runtime_config=False))

    report = await actuator.reconcile(OBSERVE, initials={}, expected_frontends=1)
    assert report.routers_present is False


async def test_a_router_without_runtime_configuration_does_not_count() -> None:
    actuator, _, client, _ = make_actuator([pod("fe-0")])
    client.frontend("fe-0").add(FakeRouter("custom", STARTUP, runtime_config=False))

    report = await actuator.reconcile(apply(CHOSEN), initials={}, expected_frontends=1)
    assert not report.converged
    assert client.writes() == []


async def test_disagg_needs_both_worker_types_and_respects_plain_decode() -> None:
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
    report = await actuator.reconcile(target, initials={}, expected_frontends=1)
    assert not report.converged  # no decode router yet
    assert prefill.current == CHOSEN

    decode = frontend.add(
        FakeRouter("d", DECODE_STARTUP, worker_type="decode", plain_decode=True)
    )
    report = await actuator.reconcile(target, initials={}, expected_frontends=1)
    assert decode.current["router_temperature"] == 0.2
    assert decode.current["overlap_score_credit"] == 0.0
    assert report.converged
    assert report.parity_ok is True


async def test_only_the_dgd_namespace_and_planner_worker_types_are_managed() -> None:
    actuator, _, client, _ = make_actuator([pod("fe-0")])
    frontend = client.frontend("fe-0")
    ours = frontend.add(FakeRouter("ours", STARTUP))
    foreign = frontend.add(FakeRouter("foreign", STARTUP, namespace="other-ns"))
    encode = frontend.add(FakeRouter("encode", STARTUP, worker_type="encode"))

    report = await actuator.reconcile(apply(CHOSEN), initials={}, expected_frontends=1)

    assert {call[2] for call in client.writes()} == {"ours"}
    assert ours.current == CHOSEN
    assert foreign.current == STARTUP and encode.current == STARTUP
    assert report.seen_router_ids == frozenset({"ours"})
    assert report.converged


async def test_frontend_restart_fills_in_without_reset() -> None:
    actuator, kube, client, _ = make_actuator([pod("fe-0")])
    client.frontend("fe-0").add(FakeRouter("r1", STARTUP))
    initials: dict[str, dict[str, float]] = {}
    report = await actuator.reconcile(
        apply(CHOSEN), initials=initials, expected_frontends=1
    )
    initials.update(report.new_initials)

    # The frontend restarts: Ready is 0 for a while.
    kube.pods = [pod("fe-0", ready=False)]
    report = await actuator.reconcile(
        apply(CHOSEN), initials=initials, expected_frontends=1
    )
    assert not report.converged

    # A new pod with a fresh router at its startup values: PATCH, no reset.
    kube.pods = [pod("fe-0b", ip="10.0.0.9")]
    client.frontend("fe-0b").add(FakeRouter("r2", STARTUP))
    report = await actuator.reconcile(
        apply(CHOSEN), initials=initials, expected_frontends=1
    )
    assert [call[0] for call in client.writes()] == ["patch", "patch"]
    assert client.frontend("fe-0b").routers["r2"].current == CHOSEN
    assert report.converged
    assert report.seen_router_ids == frozenset({"r2"})


# -------------------------------------------------------- seen router ids


async def test_seen_router_ids_drop_only_routers_that_are_provably_gone() -> None:
    pods = [pod("fe-0"), pod("fe-1", ip="10.0.0.2")]
    actuator, kube, client, _ = make_actuator(pods)
    frontend0 = client.frontend("fe-0")
    frontend0.add(FakeRouter("a", STARTUP))
    client.frontend("fe-1").add(FakeRouter("b", STARTUP))
    report = await actuator.reconcile(OBSERVE, initials={}, expected_frontends=2)
    initials = dict(report.new_initials)
    assert report.seen_router_ids == frozenset({"a", "b"})

    # fe-1 does not answer: b is kept; a left fe-0, which answered: dropped.
    del frontend0.routers["a"]
    frontend0.add(FakeRouter("a2", STARTUP))
    client.frontend("fe-1").get_failures = [0]
    report = await actuator.reconcile(OBSERVE, initials=initials, expected_frontends=2)
    assert report.seen_router_ids == frozenset({"a2", "b"})

    # fe-1 is gone from the listing: b goes too.
    kube.pods = [pod("fe-0")]
    report = await actuator.reconcile(OBSERVE, initials=initials, expected_frontends=1)
    assert report.seen_router_ids == frozenset({"a2"})


async def test_persisted_initials_of_unknown_pods_survive_an_incomplete_view() -> None:
    pods = [pod("fe-0"), pod("fe-1", ready=False, ip="10.0.0.2")]
    actuator, _, client, _ = make_actuator(pods)
    client.frontend("fe-0").add(FakeRouter("a", STARTUP))
    persisted = {"a": dict(STARTUP), "unknown": dict(STARTUP)}

    # apply does not query the NotReady fe-1, so "unknown" may live there.
    report = await actuator.reconcile(OBSERVE, initials=persisted, expected_frontends=1)
    assert report.seen_router_ids == frozenset({"a", "unknown"})

    # rollback queries every live pod; "unknown" is nowhere: dropped.
    client.frontend("fe-1")
    report = await actuator.reconcile(
        ROLLBACK, initials=persisted, expected_frontends=1
    )
    assert report.seen_router_ids == frozenset({"a"})


# --------------------------------------------------------- failure paths


async def test_list_pods_failure_reports_without_raising() -> None:
    actuator, kube, _, metrics = make_actuator([pod("fe-0")])
    kube.fail = KubeError("apiserver down")

    report = await actuator.reconcile(
        OBSERVE, initials={"r1": STARTUP}, expected_frontends=1
    )

    assert not report.converged
    assert report.errors == {"list_pods": 1}
    assert report.seen_router_ids == frozenset({"r1"})
    assert actuator.last_pods is None
    assert metrics.converged == [False]


async def test_unexpected_client_exception_never_escapes() -> None:
    actuator, _, client, metrics = make_actuator([pod("fe-0")])
    client.raise_on_get = RuntimeError("bug")

    report = await actuator.reconcile(
        apply(CHOSEN), initials={"r1": STARTUP}, expected_frontends=1
    )

    assert not report.converged
    assert report.errors == {"internal": 1}
    assert report.seen_router_ids == frozenset({"r1"})
    assert metrics.converged == [False]


async def test_cancellation_propagates() -> None:
    actuator, _, client, _ = make_actuator([pod("fe-0")])
    client.raise_on_get = asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        await actuator.reconcile(OBSERVE, initials={}, expected_frontends=1)


@pytest.mark.parametrize(
    "target",
    [
        ActuationTarget(mode="apply"),
        ActuationTarget(mode="apply", desired_by_wt={"prefill": dict(CHOSEN)}),
        ActuationTarget(
            mode="apply",
            desired_by_wt={"aggregated": {**CHOSEN, "router_temperature": -1.0}},
        ),
        ActuationTarget(
            mode="apply",
            desired_by_wt={
                "aggregated": {
                    k: v for k, v in CHOSEN.items() if k != "prefill_load_scale"
                }
            },
        ),
    ],
)
async def test_invalid_apply_target_writes_nothing(target: ActuationTarget) -> None:
    actuator, _, client, _ = make_actuator([pod("fe-0")])
    client.frontend("fe-0").add(FakeRouter("r1", STARTUP))

    report = await actuator.reconcile(target, initials={}, expected_frontends=1)

    assert client.writes() == []
    assert not report.converged
    assert report.errors == {"invalid_target": 1}
    assert report.new_initials == {"r1": STARTUP}  # still latched


async def test_max_concurrency_bounds_parallel_gets() -> None:
    pods = [pod(f"fe-{i}", ip=f"10.0.0.{i + 1}") for i in range(6)]
    actuator, _, client, _ = make_actuator(pods, max_concurrency=2)
    for i in range(6):
        client.frontend(f"fe-{i}").add(FakeRouter(f"r{i}", STARTUP))
    client.get_delay_s = 0.01

    report = await actuator.reconcile(OBSERVE, initials={}, expected_frontends=6)

    assert client.max_active == 2
    assert report.converged
    assert actuator.last_pods is not None and len(actuator.last_pods) == 6


def test_constructor_validation() -> None:
    kube, client, metrics = FakeKube(), FakePolicyClient(), RecordingMetrics()
    with pytest.raises(ValueError):
        RouterActuator(kube, client, DGD, {}, metrics)
    with pytest.raises(ValueError):
        RouterActuator(kube, client, DGD, {"aggregated": STARTUP}, metrics, 0)
    actuator = RouterActuator(
        kube, client, DGD, {"prefill": STARTUP, "decode": DECODE_STARTUP}, metrics
    )
    assert actuator.worker_types == ("prefill", "decode")
    actuator.set_startup_maps({"aggregated": STARTUP})
    assert actuator.worker_types == ("aggregated",)


async def test_close_closes_the_client_when_it_can() -> None:
    actuator, _, client, _ = make_actuator([pod("fe-0")])
    await actuator.close()  # FakePolicyClient has no close: no-op

    http = AiohttpRouterPolicyClient()
    actuator = RouterActuator(
        FakeKube(), http, DGD, {"aggregated": STARTUP}, RecordingMetrics()
    )
    session = http._ensure_session()  # what the first call does
    assert not session.closed
    await actuator.close()
    assert session.closed and http._session is None


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
        http_port=server.port,
    )
    try:
        yield frontend, client, target
    finally:
        await client.close()
        await server.close()


async def test_client_get_parses_the_router_list(frontend_server) -> None:
    frontend, client, target = frontend_server

    result = await client.get(target)

    assert result.status == 200 and result.error is None
    assert frontend.requests == [("GET", {})]  # no ?model=
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


async def test_client_patch_sends_explicit_parameters(frontend_server) -> None:
    frontend, client, target = frontend_server

    result = await client.patch(target, "6f1c", 3, {"overlap_score_credit": 0})

    assert frontend.requests == [
        (
            "PATCH",
            {
                "router_id": "6f1c",
                "expected_revision": 3,
                "parameters": {"overlap_score_credit": 0.0},
            },
        )
    ]
    assert result.status == 200
    assert result.router is not None
    assert result.router.revision == 4
    assert result.router.configuration["overlap_score_credit"] == 0.0


async def test_client_reset_sends_reset_true_without_parameters(
    frontend_server,
) -> None:
    frontend, client, target = frontend_server

    result = await client.reset(target, "6f1c", 7)

    assert frontend.requests == [
        ("PATCH", {"router_id": "6f1c", "expected_revision": 7, "reset": True})
    ]
    assert result.status == 200 and result.router.revision == 8


async def test_client_reports_the_error_status_and_message(frontend_server) -> None:
    frontend, client, target = frontend_server
    frontend.patch_status = 409

    result = await client.patch(target, "6f1c", 1, {"router_temperature": 0.1})

    assert result.status == 409
    assert result.router is None
    assert result.error == "policy revision conflict: expected 1, current 2"


@pytest.mark.parametrize(
    "body", ["not json", json.dumps({"routers": "x"}), json.dumps({"other": []})]
)
async def test_client_malformed_body_is_status_zero(frontend_server, body) -> None:
    frontend, client, target = frontend_server
    frontend.list_body = body

    result = await client.get(target)

    assert result.status == 0
    assert result.error and result.error.startswith("malformed GET response")


async def test_client_network_error_is_status_zero() -> None:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    client = AiohttpRouterPolicyClient(timeout_s=1.0)
    target = PodInfo(
        name="fe",
        uid="u",
        ip="127.0.0.1",
        ready=True,
        terminating=False,
        http_port=port,
    )
    try:
        result = await client.get(target)
    finally:
        await client.close()
    assert result.status == 0
    assert result.error


async def test_client_without_pod_ip_or_parameters_sends_nothing() -> None:
    client = AiohttpRouterPolicyClient()
    no_ip = PodInfo(name="fe", uid="u", ip=None, ready=False, terminating=False)
    result = await client.get(no_ip)
    assert result.status == 0 and "no IP" in (result.error or "")
    with_ip = PodInfo(name="fe", uid="u", ip="10.0.0.1", ready=True, terminating=False)
    result = await client.patch(with_ip, "r", 0, {})
    assert result.status == 0 and result.error == "empty parameters"
    await client.close()


def test_client_url() -> None:
    client = AiohttpRouterPolicyClient(scheme="http")
    base = dict(name="fe", uid="u", ready=True, terminating=False)
    assert (
        client.url(PodInfo(ip="10.1.2.3", http_port=8123, **base))
        == "http://10.1.2.3:8123/router_policy"
    )
    assert (
        client.url(PodInfo(ip="10.1.2.3", **base))
        == "http://10.1.2.3:8000/router_policy"
    )
    assert client.url(PodInfo(ip="fd00::5", http_port=80, **base)) == (
        "http://[fd00::5]:80/router_policy"
    )
    with pytest.raises(ValueError):
        AiohttpRouterPolicyClient(scheme="ftp")
    with pytest.raises(ValueError):
        AiohttpRouterPolicyClient(timeout_s=0)


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
