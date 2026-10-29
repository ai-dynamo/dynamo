# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402
# Optional-dependency preflight must run before the retune imports.

"""Live retune end to end: the real modules, faked only at the edges.

Real: ``bootstrap.build_supervisor`` (frontend parse, spec premises, the P11
canary), ``LiveRetuneSupervisor`` with its Lease leadership and state store,
``OrchestratorEngineAdapter`` with its ``RetuneController``,
``dgdr.run_dgdr_sweep``, the window cutter, ``RouterActuator`` and
``Reconciler``.

Fake: the Kubernetes API (``FakeRetuneKube``, with ``FakeDgdrOperator``
driving each DGDR v1beta2 Request to a Run and its Candidates), the
frontends' ``/router_policy`` (revision-checked PATCH and reset with the
semantics of ``router_policy.rs`` and ``worker_selection/runtime.rs``), the
trace claim (a temporary directory of real ``jsonl.gz`` segments) and the
clocks. The test advances the clocks in 15 s steps; each step runs every
live planner the way ``NativePlannerBase`` and the supervisor's background
tasks do: lease round, the due planner tick (its decisions go to the
outbox), outbox round, control round (revalidation, canary, actuator,
reconciler). A launched sweep runs on the controller's sweep thread and the
test waits for it before the next step, so every run is deterministic. The
DGDR operator advances one phase per poll of the sweep.
"""

from __future__ import annotations

import asyncio
import dataclasses
import gzip
import json
import math
import os
import time
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Optional

import pytest

pytest.importorskip(
    "aisimulate.sweeper",
    reason="AI Simulate is an optional Dynamo simulation dependency",
)

from dynamo.llm import KvRouterConfig
from dynamo.planner.config.planner_config import PlannerConfig
from dynamo.planner.core.types import RetuneDecision, TickInput, WorkerCounts
from dynamo.planner.plugins.clock import WallClock
from dynamo.planner.plugins.orchestrator.engine_adapter import OrchestratorEngineAdapter
from dynamo.planner.retune.live import bootstrap
from dynamo.planner.retune.live.actuator import RouterActuator
from dynamo.planner.retune.live.contracts import (
    LABEL_DGD_UID,
    LABEL_SHARD,
    LABEL_WINDOW,
    DgdRef,
    NullMetrics,
    PodInfo,
    PolicyCallResult,
    RouterInfo,
)
from dynamo.planner.retune.live.reconciler import Reconciler
from dynamo.planner.retune.live.state import (
    RetuneState,
    leadership_lease_name,
    state_lease_name,
)
from dynamo.planner.retune.live.supervisor import LiveRetuneSupervisor
from dynamo.planner.retune.live.window_cut import cut_window, cut_window_inprocess
from dynamo.planner.tests.live_retune_fakes import (
    FakeDgdrOperator,
    FakeRetuneKube,
    default_metrics,
)
from dynamo.planner.tests.retune_stubs import caps

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.integration,
    pytest.mark.planner,
]

INTERVAL = 3600.0
GRACE = 120.0
TIMEOUT = 3000.0
SCALE_INTERVAL = 60.0
# Lease renewal (a third of the lease) and the actuator cadence both fit it.
STEP = 15.0
LEASE = 60.0
# Window n is [T0 + (n - 1) * INTERVAL, T0 + n * INTERVAL), launched GRACE
# after it ends. T0 is far ahead of the real clock: files the cutter writes
# (real mtimes) look old to the reconciler, and a real deadline derived from
# the simulated one is always in the future.
T0 = 3_600_000_000.0
DGD_NAME = "graph"
K8S_NAMESPACE = "retune-e2e"
DGD_UID = "dgd-uid-1"
DYNAMO_NAMESPACE = "retune-e2e-graph"
CLAIM = "request-traces"
MODEL = "Qwen/Qwen3-0.6B"
FRONTENDS = ("graph-frontend-a", "graph-frontend-b")
WORKER_REPLICAS = 2
REQUESTS_PER_POD = 12
ROWS_PER_WINDOW = REQUESTS_PER_POD * len(FRONTENDS)
TRACE_SCHEMA = "dynamo.request.trace.v1"
TRACE_FILE_PATH = "/trace/segments/$(DYN_PARENT_DGD_K8S_NAME)/$(DYN_POD_NAME)/req"
CREDIT = "overlap_score_credit"
LOAD = "prefill_load_scale"
TEMP = "router_temperature"
STARTUP = {CREDIT: 1.0, LOAD: 1.0, TEMP: 0.0}
# The frontend runs ``--router-mode kv --router-replica-sync``.
FRONTEND_ROUTER_CONFIG = {"router_replica_sync": True}


def end_s(n: int) -> float:
    return T0 + n * INTERVAL


def end_ms(n: int) -> int:
    return int(round(end_s(n) * 1000.0))


def launch_s(n: int) -> float:
    return end_s(n) + GRACE


def decision_id(n: int) -> str:
    return f"retune-w{end_ms(n)}"


def by_wt(point: Mapping[str, float]) -> dict[str, dict[str, float]]:
    """``policy_parameters`` of the frontend's startup config with ``point``."""

    config = KvRouterConfig.from_json(json.dumps({**FRONTEND_ROUTER_CONFIG, **point}))
    return {
        "aggregated": {
            key: float(value)
            for key, value in dict(config.policy_parameters("aggregated")).items()
        }
    }


def startup_map() -> dict[str, float]:
    return by_wt({})["aggregated"]


# ------------------------------------------------------------- DGD, config


def dgd_object() -> dict[str, Any]:
    """A DGD that satisfies every spec premise (P4, P6-P10)."""

    env = [
        {
            "name": "DYN_POD_NAME",
            "valueFrom": {"fieldRef": {"fieldPath": "metadata.name"}},
        },
        {"name": "DYN_REQUEST_TRACE", "value": "1"},
        {"name": "DYN_REQUEST_TRACE_RECORDS", "value": "request_end"},
        {"name": "DYN_REQUEST_TRACE_SINKS", "value": "file"},
        {"name": "DYN_REQUEST_TRACE_FILE_FORMAT", "value": "jsonl_gz"},
        {"name": "DYN_REQUEST_TRACE_FILE_PATH", "value": TRACE_FILE_PATH},
    ]
    frontend = {
        "name": "Frontend",
        "type": "frontend",
        "replicas": len(FRONTENDS),
        "podTemplate": {
            "spec": {
                "containers": [
                    {
                        "name": "main",
                        "image": "nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.8.0",
                        "command": ["python3"],
                        "args": [
                            "-m",
                            "dynamo.frontend",
                            "--router-mode",
                            "kv",
                            "--router-replica-sync",
                        ],
                        "env": env,
                        "volumeMounts": [{"name": "traces", "mountPath": "/trace"}],
                    }
                ],
                "volumes": [
                    {"name": "traces", "persistentVolumeClaim": {"claimName": CLAIM}}
                ],
            }
        },
    }
    worker = {
        "name": "worker",
        "type": "worker",
        "replicas": WORKER_REPLICAS,
        "podTemplate": {
            "spec": {
                "containers": [
                    {
                        "name": "main",
                        "command": ["python3", "-m", "dynamo.vllm"],
                        "args": [f"--model={MODEL}"],
                    }
                ]
            }
        },
    }
    planner = {
        "name": "Planner",
        "type": "planner",
        "replicas": 1,
        "podTemplate": {"spec": {"containers": [{"name": "main"}]}},
    }
    return {
        "apiVersion": "nvidia.com/v1beta1",
        "kind": "DynamoGraphDeployment",
        "metadata": {"name": DGD_NAME, "namespace": K8S_NAMESPACE, "uid": DGD_UID},
        "spec": {"components": [frontend, worker, planner]},
    }


def planner_config(mount_path: Path, *, advisory: bool) -> PlannerConfig:
    return PlannerConfig(
        mode="agg",
        environment="kubernetes",
        advisory=advisory,
        enable_load_scaling=False,
        enable_throughput_scaling=False,
        load_adjustment_interval_seconds=int(SCALE_INTERVAL),
        scheduling={"scale_interval_seconds": SCALE_INTERVAL},
        retune={
            "interval_seconds": INTERVAL,
            "grace_seconds": GRACE,
            "timeout_seconds": TIMEOUT,
            "router": {
                "overlap_score_credit": {"choices": [0.5, 1.0]},
                "prefill_load_scale": 1.0,
                "temperature": {"choices": [0.0, 0.2]},
            },
            "executor": "dgdr",
            "dgdr": {
                "claim_name": CLAIM,
                "mount_path": str(mount_path),
                "model": MODEL,
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
                            "scheduler": {
                                "max_batched_tokens": 8192,
                                "max_sequences": 256,
                            },
                        },
                        "prefill": None,
                        "decode": None,
                    },
                },
                "run": {
                    "parallelism": 8,
                    "candidate_timeout_seconds": 600,
                    "max_shards": 3,
                },
                "cut_budget_seconds": 300,
                "job_startup_seconds": 60,
                "keep_requests": 1,
                "lease_duration_seconds": LEASE,
                "actuator_interval_seconds": STEP,
            },
        },
    )


# ------------------------------------------------------------------ fakes


class SimClock:
    """Wall and monotonic time, advanced only by the test."""

    def __init__(self, wall: float) -> None:
        self.wall = wall
        self.mono = 1_000.0

    def time(self) -> float:
        return self.wall

    def monotonic(self) -> float:
        return self.mono

    def advance(self, seconds: float) -> None:
        self.wall += seconds
        self.mono += seconds


async def inline(call):
    return call()


async def no_sleep(_seconds: float) -> None:
    return None


class RecordingMetrics(NullMetrics):
    def __init__(self) -> None:
        self.events: list[tuple[str, Any]] = []

    def decision(self, outcome: str) -> None:
        self.events.append(("decision", outcome))

    def failure(self, reason: str) -> None:
        self.events.append(("failure", reason))

    def skipped(self, reason: str) -> None:
        self.events.append(("skipped", reason))

    def patch_error(self, code: str) -> None:
        self.events.append(("patch_error", code))

    def of(self, kind: str) -> list[Any]:
        return [value for event, value in self.events if event == kind]


@dataclasses.dataclass
class FakeRouter:
    """One frontend-local router: startup values, current values, revision."""

    router_id: str
    initial: dict[str, float]
    parameters: dict[str, float]
    revision: int = 0
    worker_type: str = "aggregated"

    def info(self) -> RouterInfo:
        return RouterInfo(
            router_id=self.router_id,
            namespace=DYNAMO_NAMESPACE,
            worker_type=self.worker_type,
            revision=self.revision,
            configuration=dict(self.parameters),
            model=MODEL,
        )


@dataclasses.dataclass(frozen=True)
class PolicyWrite:
    """One PATCH or reset that reached a frontend, and the persisted retune
    state at that moment (to check persist-before-actuate)."""

    writer: str
    kind: str
    pod: str
    router_id: str
    expected_revision: int
    parameters: Optional[dict[str, Any]]
    status: int
    state: Optional[RetuneState]


class FakeFrontends:
    """The DGD's frontend pods and their routers."""

    def __init__(self, kube: FakeRetuneKube) -> None:
        self.kube = kube
        self.routers: dict[str, list[FakeRouter]] = {}
        self.writes: list[PolicyWrite] = []

    def start(self, pod: str) -> None:
        self.kube.add_pod(pod, dgd=DGD_NAME, component_type="frontend", ready=True)
        self.restart(pod)

    def restart(self, pod: str) -> None:
        """A new frontend process: a new router id at revision 0 with the
        startup values (``RouterPolicyRegistry::register``)."""

        startup = startup_map()
        self.routers[pod] = [
            FakeRouter(
                router_id=str(uuid.uuid4()),
                initial=dict(startup),
                parameters=dict(startup),
            )
        ]

    def router(self, pod: str) -> FakeRouter:
        (router,) = self.routers[pod]
        return router

    def router_ids(self) -> set[str]:
        return {self.router(pod).router_id for pod in FRONTENDS}

    def persisted_state(self) -> Optional[RetuneState]:
        lease = self.kube.leases.get(state_lease_name(DGD_NAME))
        if lease is None:
            return None
        return RetuneState.from_annotations(lease["metadata"].get("annotations"))


class FakeRouterPolicyClient:
    """``RouterPolicyClient`` with the ``/router_policy`` semantics of
    ``lib/llm/src/http/service/router_policy.rs`` and
    ``lib/kv-router/src/plugins/worker_selection/runtime.rs``: an empty patch
    is 400, an unknown router 404, a stale ``expected_revision`` 409, an
    unknown parameter or an invalid value 422; a PATCH replaces only the
    keys it names (``null`` restores that key's startup value), a reset
    restores every startup value, and both bump the revision."""

    def __init__(self, frontends: FakeFrontends, writer: str) -> None:
        self.frontends = frontends
        self.writer = writer
        self.gets = 0

    async def get(self, pod: PodInfo) -> PolicyCallResult:
        self.gets += 1
        routers = self.frontends.routers.get(pod.name)
        if routers is None:
            return PolicyCallResult(status=0, error="connection refused")
        return PolicyCallResult(
            status=200, routers=tuple(router.info() for router in routers)
        )

    async def patch(
        self,
        pod: PodInfo,
        router_id: str,
        expected_revision: int,
        parameters: Mapping[str, float],
    ) -> PolicyCallResult:
        return self._update(
            pod, "patch", router_id, expected_revision, dict(parameters)
        )

    async def reset(
        self, pod: PodInfo, router_id: str, expected_revision: int
    ) -> PolicyCallResult:
        return self._update(pod, "reset", router_id, expected_revision, None)

    def _update(
        self,
        pod: PodInfo,
        kind: str,
        router_id: str,
        expected_revision: int,
        parameters: Optional[dict[str, Any]],
    ) -> PolicyCallResult:
        status, router, error = self._apply(
            pod.name, router_id, expected_revision, parameters
        )
        self.frontends.writes.append(
            PolicyWrite(
                writer=self.writer,
                kind=kind,
                pod=pod.name,
                router_id=router_id,
                expected_revision=expected_revision,
                parameters=None if parameters is None else dict(parameters),
                status=status,
                state=self.frontends.persisted_state(),
            )
        )
        if status != 200 or router is None:
            return PolicyCallResult(status=status, error=error)
        return PolicyCallResult(status=200, router=router.info())

    def _apply(
        self,
        pod: str,
        router_id: str,
        expected_revision: int,
        parameters: Optional[dict[str, Any]],
    ) -> tuple[int, Optional[FakeRouter], Optional[str]]:
        if parameters is not None and not parameters:
            return 400, None, "provide either nonempty parameters or reset: true"
        router = next(
            (
                r
                for r in self.frontends.routers.get(pod, ())
                if r.router_id == router_id
            ),
            None,
        )
        if router is None:
            return 404, None, "router instance not found"
        if expected_revision != router.revision:
            return 409, None, "policy revision conflict"
        if parameters is None:
            updated = dict(router.initial)
        else:
            updated = dict(router.parameters)
            for key, value in parameters.items():
                if key not in router.initial:
                    return 422, None, f"unknown parameter {key!r}"
                if value is None:
                    updated[key] = router.initial[key]
                    continue
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    or value < 0
                ):
                    return 422, None, f"invalid value {value!r} for {key!r}"
                updated[key] = float(value)
        router.parameters = updated
        router.revision += 1
        return 200, router, None


class TraceClaim:
    """The RWX trace claim: ``<mount>/segments/<dgd>/<pod>/req.NNNNNN.jsonl.gz``."""

    def __init__(self, mount: Path) -> None:
        self.mount = mount
        self.segments = mount / "segments" / DGD_NAME
        self.windows = mount / "windows" / DGD_NAME
        self._index = 0
        for pod in FRONTENDS:
            (self.segments / pod).mkdir(parents=True)

    def write_window(self, n: int, now_s: float) -> None:
        """A new segment per frontend with window ``n``'s request_end records,
        one gzip member per record as the frontend's gzip sink flushes."""

        self._index += 1
        start = end_ms(n - 1)
        span = end_ms(n) - start
        for p, pod in enumerate(FRONTENDS):
            path = self.segments / pod / f"req.{self._index:06d}.jsonl.gz"
            with open(path, "wb") as sink:
                for i in range(REQUESTS_PER_POD):
                    received = start + (i + 1) * span // (REQUESTS_PER_POD + 1) + p
                    total = 400 + 25 * i
                    record = {
                        "schema": TRACE_SCHEMA,
                        "event_type": "request_end",
                        "event_time_unix_ms": received + total,
                        "request": {
                            "request_id": f"w{n}-{pod}-{i}",
                            "model": MODEL,
                            "request_received_ms": received,
                            "total_time_ms": float(total),
                            "output_tokens": 16 + i,
                            "replay": {
                                "trace_block_size": 64,
                                "input_length": 256 + 8 * i,
                                "input_sequence_hashes": [n, p, i, 11 * i + 3],
                            },
                        },
                    }
                    envelope = {"timestamp": received + total, "event": record}
                    sink.write(gzip.compress(json.dumps(envelope).encode() + b"\n"))
            os.utime(path, (now_s, now_s))

    def segment_files(self, pod: str) -> list[str]:
        return sorted(os.listdir(self.segments / pod))

    def window_files(self) -> set[int]:
        if not self.windows.is_dir():
            return set()
        return {
            int(name.split(".", 1)[0])
            for name in os.listdir(self.windows)
            if name.endswith(".jsonl.gz")
        }


# ---------------------------------------------------------------- planners


class PlannerHost:
    """The slice of ``NativePlannerBase`` the supervisor calls back into."""

    def __init__(self, engine: OrchestratorEngineAdapter, clock: SimClock) -> None:
        self._engine = engine
        self._config_lock = asyncio.Lock()
        self._clock = clock

    async def replace_live_retune(self, inputs, *, generation, last_decided_k):
        async with self._config_lock:
            self._engine.replace_retune(
                inputs,
                generation=generation,
                now_s=self._clock.time(),
                last_decided_k=last_decided_k,
            )


class Planner:
    """One planner process: its supervisor, engine, actuator and reconciler."""

    def __init__(self, world: World, name: str, *, advisory: bool) -> None:
        self.world = world
        self.name = name
        self.identity = f"{name}/{uuid.uuid4().hex[:12]}"
        self.config = planner_config(world.claim.mount, advisory=advisory)
        self.metrics = RecordingMetrics()
        self.client = FakeRouterPolicyClient(world.frontends, name)
        self.decisions: list[RetuneDecision] = []
        self.router_parameters: Optional[dict[str, dict[str, float]]] = None
        self.supervisor: LiveRetuneSupervisor
        self.engine: OrchestratorEngineAdapter
        self.host: PlannerHost

    async def boot(self) -> None:
        """``NativePlannerBase._async_init`` then ``run``'s first tick."""

        world = self.world
        clock = world.clock
        retune = self.config.retune
        assert retune is not None and retune.dgdr is not None
        reconciler = Reconciler(
            world.kube,
            world.dgd_ref,
            retune.dgdr,
            retune,
            retune.dgdr.mount_path,
            self.metrics,
            now=clock.time,
        )
        contexts = bootstrap.dgdr_context_factory(self.config, cut_window=world.cutter)

        def context_factory(supervisor: LiveRetuneSupervisor) -> Any:
            # The executor's clock is the simulated one; each poll of the
            # sweep lets the DGDR operator advance one phase.
            return dataclasses.replace(
                contexts(supervisor),
                now=clock.time,
                sleep=world.operator_sleep,
                poll_interval_s=5.0,
            )

        self.supervisor = await bootstrap.build_supervisor(
            self.config,
            kube=world.kube,
            dgd_name=DGD_NAME,
            namespace_fn=lambda: DYNAMO_NAMESPACE,
            metrics=self.metrics,
            identity=self.identity,
            actuator_factory=lambda dgd, maps: RouterActuator(
                world.kube, self.client, dgd, maps, self.metrics
            ),
            reconciler=reconciler,
            context_factory=context_factory,
            clock=clock.time,
            monotonic=clock.monotonic,
            sleep=no_sleep,
            blocking=inline,
        )
        self.engine = OrchestratorEngineAdapter(
            self.config,
            caps(),
            clock=WallClock(),
            retune_inputs=self.supervisor.retune_inputs(),
            retune_hooks=self.supervisor.hooks(),
        )
        self.host = PlannerHost(self.engine, clock)
        await self.supervisor.start(self.host, background=False)
        self.next_tick = self.engine.initial_tick(clock.time())

    async def step(self) -> None:
        supervisor = self.supervisor
        await supervisor.lease_round()
        if self.world.clock.time() >= self.next_tick.at_s:
            await self.tick()
        await supervisor.outbox_round()
        await supervisor.control_round()
        # The reconciler runs as its own task; finish it to keep each step
        # deterministic.
        if supervisor.reconcile_task is not None:
            await supervisor.reconcile_task

    async def tick(self) -> None:
        """``NativePlannerBase._run_one_tick`` without the environment."""

        counts = WorkerCounts(
            ready_num_decode=WORKER_REPLICAS, expected_num_decode=WORKER_REPLICAS
        )
        async with self.host._config_lock:
            effects = await self.engine.tick(
                self.next_tick,
                TickInput(now_s=self.world.clock.time(), worker_counts=counts),
            )
        self.supervisor.on_effects(effects)
        assert effects.next_tick is not None
        self.next_tick = effects.next_tick
        self.decisions.extend(effects.retune_decisions)
        self.router_parameters = effects.router_parameters
        # Live, the sweep runs on the controller's thread while ticks go on;
        # finishing it here keeps the interleaving deterministic.
        controller = self.engine.retune_controller
        pending = getattr(controller, "_pending", None)
        if pending is not None and not pending.future.done():
            await asyncio.wait_for(asyncio.wrap_future(pending.future), timeout=60)

    async def stop(self) -> None:
        """SIGTERM: ``_shutdown_runtime`` stops the engine, then the supervisor."""

        self.world.planners.remove(self)
        await self.engine.shutdown()
        await self.supervisor.stop()

    def crash(self) -> None:
        """SIGKILL: nothing of this process runs again; its Lease stays."""

        self.world.planners.remove(self)
        self.world.crashed.append(self)

    def decision(self, n: int) -> RetuneDecision:
        (found,) = [d for d in self.decisions if d.decision_id == decision_id(n)]
        return found


class World:
    """The cluster, the frontends, the trace claim and the clocks."""

    def __init__(self, tmp_path: Path, *, spawn_cutter: bool = False) -> None:
        # Boot inside window 0: first enable skips it (its trace starts
        # mid-way), so window 1 is the first one recorded in full.
        self.clock = SimClock(T0 - 600.0)
        self.kube = FakeRetuneKube(K8S_NAMESPACE, now=self.clock.time)
        self.kube.put_dgd(dgd_object())
        self.dgd_ref = DgdRef(DGD_NAME, K8S_NAMESPACE, DGD_UID, DYNAMO_NAMESPACE)
        self.frontends = FakeFrontends(self.kube)
        for pod in FRONTENDS:
            self.frontends.start(pod)
        self.claim = TraceClaim(tmp_path / "claim")
        # window end ms -> the router point (live names) the replay favours.
        self.preferences: dict[int, Mapping[str, float]] = {}
        self.operator = FakeDgdrOperator(self.kube, evaluate=self._evaluate)
        self.spawn_cutter = spawn_cutter
        self.cuts: list[tuple[int, int, bool]] = []
        self.polls = 0
        self.planners: list[Planner] = []
        self.crashed: list[Planner] = []
        self._next_trace_window = 1

    # ----------------------------------------------------------- edges

    def prefer(self, n: int, point: Mapping[str, float]) -> None:
        self.preferences[end_ms(n)] = dict(point)

    def _evaluate(self, request: dict, point: Mapping[str, float]) -> dict:
        """Replay summary of a router point (AISimulate public names): the
        window's preferred point scores best; deterministic, so the CRN
        probe always matches its grid twin."""

        window = int(request["metadata"]["labels"][LABEL_WINDOW])
        preferred = self.preferences.get(window, STARTUP)
        distance = abs(point["overlap_score_credit"] - preferred[CREDIT]) + abs(
            point["temperature"] - preferred[TEMP]
        )
        metrics = default_metrics(point)
        metrics["goodput_output_throughput_tok_s"] = 1000.0 - 200.0 * distance
        return metrics

    def operator_sleep(self, _seconds: float) -> None:
        self.polls += 1
        if self.polls > 10_000:
            raise AssertionError("the DGDR sweep never finished polling")
        self.operator.step()

    def cutter(
        self,
        segments_root,
        start_ms,
        end_ms_,
        as_of_ms,
        out_path,
        *,
        max_rows,
        deadline_s,
        cancel=None,
    ):
        if self.spawn_cutter:
            # The child runs on the real clock; keep the remaining budget.
            real_deadline = time.time() + (deadline_s - self.clock.time())
            result = cut_window(
                segments_root,
                start_ms,
                end_ms_,
                as_of_ms,
                out_path,
                max_rows=max_rows,
                deadline_s=real_deadline,
                cancel=cancel,
            )
        else:
            result = cut_window_inprocess(
                segments_root, start_ms, end_ms_, as_of_ms, out_path, max_rows=max_rows
            )
        self.cuts.append((int(end_ms_), result.num_requests, result.reused))
        return result

    # ------------------------------------------------------------ driving

    async def boot(self, name: str, *, advisory: bool = False) -> Planner:
        planner = Planner(self, name, advisory=advisory)
        await planner.boot()
        self.planners.append(planner)
        return planner

    async def run_until(self, until_s: float) -> None:
        while self.clock.wall < until_s:
            self.clock.advance(min(STEP, until_s - self.clock.wall))
            while self.clock.wall >= end_s(self._next_trace_window):
                self.claim.write_window(self._next_trace_window, self.clock.wall)
                self._next_trace_window += 1
            for planner in list(self.planners):
                await planner.step()

    async def close(self) -> None:
        for planner in list(self.planners):
            await planner.stop()
        for planner in self.crashed:
            await planner.engine.shutdown()

    # ---------------------------------------------------------- inspection

    def state(self) -> RetuneState:
        state = self.frontends.persisted_state()
        assert state is not None
        return state

    def history(self, decision: str) -> dict[str, Any]:
        (entry,) = [e for e in self.state().history if e["decision_id"] == decision]
        return entry

    def lease_spec(self) -> dict[str, Any]:
        return self.kube.leases[leadership_lease_name(DGD_NAME)]["spec"]

    def requests_of(self, n: int) -> dict[str, dict]:
        return {
            name: obj
            for name, obj in self.kube.requests.items()
            if obj["metadata"]["labels"][LABEL_WINDOW] == str(end_ms(n))
        }

    def request_windows(self) -> set[int]:
        return {
            int(obj["metadata"]["labels"][LABEL_WINDOW])
            for obj in self.kube.requests.values()
        }

    def created(self, n: Optional[int] = None) -> list[str]:
        """Names of the stored (non dry-run) Request creates, in order."""

        return [
            call.target["metadata"]["name"]
            for call in self.kube.calls_of("create_request")
            if not call.kwargs["dry_run"]
            and (
                n is None
                or call.target["metadata"]["labels"][LABEL_WINDOW] == str(end_ms(n))
            )
        ]

    def canaries(self) -> int:
        return sum(
            1 for call in self.kube.calls_of("create_request") if call.kwargs["dry_run"]
        )

    def deleted(self) -> set[str]:
        return {name for name, _ in self.kube.deleted_requests}

    def configurations(self) -> dict[str, tuple[int, dict[str, float]]]:
        return {
            pod: (
                self.frontends.router(pod).revision,
                self.frontends.router(pod).parameters,
            )
            for pod in FRONTENDS
        }


def converged_to(world: World, point: Mapping[str, float], revision: int) -> None:
    expected = by_wt(point)["aggregated"]
    assert world.configurations() == {pod: (revision, expected) for pod in FRONTENDS}


@pytest.fixture
async def make_world(tmp_path):
    worlds: list[World] = []

    def make(**kwargs: Any) -> World:
        world = World(tmp_path, **kwargs)
        worlds.append(world)
        return world

    yield make
    for world in worlds:
        await world.close()


# ------------------------------------------------------------------ tests


@pytest.mark.timeout(180)
async def test_advisory_records_recommendations_and_never_writes(make_world):
    world = make_world(spawn_cutter=True)
    recommended_1 = {CREDIT: 0.5, LOAD: 1.0, TEMP: 0.0}
    recommended_2 = {CREDIT: 0.5, LOAD: 1.0, TEMP: 0.2}
    world.prefer(1, recommended_1)
    world.prefer(2, recommended_2)
    planner = await world.boot("planner-a", advisory=True)

    await world.run_until(launch_s(1) - STEP)
    supervisor = planner.supervisor
    assert supervisor.is_leader() and supervisor.initialized and supervisor.ready()
    state = world.state()
    # First enable: an advisory record of the startup point; the window in
    # progress at start-up (its trace began mid-way) counts as decided.
    assert (state.mode, state.rollback_pending, state.deploy_seq) == (
        "advisory",
        False,
        1,
    )
    assert state.flat == STARTUP and state.by_wt == by_wt(STARTUP)
    assert state.last_window_end_ms == end_ms(0)
    assert state.initials == {
        router_id: startup_map() for router_id in world.frontends.router_ids()
    }
    assert planner.decisions == [] and world.created() == []
    assert world.canaries() >= 1 and world.kube.requests == {}

    # Window 1: swept through DGDR, recorded as a recommendation only.
    await world.run_until(launch_s(1) + SCALE_INTERVAL + 2 * STEP)
    decision = planner.decision(1)
    assert decision.outcome == "advisory", decision
    assert decision.previous.router == STARTUP
    assert decision.chosen.router == recommended_1
    assert decision.num_requests == ROWS_PER_WINDOW
    assert decision.failure_reason is None
    assert decision.best_score > decision.baseline_score
    details = decision.details
    assert details["window"]["cut"] is True
    assert details["window"]["num_requests"] == ROWS_PER_WINDOW
    assert details["window"]["dropped_members"] == 0
    assert details["missing_candidates"] == 0
    assert details["crn_probe"]["relative_error"] == 0.0
    assert set(details["shards"]) == {"grid-0", "probe"}
    assert world.cuts == [(end_ms(1), ROWS_PER_WINDOW, False)]

    state = world.state()
    assert state.mode == "advisory" and state.flat == STARTUP
    assert state.by_wt == by_wt(STARTUP) and state.deploy_seq == 1
    assert state.last_window_end_ms == end_ms(1)
    entry = world.history(decision_id(1))
    assert entry["outcome"] == "advisory"
    assert entry["flat"] == recommended_1
    assert entry["by_wt"] == by_wt(recommended_1)
    assert supervisor.target().mode == "observe"
    assert supervisor.last_report.converged
    assert planner.router_parameters == by_wt(STARTUP)

    shards = {
        obj["metadata"]["labels"][LABEL_SHARD] for obj in world.requests_of(1).values()
    }
    assert shards == {"grid-0", "probe"}
    assert all(
        obj["metadata"]["labels"][LABEL_DGD_UID] == DGD_UID
        for obj in world.kube.requests.values()
    )

    # Window 2: another recommendation; window 1's Requests are cleaned up
    # once window 2's are terminal (keep_requests: 1).
    window_1_requests = set(world.requests_of(1))
    await world.run_until(launch_s(2) + SCALE_INTERVAL + 2 * STEP)
    decision = planner.decision(2)
    assert decision.outcome == "advisory"
    assert decision.previous.router == STARTUP
    assert decision.chosen.router == recommended_2
    assert world.history(decision_id(2))["by_wt"] == by_wt(recommended_2)
    assert world.request_windows() == {end_ms(2)}
    assert window_1_requests <= world.deleted()
    assert world.claim.window_files() == {end_ms(2)}

    # Advisory never writes a router.
    assert world.frontends.writes == []
    converged_to(world, STARTUP, revision=0)
    assert planner.metrics.of("decision") == ["advisory", "advisory"]
    assert planner.metrics.of("patch_error") == []


@pytest.mark.timeout(180)
async def test_apply_restart_handover_and_advisory_rollback(make_world):
    world = make_world()
    applied_1 = {CREDIT: 0.5, LOAD: 1.0, TEMP: 0.0}
    applied_3 = {CREDIT: 0.5, LOAD: 1.0, TEMP: 0.2}
    world.prefer(1, applied_1)
    world.prefer(2, applied_1)
    world.prefer(3, applied_3)
    world.prefer(4, applied_1)
    pods = sorted(FRONTENDS)

    # ---------------------------------------- active: apply a better point
    p1 = await world.boot("planner-a")
    await world.run_until(launch_s(1) - STEP)
    state = world.state()
    assert (state.mode, state.rollback_pending, state.deploy_seq) == (
        "active",
        False,
        1,
    )
    assert state.flat == STARTUP and state.last_window_end_ms == end_ms(0)
    assert state.initials == {
        router_id: startup_map() for router_id in world.frontends.router_ids()
    }
    assert world.frontends.writes == [] and p1.decisions == []

    await world.run_until(launch_s(1) + SCALE_INTERVAL + 2 * STEP)
    decision = p1.decision(1)
    assert decision.outcome == "applied", decision
    assert decision.previous.router == STARTUP
    assert decision.chosen.router == applied_1
    assert decision.num_requests == ROWS_PER_WINDOW
    assert decision.best_score > decision.baseline_score
    assert decision.details["replicas"] == {"aggregated": WORKER_REPLICAS}
    # The planner started before window 1, so its replicas were observed.
    assert decision.details["replicas_verified"] is True
    assert decision.details["crn_probe"]["relative_error"] == 0.0

    state = world.state()
    assert (state.mode, state.rollback_pending, state.deploy_seq) == (
        "active",
        False,
        2,
    )
    assert state.flat == applied_1 and state.by_wt == by_wt(applied_1)
    assert state.last_window_end_ms == end_ms(1)
    assert world.history(decision_id(1))["by_wt"] == by_wt(applied_1)

    # The routers converge with explicit values, only for the changed key,
    # and only after the decision is persisted.
    writes = world.frontends.writes
    assert sorted(
        (w.pod, w.kind, w.expected_revision, w.parameters, w.status) for w in writes
    ) == [(pod, "patch", 0, {CREDIT: 0.5}, 200) for pod in pods]
    assert all(w.writer == "planner-a" for w in writes)
    assert all(w.state.flat == applied_1 and w.state.deploy_seq == 2 for w in writes)
    converged_to(world, applied_1, revision=1)
    assert p1.supervisor.last_report.converged
    assert p1.router_parameters == by_wt(applied_1)
    window_1_requests = set(world.requests_of(1))
    assert {
        obj["metadata"]["labels"][LABEL_SHARD] for obj in world.requests_of(1).values()
    } == {"grid-0", "probe"}
    assert sorted(world.created()) == sorted(window_1_requests)

    # A frontend restarts: its new router (revision 0, startup values) is
    # latched and re-applied with explicit values; the persisted initials
    # follow the new router id (the final rollback depends on them).
    restarted = FRONTENDS[1]
    gone = world.frontends.router(restarted).router_id
    world.frontends.restart(restarted)
    writes_before = len(world.frontends.writes)
    await world.run_until(launch_s(1) + SCALE_INTERVAL + 4 * STEP)
    reapplied = world.frontends.writes[writes_before:]
    assert [
        (w.pod, w.kind, w.expected_revision, w.parameters, w.status) for w in reapplied
    ] == [(restarted, "patch", 0, {CREDIT: 0.5}, 200)]
    converged_to(world, applied_1, revision=1)
    state = world.state()
    assert gone not in state.initials
    assert state.initials == {
        router_id: startup_map() for router_id in world.frontends.router_ids()
    }
    assert state.deploy_seq == 2

    # -------------------------------- restart: restore, never roll back
    restart_at = launch_s(1) + 600.0
    await world.run_until(restart_at)
    await p1.stop()
    assert not world.lease_spec().get("holderIdentity")  # released
    writes_before = len(world.frontends.writes)
    p2 = await world.boot("planner-b")
    await world.run_until(restart_at + 5 * STEP)
    assert p2.supervisor.is_leader() and p2.supervisor.initialized
    assert world.lease_spec()["holderIdentity"] == p2.identity
    assert world.lease_spec()["leaseTransitions"] == 1
    state = world.state()
    assert state.leader_term.startswith(f"1/{p2.identity}/")
    assert (state.mode, state.rollback_pending, state.deploy_seq) == (
        "active",
        False,
        2,
    )
    assert state.flat == applied_1 and state.by_wt == by_wt(applied_1)
    controller = p2.engine.retune_controller
    assert controller is not None and controller.current_point().router == applied_1
    assert p2.router_parameters == by_wt(applied_1)
    assert len(world.frontends.writes) == writes_before
    converged_to(world, applied_1, revision=1)

    # The decided window is not swept again.
    await world.run_until(launch_s(2) - STEP)
    assert p2.decisions == []
    assert sorted(world.created()) == sorted(window_1_requests)
    assert [cut[0] for cut in world.cuts] == [end_ms(1)]

    # Window 2: the restarted planner keeps the applied point.
    await world.run_until(launch_s(2) + SCALE_INTERVAL + 2 * STEP)
    kept = p2.decision(2)
    assert kept.outcome == "kept", kept
    assert kept.previous.router == applied_1 and kept.chosen.router == applied_1
    assert kept.best_score == kept.baseline_score
    state = world.state()
    assert state.deploy_seq == 2 and state.last_window_end_ms == end_ms(2)
    assert len(world.frontends.writes) == writes_before
    # Window 1's Requests and window file are gone (keep_requests: 1).
    assert world.request_windows() == {end_ms(2)}
    assert window_1_requests <= world.deleted()
    assert world.claim.window_files() == {end_ms(2)}
    window_2_requests = set(world.requests_of(2))

    # ---------------------------- leadership handover (holder crashes)
    p3 = await world.boot("planner-c")
    await world.run_until(launch_s(3))
    # The leader launched window 3 (its Requests all succeeded); the
    # standby never launched anything.
    assert not p3.supervisor.is_leader() and not p3.supervisor.initialized
    assert p3.decisions == [] and "not_leader" in p3.metrics.of("skipped")
    window_3_requests = set(world.requests_of(3))
    assert window_3_requests and set(world.created(3)) == window_3_requests
    assert [d.decision_id for d in p2.decisions] == [decision_id(2)]
    transitions = world.lease_spec()["leaseTransitions"]
    p2.crash()
    await world.run_until(launch_s(3) + LEASE + 2 * SCALE_INTERVAL + 2 * STEP)
    assert p3.supervisor.is_leader() and p3.supervisor.initialized
    assert world.lease_spec()["holderIdentity"] == p3.identity
    assert world.lease_spec()["leaseTransitions"] == transitions + 1

    decision = p3.decision(3)
    assert decision.outcome == "applied", decision
    assert decision.previous.router == applied_1
    assert decision.chosen.router == applied_3
    assert decision.num_requests == ROWS_PER_WINDOW
    # The new leader adopted the crashed leader's Requests: no new create,
    # no second cut.
    assert all(shard["adopted"] for shard in decision.details["shards"].values())
    assert decision.details["window"]["cut"] is False
    assert set(world.created(3)) == window_3_requests
    assert [cut[0] for cut in world.cuts] == [end_ms(1), end_ms(2), end_ms(3)]

    state = world.state()
    assert state.leader_term.startswith(f"{transitions + 1}/{p3.identity}/")
    assert (state.mode, state.deploy_seq) == ("active", 3)
    assert state.flat == applied_3 and state.by_wt == by_wt(applied_3)
    new_writes = world.frontends.writes[writes_before:]
    assert sorted(
        (w.pod, w.kind, w.expected_revision, w.parameters, w.status) for w in new_writes
    ) == [(pod, "patch", 1, {TEMP: 0.2}, 200) for pod in pods]
    assert all(
        w.writer == "planner-c" and w.state.flat == applied_3 for w in new_writes
    )
    converged_to(world, applied_3, revision=2)

    # ----------------------- switch to advisory: two-phase rollback
    switch_at = launch_s(3) + 600.0
    await world.run_until(switch_at)
    await p3.stop()
    writes_before = len(world.frontends.writes)
    p4 = await world.boot("planner-d", advisory=True)
    await world.run_until(switch_at + 2 * STEP)
    state = world.state()
    # Phase 1: rollback_pending persisted, then the routers are reset.
    assert (state.mode, state.rollback_pending, state.deploy_seq) == (
        "advisory",
        True,
        4,
    )
    assert state.flat == STARTUP and state.by_wt == by_wt(STARTUP)
    resets = world.frontends.writes[writes_before:]
    assert sorted(
        (w.pod, w.kind, w.expected_revision, w.parameters, w.status) for w in resets
    ) == [(pod, "reset", 2, None, 200) for pod in pods]
    assert all(
        w.writer == "planner-d"
        and w.state.rollback_pending
        and w.state.mode == "advisory"
        and w.state.deploy_seq == 4
        for w in resets
    )
    converged_to(world, STARTUP, revision=3)
    assert p4.supervisor.target().mode == "rollback"

    # Phase 2: converged for a lease duration -> rollback complete.
    await world.run_until(switch_at + 2 * STEP + LEASE + 3 * STEP)
    state = world.state()
    assert (state.mode, state.rollback_pending, state.deploy_seq) == (
        "advisory",
        False,
        4,
    )
    last = state.history[-1]
    assert (last["outcome"], last["deploy_seq"]) == ("rolled_back", 4)
    assert p4.supervisor.target().mode == "observe"
    assert len(world.frontends.writes) == writes_before + len(pods)

    # Window 4 under advisory: a recommendation, no write.
    await world.run_until(launch_s(4) + SCALE_INTERVAL + 2 * STEP)
    decision = p4.decision(4)
    assert decision.outcome == "advisory", decision
    assert decision.previous.router == STARTUP
    assert decision.chosen.router == applied_1
    state = world.state()
    assert state.flat == STARTUP and state.last_window_end_ms == end_ms(4)
    assert len(world.frontends.writes) == writes_before + len(pods)
    converged_to(world, STARTUP, revision=3)

    # Clean-up across the four windows: only the newest window's Requests
    # and window file remain, and old segments are gone (the newest of each
    # pod is never deleted).
    assert world.request_windows() == {end_ms(4)}
    assert window_1_requests | window_2_requests | window_3_requests <= world.deleted()
    assert world.claim.window_files() == {end_ms(4)}
    for pod in FRONTENDS:
        assert world.claim.segment_files(pod) == [
            "req.000003.jsonl.gz",
            "req.000004.jsonl.gz",
        ]
    for planner in (p1, p2, p3, p4):
        assert planner.metrics.of("patch_error") == []
