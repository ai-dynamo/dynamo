# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Process-level coverage for each backend worker's admission gate.

The gate sits at the backend's typed request boundary rather than in the
request plane: the mocker serves its engine through the native backend's
admission wrapper, which admits each `PreprocessedRequest` before the engine
sees it, so a CPU-only mocker worker exercises the real thing over TCP and NATS
alike. Each worker owns its gate, so workers sharing one mocker process never
share capacity.

The oracle here is the worker's own `dynamo_backend_admission_*` family, scraped
from its system port. Every series carries the worker's standard owner labels —
its namespace, component and `worker_id` — and the `dp_rank` label the gate
admitted the request under: a rank the worker serves, or `unspecified`. The
frontend routes round-robin, which assigns no rank, so a request is unspecified
unless the test pins it to a worker and rank with the
`x-dynamo-worker-instance-id` and `x-dynamo-dp-rank` headers; the frontend then
names that rank in the request body's `routing.dp_rank`, which the worker's gate
reads from the decoded request, and the mocker serves the request on the engine
of that rank, whose own per-engine gauges the multi-rank scenario compares
against the gate's.

A refusal must reach the client as an overload — HTTP 529 under the default
configuration — and the frontend must never report the refusing worker down, so
the healthy worker keeps serving once capacity returns. When the worker could
neither admit nor queue any more work, the refusal is the whole worker's, and
the frontend marks the worker overloaded instead. The frontend's router client
records both as structured `routing_event` log records, which the shared
`tests.utils.router_logs` helpers parse: each whole-worker refusal must add a
`worker_overloaded` record for the worker, and no scenario may ever produce a
`worker_inhibited` one. For the length of that overload lease (one second) the
frontend refuses new work for the worker itself, so a scenario that sends new
requests after such a refusal waits the lease out first. A refusal limited to
the request's DP rank, while the worker could still take other work, marks
nothing, so the worker's other ranks keep receiving work at once.

The gate has two engine limits, applied per rank and across the worker. A
request holds engine request capacity from admission until it finishes, and
engine waiting capacity from admission until its first generated token or
completion, whichever comes first. The mocker's first response to a request
carries its first token, so "first response" below means that token. The mocker
runs a fixed number of sequences at a time per rank: a long stream started first
keeps a sequence busy, holding request capacity but no waiting capacity once it
has answered, and a request the gate admits while every sequence is busy waits
inside the engine without a response, holding both, until the test closes a
stream ahead of it.

A caller that disconnects while its request is queued is a separate question
from the gate's own cancellation handling, which its unit tests cover: whether
the cancellation reaches the queued request at all. On the default path the
frontend forwards it to the worker before the response prologue, and the
disconnect scenario below shows the gate reclaiming the place before admission.
A request whose cancellation the frontend defers until the prologue — one
carrying a first-response guard for decoded multimodal media — cannot be
reclaimed while queued, since the worker sends the prologue only after
admission; that path is not exercised here.

Observation is metrics-driven rather than sleep-driven: each step polls until
the gauges show the state it needs, and counters are asserted as deltas from a
baseline taken after readiness, never as process-global absolutes.

Each scenario runs one mocker and one frontend of its own, on ports from
`dynamo_dynamic_ports`, and shares only the session-scoped NATS and etcd, so the
whole file is safe under xdist process parallelism; each mocker keeps its own
generated discovery namespace. `MockerProcess` and `FrontendRouterProcess` both
snapshot `os.environ` in their constructors, which is why the gate configuration
is patched around construction alone — the parallelism here is between
processes, never threads mutating one environment.
"""

import asyncio
import contextlib
import functools
import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Coroutine, Iterator

import aiohttp
import pytest

from tests.router.helper import (
    managed_runtime,
    poll_for_worker_instances,
    wait_for_frontend_ready,
)
from tests.router.mocker_process import MockerProcess
from tests.router.router_process import FrontendRouterProcess
from tests.utils.constants import FAULT_TOLERANCE_MODEL_NAME
from tests.utils.port_utils import ServicePorts
from tests.utils.router_logs import (
    LogReadable,
    wait_for_worker_routing_events,
    worker_routing_events,
)
from tests.utils.router_nvext import require_router_worker_id

logger = logging.getLogger(__name__)

MODEL_NAME = FAULT_TOLERANCE_MODEL_NAME
BLOCK_SIZE = 16

# Every setting the gate reads. A scenario sets the ones it names and clears the
# rest, so an ambient value can never decide what it is testing.
ENGINE_REQUEST_LIMIT_ENV = "DYN_BACKEND_ADMISSION_ENGINE_REQUEST_LIMIT"
LEGACY_ENGINE_REQUEST_LIMIT_ENV = "DYN_ENGINE_REQUEST_LIMIT"
ENGINE_REQUEST_WAITING_LIMIT_ENV = "DYN_BACKEND_ADMISSION_ENGINE_REQUEST_WAITING_LIMIT"
FORMER_ENGINE_WAIT_LIMIT_ENV = "DYN_BACKEND_ADMISSION_ENGINE_WAIT_LIMIT"
REQUEST_QUEUE_LIMIT_ENV = "DYN_BACKEND_ADMISSION_REQUEST_QUEUE_LIMIT"
LEGACY_REQUEST_QUEUE_LIMIT_ENV = "DYN_DYNAMO_REQUEST_QUEUE_LIMIT"
ADMISSION_SETTINGS = (
    ENGINE_REQUEST_LIMIT_ENV,
    LEGACY_ENGINE_REQUEST_LIMIT_ENV,
    ENGINE_REQUEST_WAITING_LIMIT_ENV,
    FORMER_ENGINE_WAIT_LIMIT_ENV,
    REQUEST_QUEUE_LIMIT_ENV,
    LEGACY_REQUEST_QUEUE_LIMIT_ENV,
)

# How long a poll may wait for the gate to reach an expected state.
SETTLE_TIMEOUT_S = 30
# How long a request may take to produce its outcome after the capacity it
# waits for is released.
OUTCOME_TIMEOUT_S = 60
# How long the frontend refuses new work for a worker after the worker refused
# a request for capacity (`REQUEST_PATH_OVERLOAD_COOLDOWN`), plus margin.
OVERLOAD_LEASE_S = 1.5
# The status the frontend answers an overload with by default.
OVERLOADED = 529

PREFIX = "dynamo_backend_admission"
ENGINE_REQUEST_COUNT = f"{PREFIX}_engine_request_count"
ENGINE_REQUEST_WAITING_COUNT = f"{PREFIX}_engine_request_waiting_count"
REQUEST_QUEUE_COUNT = f"{PREFIX}_request_queue_count"
REQUEST_RECEIVE_TOTAL = f"{PREFIX}_request_receive_total"
REQUEST_ADMIT_TOTAL = f"{PREFIX}_request_admit_total"
REJECTION_TOTAL = f"{PREFIX}_rejection_total"
CANCELLATION_TOTAL = f"{PREFIX}_cancellation_total"

# The whole family, by type: no limit gauges.
FAMILIES = {
    ENGINE_REQUEST_COUNT: "gauge",
    ENGINE_REQUEST_WAITING_COUNT: "gauge",
    REQUEST_QUEUE_COUNT: "gauge",
    REQUEST_RECEIVE_TOTAL: "counter",
    REQUEST_ADMIT_TOTAL: "counter",
    REJECTION_TOTAL: "counter",
    CANCELLATION_TOTAL: "counter",
}
GAUGES = (ENGINE_REQUEST_COUNT, ENGINE_REQUEST_WAITING_COUNT, REQUEST_QUEUE_COUNT)

# The `dp_rank` of requests the gate admitted without a usable rank.
UNSPECIFIED = "unspecified"

# The runtime's standard labels every series of a worker's family carries,
# naming the worker whose gate it is.
OWNER_LABELS = ("dynamo_component", "dynamo_namespace", "worker_id")

# Each outcome counter, by the short name scenarios use, as (family, labels
# beside `dp_rank`). Rejections carry no reason label.
COUNTERS = {
    "received": (REQUEST_RECEIVE_TOTAL, {}),
    "direct": (REQUEST_ADMIT_TOTAL, {"source": "direct"}),
    "queue": (REQUEST_ADMIT_TOTAL, {"source": "queue"}),
    "rejected": (REJECTION_TOTAL, {}),
    "cancelled": (CANCELLATION_TOTAL, {}),
}

# The mocker's own per-rank engine occupancy (it simulates vLLM by default).
ENGINE_RUNNING = "vllm:num_requests_running"
ENGINE_WAITING = "vllm:num_requests_waiting"

# The frontend logs JSONL, with its router client's routing decisions. The
# canonical format name is set as well as the legacy flag, which an inherited
# canonical setting would otherwise override.
FRONTEND_LOG_ENV = {
    "DYN_LOGGING_CONSOLE_FORMAT": "jsonl",
    "DYN_LOGGING_JSONL": "1",
    "DYN_LOG": "info,dynamo_runtime::component::client=debug",
}

Series = tuple[str, tuple[tuple[str, str], ...]]
_LABEL = re.compile(r'(\w+)="((?:[^"\\]|\\.)*)"')


def _parse_series(series: str) -> Series:
    """`name{a="1",b="2"}` as (`name`, sorted label pairs)."""
    name, _, labels = series.partition("{")
    return name, tuple(sorted(_LABEL.findall(labels)))


def _key(name: str, **labels: str) -> Series:
    return name, tuple(sorted(labels.items()))


def _label_value(name: str) -> str:
    """A namespace or component name as the runtime's label values spell it."""
    return re.sub(r"[^a-zA-Z0-9_]", "_", name)


def _payload(max_tokens: int, stream: bool) -> dict[str, Any]:
    return {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": "admission gate"}],
        "stream": stream,
        "max_tokens": max_tokens,
        "ignore_eos": True,
    }


def _pin(worker_id: int, dp_rank: int) -> dict[str, str]:
    """Headers that route a request to one worker and DP rank."""
    return {
        "x-dynamo-worker-instance-id": str(worker_id),
        "x-dynamo-dp-rank": str(dp_rank),
    }


async def _unary(
    session: aiohttp.ClientSession, url: str, headers: dict[str, str] | None = None
) -> tuple[int, str]:
    """A one-token request that answers in one piece. The body is read as text,
    because a generic pre-stream failure need not be JSON."""
    async with session.post(url, json=_payload(1, False), headers=headers) as response:
        return response.status, await response.text()


async def _hold(
    session: aiohttp.ClientSession, url: str, headers: dict[str, str] | None = None
) -> aiohttp.ClientResponse:
    """Start a long stream and return it once it has begun answering.

    Cancelled before then, as cleanup does to streams still waiting for their
    first response, it closes the response it already owns rather than leaving
    it to the garbage collector.
    """
    response = await session.post(url, json=_payload(2000, True), headers=headers)
    return await _first_chunk(response)


async def _first_chunk(response: aiohttp.ClientResponse) -> aiohttp.ClientResponse:
    """`response`, a long stream, once it has begun answering; closed if it
    fails, or is cancelled, first."""
    try:
        assert response.status == 200, await response.text()
        async for line in response.content:
            if line.startswith(b"data:"):
                return response
    except BaseException:
        response.close()
        raise
    response.close()
    raise AssertionError("holder stream ended before yielding a chunk")


Submit = Callable[[aiohttp.ClientSession, str], Coroutine[Any, Any, Any]]


class Gate:
    """The worker's admission metrics, the frontend's routing decisions about
    the worker, and the waits that read them."""

    def __init__(
        self,
        session: aiohttp.ClientSession,
        system_port: int,
        ranks: tuple[str, ...],
        frontend: LogReadable,
        mockers: MockerProcess,
    ) -> None:
        self._session = session
        self._url = f"http://localhost:{system_port}/metrics"
        # Every `dp_rank` the gate must expose: its served ranks and unspecified.
        self.ranks = ranks
        self.frontend = frontend
        # The worker's namespace and component, as its family's labels name them.
        self._namespace = _label_value(mockers.namespace)
        self._component = _label_value(mockers.component_name)
        # Learned from the frontend before the scenario starts.
        self.worker_id = 0
        self._baseline: dict[Series, float] = {}

    async def _scrape(self) -> str:
        async with self._session.get(self._url) as response:
            assert response.status == 200, await response.text()
            return await response.text()

    def _owner(self) -> dict[str, str]:
        """The owner labels every series of this worker's family carries."""
        return {
            "dynamo_component": self._component,
            "dynamo_namespace": self._namespace,
            "worker_id": format(self.worker_id, "x"),
        }

    async def sample(self) -> dict[Series, float]:
        """Every sample on the worker's port. Each series of the gate's family
        must carry exactly this worker's owner labels, which are dropped here, so
        the family is keyed by `dp_rank` (and `source`) alone."""
        owner = self._owner()
        samples: dict[Series, float] = {}
        for line in (await self._scrape()).splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            series, _, value = line.rpartition(" ")
            if not series:
                continue
            name, labels = _parse_series(series)
            if name.startswith(PREFIX):
                rest = dict(labels)
                found = {label: rest.pop(label, None) for label in OWNER_LABELS}
                assert found == owner, (
                    f"{name} must carry its worker's owner labels {owner}, got "
                    f"{found}"
                )
                labels = tuple(sorted(rest.items()))
            samples[(name, labels)] = float(value)
        return samples

    async def families(self) -> dict[str, str]:
        """Every exposed family in this one, by type."""
        families: dict[str, str] = {}
        for line in (await self._scrape()).splitlines():
            parts = line.split()
            if parts[:2] == ["#", "TYPE"] and parts[2].startswith(PREFIX):
                families[parts[2]] = parts[3]
        return families

    async def take_baseline(self) -> None:
        """Anchor counter deltas after readiness has done its own traffic.

        Also checks the shape of the family: exactly its seven metrics, no limit
        gauge among them, one series per served rank and `unspecified` in each
        — no aggregate series — and every counter series present before its
        first event, with no label on rejections beyond `dp_rank` and, as every
        sample is checked for, the worker's owner labels.
        """
        families = await self.families()
        assert (
            families == FAMILIES
        ), f"the gate must expose exactly its seven metric families, got {families}"

        self._baseline = await self.sample()
        expected_ranks = set(self.ranks)
        for name in FAMILIES:
            label_sets = {
                labels for (family, labels) in self._baseline if family == name
            }
            ranks = {dict(labels).get("dp_rank") for labels in label_sets}
            assert ranks == expected_ranks, (
                f"{name} must have one dp_rank series per served rank and "
                f"unspecified, got {sorted(map(str, ranks))}"
            )
            if name != REQUEST_ADMIT_TOTAL:
                assert all(
                    {key for key, _ in labels} == {"dp_rank"} for labels in label_sets
                ), f"{name} carries only dp_rank: {label_sets}"
        missing = [
            (counter, rank)
            for rank in self.ranks
            for counter in COUNTERS
            if self._counter_key(counter, rank) not in self._baseline
        ]
        assert not missing, (
            f"the gate must expose every counter series before its first event; "
            f"missing {missing}"
        )

    @staticmethod
    def _counter_key(counter: str, rank: str) -> Series:
        name, labels = COUNTERS[counter]
        return _key(name, dp_rank=rank, **labels)

    async def occupancy(self, rank: str = UNSPECIFIED) -> tuple[float, float, float]:
        """One rank's occupancy as (engine requests, engine waits, queued)."""
        samples = await self.sample()
        requests, waits, queued = (samples[_key(name, dp_rank=rank)] for name in GAUGES)
        return requests, waits, queued

    async def totals(self) -> tuple[float, float, float]:
        """The worker's occupancy: each gauge summed over its `dp_rank` series."""
        samples = await self.sample()
        requests, waits, queued = (
            sum(samples[_key(name, dp_rank=rank)] for rank in self.ranks)
            for name in GAUGES
        )
        return requests, waits, queued

    async def deltas(self, rank: str = UNSPECIFIED) -> dict[str, float]:
        samples = await self.sample()
        return {
            counter: samples.get(self._counter_key(counter, rank), 0.0)
            - self._baseline.get(self._counter_key(counter, rank), 0.0)
            for counter in COUNTERS
        }

    async def assert_counters(
        self, what: str, rank: str = UNSPECIFIED, **expected: float
    ) -> None:
        """Assert one rank's whole counter family, naming only what moved.

        Every counter not named is asserted unchanged, so a scenario that says
        nothing about cancellations is still asserting there were none.
        """
        unknown = set(expected) - set(COUNTERS)
        assert not unknown, f"unknown counter(s) {unknown}"
        actual = await self.deltas(rank)
        wanted = {name: float(expected.get(name, 0)) for name in COUNTERS}
        assert (
            actual == wanted
        ), f"dp_rank={rank} counter deltas after {what}: {actual} != {wanted}"

    async def assert_settled(self) -> None:
        """With no request still queued or in handoff, every request each rank
        received is accounted for exactly once."""
        for rank in self.ranks:
            deltas = await self.deltas(rank)
            outcomes = (
                deltas["direct"]
                + deltas["queue"]
                + deltas["rejected"]
                + deltas["cancelled"]
            )
            assert (
                deltas["received"] == outcomes
            ), f"dp_rank={rank}: receive != admit + rejection + cancellation: {deltas}"

    async def await_occupancy(
        self,
        engine_requests: int,
        engine_waits: int,
        queued: int,
        what: str,
        rank: str = UNSPECIFIED,
    ) -> None:
        """Poll until one rank's occupancy settles on the expected triple."""
        expected = (engine_requests, engine_waits, queued)
        deadline = time.monotonic() + SETTLE_TIMEOUT_S
        while True:
            observed = await self.occupancy(rank)
            if observed == expected:
                return
            assert time.monotonic() < deadline, (
                f"timed out waiting for {what}: dp_rank={rank} (engine request, "
                f"engine wait, queue) occupancy is {observed}, expected {expected}"
            )
            await asyncio.sleep(0.05)

    async def await_idle(self) -> None:
        deadline = time.monotonic() + SETTLE_TIMEOUT_S
        while (observed := await self.totals()) != (0, 0, 0):
            assert (
                time.monotonic() < deadline
            ), f"timed out waiting for the gate to drain: {observed}"
            await asyncio.sleep(0.05)

    async def await_counter(
        self, counter: str, expected: float, what: str, rank: str = UNSPECIFIED
    ) -> None:
        """Poll until one rank's `counter` has moved by at least `expected`.

        A queued admission is counted only once the request reaches the engine,
        a moment after the gauges move, and a refusal once the gate commits to
        it, before the caller sees an answer.
        """
        deadline = time.monotonic() + SETTLE_TIMEOUT_S
        while True:
            observed = (await self.deltas(rank))[counter]
            if observed >= expected:
                return
            assert time.monotonic() < deadline, (
                f"timed out waiting for {what}: dp_rank={rank} {counter} delta is "
                f"{observed}, expected at least {expected}"
            )
            await asyncio.sleep(0.05)

    async def engine_occupancy(self) -> dict[str, float]:
        """The mocker's own requests per engine rank, running or waiting."""
        occupancy: dict[str, float] = {}
        for (name, labels), value in (await self.sample()).items():
            if name in (ENGINE_RUNNING, ENGINE_WAITING):
                engine = dict(labels)["engine"]
                occupancy[engine] = occupancy.get(engine, 0.0) + value
        return occupancy

    async def await_engine_agreement(self, what: str) -> None:
        """Poll until the mocker serves on each rank exactly the requests the
        gate admitted under that rank: the gate's rank is the engine's rank."""
        deadline = time.monotonic() + SETTLE_TIMEOUT_S
        while True:
            samples = await self.sample()
            gate = {
                rank: samples[_key(ENGINE_REQUEST_COUNT, dp_rank=rank)]
                for rank in self.ranks
                if rank != UNSPECIFIED
            }
            engine = await self.engine_occupancy()
            if engine == gate:
                return
            assert time.monotonic() < deadline, (
                f"timed out waiting for {what}: the gate's per-rank engine requests "
                f"{gate} never matched the mocker's per-engine requests {engine}"
            )
            await asyncio.sleep(0.05)


async def _start_running(
    gate: Gate,
    session: aiohttp.ClientSession,
    url: str,
    occupancy: tuple[int, int, int],
    label: str,
    rank: str = UNSPECIFIED,
    headers: dict[str, str] | None = None,
) -> aiohttp.ClientResponse:
    """Start a long stream and return it open once it has answered and the gate
    shows `occupancy` for its rank. Having answered, it holds request capacity
    but no waiting capacity, however long it keeps streaming."""
    response = await _hold(session, url, headers)
    await gate.await_occupancy(*occupancy, f"the {label} stream to answer", rank)
    return response


async def _stage_waiting(
    gate: Gate,
    session: aiohttp.ClientSession,
    url: str,
    submit: Submit,
    occupancy: tuple[int, int, int],
    label: str,
    rank: str = UNSPECIFIED,
) -> asyncio.Task:
    """Submit one request that cannot answer yet — waiting in the engine for its
    first response, or in a gate queue — and wait until the gate shows
    `occupancy` for its rank.

    Staged rather than raced: only waiting for each in turn makes the queue's
    contents and its order known before the next arrives.
    """
    task = asyncio.create_task(submit(session, url))
    await gate.await_occupancy(*occupancy, f"the {label} request to settle", rank)
    assert not task.done(), (
        f"the {label} request must not have answered yet: "
        f"{task.result() if task.done() else ''}"
    )
    return task


async def _shed(
    gate: Gate,
    session: aiohttp.ClientSession,
    url: str,
    rank: str = UNSPECIFIED,
    headers: dict[str, str] | None = None,
    whole_worker: bool = True,
) -> None:
    """Submit one request the gate must reject rather than queue, and check the
    client sees the frontend's overload status for it, and that the frontend
    never reported the worker down. A whole-worker refusal must also make the
    frontend mark the worker overloaded; a refusal limited to `rank` marks
    nothing, which the caller checks once later traffic has been logged."""
    marks = len(_routing_events(gate, "worker_overloaded"))
    # Read before the request exists: a scrape awaits, and the request could
    # otherwise be rejected first and leave no further rejection to wait for.
    before = (await gate.deltas(rank))["rejected"]
    task = asyncio.create_task(_unary(session, url, headers))
    await gate.await_counter("rejected", before + 1, "the request to be rejected", rank)
    status, body = await _result(task, "shed")
    assert status == OVERLOADED, (
        f"a refusal must reach the client as an overload ({OVERLOADED}), got "
        f"{status}: {body}"
    )
    if whole_worker:
        await asyncio.to_thread(
            wait_for_worker_routing_events,
            gate.frontend,
            instance_id=gate.worker_id,
            routing_event="worker_overloaded",
            expected_count=marks + 1,
        )
    _assert_never_reported_down(gate)


def _routing_events(gate: Gate, routing_event) -> list:
    return worker_routing_events(
        gate.frontend, instance_id=gate.worker_id, routing_event=routing_event
    )


def _assert_never_reported_down(gate: Gate) -> None:
    inhibited = _routing_events(gate, "worker_inhibited")
    assert not inhibited, f"a refusing worker must never be reported down: {inhibited}"


async def _await_overload_lease() -> None:
    """Wait out the frontend's overload lease on the worker after a refusal,
    before sending it new work. Time-based: the lease is."""
    await asyncio.sleep(OVERLOAD_LEASE_S)


async def _assert_still_serving(
    gate: Gate, session: aiohttp.ClientSession, url: str, what: str, **expected
) -> None:
    """After its overload lease, the healthy worker takes new work again: a
    request reaches it, is admitted, and succeeds."""
    await _await_overload_lease()
    status, body = await _unary(session, url)
    assert status == 200, f"the worker must keep serving: {status}: {body}"
    await gate.await_idle()
    await gate.assert_counters(what, **expected)
    _assert_never_reported_down(gate)


async def _result(task: asyncio.Task, label: str) -> Any:
    try:
        return await asyncio.wait_for(task, timeout=OUTCOME_TIMEOUT_S)
    except asyncio.TimeoutError as error:
        raise AssertionError(
            f"the {label} request produced no outcome within {OUTCOME_TIMEOUT_S}s"
        ) from error


@dataclass(frozen=True)
class Deployment:
    # Where the worker's gate metrics are scraped.
    system_port: int
    # Its log records the frontend's routing decisions.
    frontend: FrontendRouterProcess
    mockers: MockerProcess


@contextlib.contextmanager
def _deployment(
    request,
    ports: ServicePorts,
    request_plane: str,
    settings: dict[str, int],
    max_num_seqs: int,
    dp_size: int = 1,
    num_workers: int = 1,
    worker_env: dict[str, str] | None = None,
) -> Iterator["Deployment"]:
    """One mocker process, running `num_workers` workers, and one frontend,
    sized for this scenario.

    The workers' gate configuration, and any `worker_env`, is patched around
    `MockerProcess` alone; the frontend is built outside it and is told
    explicitly not to inherit the workers' system port. Each worker has its own
    runtime, so only the first to bind that port serves its metrics there.
    """
    system_port = ports.system_ports[0]
    # Slowed down so a running stream lasts a whole scenario. The sequence
    # limit is always explicit: it is what keeps an admitted request waiting
    # for its first response, and it is the capacity the worker reports.
    mocker_args: dict[str, Any] = {
        "speedup_ratio": 0.1,
        "block_size": BLOCK_SIZE,
        "max_num_seqs": max_num_seqs,
        "dp_size": dp_size,
    }

    monkeypatch = request.getfixturevalue("monkeypatch")
    with monkeypatch.context() as environment:
        environment.setenv("DYN_SYSTEM_PORT", str(system_port))
        # Set explicitly or cleared, never inherited. The monkeypatch context
        # restores the original environment on exit.
        for name in ADMISSION_SETTINGS:
            if name in settings:
                environment.setenv(name, str(settings[name]))
            else:
                environment.delenv(name, raising=False)
        for name, value in (worker_env or {}).items():
            environment.setenv(name, value)
        mockers = MockerProcess(
            request,
            mocker_args=mocker_args,
            num_mockers=num_workers,
            request_plane=request_plane,
        )

    with mockers:
        with FrontendRouterProcess(
            request,
            BLOCK_SIZE,
            ports.frontend_port,
            mockers.namespace,
            request_plane=request_plane,
            router_mode="round-robin",
            extra_env={"DYN_SYSTEM_PORT": None, **FRONTEND_LOG_ENV},
        ) as frontend:
            asyncio.run(_ready(ports.frontend_port, mockers, request_plane))
            yield Deployment(system_port, frontend, mockers)


async def _ready(
    frontend_port: int, mockers: MockerProcess, request_plane: str
) -> None:
    await wait_for_frontend_ready(
        frontend_url=f"http://localhost:{frontend_port}",
        expected_num_workers=mockers.num_workers,
        engine_workers=mockers,
        request_plane=request_plane,
        test_payload=_payload(1, False),
    )


async def _worker_id(session: aiohttp.ClientSession, url: str) -> int:
    """The worker's instance ID, as the frontend reports the worker it chose."""
    payload = {**_payload(1, False), "nvext": {"extra_fields": ["worker_id"]}}
    async with session.post(url, json=payload) as response:
        body = await response.json()
        assert response.status == 200, body
    worker_id = require_router_worker_id(body)["decode_worker_id"]
    assert isinstance(worker_id, int), body
    return worker_id


def _run(
    frontend_port: int,
    deployment: Deployment,
    scenario,
    dp_size: int = 1,
) -> None:
    """Open a session, anchor the counters, and run one scenario with the
    worker's instance ID."""
    system_port, frontend = deployment.system_port, deployment.frontend

    async def main() -> None:
        url = f"http://localhost:{frontend_port}/v1/chat/completions"
        ranks = (*(str(rank) for rank in range(dp_size)), UNSPECIFIED)
        async with aiohttp.ClientSession() as session:
            gate = Gate(session, system_port, ranks, frontend, deployment.mockers)
            gate.worker_id = await _worker_id(session, url)
            # Readiness has already sent traffic, so the gate is idle but its
            # counters are not zero. Everything below is a delta from here.
            await gate.await_idle()
            await gate.take_baseline()
            await scenario(gate, session, url, gate.worker_id)
            await gate.assert_settled()
            _assert_never_reported_down(gate)

    asyncio.run(main())


pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.parallel,
    pytest.mark.gpu_0,
    pytest.mark.integration,
    pytest.mark.fault_tolerance,
    pytest.mark.mocker,
    pytest.mark.model(MODEL_NAME),
    pytest.mark.parametrize("request_plane", ["tcp", "nats"], indirect=True),
]


######################### ENGINE WAITING LIMIT #########################

WAITING_SCENARIO_QUEUE_LIMIT = 3


@pytest.mark.timeout(90)
def test_waiting_limit_holds_requests_until_a_first_response(
    request,
    runtime_services_session,
    predownload_tokenizers,
    request_plane,
    dynamo_dynamic_ports,
):
    """One request awaiting its first response, exactly Q queued, the next one
    refused as an overload, then FIFO drain — each queued request admitted by
    the first response of the one ahead of it, which keeps running and keeps
    its request capacity.

    The request limit is set far above use, so the waiting limit alone is what
    holds the queue back. The refusal leaves the worker marked overloaded and
    never reported down, and it serves new work once the lease ends.
    """
    queue_limit = WAITING_SCENARIO_QUEUE_LIMIT

    async def scenario(
        gate: Gate, session: aiohttp.ClientSession, url: str, worker_id: int
    ) -> None:
        runner = await _start_running(gate, session, url, (1, 0, 0), "running")
        await gate.assert_counters(
            "the running stream is admitted", received=1, direct=1
        )

        # Admitted directly although the running stream is unfinished: having
        # answered, it holds no waiting capacity. The engine is busy with it,
        # though, so this one waits there and holds the only waiting capacity.
        waiter = await _stage_waiting(gate, session, url, _hold, (2, 1, 0), "waiter")
        # The gauges move as capacity is reserved, a moment before the admission
        # is counted at the engine handoff.
        await gate.await_counter("direct", 2, "the waiter to be admitted")
        await gate.assert_counters("the waiter is admitted", received=2, direct=2)

        # One at a time, so each is known to hold a place before the next
        # arrives. Each follower holds its stream open once it answers, so the
        # drain below happens one release at a time under this test's control.
        queued = [
            await _stage_waiting(
                gate, session, url, _hold, (2, 1, index + 1), f"follower-{index}"
            )
            for index in range(queue_limit)
        ]
        await gate.assert_counters(
            "the queue is exactly full", received=2 + queue_limit, direct=2
        )

        # Every queue place is taken, so this one has nowhere to go. It must be
        # refused rather than displace a request ahead of it.
        await _shed(gate, session, url)
        assert await gate.occupancy() == (
            2,
            1,
            queue_limit,
        ), "a refused request must take neither engine capacity nor a queue place"
        await gate.assert_counters(
            "the overflow request is refused",
            received=3 + queue_limit,
            direct=2,
            rejected=1,
        )
        assert not any(
            task.done() for task in queued
        ), "the older queued requests must not be displaced by the refused one"

        # Free the engine, then let one request answer at a time. Each answer is
        # that request's first response, which must pass waiting capacity to the
        # oldest request still queued — and no other — while its own stream
        # keeps running and keeps its request capacity.
        chain = [waiter, *queued]
        released = runner
        for index, task in enumerate(chain):
            label = "waiter" if index == 0 else f"follower-{index - 1}"
            released.close()
            answering = await _result(task, label)
            admitted = min(index + 1, queue_limit)
            if index < queue_limit:
                await gate.await_counter(
                    "queue",
                    admitted,
                    f"{label}'s first response to admit the next queued request",
                )
                # The answering stream, and the newly admitted request waiting
                # behind it in the engine.
                expected = (2, 1, queue_limit - index - 1)
            else:
                expected = (1, 0, 0)
            await gate.await_occupancy(
                *expected, f"{label}'s first response to pass waiting capacity on"
            )
            assert not any(t.done() for t in chain[index + 1 :]), (
                f"only {label}, the oldest request awaiting a response, may answer "
                f"after this release"
            )
            await gate.assert_counters(
                f"{label} answers",
                received=3 + queue_limit,
                direct=2,
                queue=admitted,
                rejected=1,
            )
            assert (
                not answering.content.at_eof()
            ), f"{label}'s stream must still be running after its first response"
            released = answering

        released.close()
        await gate.await_idle()
        await _assert_still_serving(
            gate,
            session,
            url,
            "the worker serves new work after the refusal",
            received=4 + queue_limit,
            direct=3,
            queue=queue_limit,
            rejected=1,
        )

    settings = {
        ENGINE_REQUEST_LIMIT_ENV: 100,
        ENGINE_REQUEST_WAITING_LIMIT_ENV: 1,
        REQUEST_QUEUE_LIMIT_ENV: queue_limit,
    }
    # One running sequence, so whatever the gate admits next waits for its
    # first response until the test frees the engine.
    with _deployment(
        request, dynamo_dynamic_ports, request_plane, settings, max_num_seqs=1
    ) as deployment:
        _run(dynamo_dynamic_ports.frontend_port, deployment, scenario)


######################### ENGINE REQUEST LIMIT #########################


@pytest.mark.timeout(60)
def test_request_limit_holds_requests_until_one_finishes(
    request,
    runtime_services_session,
    predownload_tokenizers,
    request_plane,
    dynamo_dynamic_ports,
):
    """Requests that have answered keep their request capacity until they
    finish. The engine has a sequence for every admitted request and the
    waiting limit follows it, so the request limit alone holds the next one
    back, and only a stream finishing lets it in."""

    async def scenario(
        gate: Gate, session: aiohttp.ClientSession, url: str, worker_id: int
    ) -> None:
        first = await _start_running(gate, session, url, (1, 0, 0), "first")
        second = await _start_running(gate, session, url, (2, 0, 0), "second")
        await gate.assert_counters("two streams are running", received=2, direct=2)

        # Waiting capacity is free, but request capacity is not.
        queued = await _stage_waiting(gate, session, url, _hold, (2, 0, 1), "queued")
        await _shed(gate, session, url)
        assert await gate.occupancy() == (2, 0, 1)
        await gate.assert_counters(
            "the next request queues and the one after is refused",
            received=4,
            direct=2,
            rejected=1,
        )

        # A finished stream releases its request capacity to the queued
        # request, which answers at once on the freed engine sequence.
        first.close()
        third = await _result(queued, "queued")
        await gate.await_counter(
            "queue", 1, "a finished stream to admit the queued request"
        )
        await gate.await_occupancy(2, 0, 0, "the queued request to answer")
        assert not second.content.at_eof(), "the second stream is still running"

        for response in (second, third):
            response.close()
        await gate.await_idle()
        await _assert_still_serving(
            gate,
            session,
            url,
            "the worker serves new work after the refusal",
            received=5,
            direct=3,
            queue=1,
            rejected=1,
        )

    settings = {ENGINE_REQUEST_LIMIT_ENV: 2, REQUEST_QUEUE_LIMIT_ENV: 1}
    with _deployment(
        request, dynamo_dynamic_ports, request_plane, settings, max_num_seqs=2
    ) as deployment:
        _run(dynamo_dynamic_ports.frontend_port, deployment, scenario)


######################### CAPACITY-DERIVED DEFAULTS #########################


@pytest.mark.timeout(60)
def test_automatic_limits_follow_the_reported_capacity(
    request,
    runtime_services_session,
    predownload_tokenizers,
    request_plane,
    dynamo_dynamic_ports,
):
    """With no setting given, every limit follows the `max_num_seqs` the worker
    reports, one sequence: one request in the engine, one awaiting its first
    response, and a queue of twice that. With the first request running, the
    second and third queue, the fourth is refused, and the queue drains in
    order."""

    async def scenario(
        gate: Gate, session: aiohttp.ClientSession, url: str, worker_id: int
    ) -> None:
        runner = await _start_running(gate, session, url, (1, 0, 0), "running")
        queued = [
            await _stage_waiting(
                gate, session, url, _hold, (1, 0, index + 1), f"queued-{index}"
            )
            for index in range(2)
        ]
        await _shed(gate, session, url)
        await gate.assert_counters(
            "the automatic request limit queues and the automatic queue limit refuses",
            received=4,
            direct=1,
            rejected=1,
        )

        released = runner
        for index, task in enumerate(queued):
            released.close()
            released = await _result(task, f"queued-{index}")
            await gate.await_counter("queue", index + 1, f"queued-{index} to enter")
            await gate.await_occupancy(
                1, 0, 1 - index, f"queued-{index} to answer in FIFO order"
            )
        released.close()
        await gate.await_idle()
        await gate.assert_counters(
            "the queue has drained", received=4, direct=1, queue=2, rejected=1
        )

    with _deployment(
        request, dynamo_dynamic_ports, request_plane, {}, max_num_seqs=1
    ) as deployment:
        _run(dynamo_dynamic_ports.frontend_port, deployment, scenario)


######################### ALIASES #########################


@pytest.mark.timeout(60)
def test_legacy_aliases_configure_the_request_and_queue_limits(
    request,
    runtime_services_session,
    predownload_tokenizers,
    request_plane,
    dynamo_dynamic_ports,
):
    """`DYN_ENGINE_REQUEST_LIMIT` sets the full-request limit and
    `DYN_DYNAMO_REQUEST_QUEUE_LIMIT` the queue limit, with no canonical name set.
    The running stream has answered, so waiting capacity is free: it is the
    request limit of one that holds the next request back, and the queue limit
    of zero that refuses it instead of queueing it."""

    async def scenario(
        gate: Gate, session: aiohttp.ClientSession, url: str, worker_id: int
    ) -> None:
        runner = await _start_running(gate, session, url, (1, 0, 0), "running")
        await _shed(gate, session, url)
        assert await gate.occupancy() == (1, 0, 0)
        await gate.assert_counters(
            "the second request is refused", received=2, direct=1, rejected=1
        )

        runner.close()
        await gate.await_idle()

    settings = {LEGACY_ENGINE_REQUEST_LIMIT_ENV: 1, LEGACY_REQUEST_QUEUE_LIMIT_ENV: 0}
    # Spare sequences, so the engine itself never holds the second request back.
    with _deployment(
        request, dynamo_dynamic_ports, request_plane, settings, max_num_seqs=2
    ) as deployment:
        _run(dynamo_dynamic_ports.frontend_port, deployment, scenario)


@pytest.mark.timeout(60)
def test_the_former_waiting_limit_name_configures_the_waiting_limit(
    request,
    runtime_services_session,
    predownload_tokenizers,
    request_plane,
    dynamo_dynamic_ports,
):
    """`DYN_BACKEND_ADMISSION_ENGINE_WAIT_LIMIT`, the waiting limit's former
    name, still sets it: with request capacity to spare, one request awaiting
    its first response is enough to refuse the next when queueing is off."""

    async def scenario(
        gate: Gate, session: aiohttp.ClientSession, url: str, worker_id: int
    ) -> None:
        runner = await _start_running(gate, session, url, (1, 0, 0), "running")
        waiter = await _stage_waiting(gate, session, url, _hold, (2, 1, 0), "waiter")
        await _shed(gate, session, url)
        assert await gate.occupancy() == (2, 1, 0)

        runner.close()
        (await _result(waiter, "waiter")).close()
        await gate.await_idle()
        await gate.assert_counters(
            "the waiting limit refused the third request",
            received=3,
            direct=2,
            rejected=1,
        )

    settings = {
        ENGINE_REQUEST_LIMIT_ENV: 10,
        FORMER_ENGINE_WAIT_LIMIT_ENV: 1,
        REQUEST_QUEUE_LIMIT_ENV: 0,
    }
    with _deployment(
        request, dynamo_dynamic_ports, request_plane, settings, max_num_seqs=1
    ) as deployment:
        _run(dynamo_dynamic_ports.frontend_port, deployment, scenario)


######################### DP RANKS #########################


@pytest.mark.timeout(120)
def test_ranks_admit_and_queue_independently_and_share_the_aggregate(
    request,
    runtime_services_session,
    predownload_tokenizers,
    request_plane,
    dynamo_dynamic_ports,
):
    """A worker serving ranks 0 and 1, with every limit at one per rank: the
    worker-wide limits are two of each.

    Rank 0 at its own limit queues its next request but never blocks rank 1.
    The gate accounts each pinned request under the rank the mocker serves it
    on. Unspecified work has no capacity of its own: it queues behind ranked
    work that fills the aggregate, its queue shares the aggregate queue limit,
    and once admitted it in turn holds rank 1's share against a rank 1 request.
    Each rank's gauges and counters stay separate, and they sum to the worker's
    totals.
    """

    async def scenario(
        gate: Gate, session: aiohttp.ClientSession, url: str, worker_id: int
    ) -> None:
        on_rank = {rank: _pin(worker_id, rank) for rank in (0, 1)}
        hold_on = {
            rank: functools.partial(_hold, headers=on_rank[rank]) for rank in (0, 1)
        }

        hold_0 = await _start_running(
            gate, session, url, (1, 0, 0), "rank-0", "0", on_rank[0]
        )
        queued_0 = await _stage_waiting(
            gate, session, url, hold_on[0], (1, 0, 1), "queued on rank 0", "0"
        )
        # Rank 0 is at its limit, but rank 1 has room and is admitted directly.
        hold_1 = await _start_running(
            gate, session, url, (1, 0, 0), "rank-1", "1", on_rank[1]
        )
        await gate.await_engine_agreement("each pinned stream to run on its rank")
        assert await gate.totals() == (2, 0, 1)

        # Both ranks hold the whole aggregate, so unspecified work queues.
        unspecified = await _stage_waiting(
            gate, session, url, _hold, (0, 0, 1), "unspecified queued"
        )
        assert await gate.totals() == (2, 0, 2)

        # Rank 0's queue is full.
        await _shed(gate, session, url, "0", on_rank[0])
        # And unspecified work cannot queue beyond the aggregate queue limit,
        # which rank 0's queued request shares.
        await _await_overload_lease()
        await _shed(gate, session, url)
        assert await gate.totals() == (2, 0, 2)

        # Releasing rank 0 admits its own queued request, the oldest that fits:
        # the unspecified one still finds the aggregate full.
        hold_0.close()
        queued_0 = await _result(queued_0, "queued on rank 0")
        await gate.await_occupancy(1, 0, 0, "rank 0's queued request to answer", "0")
        await gate.await_occupancy(0, 0, 1, "unspecified work to keep waiting")

        # Releasing rank 1 admits the unspecified request.
        hold_1.close()
        unspecified = await _result(unspecified, "unspecified")
        await gate.await_occupancy(1, 0, 0, "the unspecified request to answer")
        await gate.await_occupancy(0, 0, 0, "rank 1 to drain", "1")

        # Rank 1 is empty, but unspecified work now holds its share of the
        # aggregate, so a rank 1 request queues.
        await _await_overload_lease()
        rank_1 = await _stage_waiting(
            gate, session, url, hold_on[1], (0, 0, 1), "queued on rank 1", "1"
        )
        unspecified.close()
        rank_1 = await _result(rank_1, "queued on rank 1")
        await gate.await_occupancy(1, 0, 0, "rank 1's queued request to answer", "1")
        await gate.await_engine_agreement("each rank's request to run on its engine")

        for response in (queued_0, rank_1):
            response.close()
        await gate.await_idle()
        await gate.await_engine_agreement("every engine to drain")

        await gate.assert_counters(
            "rank 0", "0", received=3, direct=1, queue=1, rejected=1
        )
        await gate.assert_counters("rank 1", "1", received=2, direct=1, queue=1)
        await gate.assert_counters("unspecified work", received=2, queue=1, rejected=1)

    settings = {
        ENGINE_REQUEST_LIMIT_ENV: 1,
        ENGINE_REQUEST_WAITING_LIMIT_ENV: 1,
        REQUEST_QUEUE_LIMIT_ENV: 1,
    }
    # Two sequences per rank, so an admitted request always answers at once.
    with _deployment(
        request,
        dynamo_dynamic_ports,
        request_plane,
        settings,
        max_num_seqs=2,
        dp_size=2,
    ) as deployment:
        _run(dynamo_dynamic_ports.frontend_port, deployment, scenario, dp_size=2)


@pytest.mark.timeout(120)
def test_a_full_rank_refuses_for_itself_and_its_idle_sibling_is_admitted_at_once(
    request,
    runtime_services_session,
    predownload_tokenizers,
    request_plane,
    dynamo_dynamic_ports,
):
    """A worker serving ranks 0 and 1, with every limit at one per rank.

    With rank 0's engine and queue full and rank 1 idle, the worker refuses the
    next rank 0 request for rank 0 alone: the client still sees an overload,
    but the frontend marks nothing. So a rank 1 request sent at once — without
    waiting out an overload lease, retrying or another client — is admitted
    and runs on rank 1's engine while rank 0 stays full.
    """

    async def scenario(
        gate: Gate, session: aiohttp.ClientSession, url: str, worker_id: int
    ) -> None:
        on_rank = {rank: _pin(worker_id, rank) for rank in (0, 1)}
        marks = len(_routing_events(gate, "worker_overloaded"))

        hold_0 = await _start_running(
            gate, session, url, (1, 0, 0), "rank-0", "0", on_rank[0]
        )
        queued_0 = await _stage_waiting(
            gate,
            session,
            url,
            functools.partial(_hold, headers=on_rank[0]),
            (1, 0, 1),
            "queued on rank 0",
            "0",
        )
        assert await gate.occupancy("1") == (0, 0, 0)

        await _shed(gate, session, url, "0", on_rank[0], whole_worker=False)
        hold_1 = await _start_running(
            gate, session, url, (1, 0, 0), "rank-1", "1", on_rank[1]
        )
        await gate.await_engine_agreement("rank 1's request to run on its engine")
        assert await gate.occupancy("0") == (1, 0, 1), "rank 0 stays full"

        hold_0.close()
        queued_0 = await _result(queued_0, "queued on rank 0")
        for response in (queued_0, hold_1):
            response.close()
        await gate.await_idle()
        await gate.await_engine_agreement("every engine to drain")
        assert (
            len(_routing_events(gate, "worker_overloaded")) == marks
        ), "a refusal for one rank must never mark the worker overloaded"

        await gate.assert_counters(
            "rank 0", "0", received=3, direct=1, queue=1, rejected=1
        )
        await gate.assert_counters("rank 1", "1", received=1, direct=1)
        await gate.assert_counters("unspecified work")

    settings = {
        ENGINE_REQUEST_LIMIT_ENV: 1,
        ENGINE_REQUEST_WAITING_LIMIT_ENV: 1,
        REQUEST_QUEUE_LIMIT_ENV: 1,
    }
    # Two sequences per rank, so an admitted request always answers at once.
    with _deployment(
        request,
        dynamo_dynamic_ports,
        request_plane,
        settings,
        max_num_seqs=2,
        dp_size=2,
    ) as deployment:
        _run(dynamo_dynamic_ports.frontend_port, deployment, scenario, dp_size=2)


######################### WORKERS IN ONE PROCESS #########################


async def _hold_once_admitted(
    session: aiohttp.ClientSession, url: str, headers: dict[str, str]
) -> aiohttp.ClientResponse:
    """`_hold` on a worker whose gate this test cannot scrape, so cannot wait to
    see idle: it may still be releasing earlier work, such as readiness
    traffic. Only an overload refusal is retried, each after the frontend's
    overload lease on the worker, until the settle deadline; any other failure
    fails at once."""
    deadline = time.monotonic() + SETTLE_TIMEOUT_S
    while True:
        response = await session.post(url, json=_payload(2000, True), headers=headers)
        if response.status != OVERLOADED:
            return await _first_chunk(response)
        refusal = await response.text()
        response.close()
        assert (
            time.monotonic() < deadline
        ), f"the worker still refused a stream after {SETTLE_TIMEOUT_S}s: {refusal}"
        await _await_overload_lease()


async def _scraped_worker_ids(
    session: aiohttp.ClientSession, system_port: int
) -> set[str | None]:
    """The `worker_id` of every admission series the system port serves."""
    async with session.get(f"http://localhost:{system_port}/metrics") as response:
        assert response.status == 200, await response.text()
        text = await response.text()
    return {
        dict(_parse_series(line.rpartition(" ")[0])[1]).get("worker_id")
        for line in text.splitlines()
        if line.startswith(PREFIX)
    }


@pytest.mark.timeout(120)
def test_workers_in_one_process_admit_through_their_own_gates(
    request,
    runtime_services_session,
    predownload_tokenizers,
    request_plane,
    dynamo_dynamic_ports,
):
    """Two mocker workers in one process (`--num-workers 2`), each with its own
    runtime and its own gate of one engine request and no queue.

    Each runtime keeps its own metrics, and only the first to bind the system
    port serves them there: worker A is the one whose `worker_id` the port's
    family carries, and worker B shows its gate through its answers. With A's
    gate full, B still admits a stream; with that stream filling B's own gate, B
    refuses its next request as an overload, which moves none of A's counters;
    and A refuses its own next request. Neither is ever reported down. The
    workers' instance IDs come from discovery, so the scenario sends no request
    beyond its own after readiness.
    """

    async def scenario(
        gate: Gate, session: aiohttp.ClientSession, url: str, worker_b: int
    ) -> None:
        on_a, on_b = _pin(gate.worker_id, 0), _pin(worker_b, 0)

        hold_a = await _start_running(
            gate, session, url, (1, 0, 0), "worker A's", "0", on_a
        )
        # A's gate is full, and stays full, but B's own gate has room: B admits a
        # stream, whose first response proves it, and which keeps B's gate full
        # from then on. B may still be releasing readiness traffic, so only its
        # overload refusals are retried; through A's full gate none would ever
        # pass, and none of them reaches A's counters.
        hold_b = await _hold_once_admitted(session, url, on_b)
        await gate.assert_counters("worker B's admission", "0", received=1, direct=1)
        assert await gate.occupancy("0") == (1, 0, 0)

        # B's gate is full too now: B refuses on its own gate, as an overload,
        # and A's counters do not move.
        marks = len(
            worker_routing_events(
                gate.frontend, instance_id=worker_b, routing_event="worker_overloaded"
            )
        )
        status, body = await _unary(session, url, on_b)
        assert (
            status == OVERLOADED
        ), f"worker B must refuse once its own gate is full: {status}: {body}"
        await asyncio.to_thread(
            wait_for_worker_routing_events,
            gate.frontend,
            instance_id=worker_b,
            routing_event="worker_overloaded",
            expected_count=marks + 1,
        )
        await gate.assert_counters("worker B's refusal", "0", received=1, direct=1)
        assert await gate.occupancy("0") == (1, 0, 0)

        # A refuses its own next request.
        await _shed(gate, session, url, "0", on_a)

        for response in (hold_a, hold_b):
            response.close()
        await gate.await_idle()
        await gate.assert_counters("worker A", "0", received=2, direct=1, rejected=1)
        await gate.assert_counters("no unspecified work on worker A")
        inhibited_b = worker_routing_events(
            gate.frontend, instance_id=worker_b, routing_event="worker_inhibited"
        )
        assert not inhibited_b, f"worker B must never be reported down: {inhibited_b}"

    settings = {
        ENGINE_REQUEST_LIMIT_ENV: 1,
        ENGINE_REQUEST_WAITING_LIMIT_ENV: 1,
        REQUEST_QUEUE_LIMIT_ENV: 0,
    }
    # Only the worker that binds the shared system port could serve its model
    # metadata from it, which would make the other worker's card differ, and
    # the frontend hold that worker incompatible. Both publish their metadata
    # through shared storage instead, so their cards match.
    with _deployment(
        request,
        dynamo_dynamic_ports,
        request_plane,
        settings,
        max_num_seqs=1,
        num_workers=2,
        worker_env={"DYN_SELF_HOST_METADATA": "0"},
    ) as deployment:

        async def main() -> None:
            url = f"http://localhost:{dynamo_dynamic_ports.frontend_port}/v1/chat/completions"
            # The workers' instance IDs, read from discovery: no engine traffic.
            mockers = deployment.mockers
            with managed_runtime(request_plane=request_plane) as runtime:
                endpoint = runtime.endpoint(
                    f"{mockers.namespace}.{mockers.component_name}.generate"
                )
                workers = set(await poll_for_worker_instances(endpoint, 2))
            async with aiohttp.ClientSession() as session:
                gate = Gate(
                    session,
                    deployment.system_port,
                    ("0", UNSPECIFIED),
                    deployment.frontend,
                    mockers,
                )
                scraped = await _scraped_worker_ids(session, deployment.system_port)
                owners = {
                    worker for worker in workers if format(worker, "x") in scraped
                }
                assert len(scraped) == 1 and len(owners) == 1, (
                    f"the system port must serve exactly one of the workers "
                    f"{sorted(workers)}, found worker_id labels {scraped}"
                )
                (gate.worker_id,) = owners
                (worker_b,) = workers - owners
                await gate.await_idle()
                await gate.take_baseline()
                await scenario(gate, session, url, worker_b)
                await gate.assert_settled()
                _assert_never_reported_down(gate)

        asyncio.run(main())


######################### CALLER DISCONNECT WHILE QUEUED #########################


@pytest.mark.timeout(90)
def test_a_caller_disconnecting_while_queued_frees_its_place_before_admission(
    request,
    runtime_services_session,
    predownload_tokenizers,
    request_plane,
    dynamo_dynamic_ports,
):
    """A caller that disconnects while its request waits in the gate queue.

    The frontend cancels the request when its HTTP connection closes, and on
    this default path it forwards that cancellation to the worker while still
    awaiting the response prologue. The gate therefore removes the request from
    its queue at once, as a cancellation, before it ever reaches the engine, and
    the freed place is reusable straight away.

    End-to-end evidence for the default path only: a request carrying a
    first-response guard (decoded multimodal media whose source storage must
    outlive the dispatch) has its cancellation deferred until the prologue,
    which the worker sends only after admission. Such a request stays queued
    until it is admitted, and its cancellation then takes the normal downstream
    path. That deferral is not exercised here.
    """

    async def scenario(
        gate: Gate, session: aiohttp.ClientSession, url: str, worker_id: int
    ) -> None:
        runner = await _start_running(gate, session, url, (1, 0, 0), "running")
        queued = await _stage_waiting(gate, session, url, _hold, (1, 0, 1), "queued")

        # The caller goes away while its request waits: its connection closes.
        queued.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await queued
        await gate.await_occupancy(1, 0, 0, "the disconnect to free the queue place")
        await gate.await_counter(
            "cancelled", 1, "the disconnect to count as a cancellation"
        )
        await gate.assert_counters(
            "the queued request is cancelled before admission",
            received=2,
            direct=1,
            cancelled=1,
        )

        # The freed place is reusable at once, and the drain leaves nothing.
        replacement = await _stage_waiting(
            gate, session, url, _hold, (1, 0, 1), "replacement"
        )
        runner.close()
        (await _result(replacement, "replacement")).close()
        await gate.await_idle()
        await gate.assert_counters(
            "the queue has drained", received=3, direct=1, queue=1, cancelled=1
        )

    settings = {ENGINE_REQUEST_LIMIT_ENV: 1, REQUEST_QUEUE_LIMIT_ENV: 2}
    with _deployment(
        request, dynamo_dynamic_ports, request_plane, settings, max_num_seqs=1
    ) as deployment:
        _run(dynamo_dynamic_ports.frontend_port, deployment, scenario)


######################### A WORKER THAT REALLY FAILS #########################


@pytest.mark.timeout(90)
def test_a_failed_worker_is_still_reported_down(
    request,
    runtime_services_session,
    predownload_tokenizers,
    request_plane,
    dynamo_dynamic_ports,
):
    """The positive control for the overload scenarios above: a worker that
    really fails is still reported down. With a stream confirmed admitted and
    answering, the worker's process is killed, so the stream's transport breaks
    mid-generation, and the frontend records a `worker_inhibited` routing event
    for the worker — the record no worker refusing for capacity may ever
    produce."""

    with _deployment(
        request, dynamo_dynamic_ports, request_plane, {}, max_num_seqs=1
    ) as deployment:

        async def main() -> None:
            url = f"http://localhost:{dynamo_dynamic_ports.frontend_port}/v1/chat/completions"
            async with aiohttp.ClientSession() as session:
                gate = Gate(
                    session,
                    deployment.system_port,
                    ("0", UNSPECIFIED),
                    deployment.frontend,
                    deployment.mockers,
                )
                gate.worker_id = await _worker_id(session, url)
                await gate.await_idle()
                running = await _start_running(gate, session, url, (1, 0, 0), "running")
                _assert_never_reported_down(gate)

                # Only this scenario's worker; MockerProcess exposes no public
                # handle to the process it runs.
                worker = deployment.mockers._process.proc
                worker.kill()
                await asyncio.to_thread(worker.wait, 30)

                # The broken stream ends one way or another; its transport
                # failure is what the frontend must act on.
                async def drain() -> None:
                    async for _ in running.content:
                        pass

                with contextlib.suppress(aiohttp.ClientError, asyncio.TimeoutError):
                    await asyncio.wait_for(drain(), OUTCOME_TIMEOUT_S)
                running.close()
                await asyncio.to_thread(
                    wait_for_worker_routing_events,
                    deployment.frontend,
                    instance_id=gate.worker_id,
                    routing_event="worker_inhibited",
                )

        asyncio.run(main())
