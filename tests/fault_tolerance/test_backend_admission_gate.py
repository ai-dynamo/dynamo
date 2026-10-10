# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Process-level coverage for the shared backend admission gate.

The gate lives in `Ingress::handle_payload_shared`, the one admission point both
request planes funnel through, so a CPU-only mocker worker exercises the real
thing over TCP and NATS alike.

The oracle here is the worker's own `dynamo_backend_admission_*` family, scraped
from its system port. HTTP is only a coarse cross-check: an admitted request
answers `200` and a refused one answers something else. Which typed error and
status a refusal maps to belongs to the response transport, not to the gate, and
is deliberately not pinned by these tests.

The gate has two engine limits. A request holds engine request capacity from
admission until it finishes, and engine wait capacity from admission until its
first response. The mocker runs a fixed number of sequences at a time: a long
stream started first keeps a sequence busy, holding request capacity but no wait
capacity once it has answered, and a request the gate admits while every
sequence is busy waits inside the engine without a response, holding both, until
the test closes a stream ahead of it.

Cancellation before admission is not exercised here. A request waiting in the
gate queue has not sent its response prologue yet, and the response transport
forwards a caller's stop or kill to the worker only after that prologue, so a
client that goes away while queued stays queued until it is admitted. The gate's
cancellation accounting is covered deterministically by its unit tests in
`lib/runtime/src/admission_gate.rs`.

Observation is metrics-driven rather than sleep-driven: each step polls until the
gauges show the state it needs, and counters are asserted as deltas from a
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
import logging
import time
from typing import Any, Callable, Coroutine, Iterator

import aiohttp
import pytest

from tests.router.helper import wait_for_frontend_ready
from tests.router.mocker_process import MockerProcess
from tests.router.router_process import FrontendRouterProcess
from tests.utils.constants import FAULT_TOLERANCE_MODEL_NAME
from tests.utils.port_utils import ServicePorts

logger = logging.getLogger(__name__)

MODEL_NAME = FAULT_TOLERANCE_MODEL_NAME
BLOCK_SIZE = 16

# Every setting the gate reads. A scenario sets the ones it names and clears the
# rest, so an ambient value can never decide what it is testing.
ENGINE_REQUEST_LIMIT_ENV = "DYN_BACKEND_ADMISSION_ENGINE_REQUEST_LIMIT"
LEGACY_ENGINE_REQUEST_LIMIT_ENV = "DYN_ENGINE_REQUEST_LIMIT"
ENGINE_WAIT_LIMIT_ENV = "DYN_BACKEND_ADMISSION_ENGINE_WAIT_LIMIT"
REQUEST_QUEUE_LIMIT_ENV = "DYN_BACKEND_ADMISSION_REQUEST_QUEUE_LIMIT"
LEGACY_REQUEST_QUEUE_LIMIT_ENV = "DYN_DYNAMO_REQUEST_QUEUE_LIMIT"
ADMISSION_SETTINGS = (
    ENGINE_REQUEST_LIMIT_ENV,
    LEGACY_ENGINE_REQUEST_LIMIT_ENV,
    ENGINE_WAIT_LIMIT_ENV,
    REQUEST_QUEUE_LIMIT_ENV,
    LEGACY_REQUEST_QUEUE_LIMIT_ENV,
)

# How long a poll may wait for the gate to reach an expected state.
SETTLE_TIMEOUT_S = 30
# How long a request may take to produce its outcome after the capacity it
# waits for is released.
OUTCOME_TIMEOUT_S = 60

PREFIX = "dynamo_backend_admission"
ENGINE_REQUEST_COUNT = f"{PREFIX}_engine_request_count"
ENGINE_WAIT_COUNT = f"{PREFIX}_engine_wait_count"
REQUEST_QUEUE_COUNT = f"{PREFIX}_request_queue_count"
REQUEST_RECEIVE_TOTAL = f"{PREFIX}_request_receive_total"
REQUEST_ADMIT_TOTAL = f"{PREFIX}_request_admit_total"
REJECTION_TOTAL = f"{PREFIX}_rejection_total"
CANCELLATION_TOTAL = f"{PREFIX}_cancellation_total"

# The whole family, by type: no limit gauges.
FAMILIES = {
    ENGINE_REQUEST_COUNT: "gauge",
    ENGINE_WAIT_COUNT: "gauge",
    REQUEST_QUEUE_COUNT: "gauge",
    REQUEST_RECEIVE_TOTAL: "counter",
    REQUEST_ADMIT_TOTAL: "counter",
    REJECTION_TOTAL: "counter",
    CANCELLATION_TOTAL: "counter",
}


def _series(name: str, **labels: str) -> str:
    """The exposition key for one series, e.g. `x_total{source="direct"}`."""
    if not labels:
        return name
    inner = ",".join(f'{key}="{value}"' for key, value in sorted(labels.items()))
    return f"{name}{{{inner}}}"


RECEIVED = _series(REQUEST_RECEIVE_TOTAL)
ADMITTED_DIRECT = _series(REQUEST_ADMIT_TOTAL, source="direct")
ADMITTED_QUEUE = _series(REQUEST_ADMIT_TOTAL, source="queue")
# Rejections carry no reason label.
REJECTED = _series(REJECTION_TOTAL)

# Every counter series, so a scenario can state its whole expected effect and
# have the untouched ones checked at zero rather than left unexamined.
COUNTERS = (
    RECEIVED,
    ADMITTED_DIRECT,
    ADMITTED_QUEUE,
    REJECTED,
    CANCELLATION_TOTAL,
)


def _payload(max_tokens: int, stream: bool) -> dict[str, Any]:
    return {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": "admission gate"}],
        "stream": stream,
        "max_tokens": max_tokens,
        "ignore_eos": True,
    }


async def _unary(session: aiohttp.ClientSession, url: str) -> tuple[int, str]:
    """A one-token request that answers in one piece. The body is read as text,
    because a generic pre-stream failure need not be JSON."""
    async with session.post(url, json=_payload(1, False)) as response:
        return response.status, await response.text()


async def _hold(session: aiohttp.ClientSession, url: str) -> aiohttp.ClientResponse:
    """Start a long stream and return it once it has begun answering.

    Cancelled before then, as cleanup does to streams still waiting for their
    first response, it closes the response it already owns rather than leaving
    it to the garbage collector.
    """
    response = await session.post(url, json=_payload(2000, True))
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
    """The worker's admission metrics, and the waits that read them."""

    def __init__(self, session: aiohttp.ClientSession, system_port: int) -> None:
        self._session = session
        self._url = f"http://localhost:{system_port}/metrics"
        self._baseline: dict[str, float] = {}

    async def _scrape(self) -> str:
        async with self._session.get(self._url) as response:
            assert response.status == 200, await response.text()
            return await response.text()

    async def sample(self) -> dict[str, float]:
        samples: dict[str, float] = {}
        for line in (await self._scrape()).splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            series, _, value = line.rpartition(" ")
            if series:
                samples[series] = float(value)
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
        gauge among them, every counter series present at zero before its first
        event, and no label on rejections.
        """
        families = await self.families()
        assert (
            families == FAMILIES
        ), f"the gate must expose exactly its seven metric families, got {families}"

        self._baseline = await self.sample()
        missing = [name for name in COUNTERS if name not in self._baseline]
        assert not missing, (
            f"the gate must expose every counter series at zero before its first "
            f"event; missing {missing}"
        )
        rejections = [
            series for series in self._baseline if series.startswith(REJECTION_TOTAL)
        ]
        assert rejections == [REJECTED], f"rejections carry no label: {rejections}"

    async def gauges(self) -> tuple[float, float, float]:
        """Live occupancy as (engine requests, engine waits, queued)."""
        samples = await self.sample()
        return (
            samples[ENGINE_REQUEST_COUNT],
            samples[ENGINE_WAIT_COUNT],
            samples[REQUEST_QUEUE_COUNT],
        )

    async def deltas(self) -> dict[str, float]:
        samples = await self.sample()
        return {
            name: samples.get(name, 0.0) - self._baseline.get(name, 0.0)
            for name in COUNTERS
        }

    async def assert_counters(self, what: str, **expected: float) -> None:
        """Assert the whole counter family, naming only what moved.

        Every counter not named is asserted unchanged, so a scenario that says
        nothing about cancellations is still asserting there were none.
        """
        unknown = set(expected) - set(COUNTERS)
        assert not unknown, f"unknown counter(s) {unknown}"
        actual = await self.deltas()
        wanted = {name: float(expected.get(name, 0)) for name in COUNTERS}
        assert actual == wanted, f"counter deltas after {what}: {actual} != {wanted}"

    async def await_occupancy(
        self, engine_requests: int, engine_waits: int, queued: int, what: str
    ) -> None:
        """Poll until occupancy settles on the expected triple."""
        expected = (engine_requests, engine_waits, queued)
        deadline = time.monotonic() + SETTLE_TIMEOUT_S
        while True:
            observed = await self.gauges()
            if observed == expected:
                return
            assert time.monotonic() < deadline, (
                f"timed out waiting for {what}: (engine request, engine wait, queue) "
                f"occupancy is {observed}, expected {expected}"
            )
            await asyncio.sleep(0.05)

    async def await_counter(self, name: str, expected: float, what: str) -> None:
        """Poll until `name` has moved by at least `expected`.

        A queued admission is counted only once the request reaches the engine,
        a moment after the gauges move, and a refusal once the gate commits to
        it, before the caller sees an answer.
        """
        deadline = time.monotonic() + SETTLE_TIMEOUT_S
        while True:
            observed = (await self.deltas())[name]
            if observed >= expected:
                return
            assert time.monotonic() < deadline, (
                f"timed out waiting for {what}: {name} delta is {observed}, "
                f"expected at least {expected}"
            )
            await asyncio.sleep(0.05)


async def _start_running(
    gate: Gate,
    session: aiohttp.ClientSession,
    url: str,
    occupancy: tuple[int, int, int],
    label: str,
) -> aiohttp.ClientResponse:
    """Start a long stream and return it open once it has answered and the gate
    shows `occupancy`. Having answered, it holds request capacity but no wait
    capacity, however long it keeps streaming."""
    response = await _hold(session, url)
    await gate.await_occupancy(*occupancy, f"the {label} stream to answer")
    return response


async def _stage_waiting(
    gate: Gate,
    session: aiohttp.ClientSession,
    url: str,
    submit: Submit,
    occupancy: tuple[int, int, int],
    label: str,
) -> asyncio.Task:
    """Submit one request that cannot answer yet — waiting in the engine for its
    first response, or in the gate queue — and wait until the gate shows
    `occupancy`.

    Staged rather than raced: only waiting for each in turn makes the queue's
    contents and its order known before the next arrives.
    """
    task = asyncio.create_task(submit(session, url))
    await gate.await_occupancy(*occupancy, f"the {label} request to settle")
    assert not task.done(), (
        f"the {label} request must not have answered yet: "
        f"{task.result() if task.done() else ''}"
    )
    return task


async def _shed(gate: Gate, session: aiohttp.ClientSession, url: str) -> None:
    """Submit one request the gate must reject rather than queue, and check it
    did not succeed."""
    # Read before the request exists: a scrape awaits, and the request could
    # otherwise be rejected first and leave no further rejection to wait for.
    before = (await gate.deltas())[REJECTED]
    task = asyncio.create_task(_unary(session, url))
    await gate.await_counter(REJECTED, before + 1, "the request to be rejected")
    status, body = await _result(task, "shed")
    assert status != 200, f"the shed request must not succeed, got {status}: {body}"


async def _result(task: asyncio.Task, label: str) -> Any:
    try:
        return await asyncio.wait_for(task, timeout=OUTCOME_TIMEOUT_S)
    except asyncio.TimeoutError as error:
        raise AssertionError(
            f"the {label} request produced no outcome within {OUTCOME_TIMEOUT_S}s"
        ) from error


@contextlib.contextmanager
def _deployment(
    request,
    ports: ServicePorts,
    request_plane: str,
    settings: dict[str, int],
    max_num_seqs: int,
) -> Iterator[int]:
    """One mocker and one frontend, sized for this scenario. Yields the worker's
    system port, where the gate's metrics are scraped.

    The worker's gate configuration is patched around `MockerProcess` alone; the
    frontend is built outside it and is told explicitly not to inherit the
    worker's system port.
    """
    system_port = ports.system_ports[0]
    # Slowed down so a running stream lasts a whole scenario. The sequence
    # limit is always explicit: it is what keeps an admitted request waiting
    # for its first response, and it is the capacity the worker reports.
    mocker_args: dict[str, Any] = {
        "speedup_ratio": 0.1,
        "block_size": BLOCK_SIZE,
        "max_num_seqs": max_num_seqs,
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
        mockers = MockerProcess(
            request,
            mocker_args=mocker_args,
            num_mockers=1,
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
            extra_env={"DYN_SYSTEM_PORT": None},
        ):
            asyncio.run(_ready(ports.frontend_port, mockers, request_plane))
            yield system_port


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


def _run(frontend_port: int, system_port: int, scenario) -> None:
    """Open a session, anchor the counters, and run one scenario."""

    async def main() -> None:
        url = f"http://localhost:{frontend_port}/v1/chat/completions"
        async with aiohttp.ClientSession() as session:
            gate = Gate(session, system_port)
            # Readiness has already sent traffic, so the gate is idle but its
            # counters are not zero. Everything below is a delta from here.
            await gate.await_occupancy(0, 0, 0, "the gate to settle after readiness")
            await gate.take_baseline()
            await scenario(gate, session, url)

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


######################### ENGINE WAIT LIMIT #########################

WAIT_SCENARIO_QUEUE_LIMIT = 3


@pytest.mark.timeout(60)
def test_wait_limit_holds_requests_until_a_first_response(
    request,
    runtime_services_session,
    predownload_tokenizers,
    request_plane,
    dynamo_dynamic_ports,
):
    """One request awaiting its first response, exactly Q queued, the next one
    shed, then FIFO drain — each queued request admitted by the first response
    of the one ahead of it, which keeps running and keeps its request capacity.

    The request limit stays at its default throughout, so the wait limit alone
    is what holds the queue back.
    """
    queue_limit = WAIT_SCENARIO_QUEUE_LIMIT

    async def scenario(gate: Gate, session: aiohttp.ClientSession, url: str) -> None:
        runner = await _start_running(gate, session, url, (1, 0, 0), "running")
        await gate.assert_counters(
            "the running stream is admitted", **{RECEIVED: 1, ADMITTED_DIRECT: 1}
        )

        # Admitted directly although the running stream is unfinished: having
        # answered, it holds no wait capacity. The engine is busy with it,
        # though, so this one waits there and holds the only wait capacity.
        waiter = await _stage_waiting(gate, session, url, _hold, (2, 1, 0), "waiter")
        # The gauges move as capacity is reserved, a moment before the admission
        # is counted at the engine handoff.
        await gate.await_counter(ADMITTED_DIRECT, 2, "the waiter to be admitted")
        await gate.assert_counters(
            "the waiter is admitted", **{RECEIVED: 2, ADMITTED_DIRECT: 2}
        )

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
            "the queue is exactly full",
            **{RECEIVED: 2 + queue_limit, ADMITTED_DIRECT: 2},
        )

        # Every queue place is taken, so this one has nowhere to go. It must be
        # shed rather than displace a request ahead of it.
        await _shed(gate, session, url)
        assert await gate.gauges() == (
            2,
            1,
            queue_limit,
        ), "a shed request must take neither engine capacity nor a queue place"
        await gate.assert_counters(
            "the overflow request is shed",
            **{RECEIVED: 3 + queue_limit, ADMITTED_DIRECT: 2, REJECTED: 1},
        )
        assert not any(
            task.done() for task in queued
        ), "the older queued requests must not be displaced by the shed one"

        # Free the engine, then let one request answer at a time. Each answer is
        # that request's first response, which must pass wait capacity to the
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
                    ADMITTED_QUEUE,
                    admitted,
                    f"{label}'s first response to admit the next queued request",
                )
                # The answering stream, and the newly admitted request waiting
                # behind it in the engine.
                expected = (2, 1, queue_limit - index - 1)
            else:
                expected = (1, 0, 0)
            await gate.await_occupancy(
                *expected, f"{label}'s first response to pass wait capacity on"
            )
            assert not any(t.done() for t in chain[index + 1 :]), (
                f"only {label}, the oldest request awaiting a response, may answer "
                f"after this release"
            )
            await gate.assert_counters(
                f"{label} answers",
                **{
                    RECEIVED: 3 + queue_limit,
                    ADMITTED_DIRECT: 2,
                    ADMITTED_QUEUE: admitted,
                    REJECTED: 1,
                },
            )
            assert (
                not answering.content.at_eof()
            ), f"{label}'s stream must still be running after its first response"
            released = answering

        released.close()
        await gate.await_occupancy(0, 0, 0, "the gate to drain")
        await gate.assert_counters(
            "the queue has drained",
            **{
                RECEIVED: 3 + queue_limit,
                ADMITTED_DIRECT: 2,
                ADMITTED_QUEUE: queue_limit,
                REJECTED: 1,
            },
        )

    settings = {ENGINE_WAIT_LIMIT_ENV: 1, REQUEST_QUEUE_LIMIT_ENV: queue_limit}
    # One running sequence, so whatever the gate admits next waits for its
    # first response until the test frees the engine.
    with _deployment(
        request, dynamo_dynamic_ports, request_plane, settings, max_num_seqs=1
    ) as system_port:
        _run(dynamo_dynamic_ports.frontend_port, system_port, scenario)


######################### ENGINE REQUEST LIMIT #########################


@pytest.mark.timeout(45)
def test_request_limit_holds_requests_until_one_finishes(
    request,
    runtime_services_session,
    predownload_tokenizers,
    request_plane,
    dynamo_dynamic_ports,
):
    """Requests that have answered keep their request capacity until they
    finish. The wait limit stays at its default and the engine has a sequence
    for every admitted request, so the request limit alone holds the next one
    back, and only a stream finishing lets it in."""

    async def scenario(gate: Gate, session: aiohttp.ClientSession, url: str) -> None:
        first = await _start_running(gate, session, url, (1, 0, 0), "first")
        second = await _start_running(gate, session, url, (2, 0, 0), "second")
        await gate.assert_counters(
            "two streams are running", **{RECEIVED: 2, ADMITTED_DIRECT: 2}
        )

        # Wait capacity is free, but request capacity is not.
        queued = await _stage_waiting(gate, session, url, _hold, (2, 0, 1), "queued")
        await _shed(gate, session, url)
        assert await gate.gauges() == (2, 0, 1)
        await gate.assert_counters(
            "the next request queues and the one after is shed",
            **{RECEIVED: 4, ADMITTED_DIRECT: 2, REJECTED: 1},
        )

        # A finished stream releases its request capacity to the queued
        # request, which answers at once on the freed engine sequence.
        first.close()
        third = await _result(queued, "queued")
        await gate.await_counter(
            ADMITTED_QUEUE, 1, "a finished stream to admit the queued request"
        )
        await gate.await_occupancy(2, 0, 0, "the queued request to answer")
        assert not second.content.at_eof(), "the second stream is still running"

        for response in (second, third):
            response.close()
        await gate.await_occupancy(0, 0, 0, "the gate to drain")
        await gate.assert_counters(
            "the queue has drained",
            **{RECEIVED: 4, ADMITTED_DIRECT: 2, ADMITTED_QUEUE: 1, REJECTED: 1},
        )

    settings = {ENGINE_REQUEST_LIMIT_ENV: 2, REQUEST_QUEUE_LIMIT_ENV: 1}
    with _deployment(
        request, dynamo_dynamic_ports, request_plane, settings, max_num_seqs=2
    ) as system_port:
        _run(dynamo_dynamic_ports.frontend_port, system_port, scenario)


######################### FIXED DEFAULTS #########################


@pytest.mark.timeout(45)
def test_default_limits_ignore_the_reported_engine_capacity(
    request,
    runtime_services_session,
    predownload_tokenizers,
    request_plane,
    dynamo_dynamic_ports,
):
    """With no setting given, both engine limits keep their fixed defaults. The
    worker reports a capacity of one sequence, and the gate must not size itself
    from it: requests beyond it are admitted directly and wait in the engine."""

    async def scenario(gate: Gate, session: aiohttp.ClientSession, url: str) -> None:
        runner = await _start_running(gate, session, url, (1, 0, 0), "running")
        waiting = [
            await _stage_waiting(
                gate, session, url, _hold, (2 + index, 1 + index, 0), f"waiter-{index}"
            )
            for index in range(2)
        ]
        # The gauges move as capacity is reserved, a moment before the admission
        # is counted at the engine handoff.
        await gate.await_counter(ADMITTED_DIRECT, 3, "both waiters to be admitted")
        await gate.assert_counters(
            "every request is admitted directly", **{RECEIVED: 3, ADMITTED_DIRECT: 3}
        )

        released = runner
        for index, task in enumerate(waiting):
            released.close()
            released = await _result(task, f"waiter-{index}")
        released.close()
        await gate.await_occupancy(0, 0, 0, "the gate to drain")
        await gate.assert_counters(
            "every request finished", **{RECEIVED: 3, ADMITTED_DIRECT: 3}
        )

    with _deployment(
        request, dynamo_dynamic_ports, request_plane, {}, max_num_seqs=1
    ) as system_port:
        _run(dynamo_dynamic_ports.frontend_port, system_port, scenario)


######################### LEGACY ALIASES #########################


@pytest.mark.timeout(45)
def test_legacy_aliases_configure_the_request_and_queue_limits(
    request,
    runtime_services_session,
    predownload_tokenizers,
    request_plane,
    dynamo_dynamic_ports,
):
    """`DYN_ENGINE_REQUEST_LIMIT` sets the full-request limit and
    `DYN_DYNAMO_REQUEST_QUEUE_LIMIT` the queue limit, with no canonical name set.
    The running stream has answered, so wait capacity is free: it is the request
    limit of one that holds the next request back, and the queue limit of zero
    that rejects it instead of queueing it."""

    async def scenario(gate: Gate, session: aiohttp.ClientSession, url: str) -> None:
        runner = await _start_running(gate, session, url, (1, 0, 0), "running")
        await _shed(gate, session, url)
        assert await gate.gauges() == (1, 0, 0)
        await gate.assert_counters(
            "the second request is rejected",
            **{RECEIVED: 2, ADMITTED_DIRECT: 1, REJECTED: 1},
        )

        runner.close()
        await gate.await_occupancy(0, 0, 0, "the gate to drain")

    settings = {LEGACY_ENGINE_REQUEST_LIMIT_ENV: 1, LEGACY_REQUEST_QUEUE_LIMIT_ENV: 0}
    # Spare sequences, so the engine itself never holds the second request back.
    with _deployment(
        request, dynamo_dynamic_ports, request_plane, settings, max_num_seqs=2
    ) as system_port:
        _run(dynamo_dynamic_ports.frontend_port, system_port, scenario)
