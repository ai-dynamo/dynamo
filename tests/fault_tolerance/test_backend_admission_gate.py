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

Observation is metrics-driven rather than sleep-driven: each step polls until the
gauges show the state it needs, and counters are asserted as deltas from a
baseline taken after readiness, never as process-global absolutes. Fixed sleeps
appear only where the point is to separate two deadlines by a safe margin.

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
import os
import time
from typing import Any, Callable, Coroutine, Iterator, NamedTuple, Optional
from unittest import mock

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

# One in flight, three queued, everything else shed. Three is the fewest that
# can tell the front of the queue from the back with a rejected request ahead of
# both.
ENGINE_REQUEST_LIMIT = 1
QUEUE_LIMIT = 3

# Long enough that nothing expires while a queue is staged, for the scenarios
# that are not about expiry at all.
NO_EXPIRY_DELAY_MS = 60_000

# The delay the expiry scenarios run on. Long enough to stage a queue and watch
# gauges settle well inside it, short enough to keep a CPU-only run brief.
QUEUE_DELAY_MS = 8_000
QUEUE_DELAY_S = QUEUE_DELAY_MS / 1000

# The staging shape the FIFO and adaptive-LIFO scenarios share. `front` follows
# the doomed request closely, so its deadline is close behind too and only about
# 2.0s of it is left when that one is rejected; `fresh` arrives much later and
# still holds about 6.0s. Both are wide enough to place the handover
# unambiguously on hardware slower than the machine this was tuned on, and far
# enough apart to tell the two selections apart.
FRONT_LAG_S = QUEUE_DELAY_S * 0.25
FRESH_LAG_S = QUEUE_DELAY_S * 0.5

# The mocker answers in roughly 0.16s plus 0.057s per token at the speedup ratio
# these scenarios run on, so a 68-token request takes about 4.0s. That length is
# bounded on both sides: it runs about 2.0s past the margin `front` has left,
# which is what lets a request admitted just inside its deadline finish well
# after it, and it still finishes about 2.0s inside what `fresh` holds, so the
# request behind it is unambiguously still eligible when the slot reaches it.
NEAR_FRONT_TOKENS = 68

# How long a poll may wait for the gate to reach an expected state.
SETTLE_TIMEOUT_S = 30
# How long an outcome that must arrive on its own deadline may take.
OUTCOME_TIMEOUT_S = 60

PREFIX = "dynamo_backend_admission"
ENGINE_REQUEST_COUNT = f"{PREFIX}_engine_request_count"
ENGINE_REQUEST_LIMIT_GAUGE = f"{PREFIX}_engine_request_limit"
REQUEST_QUEUE_COUNT = f"{PREFIX}_request_queue_count"
REQUEST_QUEUE_LIMIT_GAUGE = f"{PREFIX}_request_queue_limit"
REQUEST_TOTAL = f"{PREFIX}_request_total"
DEQUEUE_TOTAL = f"{PREFIX}_dequeue_total"
REJECTION_TOTAL = f"{PREFIX}_rejection_total"
CANCELLATION_TOTAL = f"{PREFIX}_cancellation_total"


def _series(name: str, **labels: str) -> str:
    """The exposition key for one series, e.g. `x_total{path="direct"}`."""
    if not labels:
        return name
    inner = ",".join(f'{key}="{value}"' for key, value in sorted(labels.items()))
    return f"{name}{{{inner}}}"


DIRECT = _series(REQUEST_TOTAL, path="direct")
QUEUED = _series(REQUEST_TOTAL, path="queue")
PRE_CANCELLED = _series(REQUEST_TOTAL, path="cancelled")
FIFO_DEQUEUE = _series(DEQUEUE_TOTAL, source="fifo")
LIFO_DEQUEUE = _series(DEQUEUE_TOTAL, source="adaptive_lifo")
QUEUE_FULL = _series(REJECTION_TOTAL, reason="queue_full")
REQUEST_EXPIRED = _series(REJECTION_TOTAL, reason="request_expired")

# Every counter in the family, so a scenario can state its whole expected effect
# and have the untouched ones checked at zero rather than left unexamined.
COUNTERS = (
    DIRECT,
    QUEUED,
    PRE_CANCELLED,
    FIFO_DEQUEUE,
    LIFO_DEQUEUE,
    QUEUE_FULL,
    REQUEST_EXPIRED,
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


def _unary_of(max_tokens: int):
    """A submit that answers in one piece. The body is read as text, because a
    generic pre-stream failure need not be JSON."""

    async def submit(session: aiohttp.ClientSession, url: str) -> tuple[int, str]:
        async with session.post(url, json=_payload(max_tokens, False)) as response:
            return response.status, await response.text()

    return submit


_unary = _unary_of(1)
_near_front = _unary_of(NEAR_FRONT_TOKENS)


async def _hold(session: aiohttp.ClientSession, url: str) -> aiohttp.ClientResponse:
    """Occupy a slot with a long stream that has already begun."""
    response = await session.post(url, json=_payload(2000, True))
    assert response.status == 200, await response.text()
    async for line in response.content:
        if line.startswith(b"data:"):
            return response
    raise AssertionError("holder stream ended before yielding a chunk")


async def _hold_when_routable(
    session: aiohttp.ClientSession, url: str
) -> aiohttp.ClientResponse:
    """Take the slot once the Frontend is routing to the worker again.

    Having just seen a refusal, the Frontend briefly answers that the model is
    not ready without consulting the worker at all, so a phase that opens
    immediately after one would fail on the Frontend rather than reach the gate.
    Any non-200 is simply waited out rather than being classified here — which
    typed error a refusal carries is not this file's business. A retry that did
    reach the gate would show up in the counter deltas each scenario asserts.
    """
    deadline = time.monotonic() + SETTLE_TIMEOUT_S
    while True:
        response = await session.post(url, json=_payload(2000, True))
        if response.status == 200:
            async for line in response.content:
                if line.startswith(b"data:"):
                    return response
            raise AssertionError("holder stream ended before yielding a chunk")
        body = await response.text()
        response.close()
        assert time.monotonic() < deadline, (
            f"the Frontend never resumed routing to the worker, last answer "
            f"{response.status}: {body}"
        )
        await asyncio.sleep(0.2)


Submit = Callable[[aiohttp.ClientSession, str], Coroutine[Any, Any, Any]]


class Gate:
    """The worker's admission metrics, and the waits that read them."""

    def __init__(self, session: aiohttp.ClientSession, system_port: int) -> None:
        self._session = session
        self._url = f"http://localhost:{system_port}/metrics"
        self._baseline: dict[str, float] = {}

    async def sample(self) -> dict[str, float]:
        async with self._session.get(self._url) as response:
            assert response.status == 200, await response.text()
            text = await response.text()
        samples: dict[str, float] = {}
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            series, _, value = line.rpartition(" ")
            if series:
                samples[series] = float(value)
        return samples

    async def take_baseline(self, engine_limit: int = ENGINE_REQUEST_LIMIT) -> None:
        """Anchor counter deltas after readiness has done its own traffic.

        Also checks the sizing gauges, which are what the live counts are read
        against: if the worker were not sized as this scenario asked, every
        occupancy assertion below would be measuring something else. The engine
        limit is polled rather than read once, because a worker sized from a
        capacity hint records that hint just after it becomes discoverable and
        so can be reachable for a moment while the gate still holds its fallback.
        """
        deadline = time.monotonic() + SETTLE_TIMEOUT_S
        while True:
            samples = await self.sample()
            sizing = (
                samples[ENGINE_REQUEST_LIMIT_GAUGE],
                samples[REQUEST_QUEUE_LIMIT_GAUGE],
            )
            if sizing == (engine_limit, QUEUE_LIMIT):
                break
            assert time.monotonic() < deadline, (
                f"the worker reports {sizing} for (engine, queue) limits, not the "
                f"expected {(engine_limit, QUEUE_LIMIT)}"
            )
            await asyncio.sleep(0.05)

        self._baseline = samples
        missing = [name for name in COUNTERS if name not in self._baseline]
        assert not missing, (
            f"the gate must expose every counter series at zero before its first "
            f"event; missing {missing}"
        )

    async def gauges(self) -> tuple[float, float]:
        """Live occupancy as (engine requests, queued)."""
        samples = await self.sample()
        return samples[ENGINE_REQUEST_COUNT], samples[REQUEST_QUEUE_COUNT]

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
        self, engine_requests: int, queued: int, what: str
    ) -> None:
        """Poll until occupancy settles on the expected pair."""
        deadline = time.monotonic() + SETTLE_TIMEOUT_S
        while True:
            observed = await self.gauges()
            if observed == (engine_requests, queued):
                return
            assert time.monotonic() < deadline, (
                f"timed out waiting for {what}: engine/queue occupancy is "
                f"{observed}, expected {(engine_requests, queued)}"
            )
            await asyncio.sleep(0.05)

    async def await_counter(self, name: str, expected: float, what: str) -> float:
        """Poll until `name` has moved by at least `expected`, and report when
        that was seen.

        The timestamp is what makes a handover causally checkable: closing a
        response only starts the release, and the slot reaches the next request
        asynchronously, so it is this observation — not the close — that dates
        the admission.
        """
        deadline = time.monotonic() + SETTLE_TIMEOUT_S
        while True:
            observed = (await self.deltas())[name]
            if observed >= expected:
                return time.monotonic()
            assert time.monotonic() < deadline, (
                f"timed out waiting for {what}: {name} delta is {observed}, "
                f"expected at least {expected}"
            )
            await asyncio.sleep(0.05)


async def _stage_queued(
    gate: Gate,
    session: aiohttp.ClientSession,
    url: str,
    submit: Submit,
    queued: int,
    label: str,
) -> asyncio.Task:
    """Submit one request and wait until the gate shows it parked in the FIFO.

    Staged rather than raced: only waiting for each in turn makes the queue's
    contents and its order known before the next arrives.
    """
    task = asyncio.create_task(submit(session, url))
    await gate.await_occupancy(ENGINE_REQUEST_LIMIT, queued, f"{label} to queue")
    assert not task.done(), (
        f"the {label} request must queue behind the holder, not answer: "
        f"{task.result() if task.done() else ''}"
    )
    return task


class NearDeadlineQueue(NamedTuple):
    """A staged queue in which the front is nearly spent and the tail is not."""

    doomed: asyncio.Task
    front: asyncio.Task
    fresh: asyncio.Task
    # `front`'s true deadline is stamped inside the worker, so it is bracketed
    # rather than computed: no earlier than a budget from just before the
    # request was sent, and no later than a budget from the moment the gate was
    # seen holding it.
    front_deadline_earliest: float
    front_deadline_latest: float


async def _stage_near_deadline_queue(
    gate: Gate,
    session: aiohttp.ClientSession,
    url: str,
    fresh_submit: Submit,
) -> NearDeadlineQueue:
    """Stage the shape the FIFO and adaptive-LIFO scenarios both run on.

    `doomed` goes first and is the one the delay gives up on. `front` follows it
    closely, so its own deadline is close behind and only a fraction of its
    budget is left when `doomed` is rejected. `fresh` arrives much later and
    still holds most of its own. Which of the two the freed slot then goes to is
    the only thing the two scenarios differ on.
    """
    doomed = await _stage_queued(gate, session, url, _unary, 1, "doomed")
    await asyncio.sleep(FRONT_LAG_S)
    front_deadline_earliest = time.monotonic() + QUEUE_DELAY_S
    front = await _stage_queued(gate, session, url, _near_front, 2, "front")
    front_deadline_latest = time.monotonic() + QUEUE_DELAY_S
    await asyncio.sleep(FRESH_LAG_S)
    fresh = await _stage_queued(gate, session, url, fresh_submit, 3, "fresh")
    return NearDeadlineQueue(
        doomed, front, fresh, front_deadline_earliest, front_deadline_latest
    )


def _assert_refused(status: int, body: Any, label: str) -> None:
    """A refused request must not succeed. Deliberately generic."""
    assert status != 200, f"the {label} request must not succeed, got {status}: {body}"


async def _result(task: asyncio.Task, label: str) -> tuple[int, str]:
    return await asyncio.wait_for(task, timeout=OUTCOME_TIMEOUT_S)


@contextlib.contextmanager
def _deployment(
    request,
    ports: ServicePorts,
    request_plane: str,
    queue_delay_ms: int = NO_EXPIRY_DELAY_MS,
    controlled_delay: Optional[str] = None,
    adaptive_lifo: Optional[str] = None,
    engine_request_limit: Optional[int] = ENGINE_REQUEST_LIMIT,
    max_num_seqs: Optional[int] = None,
    data_parallel_size: Optional[int] = None,
) -> Iterator[int]:
    """One mocker and one frontend, sized for this scenario. Yields the worker's
    system port, where the gate's metrics are scraped.

    The worker's gate configuration is patched around `MockerProcess` alone; the
    frontend is built outside it and is told explicitly not to inherit the
    worker's system port.
    """
    system_port = ports.system_ports[0]
    # The engine limit is deliberately absent here: it is set or cleared below
    # alongside the other switches, so a scenario that asks for automatic sizing
    # gets a worker with no override at all.
    worker_env = {
        "DYN_DYNAMO_REQUEST_QUEUE_LIMIT": str(QUEUE_LIMIT),
        "DYN_DYNAMO_REQUEST_QUEUE_TIMEOUT_MS": str(queue_delay_ms),
        "DYN_SYSTEM_PORT": str(system_port),
    }
    # Slowed down so a holder's stream lasts a whole scenario, but not so far
    # that an admitted one-token request approaches the queue delay.
    mocker_args: dict[str, Any] = {"speedup_ratio": 0.1, "block_size": BLOCK_SIZE}
    # Both factors the automatic limit is derived from, passed explicitly when a
    # scenario is about that derivation so neither comes from a default.
    if max_num_seqs is not None:
        mocker_args["max_num_seqs"] = max_num_seqs
    if data_parallel_size is not None:
        mocker_args["dp_size"] = data_parallel_size

    with mock.patch.dict(os.environ, worker_env):
        # Set explicitly or cleared, never inherited: an ambient value would
        # otherwise decide what these scenarios are testing. `patch.dict`
        # restores the original environment on exit.
        for name, value in (
            ("DYN_ENGINE_REQUEST_LIMIT", engine_request_limit),
            ("DYN_DYNAMO_REQUEST_QUEUE_ENABLE_CONTROLLED_DELAY", controlled_delay),
            ("DYN_DYNAMO_REQUEST_QUEUE_ENABLE_ADAPTIVE_LIFO", adaptive_lifo),
        ):
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = str(value)
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


def _run(
    frontend_port: int,
    system_port: int,
    scenario,
    engine_limit: int = ENGINE_REQUEST_LIMIT,
) -> None:
    """Open a session, anchor the counters, and run one scenario."""

    async def main() -> None:
        url = f"http://localhost:{frontend_port}/v1/chat/completions"
        async with aiohttp.ClientSession() as session:
            gate = Gate(session, system_port)
            # Readiness has already sent traffic, so the gate is idle but its
            # counters are not zero. Everything below is a delta from here.
            await gate.await_occupancy(0, 0, "the gate to settle after readiness")
            await gate.take_baseline(engine_limit)
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


######################### BOUNDED FIFO QUEUE #########################


@pytest.mark.timeout(40)
def test_bounded_fifo_queue_absorbs_burst(
    request,
    runtime_services_session,
    predownload_tokenizers,
    request_plane,
    dynamo_dynamic_ports,
):
    """N admitted, exactly Q queued, the next one shed, then FIFO drain."""

    async def scenario(gate: Gate, session: aiohttp.ClientSession, url: str) -> None:
        holder = await _hold(session, url)
        await gate.await_occupancy(1, 0, "the holder to take the only slot")
        await gate.assert_counters("the holder is admitted", **{DIRECT: 1})

        # One at a time, so each is known to hold a place before the next
        # arrives and the queue's order is known rather than raced. Each
        # follower holds its slot open once admitted, so the drain below happens
        # one release at a time under this test's control rather than running to
        # completion between two polls.
        queued = [
            await _stage_queued(
                gate, session, url, _hold, index + 1, f"follower-{index}"
            )
            for index in range(QUEUE_LIMIT)
        ]
        await gate.assert_counters(
            "the queue is exactly full", **{DIRECT: 1, QUEUED: QUEUE_LIMIT}
        )

        # Every queue place is taken, so this one has nowhere to go. It must be
        # shed rather than displace a request ahead of it.
        status, body = await _result(
            asyncio.create_task(_unary(session, url)), "overflow"
        )
        _assert_refused(status, body, "overflow")
        assert await gate.gauges() == (
            1,
            QUEUE_LIMIT,
        ), "a shed request must consume neither a slot nor a queue place"
        await gate.assert_counters(
            "the overflow request is shed",
            **{DIRECT: 1, QUEUED: QUEUE_LIMIT + 1, QUEUE_FULL: 1},
        )
        assert not any(
            task.done() for task in queued
        ), "the older queued requests must not be displaced by the shed one"

        # Release the holder, then hand the single slot on one follower at a
        # time. Each step names which request must have taken it: the oldest
        # still queued, and no other. That is the FIFO order, and the occupancy
        # between steps is the bound.
        released = holder
        for index in range(QUEUE_LIMIT):
            released.close()
            admitted = await asyncio.wait_for(queued[index], timeout=OUTCOME_TIMEOUT_S)
            remaining = QUEUE_LIMIT - index - 1
            await gate.await_occupancy(
                1, remaining, f"follower-{index} to take the freed slot"
            )
            assert not any(task.done() for task in queued[index + 1 :]), (
                f"only follower-{index}, the oldest queued request, may be served "
                f"by this release"
            )
            released = admitted

        released.close()
        await gate.await_occupancy(0, 0, "the gate to drain")
        await gate.assert_counters(
            "the queue has drained",
            **{
                DIRECT: 1,
                QUEUED: QUEUE_LIMIT + 1,
                QUEUE_FULL: 1,
                FIFO_DEQUEUE: QUEUE_LIMIT,
            },
        )

    with _deployment(request, dynamo_dynamic_ports, request_plane) as system_port:
        _run(dynamo_dynamic_ports.frontend_port, system_port, scenario)


######################### CONTROLLED DELAY DISABLED #########################


@pytest.mark.timeout(75)
def test_queue_delay_disabled_admits_waiting_requests(
    request,
    runtime_services_session,
    predownload_tokenizers,
    request_plane,
    dynamo_dynamic_ports,
):
    """With expiry off, waiting past the delay is not a reason to refuse."""

    async def scenario(gate: Gate, session: aiohttp.ClientSession, url: str) -> None:
        holder = await _hold(session, url)
        await gate.await_occupancy(1, 0, "the holder to take the only slot")

        queued = [
            await _stage_queued(
                gate, session, url, _unary, index + 1, f"waiter-{index}"
            )
            for index in range(2)
        ]

        # Well past the deadline these would have carried. Nothing may leave the
        # queue for age, so the count must be unchanged on the far side.
        await asyncio.sleep(QUEUE_DELAY_S * 1.5)
        assert await gate.gauges() == (
            1,
            2,
        ), "with controlled delay disabled, no request may leave the queue for age"
        assert not any(
            task.done() for task in queued
        ), "and none of them may have been answered"

        holder.close()
        for index, task in enumerate(queued):
            status, body = await _result(task, f"waiter-{index}")
            assert status == 200, (
                f"waiter {index} waited longer than the {QUEUE_DELAY_MS} ms delay and "
                f"must still be admitted, got {status}: {body}"
            )

        await gate.await_occupancy(0, 0, "the gate to drain")
        await gate.assert_counters(
            "both long-waiting requests are admitted",
            **{DIRECT: 1, QUEUED: 2, FIFO_DEQUEUE: 2},
        )

    with _deployment(
        request,
        dynamo_dynamic_ports,
        request_plane,
        queue_delay_ms=QUEUE_DELAY_MS,
        controlled_delay="0",
    ) as system_port:
        _run(dynamo_dynamic_ports.frontend_port, system_port, scenario)


######################### CONTROLLED DELAY ENABLED #########################


@pytest.mark.timeout(100)
def test_controlled_delay_rejects_expired_front_requests(
    request,
    runtime_services_session,
    predownload_tokenizers,
    request_plane,
    dynamo_dynamic_ports,
):
    """The due prefix is rejected from the front, and only the due prefix."""

    async def scenario(gate: Gate, session: aiohttp.ClientSession, url: str) -> None:
        holder = await _hold(session, url)
        await gate.await_occupancy(1, 0, "the holder to take the only slot")

        # Both are queued before either deadline passes. Staging the eligible
        # request only after the rejection would not reach the gate at all: the
        # Frontend backs the worker off briefly once it has seen a refusal, so
        # the request under test would never become a gate admission decision.
        doomed = await _stage_queued(gate, session, url, _unary, 1, "doomed")
        await asyncio.sleep(FRESH_LAG_S)
        later = await _stage_queued(gate, session, url, _unary, 2, "later")

        # Nothing frees a slot, so the oldest leaves on its own deadline while
        # the one behind it is still comfortably live.
        status, body = await _result(doomed, "doomed")
        _assert_refused(status, body, "doomed")
        await gate.await_occupancy(1, 1, "the expired request to leave the queue")
        assert not later.done(), "the later request must outlive the one ahead of it"
        await gate.assert_counters(
            "one front request expires",
            **{DIRECT: 1, QUEUED: 2, REQUEST_EXPIRED: 1},
        )

        holder.close()
        status, body = await _result(later, "later")
        assert status == 200, f"the later request must be admitted: {body}"
        await gate.await_occupancy(0, 0, "the gate to drain")
        await gate.assert_counters(
            "the later request is admitted",
            **{DIRECT: 1, QUEUED: 2, REQUEST_EXPIRED: 1, FIFO_DEQUEUE: 1},
        )

        # Now two close-deadline front requests, then a later one. Both fronts
        # must expire; the last must survive and be admitted. This phase opens
        # just after a refusal, so it waits for the Frontend to route again
        # rather than failing on its brief backoff.
        holder = await _hold_when_routable(session, url)
        await gate.await_occupancy(1, 0, "the second holder to take the slot")
        first = await _stage_queued(gate, session, url, _unary, 1, "first-front")
        await asyncio.sleep(FRONT_LAG_S)
        second = await _stage_queued(gate, session, url, _unary, 2, "second-front")
        # Far enough behind the pair that it is still comfortably live when
        # their deadlines pass.
        await asyncio.sleep(FRESH_LAG_S)
        survivor = await _stage_queued(gate, session, url, _unary, 3, "survivor")

        for task, label in ((first, "first-front"), (second, "second-front")):
            status, body = await _result(task, label)
            _assert_refused(status, body, label)
        await gate.await_occupancy(1, 1, "the due prefix to leave the queue")
        assert not survivor.done(), "the survivor must outlive the front pair"
        await gate.assert_counters(
            "both front requests expire",
            **{DIRECT: 2, QUEUED: 5, REQUEST_EXPIRED: 3, FIFO_DEQUEUE: 1},
        )

        holder.close()
        status, body = await _result(survivor, "survivor")
        assert status == 200, f"the survivor must be admitted: {body}"
        await gate.await_occupancy(0, 0, "the gate to drain")
        await gate.assert_counters(
            "the survivor is admitted",
            **{DIRECT: 2, QUEUED: 5, REQUEST_EXPIRED: 3, FIFO_DEQUEUE: 2},
        )

    with _deployment(
        request,
        dynamo_dynamic_ports,
        request_plane,
        queue_delay_ms=QUEUE_DELAY_MS,
        adaptive_lifo="0",
    ) as system_port:
        _run(dynamo_dynamic_ports.frontend_port, system_port, scenario)


######################### FIFO AFTER A DEADLINE #########################


@pytest.mark.timeout(70)
def test_fifo_can_complete_after_queue_deadline_without_adaptive_lifo(
    request,
    runtime_services_session,
    predownload_tokenizers,
    request_plane,
    dynamo_dynamic_ports,
):
    """Plain FIFO admits the near-expiry front, which then runs past its old
    queue deadline.

    The deadline bounds queue residence only. Once the front request is admitted
    just inside it, nothing stops it finishing afterwards — and with adaptive
    LIFO off, the freed slot goes to that front request rather than to the
    newest.
    """

    async def scenario(gate: Gate, session: aiohttp.ClientSession, url: str) -> None:
        holder = await _hold(session, url)
        await gate.await_occupancy(1, 0, "the holder to take the only slot")
        staged = await _stage_near_deadline_queue(gate, session, url, _unary)

        status, body = await _result(staged.doomed, "doomed")
        _assert_refused(status, body, "doomed")
        await gate.await_occupancy(1, 2, "the expired request to leave the queue")

        # Release, then date the handover by the counter rather than by the
        # close: the slot must have reached the front before the earliest its
        # deadline could be, or it is not established that the front was still
        # eligible when it was chosen.
        holder.close()
        handed_over_at = await gate.await_counter(
            FIFO_DEQUEUE, 1, "the freed slot to reach the front of the queue"
        )
        assert handed_over_at < staged.front_deadline_earliest, (
            f"the slot reached the front {handed_over_at - staged.front_deadline_earliest:.2f}s "
            f"after the earliest its deadline could be, so it may have been "
            f"admitted only because nothing had expired it yet"
        )

        status, body = await _result(staged.front, "front")
        assert (
            status == 200
        ), f"with adaptive LIFO off the freed slot must go to the front: {body}"
        # And the bound that is conservatively late: whatever its true deadline
        # was, it is no later than this, so finishing after it is unambiguous.
        finished_at = time.monotonic()
        assert finished_at > staged.front_deadline_latest, (
            f"the front request finished {staged.front_deadline_latest - finished_at:.2f}s "
            f"before the latest its former queue deadline could have been, so this "
            f"says nothing about running past it"
        )

        # `fresh` had the larger budget, so it is still eligible behind it.
        status, body = await _result(staged.fresh, "fresh")
        assert status == 200, f"the fresher request follows it: {body}"
        await gate.await_occupancy(0, 0, "the gate to drain")
        await gate.assert_counters(
            "the front ran past its former deadline and nothing came from the tail",
            **{DIRECT: 1, QUEUED: 3, REQUEST_EXPIRED: 1, FIFO_DEQUEUE: 2},
        )

    with _deployment(
        request,
        dynamo_dynamic_ports,
        request_plane,
        queue_delay_ms=QUEUE_DELAY_MS,
        adaptive_lifo="0",
    ) as system_port:
        _run(dynamo_dynamic_ports.frontend_port, system_port, scenario)


######################### ADAPTIVE LIFO #########################


@pytest.mark.timeout(65)
def test_adaptive_lifo_avoids_near_expiry_front_admission(
    request,
    runtime_services_session,
    predownload_tokenizers,
    request_plane,
    dynamo_dynamic_ports,
):
    """The same timing shape, with adaptive LIFO on.

    After the due prefix is rejected, the freed slot goes to the freshest queued
    request instead of to the front that has nearly spent its budget. The front
    keeps its place and then expires rather than entering the backend.
    """

    async def scenario(gate: Gate, session: aiohttp.ClientSession, url: str) -> None:
        holder = await _hold(session, url)
        await gate.await_occupancy(1, 0, "the holder to take the only slot")
        # The same shape as the FIFO scenario, except that `fresh` asks for a
        # long stream: once admitted it holds the only slot for the rest of the
        # scenario, so the front never gets one.
        staged = await _stage_near_deadline_queue(gate, session, url, _hold)

        status, body = await _result(staged.doomed, "doomed")
        _assert_refused(status, body, "doomed")
        await gate.await_occupancy(1, 2, "the expired request to leave the queue")

        # Release, then date the handover by the counter: the slot went to the
        # tail before the earliest the front's deadline could be, so the front
        # was still there to be passed over. That is what rules out the other
        # way this could look the same — a slow release that let the front
        # expire first and only then handed the slot on, which plain FIFO would
        # also do.
        holder.close()
        handed_over_at = await gate.await_counter(
            LIFO_DEQUEUE, 1, "the freed slot to be counted as a tail admission"
        )
        assert handed_over_at < staged.front_deadline_earliest, (
            f"the tail admission was seen {handed_over_at - staged.front_deadline_earliest:.2f}s "
            f"after the earliest the front's deadline could be, so the front may "
            f"simply have expired first"
        )
        assert not staged.front.done(), "the near-expiry front must still be queued"
        assert await gate.gauges() == (
            1,
            1,
        ), "the front keeps its place while the fresher request runs"
        held = await asyncio.wait_for(staged.fresh, timeout=OUTCOME_TIMEOUT_S)

        # It keeps that place until its own delay runs out.
        status, body = await _result(staged.front, "front")
        _assert_refused(status, body, "front")
        await gate.assert_counters(
            "the near-expiry front expired instead of being admitted",
            **{DIRECT: 1, QUEUED: 3, REQUEST_EXPIRED: 2, LIFO_DEQUEUE: 1},
        )

        held.close()
        await gate.await_occupancy(0, 0, "the gate to drain")

    with _deployment(
        request,
        dynamo_dynamic_ports,
        request_plane,
        queue_delay_ms=QUEUE_DELAY_MS,
    ) as system_port:
        _run(dynamo_dynamic_ports.frontend_port, system_port, scenario)


######################### ENGINE-LIMIT SIZING #########################

# The capacity the worker publishes for the sizing scenario. Both factors are
# set on the mocker rather than defaulted, so the limit the gate must derive —
# ceil(3/2 x max_num_seqs x data_parallel_size) = ceil(3/2 x 2 x 1) = 3 — follows
# from inputs this test states. The values are chosen so that the automatic
# result and the override below are different numbers: an override that happened
# to equal the automatic limit would prove nothing about which of the two the
# gate used.
SIZING_MAX_NUM_SEQS = 2
SIZING_DATA_PARALLEL_SIZE = 1
AUTOMATIC_ENGINE_LIMIT = 3
SIZING_OVERRIDE = 1


@pytest.mark.timeout(35)
@pytest.mark.parametrize(
    "engine_request_limit, expected_limit",
    [
        pytest.param(None, AUTOMATIC_ENGINE_LIMIT, id="automatic"),
        pytest.param(SIZING_OVERRIDE, SIZING_OVERRIDE, id="override"),
    ],
)
def test_engine_limit_is_sized_from_the_capacity_hint_or_the_override(
    request,
    runtime_services_session,
    predownload_tokenizers,
    request_plane,
    dynamo_dynamic_ports,
    engine_request_limit,
    expected_limit,
):
    """Where the concurrency limit comes from when it is not pinned, and that
    pinning it still wins.

    Both cases publish the same capacity. Without `DYN_ENGINE_REQUEST_LIMIT` the
    gate must size itself from it; with the override set to a different number,
    that number must win. The queue bound is configured independently in both and
    must not move with either.
    """

    async def scenario(gate: Gate, session: aiohttp.ClientSession, url: str) -> None:
        # `take_baseline` has already asserted the reported sizing. What follows
        # shows it is the limit actually in force rather than only published:
        # one more request than the limit, and exactly one of them must queue.
        inflight = [
            asyncio.create_task(_hold(session, url)) for _ in range(expected_limit + 1)
        ]
        try:
            await gate.await_occupancy(
                expected_limit, 1, "the limit to fill and the next request to queue"
            )
            await gate.assert_counters(
                "the limit is filled and the next request queues",
                **{DIRECT: expected_limit, QUEUED: 1},
            )
        finally:
            # Held streams that never had to finish: the oracle above is the
            # gauges, not any response body. Not every admitted request even
            # begins streaming, because the automatic limit sits above the
            # engine's own `max_num_seqs` by design.
            for task in inflight:
                task.cancel()
            for result in await asyncio.gather(*inflight, return_exceptions=True):
                if isinstance(result, aiohttp.ClientResponse):
                    result.close()

    with _deployment(
        request,
        dynamo_dynamic_ports,
        request_plane,
        engine_request_limit=engine_request_limit,
        max_num_seqs=SIZING_MAX_NUM_SEQS,
        data_parallel_size=SIZING_DATA_PARALLEL_SIZE,
    ) as system_port:
        _run(
            dynamo_dynamic_ports.frontend_port,
            system_port,
            scenario,
            engine_limit=expected_limit,
        )
