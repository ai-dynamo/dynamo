# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Process-level coverage for the shared backend admission gate.

The gate lives in `Ingress::handle_payload_shared`, the one admission point both
request planes funnel through, so a CPU-only mocker worker exercises the real
thing over TCP and NATS alike. The formula, precedence, FIFO order, cancellation,
resizing and queue-delay rules are covered by the Rust tests; one launch per
request plane asserts the mechanics end to end: N admitted, exactly Q queued, the
next one refused, the released slot reused, a queued request refused once it
outlives the queue delay, and — once a request has been rejected — the freed slot
going to the newest queued request rather than the oldest.

A refusal happens before the backend future is polled and surfaces through the
existing pre-stream error transport, so these assertions stay generic: a refused
request must not succeed. Which typed error and HTTP status it maps to is the
transport's business, not the gate's, and is deliberately not pinned here.

Every launch runs on the shipped default. Turning the back selection off with
`DYN_DYNAMO_REQUEST_QUEUE_ENABLE_CONTROLLED_DELAY` has no transport-visible
surface of its own, so the Rust tests cover it rather than a second pair of
launches.
"""

import asyncio
import logging
import os
import time
from typing import Any, Optional
from unittest import mock

import aiohttp
import pytest

from tests.router.e2e_harness import allocate_frontend_ports
from tests.router.helper import wait_for_frontend_ready
from tests.router.mocker_process import MockerProcess
from tests.router.router_process import FrontendRouterProcess
from tests.utils.constants import FAULT_TOLERANCE_MODEL_NAME

logger = logging.getLogger(__name__)

MODEL_NAME = FAULT_TOLERANCE_MODEL_NAME
BLOCK_SIZE = 16

# Worker-local admission: one in flight, three queued, everything else shed.
# Three is the fewest the selection check can use: one request to be rejected,
# and two live ones behind it to tell the front of the queue from the back.
ENGINE_REQUEST_LIMIT = 1
QUEUE_LIMIT = 3

# Overridden below the 5000 ms default to keep this CPU-only run short. The
# default itself, and the resolution rules around it, are covered by the Rust
# unit tests; what this proves is that the configured value is the deadline in
# force. Long enough that a refusal this slow cannot be an immediate queue-full
# shed, and that a queue can be staged and then drained through the single slot
# well inside it; far enough under the default that a run which ignored the
# override would breach the upper bound rather than pass by accident.
QUEUE_TIMEOUT_MS = 3000
EXPIRY_UPPER_BOUND_S = 4.5

# Long enough to establish that a request really is parked in the queue, short
# enough that staging a full queue cannot itself consume the queue delay.
PENDING_WINDOW_S = 0.3

# How much of its delay the oldest entry burns alone before the two behind it
# are staged, which is what leaves them most of their own once it is rejected.
HEAD_SOAK_S = QUEUE_TIMEOUT_MS / 1000 * 0.6


def _payload(max_tokens: int, stream: bool) -> dict[str, Any]:
    return {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": "admission gate"}],
        "stream": stream,
        "max_tokens": max_tokens,
        "ignore_eos": True,
    }


async def _unary(session: aiohttp.ClientSession, url: str) -> tuple[int, str]:
    """Read the body as text: a generic pre-stream failure need not be JSON."""
    async with session.post(url, json=_payload(1, False)) as response:
        return response.status, await response.text()


async def _hold_engine_slot(
    session: aiohttp.ClientSession, url: str
) -> aiohttp.ClientResponse:
    """Occupy the single engine slot with a long stream that has begun."""
    response = await session.post(url, json=_payload(2000, True))
    assert response.status == 200, await response.text()
    async for line in response.content:
        if line.startswith(b"data:"):
            return response
    raise AssertionError("holder stream ended before yielding a chunk")


async def _stage_queued(
    session: aiohttp.ClientSession, url: str, submit: Any, label: str
) -> asyncio.Task:
    """Submit one request and confirm it parked in the gate's FIFO.

    Staged rather than raced: a request that has not reached the gate holds no
    queue place, so only waiting for each in turn makes the queue's contents and
    its order known before the next arrives.
    """
    task = asyncio.create_task(submit(session, url))
    await asyncio.wait({task}, timeout=PENDING_WINDOW_S)
    assert not task.done(), (
        f"the {label} request must queue behind the holder, not answer: "
        f"{task.result() if task.done() else ''}"
    )
    return task


def _assert_refused(status: int, body: Any) -> None:
    """A refused request must not succeed.

    Deliberately generic: the gate refuses before the backend runs, and how that
    refusal is classified on the wire belongs to the response transport.
    """
    assert status >= 400, f"a refused request must not succeed, got {status}: {body}"


async def _admitted_once_settled(session: aiohttp.ClientSession, url: str) -> None:
    """Poll until a request is admitted, proving the capacity became reusable.

    Releasing a slot reaches the worker asynchronously, and the Frontend may
    briefly hold routing state from the refusals above. Any interim answer must
    still be a refusal rather than some other failure mode.
    """
    deadline = time.monotonic() + 60
    while True:
        status, body = await _unary(session, url)
        if status == 200:
            return
        _assert_refused(status, body)
        assert (
            time.monotonic() < deadline
        ), f"the released capacity never became reusable, last answer {status}: {body}"
        await asyncio.sleep(1)


async def _sheds_when_full_and_reuses_the_slot(
    session: aiohttp.ClientSession, url: str
) -> None:
    holder: Optional[aiohttp.ClientResponse] = await _hold_engine_slot(session, url)
    queued: list[asyncio.Task] = []
    overflow: Optional[asyncio.Task] = None
    try:
        # These fill the queue exactly, one at a time, so each is known to hold
        # a place before the overflow arrives. The whole sequence below has to
        # finish well inside the queue delay, or the oldest would leave on its
        # own expiry instead of on the slot.
        started = time.monotonic()
        for index in range(QUEUE_LIMIT):
            queued.append(
                await _stage_queued(session, url, _unary, f"follower {index}")
            )

        # Every queue place is taken, so this one has nowhere to go and must be
        # refused promptly rather than displacing a request ahead of it.
        overflow = asyncio.create_task(_unary(session, url))
        await asyncio.wait({overflow}, timeout=30)
        assert overflow.done(), "the overflow request must be refused promptly"
        _assert_refused(*overflow.result())

        elapsed = time.monotonic() - started
        assert elapsed < QUEUE_TIMEOUT_MS / 1000 * 0.5, (
            f"staging the queue and shedding the overflow took {elapsed:.2f}s of "
            f"the {QUEUE_TIMEOUT_MS} ms queue delay; too slow to attribute what "
            f"happens to the queued requests next"
        )
        assert not any(
            task.done() for task in queued
        ), "the older queued requests must not be displaced"

        # Release the holder; the queued followers inherit its slot in turn.
        holder.close()
        holder = None
        released = time.monotonic()

        for index, task in enumerate(queued):
            status, body = await asyncio.wait_for(task, timeout=60)
            assert status == 200, (
                f"queued request {index} must be admitted, got {status} "
                f"{time.monotonic() - released:.2f}s after the release: {body}"
            )

        # The slot is reusable once everything has drained.
        await _admitted_once_settled(session, url)
    finally:
        if holder is not None:
            holder.close()
        followers = [task for task in (*queued, overflow) if task is not None]
        for task in followers:
            if not task.done():
                task.cancel()
        await asyncio.gather(*followers, return_exceptions=True)


async def _expires_a_request_that_outlives_the_queue_delay(
    session: aiohttp.ClientSession, url: str
) -> None:
    holder = await _hold_engine_slot(session, url)
    try:
        # One follower, so it takes a queue place rather than being shed as
        # queue-full. Nothing frees a slot, so the queue delay is its only
        # way out.
        started = time.monotonic()
        status, body = await _unary(session, url)
        waited = time.monotonic() - started

        _assert_refused(status, body)
        assert waited >= QUEUE_TIMEOUT_MS / 1000 * 0.75, (
            f"the request was refused after {waited:.2f}s, too fast to be a "
            f"queue-delay expiry"
        )
        assert waited < EXPIRY_UPPER_BOUND_S, (
            f"the request was refused after {waited:.2f}s, so the configured "
            f"{QUEUE_TIMEOUT_MS} ms override was not the deadline in force"
        )
    finally:
        # Releasing the holder returns its slot to the gate.
        holder.close()

    # The expiry must leave the gate able to admit again.
    await _admitted_once_settled(session, url)


async def _admits_from_the_back_after_a_rejection(
    session: aiohttp.ClientSession, url: str
) -> None:
    """A rejection re-points the next admission at the newest queued request.

    Rejection is still only from the front; what this checks is where the freed
    slot goes next — to the request with the most of its delay budget left,
    rather than to the one waiting longest behind the request just rejected.
    """
    holder: Optional[aiohttp.ClientResponse] = await _hold_engine_slot(session, url)
    staged: list[asyncio.Task] = []
    try:
        # All three are queued before any is rejected, in a known FIFO order:
        # `doomed`, then `front`, then `back`. The oldest burns most of its delay
        # alone first, so the two behind it still hold most of their own when it
        # is rejected — that margin, not a race, is what makes the outcome below
        # attributable. `back` asks for a long stream, so once admitted it holds
        # the only slot for the rest of this check.
        started = time.monotonic()
        staged.append(await _stage_queued(session, url, _unary, "doomed"))
        await asyncio.sleep(HEAD_SOAK_S)
        assert not staged[0].done(), (
            f"the oldest request must still be queued {HEAD_SOAK_S:.2f}s into "
            f"its {QUEUE_TIMEOUT_MS} ms delay"
        )
        # Taken before the request is submitted, so the deadline derived from it
        # is conservatively early and the bound below is conservatively short.
        front_due = time.monotonic() + QUEUE_TIMEOUT_MS / 1000
        staged.append(await _stage_queued(session, url, _unary, "front"))
        staged.append(await _stage_queued(session, url, _hold_engine_slot, "back"))
        doomed, front, back = staged

        # Nothing frees a slot, so the oldest leaves on its own deadline. That
        # is the rejection this check is about, and the two behind it are still
        # live when it happens.
        status, body = await asyncio.wait_for(doomed, timeout=60)
        _assert_refused(status, body)
        rejected_after = time.monotonic() - started
        assert rejected_after >= QUEUE_TIMEOUT_MS / 1000 * 0.75, (
            f"the oldest request was refused after {rejected_after:.2f}s, too "
            f"fast to be the queue-delay rejection this check needs"
        )

        # Release the slot. The admission that follows a rejection takes the
        # newest queued request, so `back` must start streaming while `front` is
        # still live and queued. Bounding that wait well inside what `front` has
        # left, and checking `front` is still pending when it lands, is what
        # rules out the other way `back` could have been served: a slow release
        # that let `front` expire first and only then handed the slot on, which
        # plain FIFO would also do.
        holder.close()
        holder = None

        front_budget_left = front_due - time.monotonic()
        assert front_budget_left > 0, (
            f"the front request was already {-front_budget_left:.2f}s past its "
            f"deadline when the slot was released"
        )
        admitted = await asyncio.wait_for(back, timeout=front_budget_left / 2)
        assert not front.done(), (
            "the newest request was served only after the front of the queue "
            "had already completed, which proves nothing about which end the "
            "slot came from"
        )

        # `back` holds the slot from here, so `front` never gets one: it keeps
        # its place until its own delay runs out.
        status, body = await asyncio.wait_for(front, timeout=60)
        _assert_refused(status, body)
        admitted.close()
    finally:
        if holder is not None:
            holder.close()
        for task in staged:
            if not task.done():
                task.cancel()
        for result in await asyncio.gather(*staged, return_exceptions=True):
            # A stream that was admitted after all still holds a slot. Closing
            # one that the check already closed is a no-op.
            if isinstance(result, aiohttp.ClientResponse):
                result.close()

    # One rejection buys one admission from the back: the gate is ordinary
    # again, and still able to admit.
    await _admitted_once_settled(session, url)


async def _run(frontend_url: str, mockers: MockerProcess, request_plane: str) -> None:
    await wait_for_frontend_ready(
        frontend_url=frontend_url,
        expected_num_workers=mockers.num_workers,
        engine_workers=mockers,
        request_plane=request_plane,
        test_payload=_payload(1, False),
    )

    url = f"{frontend_url}/v1/chat/completions"
    async with aiohttp.ClientSession() as session:
        await _sheds_when_full_and_reuses_the_slot(session, url)
        await _expires_a_request_that_outlives_the_queue_delay(session, url)
        await _admits_from_the_back_after_a_rejection(session, url)


@pytest.mark.pre_merge
@pytest.mark.gpu_0
@pytest.mark.integration
@pytest.mark.fault_tolerance
@pytest.mark.mocker
@pytest.mark.model(MODEL_NAME)
@pytest.mark.timeout(180)
@pytest.mark.parametrize("request_plane", ["tcp", "nats"], indirect=True)
def test_backend_admission_gate(
    request,
    runtime_services,
    predownload_tokenizers,
    request_plane,
):
    """One shared gate bounds a worker identically over both request planes."""
    # `MockerProcess` snapshots `os.environ` in its constructor, so patching only
    # around construction keeps this sizing worker-local: the Frontend below is
    # built outside the patch and never sees it.
    worker_gate_config = {
        "DYN_ENGINE_REQUEST_LIMIT": str(ENGINE_REQUEST_LIMIT),
        "DYN_DYNAMO_REQUEST_QUEUE_LIMIT": str(QUEUE_LIMIT),
        "DYN_DYNAMO_REQUEST_QUEUE_TIMEOUT_MS": str(QUEUE_TIMEOUT_MS),
    }
    with mock.patch.dict(os.environ, worker_gate_config):
        # Cleared rather than set to true: this check must exercise the shipped
        # default, and an inherited value would otherwise decide the outcome.
        # `patch.dict` restores the original environment on exit.
        os.environ.pop("DYN_DYNAMO_REQUEST_QUEUE_ENABLE_CONTROLLED_DELAY", None)
        mockers = MockerProcess(
            request,
            # Slowed down so that a holder's stream lasts for a whole check,
            # but not so far that an admitted one-token request approaches the
            # queue delay: the checks below serialise several of those through
            # the single slot while other requests are still queued under it.
            # The mocker's default `max_num_seqs` is far above the gate's limit
            # of one, so the gate, not the mocker's own scheduler, is the layer
            # that refuses.
            mocker_args={"speedup_ratio": 0.1, "block_size": BLOCK_SIZE},
            num_mockers=1,
            request_plane=request_plane,
        )

    frontend_port = allocate_frontend_ports(request, 1)[0]

    with mockers:
        with FrontendRouterProcess(
            request,
            BLOCK_SIZE,
            frontend_port,
            mockers.namespace,
            request_plane=request_plane,
            router_mode="round-robin",
        ):
            asyncio.run(
                _run(f"http://localhost:{frontend_port}", mockers, request_plane)
            )
