# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Process-level coverage for the shared backend admission gate.

The gate lives in `Ingress::handle_payload_shared`, the one admission point both
request planes funnel through, so a CPU-only mocker worker exercises the real
thing over TCP and NATS alike. The formula, precedence, FIFO order, cancellation,
resizing and queue-delay rules are covered by the Rust tests; one launch per
request plane asserts the mechanics end to end: N admitted, exactly Q queued, the
next one refused, the released slot reused, and a queued request refused once it
outlives the queue delay.

A refusal happens before the backend future is polled and surfaces through the
existing pre-stream error transport, so these assertions stay generic: a refused
request must not succeed. Which typed error and HTTP status it maps to is the
transport's business, not the gate's, and is deliberately not pinned here.
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

# Worker-local admission: one in flight, one queued, everything else shed.
ENGINE_REQUEST_LIMIT = 1
QUEUE_LIMIT = 1

# Overridden below the 5000 ms default to keep this CPU-only run short. The
# default itself, and the resolution rules around it, are covered by the Rust
# unit tests; what this proves is that the configured value is the deadline in
# force. Long enough that a refusal this slow cannot be an immediate queue-full
# shed, and far enough under the default that a run which ignored the override
# would breach the upper bound rather than pass by accident.
QUEUE_TIMEOUT_MS = 2000
EXPIRY_UPPER_BOUND_S = 4.5

# Long enough to establish that a request really is parked in the queue, short
# enough that it cannot itself consume the queue delay.
PENDING_WINDOW_S = 0.3


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
    queued: Optional[asyncio.Task] = None
    overflow: Optional[asyncio.Task] = None
    try:
        # Ordered, not raced: this one arrives first and must take the single
        # queue place, so it is still pending when the next one arrives. The
        # whole sequence below has to finish well inside the queue delay, or the
        # queued request would leave on its own expiry instead of on the slot.
        started = time.monotonic()
        queued = asyncio.create_task(_unary(session, url))
        await asyncio.wait({queued}, timeout=PENDING_WINDOW_S)
        assert not queued.done(), (
            f"the first follower must queue behind the holder, not answer: "
            f"{queued.result() if queued.done() else ''}"
        )

        # The queue place is taken, so this one has nowhere to go and must be
        # refused promptly rather than displacing the request ahead of it.
        overflow = asyncio.create_task(_unary(session, url))
        await asyncio.wait({overflow}, timeout=30)
        assert overflow.done(), "the overflow request must be refused promptly"
        _assert_refused(*overflow.result())

        elapsed = time.monotonic() - started
        assert elapsed < QUEUE_TIMEOUT_MS / 1000 * 0.5, (
            f"the queue-full check took {elapsed:.2f}s of the "
            f"{QUEUE_TIMEOUT_MS} ms queue delay; too slow to attribute what "
            f"happens to the queued request next"
        )
        assert not queued.done(), "the older queued request must not be displaced"

        # Release the holder; the queued follower inherits its slot.
        holder.close()
        holder = None

        status, body = await asyncio.wait_for(queued, timeout=60)
        assert (
            status == 200
        ), f"the queued request must be admitted, got {status}: {body}"

        # The slot is reusable once everything has drained.
        await _admitted_once_settled(session, url)
    finally:
        if holder is not None:
            holder.close()
        for task in (queued, overflow):
            if task is not None and not task.done():
                task.cancel()
        followers = [task for task in (queued, overflow) if task is not None]
        await asyncio.gather(*followers, return_exceptions=True)


async def _expires_a_request_that_outlives_the_queue_delay(
    session: aiohttp.ClientSession, url: str
) -> None:
    holder = await _hold_engine_slot(session, url)
    try:
        # One follower, so it takes the only queue place rather than being shed
        # as queue-full. Nothing frees a slot, so the queue delay is its only
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
        mockers = MockerProcess(
            request,
            # A slow mocker keeps each holder generating for the whole check. Its
            # default `max_num_seqs` is far above the gate's limit of one, so the
            # gate, not the mocker's own scheduler, is the layer that refuses.
            mocker_args={"speedup_ratio": 0.01, "block_size": BLOCK_SIZE},
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
