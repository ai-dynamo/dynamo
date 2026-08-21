# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Process-level coverage for the shared backend admission gate.

The gate lives in `Ingress::handle_payload_shared`, the one admission point both
request planes funnel through, so a CPU-only mocker worker exercises the real
thing over TCP and NATS alike. The formula, precedence, FIFO order, cancellation
and resizing rules are covered by the Rust tests; this asserts only the
end-to-end contract: N in flight, exactly Q queued, the next one shed as HTTP
529, and the slot reused afterwards.
"""

import asyncio
import logging
from typing import Any, Optional

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

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.integration,
    pytest.mark.fault_tolerance,
    pytest.mark.mocker,
    pytest.mark.model(MODEL_NAME),
]


def _payload(max_tokens: int, stream: bool) -> dict[str, Any]:
    return {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": "admission gate"}],
        "stream": stream,
        "max_tokens": max_tokens,
        "ignore_eos": True,
    }


async def _await_first_chunk(response: aiohttp.ClientResponse) -> None:
    """Wait until the holder is really generating, so its slot is held."""
    async for line in response.content:
        if line.startswith(b"data:"):
            return
    raise AssertionError("holder stream ended before yielding a chunk")


async def _run(frontend_url: str, mockers: MockerProcess, request_plane: str) -> None:
    await wait_for_frontend_ready(
        frontend_url=frontend_url,
        expected_num_workers=mockers.num_workers,
        engine_workers=mockers,
        request_plane=request_plane,
        test_payload=_payload(1, False),
    )

    url = f"{frontend_url}/v1/chat/completions"
    holder_response: Optional[aiohttp.ClientResponse] = None
    followers: list[asyncio.Task] = []

    async with aiohttp.ClientSession() as session:
        try:
            # 1. Hold the single engine slot with a long streaming request.
            holder_response = await session.post(url, json=_payload(2000, True))
            assert holder_response.status == 200, await holder_response.text()
            await _await_first_chunk(holder_response)

            # 2. Two unary followers: one fills the queue, one must be shed.
            async def follower() -> tuple[int, Any]:
                async with session.post(url, json=_payload(1, False)) as response:
                    return response.status, await response.json()

            followers = [asyncio.create_task(follower()) for _ in range(2)]
            # Return as soon as the shed one answers, then settle briefly so a
            # second (incorrect) completion would still be observed.
            done, pending = await asyncio.wait(
                followers, timeout=30, return_when=asyncio.FIRST_COMPLETED
            )
            if len(done) == 1:
                await asyncio.wait(pending, timeout=1)
                done = {task for task in followers if task.done()}
                pending = {task for task in followers if not task.done()}

            assert len(done) == 1, (
                f"exactly one follower must be shed while the other queues; "
                f"{len(done)} completed"
            )
            assert len(pending) == 1, "one follower must stay queued behind the holder"

            status, body = done.pop().result()
            assert (
                status == 529
            ), f"the shed request must be HTTP 529, got {status}: {body}"
            error = body.get("error", body)
            assert str(error.get("code")) == "529", f"unexpected error body: {body}"
            assert error.get("type") == "Overloaded", f"unexpected error body: {body}"

            # 3. Cancel/release the holder; the queued follower inherits its slot.
            holder_response.close()
            holder_response = None

            queued = pending.pop()
            status, body = await asyncio.wait_for(queued, timeout=60)
            assert (
                status == 200
            ), f"the queued request must be admitted, got {status}: {body}"

            # 4. The slot is reusable once everything has drained.
            async with session.post(url, json=_payload(1, False)) as response:
                assert response.status == 200, await response.text()
        finally:
            if holder_response is not None:
                holder_response.close()
            for task in followers:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*followers, return_exceptions=True)


@pytest.mark.timeout(120)
@pytest.mark.parametrize("request_plane", ["tcp", "nats"], indirect=True)
def test_backend_admission_gate_sheds_and_reuses_slots(
    request,
    runtime_services,
    predownload_tokenizers,
    request_plane,
):
    """One shared gate bounds a worker identically over both request planes."""
    mockers = MockerProcess(
        request,
        # A slow mocker keeps the holder generating for the whole check.
        mocker_args={"speedup_ratio": 0.01, "block_size": BLOCK_SIZE},
        num_mockers=1,
        request_plane=request_plane,
        env_overrides={
            "DYN_ENGINE_REQUEST_LIMIT": str(ENGINE_REQUEST_LIMIT),
            "DYN_DYNAMO_REQUEST_QUEUE_LIMIT": str(QUEUE_LIMIT),
        },
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
