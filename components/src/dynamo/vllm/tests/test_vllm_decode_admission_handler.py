# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Decode-handler lifecycle tests for remote-prefill admission."""

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import dynamo.vllm.handlers as handlers
from dynamo.vllm.constants import DisaggregationMode

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.core,
    pytest.mark.timeout(5),
]


def _make_handler(limit: int = 1) -> handlers.DecodeWorkerHandler:
    config = SimpleNamespace(
        decode_max_remote_prefill_inflight=limit,
        disaggregation_mode=DisaggregationMode.DECODE,
    )
    with patch.object(handlers.BaseWorkerHandler, "__init__", return_value=None):
        handler = handlers.DecodeWorkerHandler(
            runtime=MagicMock(),
            config=config,
            engine=MagicMock(),
            default_sampling_params={},
        )
    handler.config = config
    handler.use_vllm_tokenizer = False
    handler.dp_range = (0, 2)
    handler.shutdown_event = None
    handler._multimodal_request_processor = MagicMock()
    return handler


def _request(
    name: str,
    dp_rank: int | None = 0,
    remote_prefill: bool = True,
) -> dict:
    return {
        "name": name,
        "routing": {"dp_rank": dp_rank},
        "prefill_result": {
            "disaggregated_params": {
                "kv_transfer_params": {
                    "do_remote_prefill": remote_prefill,
                }
            }
        },
    }


def _context(request_id: str) -> MagicMock:
    context = MagicMock()
    context.id.return_value = request_id
    context.async_killed_or_stopped.return_value = (
        asyncio.get_event_loop().create_future()
    )
    return context


async def _collect(
    handler: handlers.DecodeWorkerHandler,
    request: dict,
    context: MagicMock,
) -> list[dict]:
    return [chunk async for chunk in handler.generate(request, context)]


async def _wait_for_waiters(
    handler: handlers.DecodeWorkerHandler,
    expected: int,
) -> None:
    for _ in range(100):
        snapshot = handler._decode_remote_prefill_admission.snapshot(0)
        if snapshot.waiting == expected:
            return
        await asyncio.sleep(0)
    raise AssertionError(f"waiter count did not reach {expected}")


@pytest.mark.parametrize(
    ("request_payload", "expected"),
    [
        (_request("remote"), True),
        (_request("local", remote_prefill=False), False),
        ({"prefill_result": None}, False),
        (
            {
                **_request("bypass"),
                "annotations": [handlers.BYPASS_REMOTE_PREFILL_ANNOTATION],
            },
            False,
        ),
    ],
)
def test_remote_prefill_request_classification(request_payload, expected):
    assert handlers._has_remote_prefill_kv_transfer(request_payload) is expected


@pytest.mark.asyncio
async def test_zero_limit_preserves_the_existing_decode_path():
    handler = _make_handler(limit=0)
    handler._to_local_dp_rank = MagicMock(
        side_effect=AssertionError("disabled admission must not inspect DP routing")
    )

    async def generate(request, context, request_id):
        yield {"token_ids": [1]}

    handler._generate_token_mode = generate

    assert await _collect(handler, _request("disabled"), _context("disabled")) == [
        {"token_ids": [1]}
    ]
    handler._to_local_dp_rank.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "routing",
    [
        {},
        {"dp_rank": 4},
    ],
)
async def test_multi_rank_worker_rejects_unresolved_dp_rank(routing):
    handler = _make_handler()
    handler._generate_token_mode = MagicMock(
        side_effect=AssertionError("invalid routing must not start generation")
    )
    request = _request("invalid")
    request["routing"] = routing

    with pytest.raises(handlers.InvalidArgument, match="valid routing.dp_rank"):
        await _collect(handler, request, _context("invalid"))

    handler._generate_token_mode.assert_not_called()
    assert handler._decode_remote_prefill_admission.snapshot(0).admissions == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "routing",
    [
        {},
        {"dp_rank": 9},
    ],
)
async def test_single_rank_worker_uses_rank_zero_for_unresolved_routing(routing):
    handler = _make_handler()
    handler.dp_range = (4, 1)

    async def generate(request, context, request_id):
        yield {"token_ids": [1]}

    handler._generate_token_mode = generate
    request = _request("single")
    request["routing"] = routing

    assert await _collect(handler, request, _context("single")) == [{"token_ids": [1]}]
    assert handler._decode_remote_prefill_admission.snapshot(0).admissions == 1


@pytest.mark.asyncio
async def test_first_output_releases_waiting_request():
    handler = _make_handler()
    first_started = asyncio.Event()
    first_can_yield = asyncio.Event()
    second_started = asyncio.Event()

    async def generate(request, context, request_id):
        if request["name"] == "first":
            first_started.set()
            await first_can_yield.wait()
        else:
            second_started.set()
        yield {"token_ids": [1]}

    handler._generate_token_mode = generate
    first_task = asyncio.create_task(
        _collect(handler, _request("first"), _context("first"))
    )
    await asyncio.wait_for(first_started.wait(), timeout=1)

    second_task = asyncio.create_task(
        _collect(handler, _request("second"), _context("second"))
    )
    await _wait_for_waiters(handler, expected=1)
    assert not second_started.is_set()

    first_can_yield.set()
    await asyncio.wait_for(second_started.wait(), timeout=1)
    assert await first_task == [{"token_ids": [1]}]
    assert await second_task == [{"token_ids": [1]}]

    snapshot = handler._decode_remote_prefill_admission.snapshot(0)
    assert snapshot.active == 0
    assert snapshot.releases == 2


@pytest.mark.asyncio
async def test_failure_before_first_output_releases_waiting_request():
    handler = _make_handler()
    first_started = asyncio.Event()
    first_can_fail = asyncio.Event()
    second_started = asyncio.Event()

    async def generate(request, context, request_id):
        if request["name"] == "first":
            first_started.set()
            await first_can_fail.wait()
            raise RuntimeError("injected failure")
        second_started.set()
        yield {"token_ids": [1]}

    handler._generate_token_mode = generate
    first_task = asyncio.create_task(
        _collect(handler, _request("first"), _context("first"))
    )
    await asyncio.wait_for(first_started.wait(), timeout=1)

    second_task = asyncio.create_task(
        _collect(handler, _request("second"), _context("second"))
    )
    await _wait_for_waiters(handler, expected=1)

    first_can_fail.set()
    with pytest.raises(RuntimeError, match="injected failure"):
        await first_task

    await asyncio.wait_for(second_started.wait(), timeout=1)
    assert await second_task == [{"token_ids": [1]}]
    assert handler._decode_remote_prefill_admission.snapshot(0).active == 0


@pytest.mark.asyncio
async def test_cancellation_before_first_output_releases_waiting_request():
    handler = _make_handler()
    first_started = asyncio.Event()
    second_started = asyncio.Event()

    async def generate(request, context, request_id):
        if request["name"] == "first":
            first_started.set()
            await asyncio.Event().wait()
        second_started.set()
        yield {"token_ids": [1]}

    handler._generate_token_mode = generate
    first_task = asyncio.create_task(
        _collect(handler, _request("first"), _context("first"))
    )
    await asyncio.wait_for(first_started.wait(), timeout=1)

    second_task = asyncio.create_task(
        _collect(handler, _request("second"), _context("second"))
    )
    await _wait_for_waiters(handler, expected=1)

    first_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first_task

    await asyncio.wait_for(second_started.wait(), timeout=1)
    assert await second_task == [{"token_ids": [1]}]
    assert handler._decode_remote_prefill_admission.snapshot(0).active == 0


@pytest.mark.asyncio
async def test_client_stop_removes_queued_request_before_generation():
    handler = _make_handler()
    first_started = asyncio.Event()
    first_can_yield = asyncio.Event()
    second_started = asyncio.Event()

    async def generate(request, context, request_id):
        if request["name"] == "first":
            first_started.set()
            await first_can_yield.wait()
        else:
            second_started.set()
        yield {"token_ids": [1]}

    handler._generate_token_mode = generate
    first_task = asyncio.create_task(
        _collect(handler, _request("first"), _context("first"))
    )
    await asyncio.wait_for(first_started.wait(), timeout=1)

    second_context = _context("second")
    second_task = asyncio.create_task(
        _collect(handler, _request("second"), second_context)
    )
    await _wait_for_waiters(handler, expected=1)

    second_context.async_killed_or_stopped.return_value.set_result(None)
    with pytest.raises(asyncio.CancelledError):
        await second_task

    snapshot = handler._decode_remote_prefill_admission.snapshot(0)
    assert snapshot.waiting == 0
    assert snapshot.cancelled_waiters == 1
    assert not second_started.is_set()

    first_can_yield.set()
    assert await first_task == [{"token_ids": [1]}]


@pytest.mark.asyncio
async def test_shutdown_removes_queued_request_before_generation():
    handler = _make_handler()
    handler.shutdown_event = asyncio.Event()
    first_started = asyncio.Event()
    first_can_yield = asyncio.Event()
    second_started = asyncio.Event()

    async def generate(request, context, request_id):
        if request["name"] == "first":
            first_started.set()
            await first_can_yield.wait()
        else:
            second_started.set()
        yield {"token_ids": [1]}

    handler._generate_token_mode = generate
    first_task = asyncio.create_task(
        _collect(handler, _request("first"), _context("first"))
    )
    await asyncio.wait_for(first_started.wait(), timeout=1)

    second_task = asyncio.create_task(
        _collect(handler, _request("second"), _context("second"))
    )
    await _wait_for_waiters(handler, expected=1)

    handler.shutdown_event.set()
    with pytest.raises(
        handlers.EngineShutdown,
        match="waiting for decode admission",
    ):
        await second_task

    snapshot = handler._decode_remote_prefill_admission.snapshot(0)
    assert snapshot.waiting == 0
    assert snapshot.cancelled_waiters == 1
    assert not second_started.is_set()

    first_can_yield.set()
    assert await first_task == [{"token_ids": [1]}]


@pytest.mark.asyncio
async def test_client_stop_releases_concurrently_acquired_lease():
    handler = _make_handler()
    lease = MagicMock()

    async def acquire(dp_rank):
        return lease

    async def wait_for_all(waiters, *, return_when):
        await asyncio.gather(*waiters)
        return set(waiters), set()

    handler._decode_remote_prefill_admission.acquire = acquire
    context = _context("stopped")
    context.async_killed_or_stopped.return_value.set_result(None)

    with patch.object(handlers.asyncio, "wait", new=wait_for_all):
        with pytest.raises(asyncio.CancelledError):
            await handler._acquire_decode_remote_prefill_admission(context, 0)

    lease.release.assert_called_once_with("terminal_before_first_output")


@pytest.mark.asyncio
async def test_empty_stream_releases_waiting_request():
    handler = _make_handler()
    first_started = asyncio.Event()
    first_can_finish = asyncio.Event()
    second_started = asyncio.Event()

    async def generate(request, context, request_id):
        if request["name"] == "first":
            first_started.set()
            await first_can_finish.wait()
            return
        second_started.set()
        yield {"token_ids": [1]}

    handler._generate_token_mode = generate
    first_task = asyncio.create_task(
        _collect(handler, _request("first"), _context("first"))
    )
    await asyncio.wait_for(first_started.wait(), timeout=1)

    second_task = asyncio.create_task(
        _collect(handler, _request("second"), _context("second"))
    )
    await _wait_for_waiters(handler, expected=1)

    first_can_finish.set()
    assert await first_task == []
    await asyncio.wait_for(second_started.wait(), timeout=1)
    assert await second_task == [{"token_ids": [1]}]
    assert handler._decode_remote_prefill_admission.snapshot(0).active == 0


@pytest.mark.asyncio
async def test_non_remote_request_bypasses_full_gate():
    handler = _make_handler()
    first_started = asyncio.Event()
    first_can_yield = asyncio.Event()
    local_started = asyncio.Event()

    async def generate(request, context, request_id):
        if request["name"] == "first":
            first_started.set()
            await first_can_yield.wait()
        else:
            local_started.set()
        yield {"token_ids": [1]}

    handler._generate_token_mode = generate
    first_task = asyncio.create_task(
        _collect(handler, _request("first"), _context("first"))
    )
    await asyncio.wait_for(first_started.wait(), timeout=1)

    local_task = asyncio.create_task(
        _collect(
            handler,
            _request("local", remote_prefill=False),
            _context("local"),
        )
    )
    await asyncio.wait_for(local_started.wait(), timeout=1)
    assert await local_task == [{"token_ids": [1]}]
    assert handler._decode_remote_prefill_admission.snapshot(0).waiting == 0

    first_can_yield.set()
    assert await first_task == [{"token_ids": [1]}]
