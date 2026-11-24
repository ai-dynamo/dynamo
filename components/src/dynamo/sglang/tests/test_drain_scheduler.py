# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest
import zmq
import zmq.asyncio

from dynamo.common.forward_pass_metrics import ForwardPassMetrics, encode
from dynamo.sglang import drain_scheduler

pytestmark = [pytest.mark.unit, pytest.mark.gpu_0, pytest.mark.pre_merge]


@pytest.mark.asyncio
async def test_idle_probe_requires_ack_and_two_consecutive_worker_heartbeats(
    monkeypatch,
):
    ack_started = asyncio.Event()
    ack = asyncio.Event()
    invalid_frames_consumed = asyncio.Event()

    async def get_internal_state():
        ack_started.set()
        await ack.wait()
        return [{}]

    def observe_decode(payload):
        frame = decode(payload)
        if frame is not None and frame.counter_id == 13:
            invalid_frames_consumed.set()
        return frame

    decode = drain_scheduler.decode
    monkeypatch.setattr(drain_scheduler, "decode", observe_decode)
    address = f"inproc://sglang-drain-{uuid4()}"
    manager = SimpleNamespace(get_internal_state=get_internal_state)
    with zmq.asyncio.Context.instance().socket(zmq.XPUB) as publisher:
        publisher.bind(address)
        probe = asyncio.create_task(
            drain_scheduler.SchedulerIdleProbe(manager, address, "worker")()
        )
        try:
            async with asyncio.timeout(1):
                await ack_started.wait()
                assert not await publisher.poll(timeout=0)
                ack.set()
                assert await publisher.recv() == b"\x01"
                for seq, worker, wall_time in [
                    (10, "worker", 0),
                    (11, "other", 0),
                    (12, "worker", 0.01),
                    (13, "worker", 0),
                ]:
                    frame = ForwardPassMetrics(
                        worker_id=worker, counter_id=seq, wall_time=wall_time
                    )
                    await publisher.send_multipart(
                        [b"", seq.to_bytes(8, "big"), encode(frame)]
                    )
                await invalid_frames_consumed.wait()
                assert not probe.done()
                frame = ForwardPassMetrics(worker_id="worker", counter_id=14)
                await publisher.send_multipart(
                    [b"", (14).to_bytes(8, "big"), encode(frame)]
                )
                assert await probe
        finally:
            probe.cancel()
            await asyncio.gather(probe, return_exceptions=True)


@pytest.mark.asyncio
async def test_terminal_drain_rechecks_local_work_after_idle(monkeypatch):
    manager = SimpleNamespace(rid_to_state={})
    endpoint = Mock()
    endpoint.inflight_requests.side_effect = [1, 0, 0, 0]
    probe = drain_scheduler.SchedulerIdleProbe(manager, "unused", "worker")
    attempts = 0

    async def scheduler_idle(self):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            manager.rid_to_state["finishing"] = object()
        return True

    async def retry(_delay):
        if attempts:
            manager.rid_to_state.clear()

    monkeypatch.setattr(type(probe), "__call__", scheduler_idle)
    monkeypatch.setattr(drain_scheduler.asyncio, "sleep", retry)
    await probe.wait_for_terminal_idle(endpoint)
    assert attempts == 2
    assert endpoint.inflight_requests.call_count == 4


@pytest.mark.asyncio
async def test_terminal_drain_cancels_outstanding_scheduler_read():
    read_started = asyncio.Event()
    read_cancelled = asyncio.Event()

    async def get_internal_state():
        read_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            read_cancelled.set()

    manager = SimpleNamespace(rid_to_state={}, get_internal_state=get_internal_state)
    probe = drain_scheduler.SchedulerIdleProbe(manager, "unused", "worker")
    endpoint = SimpleNamespace(inflight_requests=lambda: 0)
    waiting = asyncio.create_task(probe.wait_for_terminal_idle(endpoint))
    await asyncio.wait_for(read_started.wait(), 1)
    waiting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiting
    await asyncio.wait_for(read_cancelled.wait(), 1)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "integrated,plane,ranks,capable,enabled",
    [
        (True, "tcp", 1, True, True),
        (False, "tcp", 1, True, False),
        (True, "quic", 1, True, False),
        (True, "tcp", 2, True, False),
        (True, "tcp", 1, False, False),
    ],
)
async def test_prefill_shutdown_tracks_tcp_completion_only_with_idle_probe(
    monkeypatch, integrated, plane, ranks, capable, enabled
):
    args = SimpleNamespace(
        enable_forward_pass_metrics=True,
        disaggregation_transfer_backend="nixl",
        nnodes=1,
        tp_size=ranks,
        dp_size=1,
        pp_size=1,
        tokenizer_worker_num=1,
        forward_pass_metrics_ipc_name="ipc://unused",
    )
    handler = SimpleNamespace(
        config=SimpleNamespace(
            server_args=args, dynamo_args=SimpleNamespace(response_plane=plane)
        ),
        engine=SimpleNamespace(tokenizer_manager=object()),
        generate_endpoint=Mock(),
    )
    shutdown = SimpleNamespace(wait_for_kv_transfers=None)
    wait = AsyncMock()
    monkeypatch.setattr(
        drain_scheduler.SchedulerIdleProbe, "wait_for_terminal_idle", wait
    )
    assert (
        drain_scheduler.configure_prefill_shutdown(
            shutdown,
            handler,
            integrated=integrated,
            supports_idle_heartbeat=lambda: capable,
        )
        is enabled
    )
    if enabled:
        await shutdown.wait_for_kv_transfers()
        wait.assert_awaited_once_with(handler.generate_endpoint)
    else:
        assert shutdown.wait_for_kv_transfers is None
