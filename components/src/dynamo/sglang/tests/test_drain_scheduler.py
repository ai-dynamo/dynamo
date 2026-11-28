# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from types import SimpleNamespace
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
