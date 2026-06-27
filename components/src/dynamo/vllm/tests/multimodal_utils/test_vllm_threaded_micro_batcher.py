# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for dynamo.vllm.multimodal_utils.threaded_micro_batcher.

Pin the execution contract: on_start + every fn call run on one dedicated thread
(so CUDA-graph capture/replay share a thread), concurrent submits coalesce into
cost-bounded same-bucket batches, errors reach every awaiting caller, and the
shutdown lifecycle (fail queued / defer reap / stop) behaves.
"""

import asyncio
import threading

import pytest

from dynamo.vllm.multimodal_utils.threaded_micro_batcher import ThreadedMicroBatcher

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.multimodal,
]


class _Recorder:
    """fn that records the threads it ran on and the batches it received."""

    def __init__(self):
        self.threads: list[int] = []
        self.batches: list[list] = []
        self.start_thread: int | None = None

    def on_start(self):
        self.start_thread = threading.get_ident()

    def fn(self, items):
        self.threads.append(threading.get_ident())
        self.batches.append(list(items))
        return [("r", x) for x in items]


async def test_submit_returns_one_result_per_item():
    rec = _Recorder()
    b = ThreadedMicroBatcher(rec.fn, on_start=rec.on_start)
    b.start()
    try:
        out = await b.submit(["a", "b", "c"])
        assert out == [("r", "a"), ("r", "b"), ("r", "c")]
    finally:
        b.shutdown()


async def test_on_start_and_fn_share_one_non_main_thread():
    rec = _Recorder()
    b = ThreadedMicroBatcher(rec.fn, on_start=rec.on_start)
    b.start()
    try:
        await asyncio.gather(b.submit(["x"]), b.submit(["y"]))
        assert rec.start_thread is not None
        assert rec.start_thread != threading.get_ident()
        assert set(rec.threads) == {rec.start_thread}
    finally:
        b.shutdown()


async def test_concurrent_submits_coalesce():
    rec = _Recorder()
    b = ThreadedMicroBatcher(rec.fn, max_wait_ms=200.0)
    b.start()
    try:
        results = await asyncio.gather(*(b.submit(["u"]) for _ in range(5)))
        assert all(len(r) == 1 for r in results)
        assert sum(len(batch) for batch in rec.batches) == 5
        assert max(len(batch) for batch in rec.batches) >= 2  # coalesced
        assert len(rec.batches) < 5
    finally:
        b.shutdown()


async def test_cost_budget_caps_each_batch():
    """cost(item)=item; with budget 5, batches never exceed summed cost 5."""
    rec = _Recorder()
    b = ThreadedMicroBatcher(
        rec.fn, max_wait_ms=200.0, cost=lambda i: i, max_batch_cost=5
    )
    b.start()
    try:
        await b.submit([3, 3, 1])  # 3 | 3,1  → two batches within budget
        assert all(sum(batch) <= 5 for batch in rec.batches)
        assert sum(len(batch) for batch in rec.batches) == 3
    finally:
        b.shutdown()


async def test_bucket_key_isolates_batches():
    """Items with different bucket_key never share a batch."""
    rec = _Recorder()
    b = ThreadedMicroBatcher(rec.fn, max_wait_ms=200.0, bucket_key=lambda i: i % 2)
    b.start()
    try:
        await asyncio.gather(*(b.submit([i]) for i in range(6)))
        for batch in rec.batches:
            assert len({i % 2 for i in batch}) == 1  # one bucket per batch
    finally:
        b.shutdown()


async def test_error_reaches_every_caller():
    def boom(items):
        raise ValueError("boom")

    b = ThreadedMicroBatcher(boom, max_wait_ms=50.0)
    b.start()
    try:
        results = await asyncio.gather(
            *(b.submit(["u"]) for _ in range(3)), return_exceptions=True
        )
        assert all(isinstance(r, ValueError) and str(r) == "boom" for r in results)
    finally:
        b.shutdown()


async def test_wrong_result_count_raises():
    b = ThreadedMicroBatcher(lambda items: [], max_wait_ms=10.0)
    b.start()
    try:
        with pytest.raises(RuntimeError, match="one result per item"):
            await b.submit(["a", "b"])
    finally:
        b.shutdown()


async def test_submit_before_start_raises():
    b = ThreadedMicroBatcher(lambda items: items)
    with pytest.raises(RuntimeError, match="before start"):
        await b.submit(["a"])


async def test_submit_after_shutdown_raises():
    b = ThreadedMicroBatcher(lambda items: items)
    b.start()
    b.shutdown()
    with pytest.raises(RuntimeError, match="after shutdown"):
        await b.submit(["a"])


def test_start_error_propagates_and_reaps():
    def bad_start():
        raise RuntimeError("start failed")

    b = ThreadedMicroBatcher(lambda items: items, on_start=bad_start)
    with pytest.raises(RuntimeError, match="start failed"):
        b.start()
    assert not b._thread.is_alive()


async def test_shutdown_fails_queued_items():
    entered = threading.Event()
    release = threading.Event()

    def blocking(items):
        entered.set()
        release.wait(timeout=5.0)
        return [("r", x) for x in items]

    b = ThreadedMicroBatcher(blocking, max_wait_ms=0.0, join_timeout_s=0.2)
    b.start()
    in_flight = asyncio.ensure_future(b.submit(["a"]))
    for _ in range(200):
        if entered.is_set():
            break
        await asyncio.sleep(0.01)
    assert entered.is_set()

    queued = [
        asyncio.ensure_future(b.submit(["b"])),
        asyncio.ensure_future(b.submit(["c"])),
    ]
    await asyncio.sleep(0.05)
    b.shutdown()  # fails b, c; a is in flight
    for q in queued:
        with pytest.raises(RuntimeError, match="shut down"):
            await q
    release.set()
    assert len(await in_flight) == 1
    b.shutdown()
    assert not b._thread.is_alive()


def test_shutdown_stops_thread():
    b = ThreadedMicroBatcher(lambda items: items)
    b.start()
    assert b._thread.is_alive()
    b.shutdown()
    assert not b._thread.is_alive()
