# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for dynamo.vllm.multimodal_utils.threaded_micro_batcher.

Pin the execution contract: on_start + every fn call (+ on_stop) run on one
dedicated thread (so CUDA-graph capture/replay share a thread), concurrent
submits coalesce into cost-bounded same-bucket batches, the graph ladder rounds a
batch's cost up to a rung (target_bucket), eager-drain pulls all queued work when
free, errors reach every awaiting caller, and the shutdown lifecycle behaves.

``fn`` is ``fn(items, target_bucket)``; ``cost`` is a precomputed scalar that rides
on ``submit(items, costs)`` (one-dimensional packing — no bucket_key).
"""

import asyncio
import threading

import pytest

from dynamo.vllm.multimodal_utils.threaded_micro_batcher import (
    BatcherOverloaded,
    ThreadedMicroBatcher,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.multimodal,
]


def _echo(items, target_bucket=None):
    return list(items)


class _Recorder:
    """fn that records the threads it ran on and the batches / target_buckets it saw."""

    def __init__(self):
        self.threads: list[int] = []
        self.batches: list[list] = []
        self.target_buckets: list = []
        self.start_thread: int | None = None
        self.stop_thread: int | None = None

    def on_start(self):
        self.start_thread = threading.get_ident()

    def on_stop(self):
        self.stop_thread = threading.get_ident()

    def fn(self, items, target_bucket=None):
        self.threads.append(threading.get_ident())
        self.batches.append(list(items))
        self.target_buckets.append(target_bucket)
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


async def test_on_start_fn_and_on_stop_share_one_non_main_thread():
    rec = _Recorder()
    b = ThreadedMicroBatcher(rec.fn, on_start=rec.on_start, on_stop=rec.on_stop)
    b.start()
    await asyncio.gather(b.submit(["x"]), b.submit(["y"]))
    b.shutdown()
    assert rec.start_thread is not None
    assert rec.start_thread != threading.get_ident()
    assert set(rec.threads) == {rec.start_thread}
    # on_stop runs on the same actor thread (so CUDA teardown is same-thread).
    assert rec.stop_thread == rec.start_thread


def test_on_stop_not_run_if_on_start_failed():
    ran = {"stop": False}

    def bad_start():
        raise RuntimeError("start failed")

    def on_stop():
        ran["stop"] = True

    b = ThreadedMicroBatcher(_echo, on_start=bad_start, on_stop=on_stop)
    with pytest.raises(RuntimeError, match="start failed"):
        b.start()
    assert ran["stop"] is False


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


async def test_eager_drain_pulls_all_queued_when_free():
    """Default (max_wait_ms=0) eager-drain: items queued while the worker is busy
    are all pulled into ONE batch on the next free iteration (no timer)."""
    entered = threading.Event()
    release = threading.Event()
    batches: list[list] = []

    def fn(items, target_bucket=None):
        batches.append(list(items))
        if "block" in items:
            entered.set()
            release.wait(timeout=5.0)
        return [("r", x) for x in items]

    b = ThreadedMicroBatcher(fn)  # default: eager-drain
    b.start()
    first = asyncio.ensure_future(b.submit(["block"]))
    for _ in range(200):
        if entered.is_set():
            break
        await asyncio.sleep(0.01)
    assert entered.is_set()
    # Queue three while the worker is blocked in fn("block").
    rest = [asyncio.ensure_future(b.submit([x])) for x in ("a", "b", "c")]
    await asyncio.sleep(0.05)
    release.set()
    await asyncio.gather(first, *rest)
    b.shutdown()
    # The post-release _collect drains a, b, c in one batch.
    assert ["a", "b", "c"] in batches


async def test_cost_budget_caps_each_batch():
    """costs ride on submit; with budget 5, batches never exceed summed cost 5."""
    rec = _Recorder()
    b = ThreadedMicroBatcher(rec.fn, max_wait_ms=200.0, max_batch_cost=5)
    b.start()
    try:
        await b.submit([3, 3, 1], costs=[3, 3, 1])  # 3 | 3,1 → two batches
        assert all(sum(batch) <= 5 for batch in rec.batches)
        assert sum(len(batch) for batch in rec.batches) == 3
    finally:
        b.shutdown()


async def test_target_bucket_rounds_packed_cost_up_to_nearest_rung():
    """Graph mode: the batcher rounds a batch's sum(cost) up to the nearest rung
    and passes it as target_bucket."""
    rec = _Recorder()
    b = ThreadedMicroBatcher(
        rec.fn, max_batch_cost=8, buckets=[2, 4, 8], max_wait_ms=200.0
    )
    b.start()
    try:
        # One coalesced batch of cost 3 → rounds up to rung 4.
        await b.submit(["x", "y", "z"], costs=[1, 1, 1])
        assert rec.batches == [["x", "y", "z"]]
        assert rec.target_buckets == [4]
    finally:
        b.shutdown()


async def test_eager_mode_passes_none_target_bucket():
    rec = _Recorder()
    b = ThreadedMicroBatcher(rec.fn)  # buckets=None ⇒ eager
    b.start()
    try:
        await b.submit(["a"])
        assert rec.target_buckets == [None]
    finally:
        b.shutdown()


def test_buckets_below_max_batch_cost_rejected():
    with pytest.raises(ValueError, match="ladder must cover"):
        ThreadedMicroBatcher(_echo, max_batch_cost=8, buckets=[2, 4])


def test_buckets_derive_max_batch_cost_when_none():
    """Graph mode with no explicit budget derives the ceiling from the ladder."""
    b = ThreadedMicroBatcher(_echo, buckets=[2, 4, 8])  # max_batch_cost=None
    assert b._max_batch_cost == 8


async def test_max_batch_cost_none_is_passthrough():
    """Default (max_batch_cost=None): no cap and no per-item ceiling — the whole
    drained same-bucket group runs as ONE fn call regardless of summed cost."""
    rec = _Recorder()
    b = ThreadedMicroBatcher(rec.fn, max_wait_ms=200.0)  # max_batch_cost=None
    b.start()
    try:
        # Big per-item costs that would be split (or rejected) under any finite cap.
        out = await b.submit([1, 2, 3, 4], costs=[1000, 1000, 1000, 1000])
        assert len(out) == 4
        assert rec.batches == [[1, 2, 3, 4]]  # one un-split batch
        assert rec.target_buckets == [None]  # eager (no ladder)
    finally:
        b.shutdown()


async def test_error_reaches_every_caller():
    def boom(items, target_bucket=None):
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
    b = ThreadedMicroBatcher(lambda items, target_bucket=None: [], max_wait_ms=10.0)
    b.start()
    try:
        with pytest.raises(RuntimeError, match="one result per item"):
            await b.submit(["a", "b"])
    finally:
        b.shutdown()


async def test_costs_length_mismatch_raises():
    b = ThreadedMicroBatcher(_echo)
    b.start()
    try:
        with pytest.raises(ValueError, match="costs has"):
            await b.submit(["a", "b"], costs=[1])
    finally:
        b.shutdown()


async def test_submit_before_start_raises():
    b = ThreadedMicroBatcher(_echo)
    with pytest.raises(RuntimeError, match="before start"):
        await b.submit(["a"])


async def test_submit_after_shutdown_raises():
    b = ThreadedMicroBatcher(_echo)
    b.start()
    b.shutdown()
    with pytest.raises(RuntimeError, match="after shutdown"):
        await b.submit(["a"])


def test_start_error_propagates_and_reaps():
    def bad_start():
        raise RuntimeError("start failed")

    b = ThreadedMicroBatcher(_echo, on_start=bad_start)
    with pytest.raises(RuntimeError, match="start failed"):
        b.start()
    assert not b._thread.is_alive()


async def test_shutdown_fails_queued_items():
    entered = threading.Event()
    release = threading.Event()

    def blocking(items, target_bucket=None):
        entered.set()
        release.wait(timeout=5.0)
        return [("r", x) for x in items]

    b = ThreadedMicroBatcher(blocking, join_timeout_s=0.2)
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
    b = ThreadedMicroBatcher(_echo)
    b.start()
    assert b._thread.is_alive()
    b.shutdown()
    assert not b._thread.is_alive()


async def test_cancelled_submit_is_retired_and_releases_admission():
    """Cancelling an await retires the request and frees its admission cost."""
    entered = threading.Event()
    release = threading.Event()

    def blocking(items, target_bucket=None):
        entered.set()
        release.wait(timeout=5.0)
        return [("r", x) for x in items]

    b = ThreadedMicroBatcher(blocking, max_outstanding_cost=10)
    b.start()
    try:
        first = asyncio.ensure_future(b.submit(["a"]))  # occupies the worker
        for _ in range(200):
            if entered.is_set():
                break
            await asyncio.sleep(0.01)
        assert entered.is_set()

        second = asyncio.ensure_future(b.submit(["b"]))  # queued behind first
        await asyncio.sleep(0.05)
        assert b._outstanding == 2  # both admitted (cost 1 each)

        second.cancel()
        release.set()  # let the worker finish "a", then collect+retire cancelled "b"
        with pytest.raises(asyncio.CancelledError):
            await second
        assert len(await first) == 1
        assert b._outstanding == 0  # cancelled request's admission released
    finally:
        b.shutdown()


async def test_max_outstanding_cost_rejects_when_full():
    """submit() raises BatcherOverloaded once accepted-but-incomplete cost is full."""
    entered = threading.Event()
    release = threading.Event()

    def blocking(items, target_bucket=None):
        entered.set()
        release.wait(timeout=5.0)
        return [("r", x) for x in items]

    b = ThreadedMicroBatcher(blocking, max_outstanding_cost=1)
    b.start()
    try:
        first = asyncio.ensure_future(b.submit(["a"]))  # cost 1 fills the budget
        for _ in range(200):
            if entered.is_set():
                break
            await asyncio.sleep(0.01)
        assert entered.is_set()

        with pytest.raises(BatcherOverloaded):
            await b.submit(["b"])

        release.set()
        assert len(await first) == 1
    finally:
        b.shutdown()


async def test_worker_supervisor_fails_awaiters_on_crash():
    """An unexpected worker crash fails live awaiters and moves to a failed state
    (later submits raise) instead of hanging."""
    b = ThreadedMicroBatcher(_echo)
    b.start()

    def explode(_works):
        raise RuntimeError("worker boom")

    b._dispatch = explode  # force a crash inside the serve loop
    try:
        with pytest.raises(RuntimeError, match="worker boom"):
            await b.submit(["a"])
        with pytest.raises(RuntimeError):  # FAILED state rejects new work
            await b.submit(["b"])
    finally:
        b.shutdown()


async def test_oversized_cost_is_rejected():
    """A per-item cost above the batch budget has no batch it can fit → rejected."""
    b = ThreadedMicroBatcher(_echo, max_batch_cost=5)
    b.start()
    try:
        with pytest.raises(ValueError, match="exceeds max_batch_cost"):
            await b.submit([6], costs=[6])
    finally:
        b.shutdown()


async def test_nonpositive_cost_is_rejected():
    b = ThreadedMicroBatcher(_echo)
    b.start()
    try:
        with pytest.raises(ValueError, match="positive int"):
            await b.submit([1], costs=[0])
    finally:
        b.shutdown()


async def test_partial_batch_failure_fails_request_once_and_releases():
    """A multi-item request split across batches where the FIRST batch raises
    fails the whole request exactly once, releases all of its admission, and the
    later sibling item is tombstoned — it never reaches fn."""
    seen: list = []

    def fn(items, target_bucket=None):
        seen.extend(items)
        if "bad" in items:
            raise ValueError("boom")
        return [("r", x) for x in items]

    # max_batch_cost=1 → "bad" and "good" are separate (cost-1) batches; "bad"
    # runs (and fails) first, tombstoning the request before "good" runs.
    b = ThreadedMicroBatcher(
        fn, max_wait_ms=200.0, max_batch_cost=1, max_outstanding_cost=10
    )
    b.start()
    try:
        with pytest.raises(ValueError, match="boom"):
            await b.submit(["bad", "good"], costs=[1, 1])
        assert b._outstanding == 0
        assert "good" not in seen  # tombstoned sibling never reached fn
    finally:
        b.shutdown()


async def test_no_fn_after_shutdown_for_collected_items():
    """Items pulled off the queue by _collect but not yet run must not reach fn
    once shutdown begins; they fail with the shutdown error."""
    entered = threading.Event()
    release = threading.Event()
    seen: list = []

    def fn(items, target_bucket=None):
        seen.extend(items)
        if "a" in items:
            entered.set()
            release.wait(timeout=5.0)
        return [("r", x) for x in items]

    # max_batch_cost=1 → one batch per item; all three collected together, the
    # "a" batch blocks in fn while shutdown() is called.
    b = ThreadedMicroBatcher(fn, max_wait_ms=50.0, max_batch_cost=1)
    b.start()
    task = asyncio.ensure_future(b.submit(["a", "b", "c"], costs=[1, 1, 1]))
    for _ in range(200):
        if entered.is_set():
            break
        await asyncio.sleep(0.01)
    assert entered.is_set()
    b.shutdown()  # b, c are collected but not yet run
    release.set()
    with pytest.raises(RuntimeError, match="shut down"):
        await task
    assert seen == ["a"]  # b and c never reached fn
    b.shutdown()


def test_double_start_raises():
    b = ThreadedMicroBatcher(_echo)
    b.start()
    try:
        with pytest.raises(RuntimeError, match="twice"):
            b.start()
    finally:
        b.shutdown()


def test_worker_thread_is_not_daemon():
    b = ThreadedMicroBatcher(_echo)
    b.start()
    try:
        assert b._thread.daemon is False
    finally:
        b.shutdown()
