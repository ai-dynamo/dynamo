# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for dynamo.vllm.multimodal_utils.threaded_micro_batcher.

Pin the execution contract: on_start + every fn call (+ on_stop) run on one
dedicated thread (so CUDA-graph capture/replay share a thread), concurrent submits
coalesce into cost-bounded batches up to max_batch_cost (or pass-through when
None), eager-drain pulls all queued work when free, errors reach every awaiting
caller, and the shutdown lifecycle behaves.

``fn`` is ``fn(items)``; ``cost`` is a precomputed scalar that rides on
``submit(items, costs)`` (one-dimensional packing — no bucket_key, no ladder).
"""

import asyncio
import threading
from typing import Callable

import pytest

from dynamo.vllm.multimodal_utils.threaded_micro_batcher import ThreadedMicroBatcher

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.multimodal,
    pytest.mark.timeout(30),
]


def _echo(items):
    return list(items)


class _Recorder:
    """fn that records the threads it ran on and the batches it received."""

    def __init__(self):
        self.threads: list[int] = []
        self.start_thread: int | None = None
        self.stop_thread: int | None = None

    def on_start(self):
        self.start_thread = threading.get_ident()

    def on_stop(self):
        self.stop_thread = threading.get_ident()

    def fn(self, items):
        self.threads.append(threading.get_ident())
        return [("r", x) for x in items]


async def _wait_until(
    predicate: Callable[[], bool],
    message: str,
    timeout_s: float = 5.0,
) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s
    while not predicate():
        if loop.time() >= deadline:
            raise AssertionError(message)
        await asyncio.sleep(0.01)


class _Gate:
    """Park the worker inside a sentinel "gate" batch until released, so a test can
    enqueue several submits that then drain together as ONE batch — deterministic
    coalescing without relying on a timed hold."""

    GATE = "__gate__"

    def __init__(self):
        self.entered = threading.Event()
        self.release = threading.Event()
        self.batches: list[list] = []  # excludes the gate batch

    def fn(self, items):
        if list(items) == [self.GATE]:
            self.entered.set()
            self.release.wait(timeout=5.0)
            return [("r", x) for x in items]
        self.batches.append(list(items))
        return [("r", x) for x in items]

    async def park(self, b):
        """Submit the gate item; return its future once the worker is parked in fn."""
        fut = asyncio.ensure_future(b.submit([self.GATE]))
        await _wait_until(self.entered.is_set, "worker never entered the gate batch")
        return fut


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


def test_on_stop_runs_if_on_start_partially_failed():
    ran = {"stop": False}

    def bad_start():
        raise RuntimeError("start failed")

    def on_stop():
        ran["stop"] = True

    b = ThreadedMicroBatcher(_echo, on_start=bad_start, on_stop=on_stop)
    with pytest.raises(RuntimeError, match="start failed"):
        b.start()
    assert ran["stop"] is True


async def test_eager_drain_pulls_all_queued_when_free():
    """Eager-drain: items queued while the worker is busy are all pulled into ONE
    batch on the next free iteration (no timer)."""
    entered = threading.Event()
    release = threading.Event()
    batches: list[list] = []

    def fn(items):
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
    """costs ride on submit; with budget 5, batches never exceed summed cost 5.

    Park the worker so [3, 3, 1] drain in one collect, deterministically
    exercising the cost-split (3 | 3,1)."""
    g = _Gate()
    b = ThreadedMicroBatcher(g.fn, max_batch_cost=5)
    b.start()
    try:
        gate = await g.park(b)
        real = asyncio.ensure_future(b.submit([3, 3, 1], costs=[3, 3, 1]))
        await asyncio.sleep(0.05)  # let all three enqueue while the worker is parked
        g.release.set()
        await asyncio.gather(gate, real)
        assert all(sum(batch) <= 5 for batch in g.batches)
        assert sum(len(batch) for batch in g.batches) == 3
        assert [3] in g.batches and [3, 1] in g.batches  # split actually happened
    finally:
        b.shutdown()


async def test_max_batch_cost_none_is_passthrough():
    """Default (max_batch_cost=None): no cap and no per-item ceiling — the whole
    drained set runs as ONE fn call regardless of summed cost."""
    g = _Gate()
    b = ThreadedMicroBatcher(g.fn)  # max_batch_cost=None
    b.start()
    try:
        gate = await g.park(b)
        # Big per-item costs that would be split (or rejected) under any finite cap.
        real = asyncio.ensure_future(
            b.submit([1, 2, 3, 4], costs=[1000, 1000, 1000, 1000])
        )
        await asyncio.sleep(0.05)  # let all four enqueue while the worker is parked
        g.release.set()
        out, _ = await asyncio.gather(real, gate)
        assert len(out) == 4
        assert g.batches == [[1, 2, 3, 4]]  # one un-split batch
    finally:
        b.shutdown()


async def test_error_reaches_every_caller():
    def boom(items):
        raise ValueError("boom")

    b = ThreadedMicroBatcher(boom)
    b.start()
    try:
        results = await asyncio.gather(
            *(b.submit(["u"]) for _ in range(3)), return_exceptions=True
        )
        assert all(isinstance(r, ValueError) and str(r) == "boom" for r in results)
    finally:
        b.shutdown()


async def test_cancelled_queued_request_cannot_poison_live_request():
    """A cancelled request is dropped before dispatch, so its bad item cannot
    fail an unrelated request that was queued alongside it."""
    gate = _Gate()

    def fn(items):
        results = gate.fn(items)
        if any(item.startswith("bad") for item in items):
            raise ValueError("bad poisoned batch")
        return results

    b = ThreadedMicroBatcher(fn)
    b.start()
    try:
        parked = await gate.park(b)
        bad = asyncio.ensure_future(b.submit(["bad-1", "bad-2"]))
        good = asyncio.ensure_future(b.submit(["good"]))
        await _wait_until(
            lambda: b._queue.qsize() == 3,
            "requests were not queued behind the gate",
        )

        bad.cancel()
        with pytest.raises(asyncio.CancelledError):
            await bad

        gate.release.set()
        await parked
        assert await good == [("r", "good")]
        assert gate.batches == [["good"]]
        assert not b._live
    finally:
        gate.release.set()
        b.shutdown()


async def test_cancellation_after_partial_dispatch_drops_later_siblings():
    """Cancellation after one item completes lets the claimed item finish, drops
    later siblings, releases the request, and leaves the actor usable."""
    blocked = threading.Event()
    release = threading.Event()
    seen: list[str] = []

    def fn(items):
        seen.extend(items)
        if items == ["blocked"]:
            blocked.set()
            release.wait(timeout=5.0)
        return [("r", item) for item in items]

    b = ThreadedMicroBatcher(fn, max_batch_cost=1)
    b.start()
    try:
        task = asyncio.ensure_future(b.submit(["first", "blocked", "later"]))
        await _wait_until(blocked.is_set, "second item never entered fn")

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        release.set()
        assert await b.submit(["probe"]) == [("r", "probe")]
        assert seen == ["first", "blocked", "probe"]
        assert not b._live
    finally:
        release.set()
        b.shutdown()


async def test_cancellation_after_batch_claim_is_best_effort():
    """Once a shared batch is claimed, cancellation cannot remove its bad item;
    the live peer may fail, but the actor remains usable afterward."""
    park_entered = threading.Event()
    park_release = threading.Event()
    batch_claimed = threading.Event()
    batch_release = threading.Event()
    seen: list[list[str]] = []

    def fn(items):
        if items == ["park"]:
            park_entered.set()
            park_release.wait(timeout=5.0)
            return [("r", "park")]
        seen.append(list(items))
        if "bad" in items:
            batch_claimed.set()
            batch_release.wait(timeout=5.0)
            raise ValueError("claimed bad poisoned batch")
        return [("r", item) for item in items]

    b = ThreadedMicroBatcher(fn)
    b.start()
    try:
        parked = asyncio.ensure_future(b.submit(["park"]))
        await _wait_until(park_entered.is_set, "worker never entered park batch")
        bad = asyncio.ensure_future(b.submit(["bad"]))
        good = asyncio.ensure_future(b.submit(["good"]))
        await _wait_until(
            lambda: b._queue.qsize() == 2,
            "shared batch was not queued behind park",
        )

        park_release.set()
        await parked
        await _wait_until(batch_claimed.is_set, "shared batch was not claimed")
        bad.cancel()
        with pytest.raises(asyncio.CancelledError):
            await bad

        batch_release.set()
        with pytest.raises(ValueError, match="claimed bad poisoned batch"):
            await good
        assert seen == [["bad", "good"]]
        assert await b.submit(["probe"]) == [("r", "probe")]
        assert b._thread.is_alive()
        assert not b._live
    finally:
        park_release.set()
        batch_release.set()
        b.shutdown()


async def test_cancellation_racing_shutdown_cleans_request():
    """Shutdown consumes every item of a canceled queued request without a hang,
    double finalization, or live-request leak."""
    entered = threading.Event()
    release = threading.Event()

    def fn(items):
        if items == ["block"]:
            entered.set()
            release.wait(timeout=5.0)
        return [("r", item) for item in items]

    b = ThreadedMicroBatcher(fn, join_timeout_s=0.1)
    b.start()
    try:
        in_flight = asyncio.ensure_future(b.submit(["block"]))
        await _wait_until(entered.is_set, "blocker never entered fn")
        queued = asyncio.ensure_future(b.submit(["cancel-1", "cancel-2"]))
        await _wait_until(
            lambda: b._queue.qsize() == 2,
            "canceled request was not queued behind blocker",
        )

        queued.cancel()
        with pytest.raises(asyncio.CancelledError):
            await queued
        b.shutdown()

        release.set()
        assert await in_flight == [("r", "block")]
        b.shutdown()
        assert not b._thread.is_alive()
        assert not b._live
    finally:
        release.set()
        b.shutdown()


async def test_cancellation_after_finalization_keeps_actor_usable():
    """Cancellation after done=True but before future settlement leaves the actor
    alive when completion observes the already-canceled bridged future."""
    completion_entered = threading.Event()
    completion_release = threading.Event()
    first_completion = True

    b = ThreadedMicroBatcher(_echo)
    original_complete = b._complete

    def gated_complete(request):
        nonlocal first_completion
        if first_completion:
            first_completion = False
            completion_entered.set()
            completion_release.wait(timeout=5.0)
        original_complete(request)

    b._complete = gated_complete
    b.start()
    try:
        task = asyncio.ensure_future(b.submit(["done"]))
        await _wait_until(
            completion_entered.is_set,
            "request did not reach finalization gate",
        )
        assert not b._live

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        completion_release.set()
        assert await b.submit(["probe"]) == ["probe"]
        assert b._thread.is_alive()
        assert not b._live
    finally:
        completion_release.set()
        b.shutdown()


async def test_wrong_result_count_raises():
    b = ThreadedMicroBatcher(lambda items: [])
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

    def blocking(items):
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


async def test_partial_batch_failure_fails_request_once():
    """A multi-item request split across batches where the FIRST batch raises
    fails the whole request exactly once, and the later sibling item is
    tombstoned — it never reaches fn."""
    seen: list = []

    def fn(items):
        seen.extend(items)
        if "bad" in items:
            raise ValueError("boom")
        return [("r", x) for x in items]

    # max_batch_cost=1 → "bad" and "good" are separate (cost-1) batches; "bad"
    # runs (and fails) first, tombstoning the request before "good" runs.
    b = ThreadedMicroBatcher(fn, max_batch_cost=1)
    b.start()
    try:
        with pytest.raises(ValueError, match="boom"):
            await b.submit(["bad", "good"], costs=[1, 1])
        assert "good" not in seen  # tombstoned sibling never reached fn
    finally:
        b.shutdown()


async def test_no_fn_after_shutdown_for_collected_items():
    """Items pulled off the queue by _collect but not yet run must not reach fn
    once shutdown begins; they fail with the shutdown error."""
    entered = threading.Event()
    release = threading.Event()
    seen: list = []

    def fn(items):
        seen.extend(items)
        if "a" in items:
            entered.set()
            release.wait(timeout=5.0)
        return [("r", x) for x in items]

    # max_batch_cost=1 → one batch per item; all three collected together, the
    # "a" batch blocks in fn while shutdown() is called.
    b = ThreadedMicroBatcher(fn, max_batch_cost=1)
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


def test_concurrent_start_starts_one_worker():
    """Two threads race start(); the winner parks inside on_start while the loser
    must already be rejected — one worker only.

    Deterministic regression guard for the double-start race: the winner blocks in
    on_start, so `_state` is still NEW when the loser races in. A state-only guard
    would let the loser through *here* (state == NEW), spawn a second worker, and
    run on_start twice; the fix also keys on `_thread` (published under the lock),
    so the loser is rejected while the winner is still parked. Blocking on_start
    forces the window every run (no reliance on scheduler timing).
    """
    entered = threading.Event()
    release = threading.Event()
    ran = {"n": 0}
    ran_lock = threading.Lock()

    def on_start():
        with ran_lock:
            ran["n"] += 1
        entered.set()
        release.wait(timeout=5.0)

    b = ThreadedMicroBatcher(_echo, on_start=on_start)
    outcomes: list[str] = []
    out_lock = threading.Lock()
    outcome_recorded = threading.Event()
    gate = threading.Barrier(2)

    def do_start():
        gate.wait()  # both threads reach start() together
        try:
            b.start()
            tag = "ok"
        except RuntimeError:
            tag = "rejected"
        with out_lock:
            outcomes.append(tag)
            outcome_recorded.set()

    t1 = threading.Thread(target=do_start)
    t2 = threading.Thread(target=do_start)
    t1.start()
    t2.start()
    try:
        assert entered.wait(timeout=5.0), "winner never reached on_start"
        # Winner is parked in on_start (state still NEW, _thread published). The
        # loser must already be rejected — the window a state-only guard misses.
        assert outcome_recorded.wait(timeout=5.0), "loser did not finish start()"
        assert outcomes == [
            "rejected"
        ], f"loser not rejected while winner parked: {outcomes}"
    finally:
        release.set()
        t1.join()
        t2.join()
    assert sorted(outcomes) == ["ok", "rejected"], outcomes
    assert ran["n"] == 1  # one worker only — on_start never ran twice
    b.shutdown()


def test_worker_thread_is_not_daemon():
    b = ThreadedMicroBatcher(_echo)
    b.start()
    try:
        assert b._thread.daemon is False
    finally:
        b.shutdown()
