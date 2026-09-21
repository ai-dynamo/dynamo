# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real, runnable tests for the transport-agnostic schema/backpressure logic
in sweeper_event_plane.py. Uses an in-memory fake transport -- no compiled
dynamo._core binding required, since this layer is deliberately transport-
agnostic (see module docstring)."""

from __future__ import annotations

import json
import threading
import time

import pytest

from dynamo.profiler.v2.sweeper_event_plane import (
    SUBJECT_SUFFIX,
    SweeperEventPublisher,
)


class FakeEventEmitter:
    """Records every (subject, payload) publish call, in order. Thread-safe."""

    def __init__(self, *, fail_on_publish: bool = False) -> None:
        self.published: list[tuple[str, bytes]] = []
        self.closed = False
        self._lock = threading.Lock()
        self._fail_on_publish = fail_on_publish

    def publish(self, subject: str, payload: bytes) -> None:
        if self._fail_on_publish:
            raise RuntimeError("simulated transport failure")
        with self._lock:
            self.published.append((subject, payload))

    def close(self) -> None:
        self.closed = True


def _wait_until(predicate, timeout=2.0, interval=0.01):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


def test_emit_does_not_block_caller():
    emitter = FakeEventEmitter()
    pub = SweeperEventPublisher("run-1", emitter)
    # No start() called -- nothing draining the queue -- emit must still
    # return immediately rather than block.
    start = time.monotonic()
    pub.emit("round.completed", {"round_no": 1, "cumulative_candidates": 1})
    assert time.monotonic() - start < 0.5


def test_events_delivered_in_order_with_correct_subjects():
    emitter = FakeEventEmitter()
    with SweeperEventPublisher("run-2", emitter) as pub:
        pub.emit("search.resolved", {"outcome": "materialized"})
        pub.emit("round.completed", {"round_no": 1, "cumulative_candidates": 1})
        pub.emit("run.completed", {"outcome": "succeeded"})
        assert _wait_until(lambda: len(emitter.published) == 3)

    subjects = [s for s, _ in emitter.published]
    assert subjects == [
        f"sweeper.run-2.{SUBJECT_SUFFIX['search.resolved']}",
        f"sweeper.run-2.{SUBJECT_SUFFIX['round.completed']}",
        f"sweeper.run-2.{SUBJECT_SUFFIX['run.completed']}",
    ]
    sequences = [json.loads(p)["sequence"] for _, p in emitter.published]
    assert sequences == [1, 2, 3]


def test_envelope_carries_run_uid_and_api_version():
    emitter = FakeEventEmitter()
    with SweeperEventPublisher("run-3", emitter) as pub:
        pub.emit("run.completed", {"outcome": "succeeded"})
        assert _wait_until(lambda: len(emitter.published) == 1)

    envelope = json.loads(emitter.published[0][1])
    assert envelope["runUID"] == "run-3"
    assert envelope["apiVersion"] == "sweeper.dynamo.nvidia.com/v1alpha1"
    assert envelope["type"] == "run.completed"
    assert envelope["data"] == {"outcome": "succeeded"}
    assert "timestamp" in envelope


def test_close_flushes_pending_queue_before_stopping():
    emitter = FakeEventEmitter()
    pub = SweeperEventPublisher("run-4", emitter)
    pub.start()
    for i in range(5):
        pub.emit("round.completed", {"round_no": i, "cumulative_candidates": i})
    pub.close()
    assert len(emitter.published) == 5
    assert emitter.closed is True


def test_pending_queue_drops_oldest_when_full_without_blocking():
    emitter = FakeEventEmitter()
    # Tiny queue, never started -- nothing drains it, so every emit past
    # capacity must drop the oldest rather than block or raise.
    pub = SweeperEventPublisher("run-5", emitter, pending_queue_size=2)
    for i in range(10):
        pub.emit("round.completed", {"round_no": i, "cumulative_candidates": i})
    assert pub._queue.qsize() == 2
    remaining = [item.data["round_no"] for item in list(pub._queue.queue)]
    assert remaining == [8, 9]


def test_non_serializable_data_is_dropped_without_killing_the_drain_loop():
    emitter = FakeEventEmitter()
    with SweeperEventPublisher("run-6", emitter) as pub:
        pub.emit("round.completed", {"round_no": 1, "bad": object()})
        pub.emit("round.completed", {"round_no": 2, "cumulative_candidates": 2})
        assert _wait_until(lambda: len(emitter.published) == 1)

    envelope = json.loads(emitter.published[0][1])
    assert envelope["data"]["round_no"] == 2


def test_unknown_event_type_raises_immediately():
    emitter = FakeEventEmitter()
    pub = SweeperEventPublisher("run-7", emitter)
    with pytest.raises(ValueError):
        pub.emit("not.a.real.type", {})


def test_publish_failure_is_logged_and_does_not_kill_the_drain_loop():
    emitter = FakeEventEmitter(fail_on_publish=True)
    with SweeperEventPublisher("run-8", emitter) as pub:
        pub.emit("round.completed", {"round_no": 1, "cumulative_candidates": 1})
        # give the drain loop a chance to run and fail without crashing
        time.sleep(0.2)
    assert emitter.published == []
    assert emitter.closed is True


def test_multiple_runs_get_independent_subjects():
    emitter_a = FakeEventEmitter()
    emitter_b = FakeEventEmitter()
    with SweeperEventPublisher("run-a", emitter_a) as pub_a, SweeperEventPublisher(
        "run-b", emitter_b
    ) as pub_b:
        pub_a.emit("run.completed", {"outcome": "succeeded"})
        pub_b.emit("run.completed", {"outcome": "failed", "error": "boom"})
        assert _wait_until(lambda: len(emitter_a.published) == 1 and len(emitter_b.published) == 1)

    assert emitter_a.published[0][0] == "sweeper.run-a.run_completed"
    assert emitter_b.published[0][0] == "sweeper.run-b.run_completed"
