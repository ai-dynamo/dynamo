# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Collection semantics without IPC or engine execution."""

import threading
import time
from unittest.mock import patch

import pytest

from dynamo.duru_router.events import CompactFeed
from dynamo.duru_router.scoring import Ingress
from dynamo.duru_router.tests.helpers import work_step

pytestmark = [
    pytest.mark.post_merge,
    pytest.mark.integration,
    pytest.mark.gpu_0,
    pytest.mark.timeout(30),
]


class Sink:
    def __init__(self):
        self.rows = []
        self.closed = False

    def submit(self, kind, value):
        assert kind == "record"
        self.rows.append(value)

    def close(self):
        self.closed = True


def test_compact_coordinates_and_clock_ordering():
    sink = Sink()
    feed = CompactFeed(sink)
    clock_started, clock_written = threading.Event(), threading.Event()
    allow_clock_write = threading.Event()
    original_emit = feed.emit

    def emit(event, **fields):
        observed_clock = event == "u_clock" and not clock_started.is_set()
        if observed_clock:
            clock_started.set()
            assert allow_clock_write.wait(timeout=1)
        original_emit(event, **fields)
        if observed_clock:
            clock_written.set()

    try:
        with patch.object(feed, "emit", emit):
            assert clock_started.wait(timeout=1)
            with feed.lock:
                end = time.time()
                step = work_step(
                    p=12,
                    root="r",
                    agent="a",
                    client_request_id="req",
                    scheduled_at=end - 0.001,
                    end=end,
                    prompt_tokens=999,
                )
                # Release the intercepted tick only while capture owns the lock.
                # Its write cannot complete before the accepted work below.
                allow_clock_write.set()
                feed.capture(
                    "accepted_output", dict(steps=[step, dict(step, accepted=False)])
                )
            assert clock_written.wait(timeout=1)
    finally:
        allow_clock_write.set()
        feed.close()
    rows = sink.rows
    index = next(i for i, row in enumerate(rows) if row["event"] == "accepted_output")
    accepted = rows[index]["steps"]
    assert len(accepted) == 1
    assert accepted[0]["p"] == 12
    assert "prompt_tokens" not in accepted[0]
    assert all(row["time"] <= end for row in rows[:index] if row["event"] == "u_clock")
    assert any(row["event"] == "u_clock" for row in rows[index + 1 :])
    assert [row["sequence"] for row in rows] == list(range(1, len(rows) + 1))
    assert rows[-1]["event"] == "footer"
    assert rows[-1]["time"] > end
    assert sink.closed


def test_router_events_validate_and_preserve_identity():
    sink = Sink()
    ingress = Ingress(sink)
    with pytest.raises(ValueError, match="program_id"):
        ingress.emit("request_arrived", 1, request_id="q")
    ingress.emit(
        "request_arrived",
        1,
        request_id="q",
        program_id="p",
        root="r",
        agent="a",
        prompt_tokens=8,
        unrelated="ignored",
    )
    ingress.emit("request_completed", 2, request_id="q")
    ingress.emit(
        "agent_end",
        3,
        program_id="p",
        root="r",
        agent="a",
        prompt_tokens=8,
        unrelated="ignored",
    )
    ingress.close()
    assert [row["event"] for row in sink.rows] == [
        "request_arrived",
        "request_completed",
        "agent_end",
        "footer",
    ]
    assert sink.rows[0]["program_id"] == "p"
    assert "unrelated" not in sink.rows[0]
    assert sink.rows[2] == {
        "sequence": 3,
        "event": "agent_end",
        "time": 3,
        "program_id": "p",
        "root": "r",
        "agent": "a",
    }
    assert sink.closed


@pytest.mark.parametrize("reject_after_close", [False, True])
def test_close_keeps_footer_last_with_pending_clock(reject_after_close):
    class ClosingSink(Sink):
        def submit(self, kind, value):
            if reject_after_close and self.closed:
                raise RuntimeError("sink is closed")
            super().submit(kind, value)

    sink = ClosingSink()
    feed = CompactFeed(sink)
    clock_started = threading.Event()
    original_emit = feed.emit
    original_join = feed.thread.join

    def emit(event, **fields):
        if event == "u_clock":
            clock_started.set()
        original_emit(event, **fields)

    try:
        with patch.object(feed, "emit", emit):
            with feed.lock:
                assert clock_started.wait(timeout=1)
                # Simulate the join timeout while the real clock is blocked on
                # the lock, without making the test wait two seconds.
                with patch.object(
                    feed.thread,
                    "join",
                    side_effect=lambda timeout: original_join(timeout=0),
                ):
                    feed.close()
                assert feed.thread.is_alive()
                assert sink.rows[-1]["event"] == "footer"
            original_join(timeout=1)
            assert not feed.thread.is_alive()
        assert feed.error is None
        assert sink.rows[-1]["event"] == "footer"
        assert sum(row["event"] == "footer" for row in sink.rows) == 1
        assert sink.closed
    finally:
        feed.stop.set()
        original_join(timeout=1)
        if not sink.closed:
            feed.close()


def test_closed_feed_rejects_producers_and_is_idempotent():
    sink = Sink()
    feed = CompactFeed(sink)
    feed.close()
    rows = list(sink.rows)

    with pytest.raises(RuntimeError, match="feed is closed"):
        feed.capture("accepted_output", dict(steps=[work_step()]))
    feed.close()

    assert sink.rows == rows
    assert feed.error is None
