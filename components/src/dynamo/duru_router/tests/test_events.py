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
    original_emit = feed.emit

    def emit(event, **fields):
        if event == "u_clock":
            clock_started.set()
        original_emit(event, **fields)
        if event == "u_clock":
            clock_written.set()

    try:
        with patch.object(feed, "emit", emit):
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
                # The real clock thread attempts a write while the callback owns
                # the lock. Its newer timestamp must follow accepted work.
                assert clock_started.wait(timeout=1)
                feed.capture(
                    "accepted_output", dict(steps=[step, dict(step, accepted=False)])
                )
            assert clock_written.wait(timeout=1)
    finally:
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
    ingress.close()
    assert [row["event"] for row in sink.rows] == [
        "request_arrived",
        "request_completed",
        "footer",
    ]
    assert sink.rows[0]["program_id"] == "p"
    assert "unrelated" not in sink.rows[0]
    assert sink.closed
