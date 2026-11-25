# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import pytest

from dynamo.duru_router.events import CompactFeed, open_journal
from dynamo.duru_router.tests.helpers import Tail, work_step

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.integration,
    pytest.mark.gpu_0,
    pytest.mark.timeout(30),
]


def test_compact_coordinates_and_clock_ordering(tmp_path):
    feed = CompactFeed(tmp_path)
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
                end = time.monotonic()
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
    tail = Tail(tmp_path / "compact-work.jsonl")
    rows = tail.read()
    index = next(i for i, row in enumerate(rows) if row["event"] == "accepted_output")
    accepted = rows[index]["steps"]
    assert len(accepted) == 1
    assert accepted[0]["p"] == 12
    assert "prompt_tokens" not in accepted[0]
    assert all(row["time"] <= end for row in rows[:index] if row["event"] == "u_clock")
    assert any(row["event"] == "u_clock" for row in rows[index + 1 :])
    assert tail.sequence == len(rows)
    assert tail.received_through > end


def test_feed_error_still_closes_writer(tmp_path):
    feed = CompactFeed(tmp_path)
    feed.error = RuntimeError("clock failed")
    with pytest.raises(RuntimeError, match="feed failed"):
        feed.close()
    assert not feed.thread.is_alive()
    assert not feed.journal.thread.is_alive()
    assert feed.journal.proc.returncode == 0


def test_close_keeps_footer_last_with_pending_clock(tmp_path):
    feed = CompactFeed(tmp_path)
    clock_started = threading.Event()
    original_emit = feed.emit
    original_join = feed.thread.join
    path = tmp_path / "compact-work.jsonl"

    def emit(event, **fields):
        if event == "u_clock":
            clock_started.set()
        original_emit(event, **fields)

    try:
        with patch.object(feed, "emit", emit):
            with feed.lock:
                assert clock_started.wait(timeout=1)
                # Simulate the join timeout while the real clock is blocked.
                with patch.object(
                    feed.thread,
                    "join",
                    side_effect=lambda timeout: original_join(timeout=0),
                ):
                    feed.close()
                assert feed.thread.is_alive()
                assert Tail(path).read()[-1]["event"] == "footer"
            original_join(timeout=1)
            assert not feed.thread.is_alive()
        rows = Tail(path).read()
        assert feed.error is None
        assert rows[-1]["event"] == "footer"
        assert sum(row["event"] == "footer" for row in rows) == 1
        assert feed.journal.closed
        assert not feed.journal.thread.is_alive()
        assert feed.journal.proc.returncode == 0
    finally:
        feed.stop.set()
        original_join(timeout=1)
        if not feed.journal.closed:
            feed.close()


def test_closed_feed_rejects_producers_and_is_idempotent(tmp_path):
    feed = CompactFeed(tmp_path)
    feed.close()
    path = tmp_path / "compact-work.jsonl"
    rows = Tail(path).read()

    with pytest.raises(RuntimeError, match="feed is closed"):
        feed.capture("accepted_output", dict(steps=[work_step()]))
    feed.close()

    assert Tail(path).read() == rows
    assert feed.error is None


@pytest.mark.parametrize("without_pythonpath", [False, True])
def test_flush_preserves_order_and_writes_only_event_stream(
    tmp_path, monkeypatch, without_pythonpath
):
    if without_pythonpath:
        monkeypatch.delenv("PYTHONPATH", raising=False)
    monkeypatch.chdir(tmp_path)
    journal = open_journal(tmp_path, name="events")
    try:
        for sequence in range(3):
            journal.submit("record", {"sequence": sequence, "event": "u_clock"})
        journal.flush(timeout=5)
        path = tmp_path / "events.jsonl"
        assert [
            json.loads(line)["sequence"] for line in path.read_text().splitlines()
        ] == [0, 1, 2]
        assert list(tmp_path.iterdir()) == [path]
    finally:
        journal.close()
    assert not journal.thread.is_alive()
    assert journal.proc.returncode == 0


@pytest.mark.parametrize("concurrent_operation", ["flush", "close"])
def test_journal_shutdown_rejects_late_records_and_drains_accepted(
    tmp_path, concurrent_operation
):
    journal = open_journal(tmp_path, name="events")
    records = [{"sequence": sequence, "event": "u_clock"} for sequence in range(3)]
    for record in records:
        journal.submit("record", record)
    stop_queued = threading.Event()
    finish_close = threading.Event()
    other_started = threading.Event()
    stop_markers = []
    original_put = journal.queue.put

    def put(item, *args, **kwargs):
        original_put(item, *args, **kwargs)
        if item is None:
            stop_markers.append(item)
            stop_queued.set()
            assert finish_close.wait(timeout=3)

    def other_operation():
        other_started.set()
        if concurrent_operation == "flush":
            journal.flush(timeout=1)
        else:
            journal.close()

    with patch.object(journal.queue, "put", put):
        with ThreadPoolExecutor(max_workers=2) as executor:
            closing = executor.submit(journal.close)
            try:
                assert stop_queued.wait(timeout=3)
                with pytest.raises(RuntimeError, match="closing|closed"):
                    journal.submit("record", {"sequence": 3, "event": "late"})
                other = executor.submit(other_operation)
                assert other_started.wait(timeout=3)
            finally:
                finish_close.set()
                closing.result(timeout=5)
            if concurrent_operation == "flush":
                with pytest.raises(RuntimeError, match="closed"):
                    other.result(timeout=5)
            else:
                other.result(timeout=5)

    assert [
        json.loads(line)
        for line in (tmp_path / "events.jsonl").read_text().splitlines()
    ] == records
    assert len(stop_markers) == 1
    assert journal.queue.empty()
    assert journal.closed
    assert not journal.thread.is_alive()
    assert journal.proc.returncode == 0
    journal.close()


def test_writer_failure_surfaces_and_close_releases_resources(tmp_path):
    journal = open_journal(tmp_path, name="events")
    journal.proc.kill()
    journal.proc.wait(timeout=5)
    with pytest.raises(RuntimeError, match="writer is closed"):
        journal.submit("record", {"event": "u_clock"})
    with pytest.raises(RuntimeError):
        journal.close()
    assert journal.closed
    assert not journal.thread.is_alive()


@pytest.fixture(autouse=True)
def journal_validation_mode(monkeypatch):
    monkeypatch.setenv("DURU_ACCOUNTING_TRANSPORT", "journal")
    monkeypatch.setenv("DURU_RECORD_EVENTS", "1")
