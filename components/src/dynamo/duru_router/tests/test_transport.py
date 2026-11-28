# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Event and score transport: in-process delivery, IPC, and recorded replay."""

import json
import os
import pickle
import queue
import select
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from dynamo.duru_router import scoring, transport
from dynamo.duru_router.events import open_feed, open_journal
from dynamo.duru_router.scoring import EventStreamReader, LiveU, Reader, run
from dynamo.duru_router.tests.helpers import arrival_record, configuration, event_feeds
from dynamo.duru_router.tests.helpers import memory_mode as memory_mode
from dynamo.duru_router.tests.helpers import replay_accounting, work_step
from dynamo.duru_router.transport import (
    MAX_FRAME,
    InProcessChannel,
    Receiver,
    ScorePublisher,
    Sender,
    channel_path,
)

pytestmark = [pytest.mark.gpu_0]


@pytest.mark.pre_merge
@pytest.mark.integration
@pytest.mark.timeout(10)
@pytest.mark.parametrize("factory", [Sender, ScorePublisher])
@pytest.mark.parametrize("existing_path", [False, True])
def test_publisher_start_failure_closes_listener_and_only_its_path(
    tmp_path, factory, existing_path
):
    path = channel_path(tmp_path / "startup")
    if existing_path:
        path.write_text("existing owner")
    listeners = []
    original_socket = socket.socket

    def make_socket(*args, **kwargs):
        listener = original_socket(*args, **kwargs)
        listeners.append(listener)
        return listener

    try:
        with (
            patch.object(transport.socket, "socket", side_effect=make_socket),
            patch.object(
                threading.Thread, "start", side_effect=RuntimeError("start failed")
            ),
        ):
            with pytest.raises(OSError if existing_path else RuntimeError):
                factory(path)
        assert len(listeners) == 1
        assert listeners[0].fileno() == -1
        if existing_path:
            assert path.read_text() == "existing owner"
        else:
            assert not path.exists()
    finally:
        for listener in listeners:
            listener.close()


@pytest.mark.pre_merge
@pytest.mark.unit
def test_local_ingress_needs_no_socket_thread_or_directory(tmp_path, monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("Local ingress attempted external I/O or started a thread")

    monkeypatch.delenv("DURU_RECORD_EVENTS", raising=False)
    monkeypatch.setattr(transport.socket, "socket", unexpected)
    monkeypatch.setattr(threading.Thread, "start", unexpected)
    monkeypatch.setattr(Path, "mkdir", unexpected)
    path = tmp_path / "unused"
    ingress = scoring.Ingress(path, in_process=True)
    owner = scoring.RouterAccounting("unused", ingress=ingress)
    reader = scoring.EventStreamReader(path, receiver=ingress.journal)
    try:
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
        rows = reader.read()
        assert [row["event"] for row in rows] == [
            "request_arrived",
            "request_completed",
            "agent_end",
            "footer",
        ]
        assert [row["sequence"] for row in rows] == [1, 2, 3, 4]
        assert rows[2] == {
            "sequence": 3,
            "event": "agent_end",
            "time": 3,
            "program_id": "p",
            "root": "r",
            "agent": "a",
        }
        assert reader.closed
        assert reader.read() == []
        assert not path.exists()
    finally:
        reader.close()
        owner.close()


@pytest.mark.pre_merge
@pytest.mark.unit
def test_local_channel_snapshots_values_and_drains_after_close():
    channel = InProcessChannel()
    row = {"event": "request_arrived", "request_id": "before"}
    channel.submit("record", row)
    row["request_id"] = "after"
    channel.submit("record", {"event": "footer"})
    channel.close()
    with pytest.raises(RuntimeError, match="closed"):
        channel.submit("record", row)
    assert channel.read() == [
        {"event": "request_arrived", "request_id": "before"},
        {"event": "footer"},
    ]
    assert channel.pending_bytes == 0
    channel.close()


@pytest.mark.pre_merge
@pytest.mark.integration
@pytest.mark.timeout(15)
@pytest.mark.parametrize("ipc", [False, True])
def test_recording_rejection_never_publishes_live_event(tmp_path, ipc):
    recorded = []

    def record(kind, value):
        if value["event"] == "rejected":
            raise RuntimeError("recording queue overflow")
        recorded.append(dict(value))

    recording = SimpleNamespace(submit=record, close=lambda: None)
    with ExitStack() as receivers, ExitStack() as cleanup:
        if ipc:
            path = channel_path(tmp_path / "recorded-events")
            feed = Sender(path, recording=recording)
            cleanup.callback(feed.close)
            receiver = Receiver(path)
            receivers.callback(receiver.close)
        else:
            feed = InProcessChannel(recording=recording)
            cleanup.callback(feed.close)
            receiver = feed
        rows = [dict(sequence=1, event="u_clock"), dict(sequence=2, event="footer")]
        feed.submit("record", rows[0])
        with pytest.raises(RuntimeError, match="recording queue overflow"):
            feed.submit("record", dict(sequence=2, event="rejected"))
        feed.submit("record", rows[1])
        feed.close()
        assert receiver.read() == rows
        assert recorded == rows
        assert feed.pending_bytes == 0


@pytest.mark.pre_merge
@pytest.mark.integration
@pytest.mark.timeout(15)
@pytest.mark.parametrize("ipc", [False, True])
def test_live_queue_rejection_never_records_event(tmp_path, ipc):
    recorded = []
    recording = SimpleNamespace(
        submit=lambda kind, value: recorded.append(dict(value)), close=lambda: None
    )
    with ExitStack() as receivers, ExitStack() as cleanup:
        if ipc:
            path = channel_path(tmp_path / "full-recorded-events")
            feed = Sender(path, recording=recording)
            feed.queue = queue.Queue(maxsize=1)
        else:
            feed = InProcessChannel(recording=recording, max_records=1)
        cleanup.callback(feed.close)
        row = dict(sequence=1, event="footer")
        try:
            feed.submit("record", row)
            with pytest.raises(RuntimeError, match="queue overflow"):
                feed.submit("record", dict(sequence=2, event="rejected"))
        finally:
            if ipc:
                # Connect only after the capacity check so the sender cannot drain it.
                receiver = Receiver(path)
                receivers.callback(receiver.close)
            else:
                receiver = feed
            feed.close()
        assert recorded == [row]
        assert receiver.read() == [row]


@pytest.mark.pre_merge
@pytest.mark.unit
@pytest.mark.parametrize("limit", ["bytes", "records"])
def test_local_channel_overflow_preserves_accepted_records(limit):
    row = {"event": "u_clock", "time": 1}
    size = len(pickle.dumps(row, protocol=5))
    channel = InProcessChannel(
        capacity=2 * size if limit == "bytes" else transport.CAPACITY,
        max_records=2 if limit == "records" else 8192,
    )
    try:
        channel.submit("record", row)
        channel.submit("record", row)
        with pytest.raises(RuntimeError, match="overflow"):
            channel.submit("record", row)
        # Reading one record frees precisely its byte/record budget.
        assert channel.read(budget=1) == [row]
        assert channel.pending_bytes == size
        channel.submit("record", row)
        assert channel.read() == [row, row]
        assert channel.pending_bytes == 0
    finally:
        channel.close()


@pytest.mark.pre_merge
@pytest.mark.unit
def test_local_channel_rejects_oversized_frames(monkeypatch):
    monkeypatch.setattr(transport, "MAX_FRAME", 8)
    channel = InProcessChannel()
    try:
        with pytest.raises(RuntimeError, match="frame exceeds memory limit"):
            channel.submit("record", {"payload": "x" * 32})
        assert channel.read() == []
        assert channel.pending_bytes == 0
    finally:
        channel.close()


@pytest.mark.pre_merge
@pytest.mark.unit
@pytest.mark.timeout(15)
def test_local_channel_concurrent_producer_and_consumer():
    channel = InProcessChannel()
    started = threading.Event()
    first_read = threading.Event()

    def produce():
        try:
            for number in range(100):
                channel.submit("record", {"number": number})
                if number == 0:
                    started.set()
                    if not first_read.wait(5):
                        raise TimeoutError("Consumer did not read the first record")
        finally:
            started.set()
            channel.close()

    rows = []
    with ThreadPoolExecutor(max_workers=1) as executor:
        producer = executor.submit(produce)
        try:
            assert started.wait(5)
            rows.extend(channel.read())
        finally:
            first_read.set()
            producer.result(timeout=5)
    rows.extend(channel.read())
    assert [row["number"] for row in rows] == list(range(100))
    assert channel.pending_bytes == 0


def read_until(receiver, count):
    rows = []
    deadline = time.monotonic() + 5
    while len(rows) < count:
        rows.extend(receiver.read())
        if len(rows) >= count:
            break
        peer = (
            receiver.receiver.peer
            if isinstance(receiver, EventStreamReader)
            else receiver.peer
        )
        remaining = deadline - time.monotonic()
        assert remaining > 0, "Timed out waiting for event records"
        ready, _, _ = select.select([peer], [], [], remaining)
        assert ready, "Timed out waiting for event records"
    assert len(rows) == count
    return rows


@pytest.fixture
def event_connection(tmp_path):
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
        listener.bind(str(channel_path(tmp_path / "partial-events")))
        listener.listen(1)
        listener.settimeout(1)
        receiver = Receiver(channel_path(tmp_path / "partial-events"))
        try:
            peer, _ = listener.accept()
            with peer:
                yield receiver, peer
        finally:
            receiver.close()


def event_frame(row, *, extended=False):
    payload = pickle.dumps(row, protocol=5)
    header = (
        struct.pack("!iQ", -1, len(payload))
        if extended
        else struct.pack("!i", len(payload))
    )
    return header + payload


@pytest.mark.pre_merge
@pytest.mark.integration
@pytest.mark.timeout(10)
@pytest.mark.parametrize(
    "extended, split",
    [(False, 1), (False, 3), (False, 4), (False, 7), (True, 4), (True, 9)],
)
def test_receiver_retains_partial_frames_without_blocking(
    event_connection, extended, split
):
    receiver, peer = event_connection
    rows = [dict(sequence=i, event="u_clock") for i in range(1, 3)]
    first = event_frame(rows[0], extended=extended)
    second = event_frame(rows[1])
    peer.sendall(first[:split])
    with ThreadPoolExecutor(max_workers=1) as executor:
        pending = executor.submit(receiver.read)
        try:
            assert pending.result(timeout=1) == []
        finally:
            # Unblock an incorrectly blocking implementation before joining it.
            if not pending.done():
                peer.shutdown(socket.SHUT_WR)
    peer.sendall(first[split:] + second[:2])
    assert receiver.read() == [rows[0]]
    assert receiver.read() == []
    peer.sendall(second[2:])
    assert read_until(receiver, 1) == [rows[1]]


@pytest.mark.pre_merge
@pytest.mark.integration
@pytest.mark.timeout(10)
def test_receiver_budget_preserves_fifo_and_footer(event_connection):
    receiver, peer = event_connection
    rows = [
        dict(sequence=1, event="u_clock"),
        dict(sequence=2, event="accepted_output", steps=[]),
        dict(sequence=3, event="footer"),
    ]
    peer.sendall(b"".join(event_frame(row) for row in rows))
    peer.shutdown(socket.SHUT_WR)
    assert receiver.read(budget=0) == []
    assert receiver.read(budget=1) == rows[:1]
    assert receiver.read() == rows[1:]


@pytest.mark.pre_merge
@pytest.mark.integration
@pytest.mark.timeout(10)
@pytest.mark.parametrize("valid_prefix", [False, True])
@pytest.mark.parametrize(
    "suffix, error, message",
    [
        (b"", EOFError, "disconnected"),
        (b"\x00\x00", EOFError, "during a frame"),
        (struct.pack("!i", 10) + b"x", EOFError, "during a frame"),
        (struct.pack("!i", MAX_FRAME + 1), ValueError, "frame length"),
        (struct.pack("!i", -2), ValueError, "frame length"),
        (struct.pack("!i", 1) + b"!", pickle.UnpicklingError, "invalid load key"),
        (struct.pack("!i", 11) + b"\x80\x04\x95" + b"\xff" * 8, OverflowError, "FRAME"),
    ],
)
def test_receiver_delivers_valid_prefix_before_terminal_error(
    event_connection, valid_prefix, suffix, error, message
):
    receiver, peer = event_connection
    row = dict(sequence=1, event="u_clock")
    peer.sendall((event_frame(row) if valid_prefix else b"") + suffix)
    peer.shutdown(socket.SHUT_WR)
    if valid_prefix:
        assert receiver.read() == [row]
    with pytest.raises(error, match=message):
        receiver.read()
    with pytest.raises(error, match=message):
        receiver.read()


@pytest.mark.pre_merge
@pytest.mark.integration
@pytest.mark.timeout(20)
def test_bounded_queue_overflow_and_cross_process(tmp_path):
    path = tmp_path / "channel"
    sender = Sender(path, capacity=512)
    try:
        for i in range(3):
            sender.submit("record", dict(number=i, payload="x" * 80))
        with pytest.raises(RuntimeError, match="overflow"):
            for _ in range(10):
                sender.submit("record", dict(number=9, payload="x" * 80))
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                """
import select
import sys
import time

from dynamo.duru_router.transport import Receiver

receiver = Receiver(sys.argv[1])
try:
    rows = []
    deadline = time.monotonic() + 5
    while len(rows) < 3:
        rows.extend(receiver.read())
        if len(rows) >= 3:
            break
        remaining = deadline - time.monotonic()
        assert remaining > 0, "Timed out waiting for event records"
        ready, _, _ = select.select([receiver.peer], [], [], remaining)
        assert ready, "Timed out waiting for event records"
    assert [row["number"] for row in rows[:3]] == [0, 1, 2]
finally:
    receiver.close()
""",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0, result.stderr
    finally:
        sender.close()
    assert not sender.thread.is_alive()
    assert sender.pending_bytes == 0


@pytest.mark.pre_merge
@pytest.mark.integration
@pytest.mark.timeout(30)
@pytest.mark.parametrize("record", [False, True])
def test_sender_close_rejects_late_records_and_drains(tmp_path, record):
    journal = open_journal(tmp_path, name="recording") if record else None
    sender = Sender(channel_path(tmp_path / "sender"), recording=journal)
    receiver = Receiver(channel_path(tmp_path / "sender"))
    rows = [dict(sequence=i, event="u_clock") for i in range(2)]
    stop_queued, finish_close = threading.Event(), threading.Event()
    original_put = sender.queue.put

    def put(item, *args, **kwargs):
        original_put(item, *args, **kwargs)
        if item is None:
            stop_queued.set()
            assert finish_close.wait(timeout=3)

    try:
        for row in rows:
            sender.submit("record", row)
        with patch.object(sender.queue, "put", put):
            with ThreadPoolExecutor(max_workers=1) as executor:
                closing = executor.submit(sender.close)
                try:
                    assert stop_queued.wait(timeout=3)
                    with pytest.raises(RuntimeError, match="closing|closed"):
                        sender.submit("record", dict(sequence=2, event="late"))
                finally:
                    finish_close.set()
                    closing.result(timeout=5)
        assert receiver.read() == rows
        assert sender.queue.empty()
        assert sender.pending_bytes == 0
        assert not sender.thread.is_alive()
        if record:
            assert [
                json.loads(line)
                for line in (tmp_path / "recording.jsonl").read_text().splitlines()
            ] == rows
    finally:
        receiver.close()
        sender.close()


@pytest.mark.pre_merge
@pytest.mark.integration
@pytest.mark.timeout(15)
def test_sender_close_drains_full_queue_without_holding_submit_lock(
    tmp_path, monkeypatch
):
    sending, release_send, stop_attempted = (
        threading.Event(),
        threading.Event(),
        threading.Event(),
    )
    original_send = transport.Connection.send_bytes

    def send(connection, frame):
        sending.set()
        assert release_send.wait(timeout=3)
        original_send(connection, frame)

    monkeypatch.setattr(transport.Connection, "send_bytes", send)
    sender = Sender(channel_path(tmp_path / "full"))
    sender.queue = queue.Queue(maxsize=1)
    receiver = Receiver(channel_path(tmp_path / "full"))
    original_put = sender.queue.put
    rows = [dict(sequence=i, event="u_clock") for i in range(2)]

    def put(item, *args, **kwargs):
        if item is None:
            stop_attempted.set()
        original_put(item, *args, **kwargs)

    try:
        sender.submit("record", rows[0])
        assert sending.wait(timeout=3)
        sender.submit("record", rows[1])
        assert sender.queue.full()
        with patch.object(sender.queue, "put", put):
            with ThreadPoolExecutor(max_workers=1) as executor:
                closing = executor.submit(sender.close)
                try:
                    assert stop_attempted.wait(timeout=3)
                finally:
                    release_send.set()
                closing.result(timeout=3)
        assert receiver.read() == rows
        assert sender.pending_bytes == 0
        assert sender.queue.empty()
        assert not sender.thread.is_alive()
    finally:
        release_send.set()
        receiver.close()
        sender.close()


@pytest.mark.pre_merge
@pytest.mark.integration
@pytest.mark.timeout(15)
def test_default_feed_needs_no_journal_and_rejects_second_producer(tmp_path):
    feed = open_feed(tmp_path / "events", name="ingress")
    tail = EventStreamReader(tmp_path / "events/ingress.jsonl")
    try:
        feed.submit("record", dict(sequence=1, time=1, event="footer"))
        with pytest.raises(OSError):
            open_feed(tmp_path / "events", name="ingress")
        assert read_until(tail, 1) == [dict(sequence=1, time=1, event="footer")]
        assert tail.closed
        assert tail.read() == []
    finally:
        feed.close()
        tail.close()
    assert not (tmp_path / "events").exists()


@pytest.mark.pre_merge
@pytest.mark.integration
@pytest.mark.timeout(15)
def test_memory_sequence_gap_fails_closed(tmp_path):
    feed = open_feed(tmp_path, name="worker")
    feed.submit("record", dict(sequence=2, event="footer", time=2))
    tail = EventStreamReader(tmp_path / "worker.jsonl")
    try:
        with pytest.raises(ValueError, match="sequence gap"):
            read_until(tail, 1)
    finally:
        feed.close()
        tail.close()


def run_memory(path, *, ingress_source=None):
    sources = []
    expected = 1 + len(json.loads(Path(path).read_text())["ledgers"])
    deadline = time.monotonic() + 5

    def reader(*args, **kwargs):
        source = EventStreamReader(*args, **kwargs)
        sources.append(source)
        return source

    class CompletedInputs(threading.Event):
        def is_set(self):
            if len(sources) == expected and all(source.closed for source in sources):
                self.set()
            if not super().is_set() and time.monotonic() >= deadline:
                raise TimeoutError("Accounting did not receive all input footers")
            return super().is_set()

    stop = CompletedInputs()
    with patch.object(scoring, "EventStreamReader", side_effect=reader):
        run(path, stop, ingress_source=ingress_source)


@pytest.mark.pre_merge
@pytest.mark.unit
def test_root_cleanup_retains_tool_gap_then_releases_all_history():
    book = LiveU(2, {})
    book.cost = lambda *args: 0.01
    for i in range(1000):
        root, agent = f"root-{i}", f"agent-{i}"
        t = i * 10
        book.submit("request", root, agent, 10, t)
        book.accepted(work_step(p=1, d=10, root=root, agent=agent, phase="decode"))
        with pytest.raises(ValueError, match="outstanding requests"):
            book.finish_root(root, t)
        book.finish("request", t + 1)
        book.advance(t + 5)
        assert book.measured(root) == 100
        book.submit("return", root, agent, 11, t + 5)
        book.finish("return", t + 6)
        assert book.measured(root) == 200
        book.finish_root(root, t + 7)
        assert not any(
            (
                book.active,
                book.agent_root,
                book.agent_active_requests,
                book.root_active,
                book.work,
                book.entitlement,
                book.invalid,
            )
        )
    assert book.output_tokens == book.accepted_steps == 1000


@pytest.mark.pre_merge
@pytest.mark.integration
@pytest.mark.timeout(30)
@pytest.mark.parametrize("completed", [False, True])
@pytest.mark.parametrize("in_process", [False, True])
def test_identical_events_have_identical_u_in_ipc_and_recorded_replay(
    tmp_path, monkeypatch, completed, in_process
):
    path, cfg = configuration(tmp_path)
    monkeypatch.setenv("DURU_RECORD_EVENTS", "1")
    ingress = [
        arrival_record(100, "q", "p", root="r", agent="a"),
        dict(event="request_completed", time=101, request_id="q"),
        # Callback at 102 arrives after ingress completion at 101.
        arrival_record(103, "q2", "p", root="r", agent="a"),
        arrival_record(103, "child", "pb", root="r", agent="b"),
        dict(event="request_completed", time=104, request_id="q2"),
        dict(event="request_completed", time=105, request_id="child"),
    ]
    if completed:
        ingress += [
            dict(event="agent_end", time=106, program_id="p"),
            dict(event="agent_group_end", time=107, root="r"),
        ]
    ingress.append(dict(event="footer", time=108))
    step = work_step(
        p=8, root="r", agent="a", client_request_id="q", scheduled_at=100.5, end=102
    )
    worker = [
        dict(event="accepted_output", time=102, steps=[step]),
        dict(event="footer", time=109),
    ]
    if completed:
        ingress.insert(1, dict(event="request_dispatched", time=100.1, request_id="q"))
        # Final ingress can precede an engine callback. Keep the root until
        # its worker acknowledgement, including across replay.
        worker[-1:-1] = [
            dict(event="accepted_output", time=107.2, steps=[dict(step, end=107.2)]),
            dict(
                event="worker_request_finished",
                time=107.3,
                requests=[dict(client_request_id="q", root="r", agent="a")],
            ),
        ]
    snapshots = []
    stop = threading.Event()
    stop.set()
    for mode in ("memory", "journal"):
        monkeypatch.setenv("DURU_ACCOUNTING_TRANSPORT", mode)
        if mode == "memory":
            with event_feeds(cfg, ingress, worker, in_process=in_process) as local:
                run_memory(path, ingress_source=local.journal if local else None)
        else:
            replay_accounting(path, stop)
        footer = json.loads(
            (Path(cfg["audit_dir"]) / "duru-control.jsonl").read_text().splitlines()[-1]
        )
        assert footer["complete"]
        snapshots.append(footer["last_snapshot"])
    for key in (
        "roots",
        "programs",
        "requests",
        "accounting_state",
        "accepted_steps",
        "output_tokens",
    ):
        assert snapshots[0][key] == snapshots[1][key]
    if completed:
        assert snapshots[0]["accepted_steps"] == 2
        assert not snapshots[0]["roots"]
        assert not any(snapshots[0]["accounting_state"].values())
    else:
        assert snapshots[0]["roots"]["r"]["U"] == pytest.approx(300)
        assert snapshots[0]["programs"]["p"]["kind"] == "idle_measured_only"


@pytest.mark.pre_merge
@pytest.mark.integration
@pytest.mark.timeout(15)
@pytest.mark.parametrize("producer", ["ingress", "local_ingress", "worker"])
def test_pending_limit_applies_with_missing_peer_watermark(
    tmp_path, monkeypatch, producer
):
    path, cfg = configuration(tmp_path)
    monkeypatch.setattr(scoring, "MAX_PENDING_EVENTS", 2)
    logical = Path(cfg["ingress"] if producer != "worker" else cfg["ledgers"][0])
    if producer == "local_ingress":
        feed = scoring.Ingress(logical.parent, in_process=True).journal
    else:
        feed = open_feed(logical.parent, name=logical.stem)
    try:
        for sequence in range(1, 4):
            row = (
                dict(
                    event="request_arrived",
                    request_id=str(sequence),
                    program_id="p",
                    root="r",
                    agent="a",
                    prompt_tokens=8,
                )
                if producer != "worker"
                else dict(event="accepted_output", steps=[dict(end=1)])
            )
            feed.submit("record", dict(sequence=sequence, time=1, **row))
        with pytest.raises(RuntimeError, match="pending-event limit"):
            run_memory(
                path, ingress_source=feed if producer == "local_ingress" else None
            )
    finally:
        feed.close()


@pytest.fixture
def endpoint(monkeypatch):
    with tempfile.TemporaryDirectory(prefix="duru-score-test-") as directory:
        monkeypatch.setenv("DURU_IPC_DIR", directory)
        logical = Path(directory) / "scores"
        monkeypatch.setenv("DURU_SCORES", str(logical))
        yield logical, channel_path(logical)


@pytest.mark.pre_merge
@pytest.mark.unit
@pytest.mark.parametrize("wall_jump", [-60, 60])
def test_score_freshness_and_ingress_footer_use_monotonic_time(
    tmp_path, monkeypatch, wall_jump
):
    clock = SimpleNamespace(wall=1000.0, elapsed=100.0)
    monkeypatch.setattr(
        scoring,
        "time",
        SimpleNamespace(time=lambda: clock.wall, monotonic=lambda: clock.elapsed),
    )
    value = dict(observation_time=clock.elapsed, requests={})
    reader = Reader(snapshot_source=lambda: value)
    ingress = scoring.Ingress(tmp_path, in_process=True)
    try:
        assert reader.refresh() is value
        clock.wall += wall_jump
        clock.elapsed += 1
        assert reader.refresh() is value
        assert reader.age == 1
        clock.elapsed += 2
        assert reader.refresh() == {}
        assert reader.reason == "stale_snapshot"
    finally:
        ingress.close()
        reader.close()
    assert ingress.journal.read()[-1]["time"] == 103.0


def snapshot(value, at=None):
    return dict(
        run_id="test",
        observation_time=time.monotonic() if at is None else at,
        requests={"r": dict(score=value, root="not-needed-by-worker")},
        roots={"r": {"U": value}},
    )


def wait(reader, condition):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        current = reader.refresh()
        if condition(current):
            return current
        time.sleep(0.01)
    pytest.fail("Score IPC did not reach expected state")


@pytest.mark.pre_merge
@pytest.mark.integration
@pytest.mark.timeout(30)
def test_latest_only_projection_and_no_score_file(endpoint):
    logical, address = endpoint
    publisher = ScorePublisher(address)
    reader = Reader()
    try:
        publisher.publish(snapshot(1))
        first = wait(
            reader, lambda s: s.get("requests", {}).get("r", {}).get("score") == 1
        )
        assert first["requests"] == {"r": {"score": 1}}
        assert "roots" not in first
        publisher.publish(snapshot(2))
        with reader.pin(first):
            assert reader.refresh() is first
        second = wait(
            reader, lambda s: s.get("requests", {}).get("r", {}).get("score") == 2
        )
        assert first["requests"]["r"]["score"] == 1
        assert second is not first
        assert not logical.exists()
        publisher.publish(snapshot(None))
        wait(reader, lambda s: s.get("requests", {}).get("r") == {"score": None})
        failures = reader._subscriber.failures
        publisher.publish({})
        wait(reader, lambda s: not s)
        assert reader._subscriber.failures == failures
    finally:
        reader.close()
        publisher.close()
    assert not reader._subscriber._thread.is_alive()
    assert not publisher._thread.is_alive()


@pytest.mark.pre_merge
@pytest.mark.integration
@pytest.mark.timeout(20)
@pytest.mark.parametrize("rejection", ["wrong_uid", "credentials_unavailable"])
def test_score_server_identity_checked_before_decode(endpoint, monkeypatch, rejection):
    _, address = endpoint
    trusted = threading.Event()
    original_getsockopt = socket.socket.getsockopt

    def getsockopt(peer, level, option, *args):
        credentials = original_getsockopt(peer, level, option, *args)
        if level == socket.SOL_SOCKET and option == socket.SO_PEERCRED:
            if not trusted.is_set():
                if rejection == "credentials_unavailable":
                    raise OSError("Peer credentials unavailable")
                pid, uid, gid = struct.unpack("iII", credentials)
                return struct.pack("iII", pid, uid + 1, gid)
        return credentials

    monkeypatch.setattr(socket.socket, "getsockopt", getsockopt)
    with ExitStack() as cleanup:
        publisher = ScorePublisher(address)
        cleanup.callback(publisher.close)
        publisher.publish(snapshot(7))
        decoded = cleanup.enter_context(
            patch.object(pickle, "loads", wraps=pickle.loads)
        )
        reader = Reader()
        cleanup.callback(reader.close)
        wait(reader, lambda s: bool(s) or reader._subscriber.failures > 0)
        assert reader._subscriber.failures > 0
        assert reader.refresh() == {}
        decoded.assert_not_called()

        trusted.set()
        wait(reader, lambda s: s.get("requests", {}).get("r", {}).get("score") == 7)
        assert decoded.called

        failures = reader._subscriber.failures
        trusted.clear()
        wait(reader, lambda s: reader._subscriber.failures > failures)
        assert reader.refresh() == {}
        calls = decoded.call_count
        failures = reader._subscriber.failures
        wait(reader, lambda s: reader._subscriber.failures > failures)
        assert decoded.call_count == calls


@pytest.mark.pre_merge
@pytest.mark.integration
@pytest.mark.parametrize(
    "invalid_fields",
    [
        {"observation_time": "not-a-number"},
        {"observation_time": True},
        {"observation_time": float("nan")},
        {"observation_time": float("inf")},
        {"observation_time": 10**1000},
        {"requests": []},
        {"requests": {"r": None}},
        {"requests": {"r": {}}},
        {"requests": {"r": {"score": "bad"}}},
        {"requests": {"r": {"score": float("nan")}}},
        {"requests": {"r": {"score": -1}}},
        {"requests": {"r": {"score": True}}},
        {"requests": {1: {"score": 1}}},
    ],
)
def test_invalid_score_snapshot_clears_cache_and_recovers(endpoint, invalid_fields):
    _, address = endpoint
    publisher = ScorePublisher(address)
    reader = Reader()
    try:
        publisher.publish(snapshot(1))
        wait(reader, lambda s: bool(s))
        failures = reader._subscriber.failures
        malformed = dict(
            observation_time=time.monotonic(), requests={"r": {"score": 1}}
        )
        malformed.update(invalid_fields)
        # Inject malformed wire data without the publisher's request projection.
        publisher._frame = pickle.dumps(malformed, protocol=5)
        wait(reader, lambda s: reader._subscriber.failures > failures)
        assert reader.refresh() == {}
        assert reader.reason == "missing_snapshot"
        publisher.publish(snapshot(2))
        wait(reader, lambda s: s.get("requests", {}).get("r") == {"score": 2})
    finally:
        reader.close()
        publisher.close()


@pytest.mark.pre_merge
@pytest.mark.integration
@pytest.mark.timeout(30)
def test_missing_stale_future_and_disconnect_never_use_file(endpoint, monkeypatch):
    logical, address = endpoint
    logical.write_text(json.dumps(snapshot(999)))
    monkeypatch.setattr(
        Path, "read_text", lambda *a, **k: pytest.fail("Worker read a score file")
    )
    reader = Reader()
    publisher = None
    try:
        assert reader.refresh() == {}
        publisher = ScorePublisher(address)
        publisher.publish(snapshot(1, time.monotonic() - 10))
        assert wait(reader, lambda s: reader.reason == "stale_snapshot") == {}
        assert reader.snapshot["requests"]["r"]["score"] == 1
        publisher.publish(snapshot(2, time.monotonic() + 10))
        assert wait(reader, lambda s: reader.reason == "future_snapshot") == {}
        publisher.publish(snapshot(3))
        wait(reader, lambda s: bool(s))
        publisher.close()
        publisher = None
        assert wait(reader, lambda s: reader.reason == "missing_snapshot") == {}
    finally:
        reader.close()
        if publisher is not None:
            publisher.close()


@pytest.mark.pre_merge
@pytest.mark.integration
@pytest.mark.timeout(30)
def test_scores_reach_separate_worker_processes(endpoint):
    _, address = endpoint
    code = """import os,time,json
from dynamo.duru_router.scoring import Reader
r=Reader();deadline=time.monotonic()+5
try:
 print('READY',flush=True)
 while time.monotonic()<deadline:
  s=r.refresh()
  if s.get('requests',{}).get('r',{}).get('score')==7:
   print(json.dumps(dict(pid=os.getpid(),score=7)));break
  time.sleep(.01)
 else: raise RuntimeError('No scores')
finally: r.close()
"""
    with ExitStack() as cleanup:
        publisher = ScorePublisher(address)
        cleanup.callback(publisher.close)
        processes = []
        for _ in range(3):
            process = cleanup.enter_context(
                subprocess.Popen(
                    [sys.executable, "-c", code],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
            )
            # Kill unfinished workers before Popen.__exit__ waits and closes pipes.
            cleanup.callback(process.kill)
            processes.append(process)
        deadline = time.monotonic() + 5
        for process in processes:
            ready, _, _ = select.select(
                [process.stdout], [], [], max(0, deadline - time.monotonic())
            )
            assert ready, "Worker did not become ready"
            assert process.stdout.readline() == "READY\n"
        publisher.publish(snapshot(7))
        for process in processes:
            output, error = process.communicate(timeout=10)
            assert process.returncode == 0, error
            result = json.loads(output)
            assert result["pid"] != os.getpid()
            assert result["score"] == 7


@pytest.mark.pre_merge
@pytest.mark.integration
@pytest.mark.timeout(30)
@pytest.mark.parametrize(
    "frame",
    [
        struct.pack("!I", MAX_FRAME + 1),
        struct.pack("!I", 100) + b"truncated",
        struct.pack("!I", 11) + b"\x80\x04\x95" + b"\xff" * 8,
    ],
)
def test_bad_frame_fails_closed(endpoint, frame):
    _, address = endpoint
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(str(address))
    listener.listen(1)
    listener.settimeout(2)

    def serve():
        with listener:
            peer, _ = listener.accept()
            with peer:
                peer.sendall(frame)

    with ThreadPoolExecutor(max_workers=1) as executor:
        serving = executor.submit(serve)
        reader = Reader()
        try:
            wait(reader, lambda s: reader._subscriber.failures > 0)
            assert reader.refresh() == {}
        finally:
            reader.close()
            serving.result(timeout=3)
