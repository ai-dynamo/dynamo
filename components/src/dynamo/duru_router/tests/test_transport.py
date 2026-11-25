# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Event and score transport: in-process delivery, IPC, and recorded replay."""

import json
import os
import pickle
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import pytest

from dynamo.duru_router import scoring, transport
from dynamo.duru_router.events import open_feed
from dynamo.duru_router.scoring import EventStreamReader, Reader
from dynamo.duru_router.tests.helpers import memory_mode as memory_mode
from dynamo.duru_router.transport import (
    MAX_FRAME,
    InProcessChannel,
    ScorePublisher,
    Sender,
    channel_path,
)

pytestmark = [pytest.mark.gpu_0]


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
    reader = scoring.EventStreamReader(path, receiver=ingress.journal)
    try:
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
        ingress.close()
        rows = reader.read()
        assert [row["event"] for row in rows] == [
            "request_arrived",
            "request_completed",
            "footer",
        ]
        assert [row["sequence"] for row in rows] == [1, 2, 3]
        assert reader.closed
        assert reader.read() == []
        assert not path.exists()
    finally:
        reader.close()


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
def test_local_channel_concurrent_producer_and_consumer():
    channel = InProcessChannel()
    started = threading.Event()
    first_read = threading.Event()
    errors = []

    def produce():
        try:
            for number in range(100):
                channel.submit("record", {"number": number})
                if number == 0:
                    started.set()
                    if not first_read.wait(5):
                        raise TimeoutError("Consumer did not read the first record")
        except Exception as exc:
            errors.append(exc)
        finally:
            channel.close()

    producer = threading.Thread(target=produce)
    producer.start()
    rows = []
    try:
        assert started.wait(5)
        rows.extend(channel.read())
    finally:
        first_read.set()
        producer.join(5)
    assert not producer.is_alive()
    assert not errors
    rows.extend(channel.read())
    assert [row["number"] for row in rows] == list(range(100))
    assert channel.pending_bytes == 0


def read_until(receiver, count):
    rows = []
    deadline = time.monotonic() + 5
    while len(rows) < count and time.monotonic() < deadline:
        rows.extend(receiver.read())
        time.sleep(0.001)
    assert len(rows) == count
    return rows


@pytest.mark.pre_merge
@pytest.mark.unit
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
                "from dynamo.duru_router.transport import Receiver; import sys,time; "
                "r=Receiver(sys.argv[1]); time.sleep(.1); rows=r.read(); "
                'assert rows[0]["number"]==0; assert len(rows)>=3; r.close()',
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
@pytest.mark.unit
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
@pytest.mark.unit
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


@pytest.fixture
def endpoint(monkeypatch):
    with tempfile.TemporaryDirectory(prefix="duru-score-test-") as directory:
        monkeypatch.setenv("DURU_IPC_DIR", directory)
        logical = Path(directory) / "scores"
        monkeypatch.setenv("DURU_SCORES", str(logical))
        yield logical, channel_path(logical)


def snapshot(value, at=None):
    return dict(
        run_id="test",
        observation_time=time.time() if at is None else at,
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


@pytest.mark.post_merge
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
        publisher.publish({})
        wait(reader, lambda s: not s)
    finally:
        reader.close()
        publisher.close()
    assert not reader._subscriber._thread.is_alive()
    assert not publisher._thread.is_alive()


@pytest.mark.post_merge
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
        publisher.publish(snapshot(1, time.time() - 10))
        wait(reader, lambda s: reader.reason == "stale_snapshot")
        assert reader.snapshot["requests"]["r"]["score"] == 1
        publisher.publish(snapshot(2, time.time() + 10))
        wait(reader, lambda s: reader.reason == "future_snapshot")
        publisher.publish(snapshot(3))
        wait(reader, lambda s: bool(s))
        publisher.close()
        publisher = None
        wait(reader, lambda s: not s)
        assert reader.reason == "missing_snapshot"
    finally:
        reader.close()
        if publisher is not None:
            publisher.close()


@pytest.mark.post_merge
@pytest.mark.integration
@pytest.mark.timeout(30)
def test_scores_reach_separate_worker_processes(endpoint):
    _, address = endpoint
    publisher = ScorePublisher(address)
    publisher.publish(snapshot(7))
    code = """import os,time,json
from dynamo.duru_router.scoring import Reader
r=Reader();deadline=time.monotonic()+5
try:
 while time.monotonic()<deadline:
  s=r.refresh()
  if s.get('requests',{}).get('r',{}).get('score')==7:
   print(json.dumps(dict(pid=os.getpid(),score=7)));break
  time.sleep(.01)
 else: raise RuntimeError('No scores')
finally: r.close()
"""
    processes = [
        subprocess.Popen(
            [sys.executable, "-c", code],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for _ in range(3)
    ]
    try:
        for process in processes:
            output, error = process.communicate(timeout=10)
            assert process.returncode == 0, error
            result = json.loads(output)
            assert result["pid"] != os.getpid()
            assert result["score"] == 7
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.wait()
        publisher.close()


@pytest.mark.post_merge
@pytest.mark.integration
@pytest.mark.timeout(30)
@pytest.mark.parametrize(
    "frame", [struct.pack("!I", MAX_FRAME + 1), struct.pack("!I", 100) + b"truncated"]
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

    thread = threading.Thread(target=serve)
    thread.start()
    reader = Reader()
    try:
        wait(reader, lambda s: reader._subscriber.failures > 0)
        assert reader.refresh() == {}
    finally:
        reader.close()
        thread.join(3)
    assert not thread.is_alive()
