# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import builtins
import importlib
import io
import logging
import queue
import signal
import threading
from unittest.mock import Mock

import msgspec
import pytest

from dynamo.common import recv_forward_pass_metrics as recorder
from dynamo.common.forward_pass_metrics import ForwardPassMetrics, encode

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.timeout(10),
]


def test_capture_import_does_not_require_plotting(monkeypatch):
    original_import = builtins.__import__

    def without_matplotlib(name, *args, **kwargs):
        if name == "matplotlib" or name.startswith("matplotlib."):
            raise ModuleNotFoundError("plotting is unavailable in this runtime")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_matplotlib)
    importlib.reload(recorder)
    assert recorder._parse_args([]).output is None


class Subscriber:
    def __init__(self, messages=()):
        self.messages = queue.Queue()
        for message in messages:
            self.messages.put(message)
        self.started = threading.Event()
        self.waiting = threading.Event()
        self.stopped = False

    def recv(self):
        self.started.set()
        if self.messages.empty():
            self.waiting.set()
        return self.messages.get()

    def shutdown(self):
        self.stopped = True
        self.messages.put(None)


def payload(counter=0, worker="worker-a", dp_rank=0):
    return encode(
        ForwardPassMetrics(
            worker_id=worker,
            dp_rank=dp_rank,
            counter_id=counter,
            wall_time=0.001,
        )
    )


def test_jsonl_capture_is_compact_and_quiet(tmp_path, caplog):
    path = tmp_path / "fpm.jsonl"
    args = recorder._parse_args(["--output", str(path)])
    subscriber = Subscriber([payload(20), payload(21), None])
    with caplog.at_level(logging.INFO):
        asyncio.run(recorder._run_recv(subscriber, args))
    lines = path.read_bytes().splitlines()
    assert len(lines) == 2
    records = [msgspec.json.decode(line) for line in lines]
    assert [r["metrics"]["counter_id"] for r in records] == [20, 21]
    assert all(isinstance(r["received_at_ns"], int) for r in records)
    assert all(r["received_at_ns"] > 0 for r in records)
    assert "[worker=" not in caplog.text
    assert "messages=2 rejected=0 streams=1 counter_gaps=0" in caplog.text
    assert subscriber.stopped


def test_existing_file_is_not_overwritten(tmp_path):
    path = tmp_path / "existing.jsonl"
    path.write_bytes(b"previous capture\n")
    args = recorder._parse_args(["--output", str(path)])
    with pytest.raises(FileExistsError):
        asyncio.run(recorder._run_recv(Subscriber(), args))
    assert path.read_bytes() == b"previous capture\n"


@pytest.mark.parametrize("value", ["0", "nan"])
def test_invalid_flush_interval(value):
    with pytest.raises(SystemExit):
        recorder._parse_args(["--flush-interval", value])


def test_tracking_rejects_output():
    with pytest.raises(SystemExit):
        recorder._parse_args(["--mode", "tracking", "--output", "unused.jsonl"])


@pytest.mark.parametrize("save", [False, True])
def test_debug_logging_remains_available(tmp_path, caplog, save):
    argv = ["--output", str(tmp_path / "fpm.jsonl"), "--log-metrics"] if save else []
    args = recorder._parse_args(argv)
    with caplog.at_level(logging.INFO):
        asyncio.run(recorder._run_recv(Subscriber([payload(), None]), args))
    assert "[worker=worker-a dp=0 counter=0]" in caplog.text


def test_stream_counter_diagnostics():
    stats = recorder._RecordingStats()
    # A late subscription must not count messages before the first observation.
    for counter, worker, dp in [
        (100, "a", 0),
        (103, "a", 0),
        (7, "a", 1),
        (0, "b", 0),
        (103, "a", 0),
        (0, "a", 0),
        (104, "a", 0),
    ]:
        stats.observe(
            ForwardPassMetrics(worker_id=worker, dp_rank=dp, counter_id=counter)
        )
    assert stats.messages == 7
    assert stats.counter_gaps == 2
    assert stats.non_increasing_counters == 2
    assert len(stats.last_counter) == 3


def test_invalid_payload_is_counted(tmp_path, caplog):
    args = recorder._parse_args(["--output", str(tmp_path / "fpm.jsonl")])
    with caplog.at_level(logging.INFO):
        asyncio.run(recorder._run_recv(Subscriber([b"\xc1", payload(), None]), args))
    assert "messages=1 rejected=1" in caplog.text


def test_flush_while_idle_does_not_spawn_more_receivers():
    subscriber = Subscriber()
    subscriber.recv = Mock(wraps=subscriber.recv)
    output = io.BytesIO()
    output.flush = Mock(side_effect=subscriber.shutdown)

    async def receive():
        return [item async for item in recorder._receive(subscriber, output, 0.001)]

    assert asyncio.run(receive()) == []
    assert output.flush.called
    assert subscriber.recv.call_count == 1


def test_cancellation_unblocks_receiver_and_closes_file(tmp_path):
    args = recorder._parse_args(["--output", str(tmp_path / "fpm.jsonl")])
    subscriber = Subscriber([payload()])

    async def cancel():
        task = asyncio.create_task(recorder._run_recv(subscriber, args))
        await asyncio.to_thread(subscriber.waiting.wait)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(cancel())
    assert subscriber.stopped
    records = args.output.read_bytes().splitlines()
    assert len(records) == 1
    assert msgspec.json.decode(records[0])["metrics"]["counter_id"] == 0


def test_write_failure_propagates_and_stops_subscriber():
    args = recorder._parse_args([])
    output = io.BytesIO()
    output.write = Mock(side_effect=OSError("disk full"))
    args.output = Mock()
    args.output.open.return_value = output
    subscriber = Subscriber([payload()])
    with pytest.raises(OSError, match="disk full"):
        asyncio.run(recorder._run_recv(subscriber, args))
    assert output.closed
    assert subscriber.stopped


def test_sigterm_cancels_run_and_removes_handler(monkeypatch):
    async def exercise():
        loop = asyncio.get_running_loop()
        install = Mock()
        remove = Mock()
        monkeypatch.setattr(loop, "add_signal_handler", install)
        monkeypatch.setattr(loop, "remove_signal_handler", remove)

        async def interrupted_run(args):
            install.call_args.args[1]()
            await asyncio.Future()

        monkeypatch.setattr(recorder, "run", interrupted_run)
        await recorder._run_with_signals(recorder._parse_args([]))
        assert install.call_args.args[0] == signal.SIGTERM
        remove.assert_called_once_with(signal.SIGTERM)

    asyncio.run(exercise())
