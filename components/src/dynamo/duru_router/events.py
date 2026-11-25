# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Record completed work and clock updates for U accounting.

Keep timestamps ordered so the scorer cannot advance past work that
has not yet been recorded. Clock updates require no GPU operations.
"""

import atexit
import json
import os
import pickle
import queue
import select
import signal
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path

from dynamo.duru_router.transport import Sender, channel_path, transport

FIELDS = (
    "client_request_id",
    "root",
    "agent",
    "scheduled_at",
    "p",
    "d",
    "k",
    "phase",
    "end",
    "generated_tokens",
    "accepted",
)


class AsyncJournal:
    """Bounded event queue with serialization and file writes off the caller thread.

    Only submit captured values that will not be mutated. A flush waits for the
    writer process to flush its buffers; it does not promise fsync durability.
    """

    def __init__(self, path, capacity=8192):
        self.queue = queue.Queue(maxsize=capacity)
        self.error = None
        self.closed = False
        # Group signals must reach the owner first so it can drain the writer.
        previous = signal.pthread_sigmask(
            signal.SIG_BLOCK, {signal.SIGINT, signal.SIGTERM}
        )
        try:
            self.proc = subprocess.Popen(
                [sys.executable, "-S", "-u", str(Path(__file__).resolve()), str(path)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
            )
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK, previous)
        self.thread = threading.Thread(
            target=self._send, name="duru-events", daemon=True
        )
        self.thread.start()
        atexit.register(self.close)

    def check(self):
        if self.error is not None:
            raise RuntimeError("Duru event writer failed") from self.error
        if self.closed or self.proc.poll() is not None:
            raise RuntimeError("Duru event writer is closed")

    def submit(self, kind, value):
        self.check()
        if kind != "record":
            raise ValueError(f"Unsupported event message: {kind}")
        try:
            self.queue.put_nowait((value, None))
        except queue.Full as exc:
            raise RuntimeError(
                "Duru event queue overflow; accounting cannot continue"
            ) from exc

    def _send(self):
        try:
            while not self.closed:
                try:
                    item = self.queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                if item is None:
                    return
                value, ack = item
                frame = pickle.dumps(value, protocol=5)
                self.proc.stdin.write(struct.pack("!Q", len(frame)))
                self.proc.stdin.write(frame)
                self.proc.stdin.flush()
                if ack is not None:
                    if self.proc.stdout.readline() != b"OK\n":
                        raise RuntimeError(
                            "Duru writer failed before flush acknowledgement"
                        )
                    ack.set()
        except Exception as exc:
            self.error = exc

    def flush(self, timeout=30):
        self.check()
        ack = threading.Event()
        deadline = time.monotonic() + timeout
        self.queue.put((None, ack), timeout=timeout)
        while not ack.wait(0.02):
            self.check()
            if time.monotonic() >= deadline:
                raise TimeoutError("Duru event flush timed out")

    def close(self):
        if self.closed:
            return
        try:
            self.flush()
            self.queue.put(None, timeout=30)
            self.thread.join(30)
            if self.thread.is_alive():
                raise TimeoutError("Duru event sender did not stop")
            self.check()
            self.proc.stdin.close()
            if self.proc.wait(timeout=30):
                raise RuntimeError("Duru event writer failed on close")
        finally:
            self.closed = True
            if self.proc.poll() is None:
                self.proc.kill()
                self.proc.wait(timeout=5)
            self.thread.join(5)
            self.proc.stdin.close()
            self.proc.stdout.close()
            atexit.unregister(self.close)


def open_journal(directory, *, name, fresh=True):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.jsonl"
    if fresh:
        # Each producer owns one immutable sequence, including across restarts.
        path.touch(exist_ok=False)
    return AsyncJournal(path)


def _read_exact(source, size):
    chunks = []
    while size:
        chunk = source.read(size)
        if not chunk:
            raise EOFError("Truncated Duru event frame")
        chunks.append(chunk)
        size -= len(chunk)
    return b"".join(chunks)


def _write_events(path):
    # Unbuffered reads keep select from overlooking prefetched frames.
    with (
        os.fdopen(os.dup(0), "rb", buffering=0) as source,
        Path(path).open("a") as output,
    ):
        last_flush = time.monotonic()
        while True:
            if not select.select([source], [], [], 0.1)[0]:
                output.flush()
                continue
            first = source.read(1)
            if not first:
                break
            size = struct.unpack("!Q", first + _read_exact(source, 7))[0]
            value = pickle.loads(_read_exact(source, size))
            if value is None:
                output.flush()
                sys.stdout.buffer.write(b"OK\n")
                sys.stdout.buffer.flush()
            else:
                output.write(json.dumps(value, separators=(",", ":")) + "\n")
            if time.monotonic() - last_flush >= 0.1:
                output.flush()
                last_flush = time.monotonic()


def open_feed(directory, *, name):
    if transport() == "journal":
        return open_journal(directory, name=name)
    recording = (
        open_journal(directory, name=name)
        if os.environ.get("DURU_RECORD_EVENTS") == "1"
        else None
    )
    try:
        return Sender(
            channel_path(Path(directory) / f"{name}.jsonl"), recording=recording
        )
    except BaseException:
        if recording is not None:
            recording.close()
        raise


class CompactFeed:
    def __init__(self, directory):
        self.lock = threading.RLock()
        self.stop = threading.Event()
        self.error = None
        self.sequence = 0
        self.journal = open_feed(directory, name="compact-work")
        self.emit("header", schema=1)
        self.thread = threading.Thread(
            target=self._clock, name="compact-U-clock", daemon=True
        )
        self.thread.start()

    def emit(self, event, **fields):
        with self.lock:
            if self.error:
                raise RuntimeError("Compact live feed failed") from self.error
            self.sequence += 1
            self.journal.submit(
                "record",
                dict(sequence=self.sequence, event=event, time=time.time(), **fields),
            )

    def capture(self, event, fields):
        if event != "accepted_output":
            return
        # Immutable small records only: no token arrays, cache state or runtime strings.
        steps = [
            {k: s[k] for k in FIELDS if k in s}
            for s in fields["steps"]
            if s["accepted"]
        ]
        self.emit(event, steps=steps)

    def _clock(self):
        try:
            while not self.stop.wait(0.1):
                self.emit("u_clock")
        except BaseException as exc:
            self.error = exc

    def close(self):
        self.stop.set()
        self.thread.join(timeout=2)
        try:
            self.emit("footer")
        finally:
            self.journal.close()


if __name__ == "__main__":
    _write_events(sys.argv[1])
