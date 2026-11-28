# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Local IPC for accounting events and current request scores.

Events use a bounded, ordered queue: every accepted-work and lifecycle record
must reach accounting. Scores retain only the latest snapshot, which engine
threads refresh in the background without blocking scheduling on socket I/O.
"""

import hashlib
import math
import os
import pickle
import queue
import socket
import struct
import threading
from collections import deque
from multiprocessing.connection import Connection
from pathlib import Path

CAPACITY = 16 * 1024 * 1024
MAX_FRAME = 2 * 1024 * 1024
_LENGTH = struct.Struct("!I")
_PEERCRED = struct.Struct("iII")


def transport():
    value = os.environ.get("DURU_ACCOUNTING_TRANSPORT", "memory")
    if value not in ("memory", "journal"):
        raise ValueError("DURU_ACCOUNTING_TRANSPORT must be memory or journal")
    return value


def channel_path(logical_path):
    directory = Path(os.environ.get("DURU_IPC_DIR", f"/tmp/duru-{os.getuid()}"))
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    if directory.stat().st_uid != os.getuid() or directory.stat().st_mode & 0o077:
        raise ValueError("Duru IPC directory must be private to its owner")
    name = hashlib.sha256(str(Path(logical_path).absolute()).encode()).hexdigest()[:32]
    path = directory / (name + ".sock")
    if len(os.fsencode(path)) >= 108:
        raise ValueError(
            "DURU_IPC_DIR is too long for a Unix socket; use a short local path"
        )
    return path


class InProcessChannel:
    """Ordered router events without a socket or a sender thread.

    Keep Sender's record/byte limits and serialized value snapshots. Closing the
    producer rejects new records but lets accounting drain the accepted records,
    including its footer, before the accounting thread stops.
    """

    def __init__(self, capacity=CAPACITY, max_records=8192, recording=None):
        self.capacity = capacity
        self.max_records = max_records
        self.recording = recording
        self.pending_bytes = 0
        self.queue = deque()
        self.lock = threading.Lock()
        self.closed = False

    def submit(self, kind, value):
        if kind != "record":
            raise ValueError("Unsupported Duru event message: " + kind)
        frame = pickle.dumps(value, protocol=5)
        if len(frame) > MAX_FRAME:
            raise RuntimeError("Duru accounting frame exceeds memory limit")
        with self.lock:
            if self.closed:
                raise RuntimeError("Duru event transport is closed")
            if self.pending_bytes + len(frame) > self.capacity:
                raise RuntimeError(
                    "Duru accounting buffer overflow; scorer must catch up"
                )
            if len(self.queue) >= self.max_records:
                raise RuntimeError("Duru accounting queue overflow")
            if self.recording is not None:
                self.recording.submit(kind, value)
            self.queue.append(frame)
            self.pending_bytes += len(frame)

    def read(self, budget=4 * 1024 * 1024):
        frames = []
        size = 0
        with self.lock:
            while self.queue and size < budget:
                frame = self.queue.popleft()
                frames.append(frame)
                size += len(frame)
            self.pending_bytes -= size
        return [pickle.loads(frame) for frame in frames]

    def close(self):
        with self.lock:
            if self.closed:
                return
            self.closed = True
        if self.recording is not None:
            self.recording.close()


class Sender:
    """Serialize captured batches, then send them outside the engine callback."""

    def __init__(self, path, capacity=CAPACITY, recording=None):
        self.recording = recording
        self.capacity = capacity
        self.pending_bytes = 0
        self.lock = threading.Lock()
        self.queue = queue.Queue(maxsize=8192)
        self.closed = False
        self._closing = False
        self._close_lock = threading.Lock()
        self.error = None
        self.peer = None
        self.listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        bound = False
        try:
            self.listener.bind(str(path))
            bound = True
            self.listener.listen(1)
            self.listener.settimeout(0.1)
            self.thread = threading.Thread(
                target=self._send, name="duru-events", daemon=True
            )
            self.thread.start()
        except BaseException:
            self.listener.close()
            if bound:
                Path(path).unlink(missing_ok=True)
            raise

    def check(self):
        if self.error is not None:
            raise RuntimeError("Duru event transport failed") from self.error
        if self.closed:
            raise RuntimeError("Duru event transport is closed")

    def submit(self, kind, value):
        if kind != "record":
            raise ValueError("Unsupported Duru event message: " + kind)
        frame = pickle.dumps(value, protocol=5)
        if len(frame) > MAX_FRAME:
            raise RuntimeError("Duru accounting frame exceeds memory limit")
        with self.lock:
            self.check()
            if self._closing:
                raise RuntimeError("Duru event transport is closing")
            if self.pending_bytes + len(frame) > self.capacity:
                raise RuntimeError(
                    "Duru accounting buffer overflow; scorer must catch up"
                )
            # Other submitters and close are gated by this lock; only draining can race.
            if self.queue.full():
                raise RuntimeError("Duru accounting queue overflow")
            if self.recording is not None:
                self.recording.submit(kind, value)
            self.queue.put_nowait(frame)
            self.pending_bytes += len(frame)

    def _send(self):
        connection = None
        try:
            while not self.closed:
                try:
                    self.peer, _ = self.listener.accept()
                    break
                except socket.timeout:
                    continue
            if self.peer is None:
                return
            # Connection provides the standard-library length-framed protocol.
            self.listener.close()
            connection = Connection(os.dup(self.peer.fileno()))
            while not self.closed:
                frame = self.queue.get()
                if frame is None:
                    return
                connection.send_bytes(frame)
                with self.lock:
                    self.pending_bytes -= len(frame)
        except Exception as exc:
            self.error = exc
        finally:
            if connection is not None:
                connection.close()
            if self.peer is not None:
                self.peer.close()

    def close(self):
        with self._close_lock:
            with self.lock:
                if self.closed:
                    return
                self._closing = True
            try:
                self.check()
                # A full queue must not keep the sender from updating pending_bytes.
                self.queue.put(None, timeout=5)
                self.thread.join(5)
                if self.thread.is_alive():
                    raise TimeoutError("Duru event consumer did not drain before close")
                self.check()
            finally:
                with self.lock:
                    self.closed = True
                if self.peer is not None:
                    try:
                        self.peer.shutdown(socket.SHUT_RDWR)
                    except OSError:
                        pass  # Already closed by the sender thread.
                self.listener.close()
                self.thread.join(1)
                if self.recording is not None:
                    self.recording.close()
                # Keep the pathname to reject accidental reuse within this session.
                # The launcher removes the run's directory after participants exit.


class Receiver:
    """Read Connection-framed events without waiting for a partial frame."""

    def __init__(self, path):
        self.peer = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            self.peer.connect(str(path))
            self.peer.setblocking(False)
        except BaseException:
            self.peer.close()
            raise
        self._buffer = bytearray()
        self._header_size = 4
        self._frame_size = None
        self._error = None

    def _read_frame(self):
        while True:
            needed = self._header_size if self._frame_size is None else self._frame_size
            if len(self._buffer) < needed:
                try:
                    chunk = self.peer.recv(needed - len(self._buffer))
                except BlockingIOError:
                    return None
                if not chunk:
                    if (
                        self._buffer
                        or self._frame_size is not None
                        or self._header_size != 4
                    ):
                        raise EOFError(
                            "Duru event producer disconnected during a frame"
                        )
                    raise EOFError("Duru event producer disconnected")
                self._buffer.extend(chunk)
                continue
            if self._frame_size is None:
                # Match Connection.send_bytes, including its extended length header.
                length = struct.unpack(
                    "!i" if self._header_size == 4 else "!Q", self._buffer
                )[0]
                self._buffer.clear()
                if self._header_size == 4 and length == -1:
                    self._header_size = 8
                    continue
                self._header_size = 4
                if not 0 <= length <= MAX_FRAME:
                    raise ValueError("Invalid Duru accounting frame length")
                self._frame_size = length
                continue
            frame = self._buffer
            self._buffer = bytearray()
            self._frame_size = None
            return frame

    def read(self, budget=4 * 1024 * 1024):
        if self._error is not None:
            raise self._error
        rows = []
        size = 0
        try:
            while size < budget:
                frame = self._read_frame()
                if frame is None:
                    break
                size += len(frame)
                row = pickle.loads(frame)
                if not isinstance(row, dict):
                    raise ValueError("Duru event frame must contain a record")
                rows.append(row)
                if row.get("event") == "footer":
                    break
        except (
            EOFError,
            OSError,
            ValueError,
            OverflowError,
            pickle.UnpicklingError,
        ) as error:
            self._error = error
            self.close()
            if not rows:
                raise
            # Deliver the valid prefix before reporting failure on the next read.
        return rows

    def close(self):
        self.peer.close()
        self._buffer.clear()


def _receive(peer, size):
    result = bytearray()
    while len(result) < size:
        part = peer.recv(size - len(result))
        if not part:
            raise EOFError("Score publisher disconnected")
        result.extend(part)
    return bytes(result)


class ScorePublisher:
    def __init__(self, path):
        self._path = Path(path)
        self._frame = pickle.dumps({}, protocol=5)
        self._stop = threading.Event()
        self._error = None
        self._listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        bound = False
        try:
            self._listener.bind(str(path))
            bound = True
            self._listener.listen(16)
            self._listener.settimeout(0.1)
            self._thread = threading.Thread(
                target=self._serve, name="duru-score-publisher", daemon=True
            )
            self._thread.start()
        except BaseException:
            self._listener.close()
            if bound:
                self._path.unlink(missing_ok=True)
            raise

    def publish(self, snapshot):
        if self._error is not None:
            raise RuntimeError("Worker score publisher failed") from self._error
        # Engine queues only need request existence, score, and observation time.
        # Full E/W/U and program state stay in the router, not in every engine.
        worker = (
            dict(
                schema=1,
                run_id=snapshot.get("run_id"),
                observation_time=snapshot.get("observation_time", 0),
                requests={
                    rid: {"score": row["score"]}
                    for rid, row in snapshot.get("requests", {}).items()
                },
            )
            if snapshot
            else {}
        )
        frame = pickle.dumps(worker, protocol=5)
        if len(frame) > MAX_FRAME:
            raise RuntimeError("Worker score snapshot exceeds IPC memory limit")
        self._frame = frame

    def _serve(self):
        try:
            while not self._stop.is_set():
                try:
                    peer, _ = self._listener.accept()
                except socket.timeout:
                    continue
                with peer:
                    peer.settimeout(0.2)
                    frame = self._frame
                    try:
                        peer.sendall(_LENGTH.pack(len(frame)) + frame)
                    except OSError:
                        # A disconnected/slow engine gets a fresh snapshot next time.
                        continue
        except OSError as exc:
            if not self._stop.is_set():
                self._error = exc

    def close(self):
        self._stop.set()
        self._listener.close()
        self._thread.join(1)
        if self._thread.is_alive():
            raise TimeoutError("Worker score publisher did not stop")
        self._path.unlink(missing_ok=True)
        self._frame = b""


def _finite_number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _validate_score_snapshot(snapshot):
    if not isinstance(snapshot, dict):
        raise ValueError("Invalid worker score snapshot")
    if not snapshot:
        return
    if not _finite_number(snapshot.get("observation_time")):
        raise ValueError("Invalid worker score observation time")
    requests = snapshot.get("requests")
    if not isinstance(requests, dict):
        raise ValueError("Invalid worker score requests")
    for request_id, row in requests.items():
        if (
            not isinstance(request_id, str)
            or not request_id
            or not isinstance(row, dict)
            or "score" not in row
        ):
            raise ValueError("Invalid worker request score")
        score = row["score"]
        if score is not None and (not _finite_number(score) or score < 0):
            raise ValueError("Invalid worker request score")


class ScoreSubscriber:
    def __init__(self, path):
        self._path = str(path)
        self._snapshot = {}
        self._stop = threading.Event()
        self.failures = 0
        self._thread = threading.Thread(
            target=self._refresh, name="duru-worker-scores", daemon=True
        )
        self._thread.start()

    def snapshot(self):
        return self._snapshot

    def _refresh(self):
        while not self._stop.is_set():
            try:
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as peer:
                    peer.settimeout(0.2)
                    peer.connect(self._path)
                    # Authenticate each connected peer before decoding pickle data.
                    credentials = peer.getsockopt(
                        socket.SOL_SOCKET, socket.SO_PEERCRED, _PEERCRED.size
                    )
                    _, uid, _ = _PEERCRED.unpack(credentials)
                    if uid != os.getuid():
                        raise PermissionError(
                            "Duru score server belongs to another user"
                        )
                    size = _LENGTH.unpack(_receive(peer, _LENGTH.size))[0]
                    if size > MAX_FRAME:
                        raise ValueError(
                            "Worker score snapshot exceeds IPC memory limit"
                        )
                    snapshot = pickle.loads(_receive(peer, size))
                    _validate_score_snapshot(snapshot)
                    self._snapshot = snapshot
            except (
                OSError,
                EOFError,
                ValueError,
                OverflowError,
                pickle.UnpicklingError,
            ):
                self.failures += 1
                self._snapshot = {}
            self._stop.wait(0.02)

    def close(self):
        self._stop.set()
        self._thread.join(1)
        if self._thread.is_alive():
            raise TimeoutError("Worker score subscriber did not stop")
        self._snapshot = {}
