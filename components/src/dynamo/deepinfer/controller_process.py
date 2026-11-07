# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The lockstep controller in a process of its own, next to the Dynamo worker.

    python -m dynamo.deepinfer.controller_process --endpoint ipc://PATH \\
        --parent-pid PID -- HEAD_ARGV...

With ``--no-rank0-in-process`` the head (the Dynamo worker) runs no rank, and
its Python serves requests and streams outputs. A controller thread there
shares that GIL, so its Cmds go out late and staggered and every rank waits.
Instead the head starts this process before the Rust worker and Tokio exist
(like rank 0). It reads the DeepInfer options from the head's command line,
starts the ranks (``launch_ranks``), owns their sockets and runs the
``Controller``.

One ZMQ ``PAIR`` over ``ipc://`` links it to the head. The head sends the
controller's inbox items (``Submit``, ``Abort``, ``Release``, ``Close``). The
controller sends ``Started`` (or ``Failed``) once the ranks are ready, then
per iteration one ``Outputs`` message carrying every rank's outputs as the
rank encoded them (it decodes only the reply headers), and ``Stats`` once per
snapshot interval. The head decodes an iteration's outputs once and hands
them to its event loop in one call.

If the controller dies, the head exits (``exit_process``). If the head dies,
the controller exits within ``ChildWatch``'s interval, and its ranks with it
(``PR_SET_PDEATHSIG``). Like the ranks it ignores SIGINT/SIGTERM: the head's
CLOSE ends it. Must not import ``dynamo._core`` or ``dynamo.common.backend``.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import queue
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Sequence
from typing import Any, Optional, Union

import msgspec
import zmq

from dynamo.deepinfer.controller import (
    Abort,
    Close,
    Controller,
    InboxItem,
    OutputSink,
    Release,
    StatsSink,
    Submit,
    _percentiles,
    exit_process,
)
from dynamo.deepinfer.launch import launch_ranks
from dynamo.deepinfer.options import DeepInferOptions, add_deepinfer_arguments
from dynamo.deepinfer.protocol import (
    AddReq,
    EngineMeta,
    Output,
    RankFailed,
    RankStats,
    child_env,
    decode_outputs,
    encode,
    trace_lockstep,
)

logger = logging.getLogger(__name__)


class Started(msgspec.Struct, tag="started", array_like=True):
    meta: EngineMeta
    dp_size: int
    # dp rank -> KV event endpoint of a rank on another node.
    kv_endpoints: dict[int, str] = {}


class Failed(msgspec.Struct, tag="failed", array_like=True):
    error: str


class Outputs(msgspec.Struct, tag="outputs", array_like=True):
    # When the controller sent it (time.time()): how far the head lags.
    sent: float
    # Per rank with outputs, its msgpack array of ``Output`` as it encoded it.
    # One frame: every frame a thread receives retakes the head's GIL.
    outputs: list[msgspec.Raw]


class Stats(msgspec.Struct, tag="stats", array_like=True):
    stats: list[RankStats]


_TO_HEAD = msgspec.msgpack.Decoder(Union[Started, Failed, Outputs, Stats])
_TO_CONTROLLER = msgspec.msgpack.Decoder(InboxItem)


def _pair(endpoint: str, bind: bool, ctx: Optional[zmq.Context] = None) -> zmq.Socket:
    sock = (ctx or zmq.Context.instance()).socket(zmq.PAIR)
    sock.setsockopt(zmq.LINGER, 0)
    # Never block either side: a slow head queues outputs, it does not stall
    # the lockstep.
    sock.setsockopt(zmq.SNDHWM, 0)
    sock.setsockopt(zmq.RCVHWM, 0)
    if bind:
        sock.bind(endpoint)
    else:
        sock.connect(endpoint)
    return sock


class ControllerProcess:
    """The head's side: what the engine uses of :class:`Controller`.

    One thread (started by :meth:`start`) owns the socket: it sends what the
    other threads queue and wakes it for (an eventfd), receives, decodes and
    emits outputs, and watches the controller process.
    """

    _POLL_MS = 500

    def __init__(
        self,
        proc: subprocess.Popen,
        sock: zmq.Socket,
        ipc_dir: str,
        started: Started,
        on_fatal: Callable[[str], None],
    ) -> None:
        self.proc = proc
        self.ipc_dir = ipc_dir
        self.meta = started.meta
        self.dp_size = started.dp_size
        self.kv_endpoints = started.kv_endpoints
        # Called ~once per snapshot interval per rank from the link thread.
        self.on_stats: Optional[StatsSink] = None
        self._sock = sock
        self._on_fatal = on_fatal
        self._stats = [RankStats() for _ in range(self.dp_size)]
        # (when queued, message) for the link thread to send.
        self._outbox: queue.SimpleQueue[tuple[float, bytes]] = queue.SimpleQueue()
        self._wake = os.eventfd(0, os.EFD_NONBLOCK | os.EFD_CLOEXEC)
        self._emit: Optional[OutputSink] = None
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._closed = False
        self._failed = False
        # Set by close(): the controller exits from then on, and takes no
        # more items.
        self._closing = False
        self._trace = LinkTrace() if trace_lockstep() else None

    @classmethod
    def spawn(
        cls,
        argv: Sequence[str],
        startup_timeout_s: float,
        on_fatal: Callable[[str], None] = exit_process,
    ) -> ControllerProcess:
        """Start the controller for the head's command line ``argv``; return
        once every rank is ready (or raise ``RankFailed``)."""
        ipc_dir = tempfile.mkdtemp(prefix="dynamo-deepinfer-head-")
        endpoint = f"ipc://{os.path.join(ipc_dir, 'controller.sock')}"
        sock = _pair(endpoint, bind=True)
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "dynamo.deepinfer.controller_process",
                "--endpoint",
                endpoint,
                "--parent-pid",
                str(os.getpid()),
                "--",
                *argv,
            ],
            env=child_env(),
        )
        logger.info("Spawned the DeepInfer controller (pid %d)", proc.pid)
        try:
            started = cls._wait_started(proc, sock, startup_timeout_s)
        except BaseException:
            if proc.poll() is None:
                proc.kill()
            proc.wait()
            sock.close(linger=0)
            shutil.rmtree(ipc_dir, ignore_errors=True)
            raise
        return cls(proc, sock, ipc_dir, started, on_fatal)

    @classmethod
    def _wait_started(
        cls, proc: subprocess.Popen, sock: zmq.Socket, timeout: float
    ) -> Started:
        # The controller bounds the ranks' startup itself; this bounds a
        # controller that hangs.
        deadline = time.monotonic() + timeout + 60.0
        while not sock.poll(cls._POLL_MS):
            code = proc.poll()
            if code is not None:
                raise RankFailed(f"the DeepInfer controller exited with code {code}")
            if time.monotonic() > deadline:
                raise RankFailed(
                    f"the DeepInfer controller did not start in {timeout}s"
                )
        msg = _TO_HEAD.decode(sock.recv())
        if isinstance(msg, Failed):
            raise RankFailed(msg.error)
        if not isinstance(msg, Started):
            raise RankFailed(f"the DeepInfer controller sent {type(msg).__name__}")
        return msg

    # -- the Controller interface ---------------------------------------------

    def start(self, emit: OutputSink) -> None:
        self._emit = emit
        self._thread = threading.Thread(
            target=self._run, name="di-controller-link", daemon=True
        )
        self._thread.start()

    def submit(self, add: AddReq, dp_rank: Optional[int] = None) -> None:
        self._send(Submit(add, dp_rank))

    def abort(self, request_id: str) -> None:
        self._send(Abort(request_id))

    def release(self, handoff: dict[str, Any], rank: Optional[int] = None) -> None:
        self._send(Release(handoff, rank))

    def stats(self) -> list[RankStats]:
        return list(self._stats)

    def close(self, timeout: float = 60.0) -> None:
        """Have the controller run the CLOSE iteration and wait for it to
        exit. Idempotent; works before :meth:`start`."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
        try:
            if self.proc.poll() is None:
                self._closing = True
                self._post(encode(Close(timeout)))
                # The controller bounds the CLOSE iteration by ``timeout``.
                try:
                    code = self.proc.wait(timeout + 10.0)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
                    code = self.proc.wait()
                if code != 0:
                    self._fail(f"the DeepInfer controller exited with code {code}")
            else:
                self._fail(
                    f"the DeepInfer controller exited with code {self.proc.returncode}"
                )
        finally:
            if self._thread is not None:
                self._thread.join(2 * self._POLL_MS / 1000)
            self._sock.close(linger=0)
            os.close(self._wake)

    # -- the link thread --------------------------------------------------------

    def _send(self, item: InboxItem) -> None:
        if not self._closing:
            self._post(encode(item))

    def _post(self, data: bytes) -> None:
        if self._thread is None:
            # Not started: the calling thread is the socket's only user.
            self._sock.send(data)
        else:
            self._outbox.put((time.monotonic(), data))
            os.eventfd_write(self._wake, 1)

    def _run(self) -> None:
        poller = zmq.Poller()
        poller.register(self._sock, zmq.POLLIN)
        poller.register(self._wake, zmq.POLLIN)
        try:
            while True:
                ready = dict(poller.poll(self._POLL_MS))
                if self._wake in ready:
                    os.eventfd_read(self._wake)
                    self._flush()
                if self._sock in ready:
                    self._receive()
                code = self.proc.poll()
                if code is not None:
                    self._receive()  # what it sent before it exited
                    if not self._closing or code != 0:
                        self._fail(f"the DeepInfer controller exited with code {code}")
                    return
        except Exception as e:
            logger.exception("DeepInfer controller link failed")
            self._fail(repr(e))

    def _flush(self) -> None:
        while True:
            try:
                queued, data = self._outbox.get_nowait()
            except queue.Empty:
                return
            self._sock.send(data)
            if self._trace is not None:
                self._trace.waited(time.monotonic() - queued)

    def _receive(self) -> None:
        while True:
            # Between any two messages: a head that lags the controller never
            # finds the socket empty, and must not hold back submits until it
            # does.
            self._flush()
            try:
                data = self._sock.recv(zmq.NOBLOCK)
            except zmq.Again:
                return
            msg = _TO_HEAD.decode(data)
            if isinstance(msg, Outputs):
                if self._trace is not None:
                    self._trace.lagged(time.time() - msg.sent)
                merged: dict[str, Output] = {}
                for raw in msg.outputs:
                    for out in decode_outputs(raw):
                        merged[out.request_id] = out
                if merged and self._emit is not None:
                    self._emit(merged)
            elif isinstance(msg, Stats):
                self._stats = msg.stats
                if self.on_stats is not None:
                    for rank, stats in enumerate(msg.stats):
                        self.on_stats(rank, stats)
            else:
                logger.warning("Ignoring %s from the DeepInfer controller", msg)

    def _fail(self, reason: str) -> None:
        with self._lock:
            first, self._failed = not self._failed, True
        if first:
            self._on_fatal(reason)


class LinkTrace:
    """``DYN_DEEPINFER_TRACE_LOCKSTEP=1`` on the head: every ``interval_s``,
    how long after the controller sent them the head handled an iteration's
    outputs, and how long submits and aborts waited to go out."""

    def __init__(self, interval_s: float = 10.0) -> None:
        self.interval_s = interval_s
        self._since = time.monotonic()
        self._lags: list[float] = []
        self._waits: list[float] = []

    def waited(self, seconds: float) -> None:
        self._waits.append(seconds * 1e3)

    def lagged(self, seconds: float) -> None:
        self._lags.append(seconds * 1e3)
        now = time.monotonic()
        if now - self._since >= self.interval_s:
            logger.info(
                "lockstep link messages=%d lag_ms p50=%.1f p99=%.1f max=%.1f "
                "sends=%d send_wait_ms p50=%.2f p99=%.2f max=%.2f",
                len(self._lags),
                *_percentiles(self._lags, (50, 99, 100)),
                len(self._waits),
                *_percentiles(self._waits, (50, 99, 100)),
            )
            self._lags, self._waits, self._since = [], [], now


class _HeadInbox:
    """A controller process's inbox: the head's socket."""

    def __init__(self, sock: zmq.Socket) -> None:
        self._sock = sock

    def get(self, timeout: float) -> InboxItem:
        if not self._sock.poll(max(1, math.ceil(timeout * 1000))):
            raise queue.Empty
        return self.get_nowait()

    def get_nowait(self) -> InboxItem:
        try:
            data = self._sock.recv(zmq.NOBLOCK)
        except zmq.Again:
            raise queue.Empty from None
        return _TO_CONTROLLER.decode(data)


def head_options(argv: Sequence[str]) -> DeepInferOptions:
    """The DeepInfer options of the head's command line (its Dynamo flags are
    ignored)."""
    parser = argparse.ArgumentParser(allow_abbrev=False)
    add_deepinfer_arguments(parser)
    ns, _ = parser.parse_known_args(argv)
    return DeepInferOptions.from_namespace(ns)


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    split = argv.index("--") if "--" in argv else len(argv)
    parser = argparse.ArgumentParser(description="DeepInfer lockstep controller")
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--parent-pid", type=int, required=True)
    args = parser.parse_args(argv[:split])
    head_argv = argv[split + 1 :]
    # A group-wide SIGINT/SIGTERM must not stop the lockstep: the head shuts
    # it down with its CLOSE iteration.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    # DYN_LOG may carry per-target filters ("info,dynamo=debug"); use the default.
    level = os.environ.get("DYN_LOG", "info").split(",")[0].upper()
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s %(levelname)s [deepinfer controller] %(name)s: %(message)s",
    )
    if os.getppid() != args.parent_pid:
        return 1  # the head is gone already
    ctx = zmq.Context()
    sock = _pair(args.endpoint, bind=False, ctx=ctx)
    try:
        opts = head_options(head_argv)
        launch = launch_ranks(opts, raw_outputs=True, parent_pid=args.parent_pid)
    except Exception as e:
        logger.exception("DeepInfer ranks failed to start")
        sock.send(encode(Failed(str(e) or repr(e))))
        sock.close(linger=5000)
        ctx.term()
        return 1
    sock.send(encode(Started(launch.meta, len(launch.handles), launch.kv_endpoints)))

    def forward(frames: list[Any]) -> None:
        sock.send(encode(Outputs(time.time(), frames)))

    def send_stats(rank: int, stats: RankStats) -> None:
        # Called for every rank in turn: send them together, after the last.
        if rank == controller.dp_size - 1:
            sock.send(encode(Stats(controller.stats())))

    controller = Controller(
        launch.handles,
        launch.meta,
        engine_type=opts.engine_type,
        on_stats=send_stats,
        inbox=_HeadInbox(sock),
        raw_outputs=True,
    )
    controller.run(forward)  # until the head's CLOSE
    controller.close(controller.close_timeout or 60.0)
    launch.close()
    sock.close(linger=5000)  # flush the CLOSE iteration's outputs
    ctx.term()
    return 0


if __name__ == "__main__":
    sys.exit(main())
