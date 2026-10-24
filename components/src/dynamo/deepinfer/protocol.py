# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Controller <-> rank lockstep protocol and rank handles.

Every iteration the controller sends one ``Cmd`` to every rank and waits for
one ``Reply`` from each. Rank 0 may run in the controller's process
(``LocalRankHandle``); the other ranks on the head's node are child processes
reached over a ZMQ ``PAIR`` socket on ``ipc://`` (``ZmqRankHandle``), and ranks
on other nodes connect to the head's ``ControlServer`` over TCP
(``RemoteRankHandle``). The ``Cmd`` carries the iteration's global flags
(``Sync``), so ranks need no all-reduce to agree on them.

Child processes import this module, so it must not import ``dynamo._core``,
``dynamo.common.backend``, torch or DeepInfer.
"""

from __future__ import annotations

import enum
import logging
import os
import queue
import subprocess
import sys
import threading
import time
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Union

import msgspec
import zmq
from zmq.utils.monitor import recv_monitor_message

if TYPE_CHECKING:
    from dynamo.deepinfer.rank import RankRunner, RankSpec

logger = logging.getLogger(__name__)


class Op(enum.IntEnum):
    STEP = 0
    CLOSE = 1


class AddReq(msgspec.Struct):
    request_id: str
    # Prompt as little-endian int32 bytes (cheaper than a msgpack int list).
    token_ids: bytes
    # deepinfer SamplingParams kwargs.
    sampling: dict[str, Any] = {}


class Sync(msgspec.Struct):
    """The global facts of one iteration, the same on every rank of the engine.

    The controller sends them in ``Cmd.sync``; without them every rank
    all-reduces (MAX) its local values.
    """

    # Some rank has unfinished requests: every rank steps, else flushes.
    unfinished: bool = False
    # Some rank got CLOSE: every rank runs DeepInfer's (collective) close().
    close: bool = False
    # A prompt entered some rank's scheduler: NORMAL engines make the next
    # step an ANY prefill vote point on every rank.
    new_prompts: bool = False
    # Some rank has requests outside the scheduler (awaiting KV admission):
    # the controller keeps polling instead of idling.
    pending: bool = False


class Cmd(msgspec.Struct):
    seq: int
    op: int = Op.STEP
    adds: list[AddReq] = []
    aborts: list[str] = []
    # This iteration's global flags; ranks skip the all-reduce.
    sync: Optional[Sync] = None


class Output(msgspec.Struct):
    """Copied step output for one request (DeepInfer's pointers die next step)."""

    request_id: str
    token_ids: list[int] = []
    logprobs: list[float] = []
    top_token_ids: list[list[int]] = []
    top_logprobs: list[list[float]] = []
    # None while running; else a Dynamo finish reason (stop/length/cancelled/error).
    finish_reason: Optional[str] = None


class RankStats(msgspec.Struct):
    num_running: int = 0
    num_waiting: int = 0
    # Requests outside the scheduler: awaiting KV admission.
    num_pending: int = 0
    kv_used_blocks: int = 0
    kv_total_blocks: int = 0


class EngineMeta(msgspec.Struct):
    block_size: int
    num_gpu_blocks: int
    max_num_seqs: int
    max_model_len: int
    max_num_batched_tokens: int
    prefill_any_attempt_interval: int
    bos_token_id: int = 0


class Ready(msgspec.Struct, tag="ready"):
    rank: int
    meta: EngineMeta
    # Ranks on other nodes: their host, a digest of their RankSpec (every
    # node must agree) and their DEEPINFER_* environment (the head refuses a
    # rank that differs in LOCKSTEP_ENV and warns about other differences).
    host: Optional[str] = None
    spec_digest: Optional[str] = None
    env: Optional[dict[str, str]] = None


class Reply(msgspec.Struct, tag="reply"):
    seq: int
    rank: int
    # The global unfinished flag this iteration used (stepped or flushed).
    u: bool = False
    outputs: list[Output] = []
    stats: RankStats = msgspec.field(default_factory=RankStats)
    error: Optional[str] = None
    # This rank ran DeepInfer's close() because the global close flag was set.
    closed: bool = False
    # This rank's own unfinished requests after its step (a central head's
    # next ``Sync.unfinished``).
    unfinished: bool = False
    # This rank's own requests outside the scheduler.
    pending: bool = False
    # Requests that finished in this iteration (their owner can be forgotten).
    finished: list[str] = []


class RawReply(msgspec.Struct, tag="reply"):
    """A ``Reply`` read by a controller in its own process: the outputs stay
    as the rank encoded them (a msgpack array of ``Output``) and go to the
    head undecoded. Same fields as ``Reply``."""

    seq: int
    rank: int
    u: bool = False
    outputs: msgspec.Raw = msgspec.Raw()
    stats: RankStats = msgspec.field(default_factory=RankStats)
    error: Optional[str] = None
    closed: bool = False
    unfinished: bool = False
    pending: bool = False
    finished: list[str] = []


# What a controller receives per iteration: decoded in the Dynamo worker's
# process, raw in a controller process of its own.
AnyReply = Union[Reply, RawReply]

_ENCODER = msgspec.msgpack.Encoder()
_CMD_DECODER = msgspec.msgpack.Decoder(Cmd)
_UP_DECODER = msgspec.msgpack.Decoder(Union[Ready, Reply])
_UP_RAW_DECODER = msgspec.msgpack.Decoder(Union[Ready, RawReply])
_OUTPUTS_DECODER = msgspec.msgpack.Decoder(list[Output])


def encode(msg: msgspec.Struct) -> bytes:
    return _ENCODER.encode(msg)


def decode_cmd(data: bytes) -> Cmd:
    return _CMD_DECODER.decode(data)


def decode_up(data: bytes) -> Union[Ready, Reply]:
    return _UP_DECODER.decode(data)


def decode_up_raw(data: bytes) -> Union[Ready, RawReply]:
    return _UP_RAW_DECODER.decode(data)


def decode_outputs(data: Any) -> list[Output]:
    """A ``RawReply``'s outputs."""
    return _OUTPUTS_DECODER.decode(data)


def num_outputs(outputs: Any) -> int:
    """How many outputs a reply carries, decoded or not (read from the msgpack
    array header)."""
    if isinstance(outputs, list):
        return len(outputs)
    data = memoryview(outputs)
    if not data:
        return 0
    head = data[0]
    if head & 0xF0 == 0x90:
        return head & 0x0F
    if head in (0xDC, 0xDD):
        return int.from_bytes(data[1 : 3 if head == 0xDC else 5], "big")
    raise ValueError(f"outputs are not a msgpack array (0x{head:02x})")


def trace_lockstep() -> bool:
    """``DYN_DEEPINFER_TRACE_LOCKSTEP=1``: log per-iteration timestamps on every
    rank (Cmd received, flags agreed, step done) and the controller's period,
    send and receive spans, to tell a staggered head from a slow rank."""
    return os.environ.get("DYN_DEEPINFER_TRACE_LOCKSTEP", "0") not in (
        "0",
        "",
        "false",
        "False",
    )


class RankFailed(RuntimeError):
    """A rank died, timed out or reported an error; lockstep cannot continue."""


class RankHandle(ABC):
    """One DP rank as seen by the controller."""

    rank: int
    # Inline handles execute the command inside ``recv``; the controller must
    # receive from them first so peers are already inside the collectives.
    inline: bool = False

    @abstractmethod
    def wait_ready(self, timeout: Optional[float] = None) -> EngineMeta:
        ...

    def poll_ready(self) -> Optional[EngineMeta]:
        """Non-blocking :meth:`wait_ready`: None while the rank is starting."""
        return self.wait_ready(0.0)

    @abstractmethod
    def send(self, cmd: Cmd) -> None:
        ...

    @abstractmethod
    def recv(self, timeout: Optional[float] = None) -> AnyReply:
        ...

    @abstractmethod
    def close(self, timeout: float = 30.0) -> None:
        """Release the handle after the CLOSE iteration (or on failure)."""


class LocalRankHandle(RankHandle):
    """A rank driven in this process.

    Inline (default): ``recv`` applies the command on the calling thread; used
    for rank 0 on the controller thread. Threaded: a dedicated thread applies
    commands, so several in-process ranks can run collectives concurrently.
    """

    def __init__(self, runner: RankRunner, threaded: bool = False):
        self.runner = runner
        self.rank = runner.spec.rank
        self.inline = not threaded
        self._cmd: Optional[Cmd] = None
        self._thread: Optional[threading.Thread] = None
        self._cmds: queue.SimpleQueue[Optional[Cmd]] = queue.SimpleQueue()
        self._replies: queue.SimpleQueue[Reply] = queue.SimpleQueue()
        if threaded:
            self._thread = threading.Thread(
                target=self._serve, name=f"di-rank{self.rank}", daemon=True
            )
            self._thread.start()

    def wait_ready(self, timeout: Optional[float] = None) -> EngineMeta:
        return self.runner.meta

    def send(self, cmd: Cmd) -> None:
        if self._thread is not None:
            self._cmds.put(cmd)
        else:
            self._cmd = cmd

    def recv(self, timeout: Optional[float] = None) -> Reply:
        if self._thread is None:
            cmd, self._cmd = self._cmd, None
            if cmd is None:
                raise RankFailed(f"rank {self.rank}: recv without send")
            return self._apply(cmd)
        try:
            return self._replies.get(timeout=timeout)
        except queue.Empty:
            raise RankFailed(f"rank {self.rank}: no reply within {timeout}s") from None

    def _apply(self, cmd: Cmd) -> Reply:
        try:
            return self.runner.apply(cmd)
        except Exception as e:
            logger.exception("rank %d failed applying cmd %d", self.rank, cmd.seq)
            return Reply(seq=cmd.seq, rank=self.rank, error=repr(e))

    def _serve(self) -> None:
        while (cmd := self._cmds.get()) is not None:
            self._replies.put(self._apply(cmd))

    def close(self, timeout: float = 30.0) -> None:
        if self._thread is not None:
            self._cmds.put(None)
            self._thread.join(timeout)
            self._thread = None


def spawn_rank_process(
    spec: RankSpec,
    endpoint: str,
    env: Optional[dict[str, str]] = None,
    args: Sequence[str] = (),
) -> subprocess.Popen:
    """Start ``python -m dynamo.deepinfer.rank_worker`` for ``spec``, talking
    to the controller on ``endpoint``."""
    argv = [
        sys.executable,
        "-m",
        "dynamo.deepinfer.rank_worker",
        "--endpoint",
        endpoint,
        "--parent-pid",
        str(os.getpid()),
        "--spec",
        spec.to_json(),
        *args,
    ]
    proc = subprocess.Popen(argv, env=child_env(env))
    logger.info("Spawned DeepInfer rank %d (pid %d)", spec.rank, proc.pid)
    return proc


def child_env(env: Optional[dict[str, str]] = None) -> dict[str, str]:
    """This process's environment plus ``env``, with this checkout of
    ``dynamo.deepinfer`` importable."""
    child = dict(os.environ)
    child.update(env or {})
    src = str(Path(__file__).resolve().parents[2])
    child["PYTHONPATH"] = os.pathsep.join(
        p for p in (src, child.get("PYTHONPATH")) if p
    )
    return child


class ZmqRankHandle(RankHandle):
    """A rank in a ``python -m dynamo.deepinfer.rank_worker`` child process."""

    _POLL_S = 0.5

    def __init__(
        self,
        rank: int,
        proc: subprocess.Popen,
        sock: zmq.Socket,
        path: str,
        raw_outputs: bool = False,
    ):
        self.rank = rank
        self.proc = proc
        # Set once CLOSE went out: the child may exit from then on.
        self.closing = False
        self._sock = sock
        self._path = path
        # Replies as RawReply (outputs left encoded) instead of Reply.
        self._decode = decode_up_raw if raw_outputs else decode_up

    @classmethod
    def spawn(
        cls,
        spec: RankSpec,
        ipc_dir: str,
        env: Optional[dict[str, str]] = None,
        raw_outputs: bool = False,
    ) -> ZmqRankHandle:
        path = os.path.join(ipc_dir, f"rank{spec.rank}.sock")
        sock = zmq.Context.instance().socket(zmq.PAIR)
        sock.setsockopt(zmq.LINGER, 0)
        sock.bind(f"ipc://{path}")
        proc = spawn_rank_process(spec, f"ipc://{path}", env)
        return cls(spec.rank, proc, sock, path, raw_outputs)

    def _recv_msg(self, timeout: Optional[float]) -> Union[Ready, AnyReply]:
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            if self._sock.poll(int(self._POLL_S * 1000)):
                return self._decode(self._sock.recv())
            self._check_alive()
            if deadline is not None and time.monotonic() > deadline:
                raise RankFailed(f"rank {self.rank}: no reply within {timeout}s")

    def wait_ready(self, timeout: Optional[float] = None) -> EngineMeta:
        return self._ready(self._recv_msg(timeout))

    def poll_ready(self) -> Optional[EngineMeta]:
        # Check the exit first: what the child sent before exiting is queued.
        code = self.proc.poll()
        if self._sock.poll(0):
            return self._ready(self._decode(self._sock.recv()))
        if code is not None:
            raise RankFailed(f"rank {self.rank} exited with code {code}")
        return None

    def _ready(self, msg: Union[Ready, AnyReply]) -> EngineMeta:
        if not isinstance(msg, Ready):
            raise RankFailed(f"rank {self.rank} failed to start: {msg.error}")
        return msg.meta

    def send(self, cmd: Cmd) -> None:
        if cmd.op == Op.CLOSE:
            self.closing = True
        data = encode(cmd)
        # A PAIR socket blocks on send while its peer is gone; watch the child.
        while True:
            try:
                self._sock.send(data, zmq.NOBLOCK)
                return
            except zmq.Again:
                self._check_alive()
                self._sock.poll(int(self._POLL_S * 1000), zmq.POLLOUT)

    def _check_alive(self) -> None:
        code = self.proc.poll()
        if code is not None:
            raise RankFailed(f"rank {self.rank} exited with code {code}")

    def recv(self, timeout: Optional[float] = None) -> AnyReply:
        msg = self._recv_msg(timeout)
        if isinstance(msg, Ready):
            raise RankFailed(f"rank {self.rank}: unexpected {type(msg).__name__}")
        return msg

    def close(self, timeout: float = 30.0) -> None:
        self.closing = True
        try:
            self.proc.wait(timeout)
        except subprocess.TimeoutExpired:
            logger.warning("rank %d did not exit in %.0fs; killing", self.rank, timeout)
            self.proc.kill()
            self.proc.wait()
        if not self._sock.closed:
            self._sock.close(linger=0)
        try:
            os.unlink(self._path)
        except OSError:
            pass


_HANDSHAKE_FAILED = (
    zmq.EVENT_HANDSHAKE_FAILED_NO_DETAIL,
    zmq.EVENT_HANDSHAKE_FAILED_PROTOCOL,
    zmq.EVENT_HANDSHAKE_FAILED_AUTH,
)
CONNECTION_EVENTS = (
    zmq.EVENT_HANDSHAKE_SUCCEEDED
    | zmq.EVENT_DISCONNECTED
    | zmq.EVENT_HANDSHAKE_FAILED_NO_DETAIL
    | zmq.EVENT_HANDSHAKE_FAILED_PROTOCOL
    | zmq.EVENT_HANDSHAKE_FAILED_AUTH
)


# DeepInfer settings that change the KV block or state layout.
LAYOUT_ENV = (
    "DEEPINFER_USE_NEXTN_DECODE",
    "DEEPINFER_MAX_BACKTRACKING_TOKENS",
    "DEEPINFER_YOCO_PREFILL_TRUNCATION_MODE",
    "DEEPINFER_YOCO_PREFILL_KEEP_TOKENS",
    "DEEPINFER_PREFILL_STORE_EXTRA_STATE",
)
# Settings every rank of one engine must share: the KV layout and the step
# schedule. Each rank derives the steps that run a prefill vote (an
# all-reduce) from its own intervals (scheduler.hpp ``_schedule``), so ranks
# that disagree run mismatched collectives.
LOCKSTEP_ENV = frozenset(
    (
        *LAYOUT_ENV,
        "DEEPINFER_PREFILL_ATTEMPT_INTERVAL",
        "DEEPINFER_PREFILL_ANY_ATTEMPT_INTERVAL",
        "DEEPINFER_ZERO_OVERHEAD_SCHEDULING",
    )
)


# Set by DeepInfer itself (``EngineArgs.__post_init__``, ``os.environ.setdefault``)
# from the engine type when a rank builds its engine. A head that runs no rank
# (``--no-rank0-in-process``) never builds one, so it lacks them while every rank
# of the engine derives the same value; compare them only when the head set them.
ENGINE_DERIVED_ENV = frozenset(
    ("DEEPINFER_ZERO_OVERHEAD_SCHEDULING", "DEEPINFER_USE_MULTIMODAL_POOL")
)


def deepinfer_environ() -> dict[str, str]:
    """The ``DEEPINFER_*`` variables DeepInfer reads; ranks of one engine
    must agree on ``LOCKSTEP_ENV`` and should agree on the rest."""
    return {k: v for k, v in os.environ.items() if k.startswith("DEEPINFER_")}


def tcp_endpoint(host: str, port: int) -> str:
    return f"tcp://[{host}]:{port}" if ":" in host else f"tcp://{host}:{port}"


def routing_id(rank: int) -> bytes:
    """ZMQ routing id of a remote rank's DEALER socket."""
    return f"rank{rank}".encode()


def set_heartbeat(sock: zmq.Socket, timeout_s: float) -> None:
    """ZMTP heartbeats: drop a connection whose peer stopped answering (host
    down, network partition) after ``timeout_s``. A process that dies closes
    its TCP connection at once."""
    timeout_ms = max(100, int(timeout_s * 1000))
    sock.setsockopt(zmq.HEARTBEAT_IVL, min(1000, timeout_ms // 3))
    sock.setsockopt(zmq.HEARTBEAT_TIMEOUT, timeout_ms)
    sock.setsockopt(zmq.HEARTBEAT_TTL, timeout_ms)


class ConnectionEvents:
    """Reads a socket monitor (``CONNECTION_EVENTS``) for lost peers.

    A connection that completed the ZMTP handshake and then dropped is lost.
    One that never completed it (a port scanner, a probe) reports the failed
    handshake right before its disconnect; with one I/O thread per context
    nothing comes in between.
    """

    def __init__(self, monitor: zmq.Socket) -> None:
        self.monitor = monitor
        self.handshakes = 0
        self._after_failed_handshake = False

    def poll(self, timeout_ms: int) -> int:
        """Handle the events of the next ``timeout_ms``; returns how many
        established connections dropped."""
        lost = 0
        if not self.monitor.poll(timeout_ms):
            return 0
        while self.monitor.poll(0):
            event = recv_monitor_message(self.monitor)["event"]
            if event == zmq.EVENT_HANDSHAKE_SUCCEEDED:
                self.handshakes += 1
            elif event == zmq.EVENT_DISCONNECTED and not self._after_failed_handshake:
                lost += 1
            self._after_failed_handshake = event in _HANDSHAKE_FAILED
        return lost


class ControlServer:
    """The head's TCP endpoint for DP ranks on other nodes.

    One ROUTER socket on ``--control-port``. Every remote rank connects a
    DEALER whose routing id names its rank and speaks the same Ready/Cmd/Reply
    protocol as a child on ``ipc://``. One thread uses the socket at a time:
    the launcher until every rank is ready, then the controller thread.

    The watch thread calls :meth:`watch`: a rank connection that drops before
    a CLOSE went out means its process exited, its host went down or the
    network failed for ``timeout_s`` (ZMTP heartbeats).
    """

    _POLL_MS = 500

    def __init__(
        self,
        port: int,
        ranks: Iterable[int],
        timeout_s: float = 30.0,
        raw_outputs: bool = False,
    ):
        # Replies as RawReply (outputs left encoded) instead of Reply.
        self._decode = decode_up_raw if raw_outputs else decode_up
        # Its own context with one I/O thread (see ConnectionEvents).
        self._ctx = zmq.Context(io_threads=1)
        sock = self._ctx.socket(zmq.ROUTER)
        sock.setsockopt(zmq.LINGER, 0)
        # Fail a send to a rank that is not connected instead of dropping it.
        sock.setsockopt(zmq.ROUTER_MANDATORY, 1)
        set_heartbeat(sock, timeout_s)
        self._events = ConnectionEvents(sock.get_monitor_socket(CONNECTION_EVENTS))
        try:
            sock.bind(f"tcp://*:{port}")
        except zmq.ZMQError as e:
            self._ctx.destroy(linger=0)
            raise RuntimeError(f"cannot listen on --control-port {port}: {e}") from e
        self.port = port
        self._sock = sock
        self._inbox: dict[int, deque[Union[Ready, AnyReply]]] = {
            rank: deque() for rank in ranks
        }
        # Set once a CLOSE went out: ranks disconnect on purpose from then on.
        self.closing = False
        self.lost: Optional[str] = None
        self._closed = False

    def handle(self, rank: int, spec_digest: Optional[str] = None) -> RemoteRankHandle:
        return RemoteRankHandle(rank, self, spec_digest)

    def watch(self, timeout_s: float) -> Optional[str]:
        """Watch thread only: wait up to ``timeout_s`` for connection events;
        returns why lockstep cannot continue, if a rank was lost."""
        if self._events.poll(int(timeout_s * 1000)) and not self.closing:
            self.lost = (
                "a DeepInfer rank on another node lost its connection to the "
                "controller (its process exited, its host went down, or the "
                "network was silent for --node-timeout-s)"
            )
        return self.lost

    def send(self, rank: int, cmd: Cmd) -> None:
        if cmd.op == Op.CLOSE:
            self.closing = True
        data = encode(cmd)
        while True:
            self._check_lost()
            try:
                self._sock.send_multipart([routing_id(rank), data], zmq.NOBLOCK)
                return
            except zmq.Again:
                self._sock.poll(self._POLL_MS, zmq.POLLOUT)
            except zmq.ZMQError as e:
                if e.errno == zmq.EHOSTUNREACH:
                    raise RankFailed(f"rank {rank} is not connected") from None
                raise

    def recv(self, rank: int, timeout: Optional[float]) -> Union[Ready, AnyReply]:
        deadline = None if timeout is None else time.monotonic() + timeout
        inbox = self._inbox[rank]
        while not inbox:
            # What already arrived comes first: a rank's error reply says
            # more than its lost connection.
            if self._sock.poll(0 if self.lost else self._POLL_MS):
                self._pump()
                continue
            self._check_lost()
            if deadline is not None and time.monotonic() > deadline:
                raise RankFailed(f"rank {rank}: no reply within {timeout}s")
        return inbox.popleft()

    def poll(self, rank: int) -> Optional[Union[Ready, AnyReply]]:
        self._pump()
        inbox = self._inbox[rank]
        if inbox:
            return inbox.popleft()
        self._check_lost()
        return None

    def _pump(self) -> None:
        """Queue every message that arrived, by sender."""
        while True:
            try:
                frames = self._sock.recv_multipart(zmq.NOBLOCK)
            except zmq.Again:
                return
            name = frames[0].decode(errors="replace")
            rank = int(name[4:]) if name[4:].isdigit() else -1
            if routing_id(rank) != frames[0] or rank not in self._inbox:
                raise RankFailed(
                    f"unexpected peer {name!r} on --control-port {self.port}"
                )
            msg = self._decode(frames[-1])
            if msg.rank != rank:
                raise RankFailed(f"peer {name!r} sent a message of rank {msg.rank}")
            self._inbox[rank].append(msg)

    def _check_lost(self) -> None:
        if self.lost is None:
            return
        for rank, inbox in self._inbox.items():
            for msg in inbox:
                if not isinstance(msg, Ready) and msg.error is not None:
                    what = "failed to start" if msg.seq < 0 else "failed"
                    raise RankFailed(f"rank {rank} {what}: {msg.error}")
        raise RankFailed(self.lost)

    def close(self) -> None:
        """Once the watch thread stopped. Idempotent."""
        if self._closed:
            return
        self._closed = True
        self.closing = True
        self._ctx.destroy(linger=0)


class RemoteRankHandle(RankHandle):
    """A rank on another node, connected to the head's :class:`ControlServer`."""

    def __init__(
        self, rank: int, server: ControlServer, spec_digest: Optional[str] = None
    ):
        self.rank = rank
        self.ready: Optional[Ready] = None
        self._server = server
        self._spec_digest = spec_digest

    def wait_ready(self, timeout: Optional[float] = None) -> EngineMeta:
        if self.ready is None:
            self._ready(self._server.recv(self.rank, timeout))
        assert self.ready is not None
        return self.ready.meta

    def poll_ready(self) -> Optional[EngineMeta]:
        if self.ready is None:
            msg = self._server.poll(self.rank)
            if msg is None:
                return None
            self._ready(msg)
        assert self.ready is not None
        return self.ready.meta

    def _ready(self, msg: Union[Ready, AnyReply]) -> None:
        if not isinstance(msg, Ready):
            raise RankFailed(f"rank {self.rank} failed to start: {msg.error}")
        if self._spec_digest is not None and msg.spec_digest != self._spec_digest:
            raise RankFailed(
                f"rank {self.rank} on {msg.host} was launched with other DeepInfer "
                "options than the head; give every node the same arguments"
            )
        if msg.env is not None:
            here = deepinfer_environ()
            differ = {
                k: f"{k}={msg.env.get(k)!r} (head: {here.get(k)!r})"
                for k in sorted(here.keys() | msg.env.keys())
                if here.get(k) != msg.env.get(k)
                and not (k in ENGINE_DERIVED_ENV and k not in here)
            }
            lockstep = [d for k, d in differ.items() if k in LOCKSTEP_ENV]
            if lockstep:
                raise RankFailed(
                    f"rank {self.rank} on {msg.host} has another DeepInfer "
                    "environment than the head, and every rank needs the same "
                    f"layout and schedule settings: {', '.join(lockstep)}"
                )
            if differ:
                logger.warning(
                    "DeepInfer rank %d on %s has another environment than the "
                    "head: %s",
                    self.rank,
                    msg.host,
                    ", ".join(differ.values()),
                )
        logger.info("DeepInfer rank %d ready on %s", self.rank, msg.host)
        self.ready = msg

    def send(self, cmd: Cmd) -> None:
        self._server.send(self.rank, cmd)

    def recv(self, timeout: Optional[float] = None) -> AnyReply:
        msg = self._server.recv(self.rank, timeout)
        if isinstance(msg, Ready):
            raise RankFailed(f"rank {self.rank}: unexpected {type(msg).__name__}")
        return msg

    def close(self, timeout: float = 30.0) -> None:
        """Nothing to reap here: the rank's node launcher does, and the
        engine closes the shared server."""


def wait_all_ready(
    handles: list[RankHandle], timeout: Optional[float] = None
) -> list[EngineMeta]:
    """Wait for READY from every rank under one deadline; fail as soon as any
    rank reports an error or exits."""
    deadline = None if timeout is None else time.monotonic() + timeout
    metas: dict[int, EngineMeta] = {}
    while True:
        for handle in handles:
            if handle.rank not in metas:
                meta = handle.poll_ready()
                if meta is not None:
                    metas[handle.rank] = meta
        if len(metas) == len(handles):
            return [metas[h.rank] for h in handles]
        if deadline is not None and time.monotonic() > deadline:
            missing = sorted(h.rank for h in handles if h.rank not in metas)
            raise RankFailed(f"ranks {missing} not ready within {timeout}s")
        time.sleep(0.05)
