# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Child-process entry point for every DP rank but an in-process rank 0.

    python -m dynamo.deepinfer.rank_worker --endpoint ipc://... --parent-pid P --spec JSON
    python -m dynamo.deepinfer.rank_worker --endpoint tcp://HEAD:PORT --parent-pid P \\
        --spec JSON [--advertise-addr ADDR] [--node-timeout-s S] [--startup-timeout-s S]

Builds the rank's engine, reports ``Ready``, then answers one ``Reply`` per
``Cmd`` until the CLOSE iteration. On ``ipc://`` it is a child of the head's
Dynamo worker. On ``tcp://`` it is a rank on another node (spawned by
``dynamo.deepinfer.node``): it connects to the head's ``ControlServer`` and
exits when that connection is lost. Must not import ``dynamo._core`` or
``dynamo.common.backend``: only ``dynamo.deepinfer.{rank,protocol,kv_cache,
kv_transfer}`` and what they need.
"""

from __future__ import annotations

import argparse
import ctypes
import logging
import os
import signal
import socket
import sys
import threading
import time
import traceback
from typing import NoReturn, Optional

import zmq

from dynamo.deepinfer.protocol import (
    CONNECTION_EVENTS,
    ConnectionEvents,
    Op,
    Ready,
    Reply,
    decode_cmd,
    deepinfer_environ,
    encode,
    routing_id,
    set_heartbeat,
    tcp_endpoint,
)
from dynamo.deepinfer.rank import RankRunner, RankSpec

logger = logging.getLogger(__name__)

_PR_SET_PDEATHSIG = 1


def _die_with_parent(parent_pid: int) -> None:
    """SIGKILL this process when the parent's spawning thread exits."""
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(_PR_SET_PDEATHSIG, signal.SIGKILL, 0, 0, 0) != 0:
        logger.warning("prctl(PR_SET_PDEATHSIG) failed: errno %d", ctypes.get_errno())
    if os.getppid() != parent_pid:
        os._exit(1)  # the parent died before prctl took effect


class _ControllerWatch:
    """TCP ranks: exit when the connection to the head's controller is lost.

    The main thread may be blocked in a collective, so a thread watches the
    socket monitor. A rank that cannot reach the head within the startup
    timeout exits too. ZMTP heartbeats bound how long a silent head (host
    down, network split) goes unnoticed.
    """

    def __init__(self, sock: zmq.Socket, endpoint: str, startup_timeout_s: float):
        # Before connect(), so the first connection is seen.
        self._events = ConnectionEvents(sock.get_monitor_socket(CONNECTION_EVENTS))
        self._endpoint = endpoint
        self._deadline = time.monotonic() + startup_timeout_s
        # Set once CLOSE arrived: the head may go away from then on.
        self.closing = False
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name="di-controller-watch", daemon=True
        )
        self._thread.start()

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                lost = self._events.poll(200)
                if self.closing:
                    continue
                if lost:
                    self._exit("lost the connection to the DeepInfer controller")
                if not self._events.handshakes and time.monotonic() > self._deadline:
                    self._exit("no DeepInfer controller within --startup-timeout-s")
        finally:
            self._events.monitor.close(linger=0)

    def _exit(self, reason: str) -> NoReturn:
        logger.critical("%s at %s; exiting", reason, self._endpoint)
        logging.shutdown()
        os._exit(1)

    def stop(self, sock: zmq.Socket) -> None:
        self.closing = True
        self._stop.set()
        self._thread.join(5.0)
        sock.disable_monitor()


def _fail(
    ctx: zmq.Context,
    sock: zmq.Socket,
    reply: Reply,
    watch: Optional[_ControllerWatch] = None,
) -> NoReturn:
    """Report the error and exit hard. A normal exit would run DeepInfer's
    ``LLMEngine.__del__`` -> ``close()`` barrier and hang for the gloo timeout
    instead of failing the peers' collectives now."""
    if watch is not None:
        watch.stop(sock)
    sock.setsockopt(zmq.SNDTIMEO, 5000)  # the parent may be gone
    try:
        sock.send(encode(reply))
    except zmq.ZMQError:
        pass
    sock.close(linger=5000)  # flush the reply
    ctx.term()
    logging.shutdown()
    os._exit(1)


def _advertised(bind: Optional[str], host: Optional[str]) -> Optional[str]:
    """The KV event endpoint ``tcp://*:PORT`` as the head reaches it."""
    if bind is None:
        return None
    return tcp_endpoint(host or socket.gethostname(), int(bind.rsplit(":", 1)[1]))


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="DeepInfer DP rank worker")
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--parent-pid", type=int, required=True)
    parser.add_argument("--spec", required=True)
    # tcp:// only.
    parser.add_argument("--advertise-addr", default=None)
    parser.add_argument("--node-timeout-s", type=float, default=30.0)
    parser.add_argument("--startup-timeout-s", type=float, default=3600.0)
    args = parser.parse_args(argv)
    _die_with_parent(args.parent_pid)
    # A group-wide SIGINT/SIGTERM (Ctrl-C, scancel) must not kill a rank
    # mid-iteration: the head shuts every rank down with a CLOSE iteration,
    # and PR_SET_PDEATHSIG covers the parent dying.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)

    spec = RankSpec.from_json(args.spec)
    # DYN_LOG may carry per-target filters ("info,dynamo=debug"); use the default.
    level = os.environ.get("DYN_LOG", "info").split(",")[0].upper()
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format=f"%(asctime)s %(levelname)s [rank {spec.rank}] %(name)s: %(message)s",
    )
    ctx = zmq.Context()
    remote = args.endpoint.startswith("tcp://")
    watch: Optional[_ControllerWatch] = None
    if remote:
        sock = ctx.socket(zmq.DEALER)
        sock.setsockopt(zmq.ROUTING_ID, routing_id(spec.rank))
        set_heartbeat(sock, args.node_timeout_s)
        watch = _ControllerWatch(sock, args.endpoint, args.startup_timeout_s)
    else:
        sock = ctx.socket(zmq.PAIR)
    sock.setsockopt(zmq.LINGER, 5000)
    sock.connect(args.endpoint)
    try:
        runner = RankRunner(spec)
    except Exception:
        logger.exception("rank %d failed to start", spec.rank)
        _fail(
            ctx,
            sock,
            Reply(seq=-1, rank=spec.rank, error=traceback.format_exc()),
            watch,
        )
    ready = Ready(rank=spec.rank, meta=runner.meta)
    if remote:
        ready.host = socket.gethostname()
        ready.spec_digest = spec.digest()
        ready.env = deepinfer_environ()
        ready.kv_events_endpoint = _advertised(
            spec.kv_events_endpoint, args.advertise_addr
        )
    sock.send(encode(ready))
    while True:
        cmd = decode_cmd(sock.recv())
        if cmd.op == Op.CLOSE and watch is not None:
            watch.closing = True
        try:
            reply = runner.apply(cmd)
        except Exception:
            logger.exception("rank %d failed in iteration %d", spec.rank, cmd.seq)
            _fail(
                ctx,
                sock,
                Reply(seq=cmd.seq, rank=spec.rank, error=traceback.format_exc()),
                watch,
            )
        sock.send(encode(reply))
        if cmd.op == Op.CLOSE:
            if watch is not None:
                watch.stop(sock)
            sock.close()
            ctx.term()
            return 0


if __name__ == "__main__":
    sys.exit(main())
