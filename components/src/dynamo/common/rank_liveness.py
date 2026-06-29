# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ZMQ-based cross-rank liveness watcher for fast detection of a non-leader rank's
death in a multi-node tensor-parallel cohort.

Motivation
----------
When a rank>0 worker on another node dies, the leader (rank 0) only learns via the
engine's NCCL collective timeout — 600s by default, ~20s even when tuned down — and
in-flight requests stall for that whole window. The flock-based GMS failover only
covers rank-0 death (the OS releases the flock on the leader's exit); it does not see
a remote worker die.

This module adds a *GPU-independent* liveness channel. Each worker rank holds a ZMQ
connection to a leader-side monitor and sends a heartbeat on a plain CPU thread. When
the worker process dies, its heartbeats stop and the monitor fires within one
heartbeat-timeout (sub-second), letting the leader proactively fence + release the GMS
failover lock so the warm shadow takes over immediately — instead of waiting out the
NCCL timeout.

Two properties make this better than both the NFS-flock idea and the NCCL timeout:
  * The heartbeat runs on a CPU thread, so it keeps beating even while the GPU is busy
    in a long legitimate collective (warmup, load spike) — it does NOT false-positive
    the way an aggressive NCCL/engine watchdog does during init.
  * A peer process exit drops the heartbeat promptly, so a *crash* is detected in
    ~one interval rather than ~one collective-timeout.

It does NOT replace the NCCL/engine watchdog: a rank that is hung-but-alive with its
heartbeat thread still running is invisible here (only a timeout catches a true hang).
This is the crash detector; the (dynamically-lowered) engine watchdog stays the hang
detector.

Reuses pyzmq, already a dependency of both vLLM and SGLang. Engine-agnostic: the
leader supplies an ``on_rank_lost`` callback that wires into the existing failover
trigger (fence children + release the failover lock).
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Callable, Iterable, Optional

logger = logging.getLogger(__name__)

# Defaults: ~750ms detection (timeout) with a 250ms heartbeat. Well under the ~20s
# tuned NCCL window, and the CPU-thread heartbeat is immune to GPU-collective stalls.
DEFAULT_HEARTBEAT_MS = 250
DEFAULT_TIMEOUT_MS = 750
DEFAULT_LIVENESS_PORT = 29555
DEFAULT_STARTUP_GRACE_MS = 30_000


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("Ignoring invalid %s=%r; using %d", name, raw, default)
        return default


def liveness_enabled() -> bool:
    val = os.environ.get("DYN_GMS_RANK_LIVENESS")
    if val is None:
        return False
    return val.lower() in {"1", "true", "yes", "on"}


def heartbeat_ms() -> int:
    return max(20, _int_env("DYN_GMS_RANK_LIVENESS_HEARTBEAT_MS", DEFAULT_HEARTBEAT_MS))


def timeout_ms() -> int:
    return max(
        heartbeat_ms() * 2,
        _int_env("DYN_GMS_RANK_LIVENESS_TIMEOUT_MS", DEFAULT_TIMEOUT_MS),
    )


def startup_grace_ms() -> int:
    return max(
        timeout_ms(),
        _int_env("DYN_GMS_RANK_LIVENESS_STARTUP_GRACE_MS", DEFAULT_STARTUP_GRACE_MS),
    )


def liveness_port() -> int:
    return _int_env("DYN_GMS_RANK_LIVENESS_PORT", DEFAULT_LIVENESS_PORT)


def leader_bind_addr() -> str:
    return os.environ.get(
        "DYN_GMS_RANK_LIVENESS_BIND_ADDR", f"tcp://*:{liveness_port()}"
    )


def leader_connect_addr(leader_host: str) -> str:
    template = os.environ.get("DYN_GMS_RANK_LIVENESS_CONNECT_ADDR")
    if template:
        return template.format(leader_host=leader_host)
    return f"tcp://{leader_host}:{liveness_port()}"


class RankLivenessClient:
    """Runs on a worker rank (rank>0). Holds a DEALER connection to the leader and
    heartbeats on a daemon thread. On process death the socket drops and the leader's
    monitor fires within one timeout window."""

    def __init__(
        self,
        leader_host: str,
        rank: int,
        *,
        interval_ms: Optional[int] = None,
        connect_addr: Optional[str] = None,
    ):
        self._leader_host = leader_host
        self._rank = int(rank)
        self._interval = (interval_ms or heartbeat_ms()) / 1000.0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._connect_addr = connect_addr or leader_connect_addr(leader_host)

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run, name=f"gms-rank-liveness-client-{self._rank}", daemon=True
        )
        self._thread.start()
        logger.info(
            "[GMS liveness] rank %d heartbeating leader %s every %dms",
            self._rank,
            self._connect_addr,
            int(self._interval * 1000),
        )

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=0.5)
            self._thread = None

    def _run(self) -> None:
        import zmq

        ctx = zmq.Context.instance()
        sock = ctx.socket(zmq.DEALER)
        # Identity = rank, so the leader maps heartbeats -> rank with no correlation.
        sock.setsockopt(zmq.IDENTITY, f"rank-{self._rank}".encode())
        sock.setsockopt(zmq.LINGER, 0)
        # ZMTP-level heartbeats make ZMQ itself notice a dead peer quickly too.
        sock.setsockopt(zmq.HEARTBEAT_IVL, int(self._interval * 1000))
        sock.setsockopt(zmq.HEARTBEAT_TIMEOUT, int(self._interval * 1000) * 3)
        sock.connect(self._connect_addr)
        try:
            while not self._stop.is_set():
                try:
                    sock.send(b"hb", flags=zmq.NOBLOCK)
                except zmq.ZMQError:
                    logger.debug(
                        "[GMS liveness] rank %d heartbeat send failed",
                        self._rank,
                        exc_info=True,
                    )
                self._stop.wait(self._interval)
        finally:
            sock.close(0)


class RankLivenessMonitor:
    """Monitor non-leader ranks and report the first lost rank exactly once.

    When ``expected_ranks`` is provided, ranks that never register are reported after
    the startup grace period. Omitting it preserves the legacy behavior: only ranks
    observed at least once are armed. ``bind_addr`` allows each colocated replica to
    use its own endpoint.
    """

    def __init__(
        self,
        on_rank_lost: Callable[[int, str], None],
        *,
        bind_addr: Optional[str] = None,
        timeout_ms_override: Optional[int] = None,
        expected_ranks: Optional[Iterable[int]] = None,
        startup_grace_ms_override: Optional[int] = None,
    ):
        self._on_rank_lost = on_rank_lost
        self._bind_addr = bind_addr or leader_bind_addr()
        timeout_value = (
            timeout_ms() if timeout_ms_override is None else max(1, timeout_ms_override)
        )
        self._timeout = timeout_value / 1000.0
        self._expected_ranks = (
            None
            if expected_ranks is None
            else frozenset(int(rank) for rank in expected_ranks)
        )
        grace_value = (
            startup_grace_ms()
            if startup_grace_ms_override is None
            else max(0, startup_grace_ms_override)
        )
        self._startup_grace = grace_value / 1000.0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._fired = False

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run, name="gms-rank-liveness-monitor", daemon=True
        )
        self._thread.start()
        logger.info(
            "[GMS liveness] leader monitor bound %s (timeout %dms)",
            self._bind_addr,
            int(self._timeout * 1000),
        )

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=0.5)
            self._thread = None

    def _run(self) -> None:
        import zmq

        ctx = zmq.Context.instance()
        sock = ctx.socket(zmq.ROUTER)
        sock.setsockopt(zmq.LINGER, 0)
        sock.setsockopt(zmq.HEARTBEAT_IVL, int(self._timeout * 1000 / 3))
        sock.setsockopt(zmq.HEARTBEAT_TIMEOUT, int(self._timeout * 1000))
        sock.bind(self._bind_addr)
        poller = zmq.Poller()
        poller.register(sock, zmq.POLLIN)

        last_seen: dict[int, float] = {}
        started = time.monotonic()
        poll_ms = max(10, int(self._timeout * 1000 / 5))
        try:
            while not self._stop.is_set():
                events = dict(poller.poll(poll_ms))
                now = time.monotonic()
                if sock in events:
                    while True:
                        try:
                            frames = sock.recv_multipart(flags=zmq.NOBLOCK)
                        except zmq.Again:
                            break
                        if len(frames) < 2 or frames[-1] != b"hb":
                            continue
                        rank = self._rank_of(frames[0])
                        if rank is None:
                            continue
                        if (
                            self._expected_ranks is not None
                            and rank not in self._expected_ranks
                        ):
                            logger.debug(
                                "[GMS liveness] ignoring unexpected rank %d", rank
                            )
                            continue
                        if rank not in last_seen:
                            logger.info("[GMS liveness] rank %d registered", rank)
                        last_seen[rank] = now

                if (
                    self._expected_ranks is not None
                    and now - started > self._startup_grace
                ):
                    missing = sorted(self._expected_ranks.difference(last_seen))
                    if missing:
                        self._fire(missing[0], "startup-timeout")
                        return

                for rank, seen in list(last_seen.items()):
                    if now - seen > self._timeout:
                        logger.warning(
                            "[GMS liveness] rank %d silent for %.0fms (>%.0fms)",
                            rank,
                            (now - seen) * 1000,
                            self._timeout * 1000,
                        )
                        self._fire(rank, "liveness-timeout")
                        return
        finally:
            sock.close(0)

    def _fire(self, rank: int, reason: str) -> bool:
        if self._fired:
            return False
        self._fired = True
        logger.warning("[GMS liveness] rank %d lost (%s)", rank, reason)
        try:
            self._on_rank_lost(rank, reason)
        except Exception:
            logger.exception("[GMS liveness] on_rank_lost callback failed")
        return True

    @staticmethod
    def _rank_of(identity: bytes) -> Optional[int]:
        try:
            text = identity.decode()
        except Exception:
            return None
        if text.startswith("rank-"):
            try:
                return int(text[len("rank-") :])
            except ValueError:
                return None
        return None
