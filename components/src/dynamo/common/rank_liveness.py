# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bidirectional ZMQ liveness for multi-node tensor-parallel cohorts.

Motivation
----------
When a rank>0 worker on another node dies, the leader (rank 0) only learns via the
engine's NCCL collective timeout — 600s by default, ~20s even when tuned down — and
in-flight requests stall for that whole window. The flock-based GMS failover only
covers rank-0 death (the OS releases the flock on the leader's exit); it does not see
a remote worker die.

This module adds a *GPU-independent* liveness channel. Each worker rank holds a ZMQ
connection to a leader-side monitor and sends a heartbeat on a plain CPU thread. The
leader acknowledges each heartbeat. Loss in either direction therefore fires within one
heartbeat timeout: the leader fences a dead worker, while a surviving worker fences its
orphaned local cohort when rank 0 dies. Both paths release pod-local ownership so the
complete warm-shadow TP cohort can take over without waiting for the NCCL timeout.

Two properties make this better than both the NFS-flock idea and the NCCL timeout:
  * The heartbeat does not wait for a CUDA collective. It still competes for
    Python's GIL and CPU scheduling, so an aggressive deadline CAN false-positive
    under legitimate load. Timeout logs include local observer delay to help
    distinguish it from peer silence.
  * A peer process exit drops the heartbeat promptly, so a *crash* is detected in
    ~one heartbeat-timeout window rather than ~one collective-timeout.

Missing heartbeats indicate suspicion, not proof of process death or CUDA drain.
Callbacks must fence writers before handing off writable KV ownership.
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

from gpu_memory_service.common.gpu_failure_marker import (
    gpu_failure_marker_path,
    read_gpu_failure_marker,
)

from dynamo.common.utils.env import env_bool
from dynamo.common.utils.env import env_int as _int_env

logger = logging.getLogger(__name__)

DEFAULT_HEARTBEAT_MS = 250
DEFAULT_TIMEOUT_MS = 750
DEFAULT_LIVENESS_PORT = 29555
DEFAULT_STARTUP_GRACE_MS = 30_000
DEFAULT_STARTUP_TIMEOUT_MS = 5_000
# Limit receive work per poll so a busy peer cannot starve lost-rank deadlines.
_MAX_HEARTBEATS_PER_POLL = 64


def liveness_enabled() -> bool:
    return env_bool("DYN_GMS_RANK_LIVENESS")


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


def startup_timeout_ms() -> int:
    """Deadline for already-connected peers until serving is armed."""

    return max(
        timeout_ms(),
        _int_env(
            "DYN_GMS_RANK_LIVENESS_STARTUP_TIMEOUT_MS", DEFAULT_STARTUP_TIMEOUT_MS
        ),
    )


_MIN_UNPRIVILEGED_PORT = 1024
_UNPRIVILEGED_PORT_COUNT = 65535 - _MIN_UNPRIVILEGED_PORT + 1
_COHORT_PORT_OFFSET = _UNPRIVILEGED_PORT_COUNT // 2


def liveness_port(cohort_identity: str | None = None) -> int:
    """Return an explicit port, or derive one from the TP cohort init address."""

    if "DYN_GMS_RANK_LIVENESS_PORT" in os.environ:
        return _int_env("DYN_GMS_RANK_LIVENESS_PORT", DEFAULT_LIVENESS_PORT)
    if cohort_identity:
        try:
            dist_port = int(cohort_identity.rsplit(":", 1)[1])
        except (IndexError, ValueError):
            pass
        else:
            if _MIN_UNPRIVILEGED_PORT <= dist_port <= 65535:
                return _MIN_UNPRIVILEGED_PORT + (
                    (dist_port - _MIN_UNPRIVILEGED_PORT + _COHORT_PORT_OFFSET)
                    % _UNPRIVILEGED_PORT_COUNT
                )
    return DEFAULT_LIVENESS_PORT


def leader_bind_addr(cohort_identity: str | None = None) -> str:
    return os.environ.get(
        "DYN_GMS_RANK_LIVENESS_BIND_ADDR",
        f"tcp://*:{liveness_port(cohort_identity)}",
    )


def leader_connect_addr(leader_host: str, cohort_identity: str | None = None) -> str:
    template = os.environ.get("DYN_GMS_RANK_LIVENESS_CONNECT_ADDR")
    if template:
        return template.format(leader_host=leader_host)
    return f"tcp://{leader_host}:{liveness_port(cohort_identity)}"


def configured_gpu_failure_marker() -> str | None:
    """Return this boot's GMS crash marker path, when shared KV is active."""

    for name in (
        "GMS_VLLM_WRITER_COHORT_PATH",
        "GMS_SGLANG_WRITER_COHORT_PATH",
    ):
        cohort = os.environ.get(name)
        if cohort:
            return str(gpu_failure_marker_path(cohort))
    return None


class RankLivenessClient:
    """Worker heartbeat client with optional leader-loss detection.

    When ``on_leader_lost`` is supplied, missing acknowledgements fence an orphaned
    worker cohort within one timeout window.
    """

    def __init__(
        self,
        leader_host: str,
        rank: int,
        *,
        interval_ms: Optional[int] = None,
        connect_addr: Optional[str] = None,
        on_leader_lost: Callable[[int, str], None] | None = None,
        timeout_ms_override: Optional[int] = None,
        startup_grace_ms_override: Optional[int] = None,
        startup_timeout_ms_override: Optional[int] = None,
    ):
        self._leader_host = leader_host
        self._rank = int(rank)
        self._interval = (interval_ms or heartbeat_ms()) / 1000.0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._connect_addr = connect_addr or leader_connect_addr(leader_host)
        self._on_leader_lost = on_leader_lost
        timeout_value = (
            timeout_ms() if timeout_ms_override is None else max(1, timeout_ms_override)
        )
        self._timeout = timeout_value / 1000.0
        grace_value = (
            startup_grace_ms()
            if startup_grace_ms_override is None
            else max(0, startup_grace_ms_override)
        )
        self._startup_grace = grace_value / 1000.0
        startup_timeout_value = (
            startup_timeout_ms()
            if startup_timeout_ms_override is None
            else max(1, startup_timeout_ms_override)
        )
        self._startup_timeout = startup_timeout_value / 1000.0
        self._runtime_armed = False
        self._runtime_armed_event = threading.Event()
        self._fired = False

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
        self._runtime_armed_event.set()
        if self._thread is not None:
            self._thread.join(timeout=0.5)
            self._thread = None

    def wait_for_runtime_arm(self, timeout: float | None = None) -> bool:
        """Wait until the leader releases this rank into serving runtime."""

        self._runtime_armed_event.wait(timeout)
        return self._runtime_armed and not self._stop.is_set() and not self._fired

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
        poller = zmq.Poller()
        poller.register(sock, zmq.POLLIN)
        started = time.monotonic()
        last_ack: float | None = None
        previous_cycle_started = started
        failure_marker = configured_gpu_failure_marker()
        heartbeat = None if failure_marker is None else [failure_marker.encode(), b"hb"]
        try:
            while not self._stop.is_set():
                cycle_started = time.monotonic()
                scheduling_gap_ms = (
                    max(0.0, cycle_started - previous_cycle_started - self._interval)
                    * 1000
                )
                previous_cycle_started = cycle_started
                try:
                    if heartbeat is None:
                        sock.send(b"hb", flags=zmq.NOBLOCK)
                    else:
                        sock.send_multipart(heartbeat, flags=zmq.NOBLOCK)
                except zmq.ZMQError:
                    logger.debug(
                        "[GMS liveness] rank %d heartbeat send failed",
                        self._rank,
                        exc_info=True,
                    )
                events = dict(poller.poll(max(1, int(self._interval * 1000))))
                now = time.monotonic()
                if sock in events:
                    for _ in range(_MAX_HEARTBEATS_PER_POLL):
                        try:
                            frame = sock.recv(flags=zmq.NOBLOCK)
                        except zmq.Again:
                            break
                        if frame == b"fence":
                            logger.warning(
                                "[GMS liveness] leader fenced the TP cohort after "
                                "a peer-rank failure"
                            )
                            try:
                                sock.send(b"fence-ack", flags=zmq.NOBLOCK)
                            except zmq.ZMQError:
                                logger.debug(
                                    "[GMS liveness] rank %d fence acknowledgement "
                                    "failed",
                                    self._rank,
                                    exc_info=True,
                                )
                            self._fire(0, "peer-rank-lost")
                            return
                        if frame in (b"startup-ack", b"ack"):
                            if frame == b"ack":
                                self._runtime_armed = True
                                self._runtime_armed_event.set()
                            if last_ack is None:
                                logger.info(
                                    "[GMS liveness] rank %d received leader acknowledgement",
                                    self._rank,
                                )
                            last_ack = now
                if self._on_leader_lost is not None and not self._stop.is_set():
                    if last_ack is None and now - started > self._startup_grace:
                        self._fire(0, "startup-timeout")
                        return
                    deadline = (
                        self._timeout if self._runtime_armed else self._startup_timeout
                    )
                    if last_ack is not None and now - last_ack > deadline:
                        logger.warning(
                            "[GMS liveness] leader silent %.0fms (deadline %.0fms); "
                            "local scheduling gap %.0fms, poll cycle %.0fms; "
                            "suspected failure, writer fencing still required",
                            (now - last_ack) * 1000,
                            deadline * 1000,
                            scheduling_gap_ms,
                            (now - cycle_started) * 1000,
                        )
                        self._fire(0, "liveness-timeout")
                        return
                # An acknowledgement normally arrives immediately. Preserve the
                # configured send cadence instead of turning the client into a
                # busy heartbeat/ack loop.
                remaining = self._interval - (time.monotonic() - cycle_started)
                if remaining > 0:
                    self._stop.wait(remaining)
        finally:
            sock.close(0)

    def _fire(self, rank: int, reason: str) -> bool:
        if self._fired or self._stop.is_set():
            return False
        self._fired = True
        logger.warning("[GMS liveness] leader rank %d lost (%s)", rank, reason)
        try:
            assert self._on_leader_lost is not None
            self._on_leader_lost(rank, reason)
        except Exception:
            logger.exception("[GMS liveness] on_leader_lost callback failed")
        return True


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
        runtime_armed: bool = True,
        broadcast_fence: bool = False,
        failure_marker_path: Optional[str] = None,
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
        self._runtime_armed = bool(runtime_armed)
        self._broadcast_fence_enabled = bool(broadcast_fence)
        self._stop = threading.Event()
        self._failure_marker_path = (
            configured_gpu_failure_marker()
            if failure_marker_path is None
            else failure_marker_path
        )
        self._thread: Optional[threading.Thread] = None
        self._fired = False
        self._bind_ready = threading.Event()
        self._bind_error: Optional[BaseException] = None
        self._seen_ranks: set[int] = set()
        self._rank_failure_marker_paths: dict[int, str] = {}
        self._seen_changed = threading.Condition()

    def start(self) -> None:
        if self._thread is not None:
            return
        self._bind_ready = threading.Event()
        self._bind_error: Optional[BaseException] = None
        self._thread = threading.Thread(
            target=self._run, name="gms-rank-liveness-monitor", daemon=True
        )
        self._thread.start()
        # Wait for bind completion so start() can surface bind failures.
        if not self._bind_ready.wait(timeout=5.0):
            raise RuntimeError(
                f"[GMS liveness] monitor bind to {self._bind_addr} timed out"
            )
        if self._bind_error is not None:
            raise RuntimeError(
                f"[GMS liveness] monitor bind to {self._bind_addr} failed: "
                f"{self._bind_error}"
            )
        logger.info(
            "[GMS liveness] leader monitor bound %s (timeout %dms)",
            self._bind_addr,
            int(self._timeout * 1000),
        )

    def stop(self) -> None:
        self._stop.set()
        with self._seen_changed:
            self._seen_changed.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=0.5)
            self._thread = None

    @staticmethod
    def _poll_interval_ms(timeout_seconds: float) -> int:
        # Keep runtime deadline tightening responsive even when the monitor was
        # created with a much more conservative startup timeout.
        return max(10, min(50, int(timeout_seconds * 1000 / 5)))

    def set_timeout_ms(self, value: int) -> None:
        """Update the loss deadline without restarting the liveness socket.

        Engine process creation can briefly starve Python heartbeat threads. The
        leader can therefore start with a conservative deadline and tighten it
        after the serving handler is attached.
        """
        self._timeout = max(1, int(value)) / 1000.0
        self.arm_runtime()

    def arm_runtime(self) -> None:
        """Release registered ranks from startup fencing into serving runtime."""

        self._runtime_armed = True

    def wait_for_ranks(
        self, expected_ranks: Iterable[int], timeout: float | None = None
    ) -> bool:
        """Wait until every expected rank has heartbeated this monitor.

        This is also the TP activation barrier: a non-leader starts heartbeating
        only after it owns and has fenced its pod-local failover namespace.
        """

        expected = frozenset(int(rank) for rank in expected_ranks)
        deadline = None if timeout is None else time.monotonic() + max(0.0, timeout)
        with self._seen_changed:
            while True:
                if self._stop.is_set() or self._fired:
                    return False
                if expected.issubset(self._seen_ranks):
                    return True
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return False
                self._seen_changed.wait(remaining)

    def _run(self) -> None:
        import zmq

        ctx = zmq.Context.instance()
        sock = ctx.socket(zmq.ROUTER)
        sock.setsockopt(zmq.LINGER, 0)
        sock.setsockopt(zmq.HEARTBEAT_IVL, int(self._timeout * 1000 / 3))
        sock.setsockopt(zmq.HEARTBEAT_TIMEOUT, int(self._timeout * 1000))
        try:
            sock.bind(self._bind_addr)
        except Exception as exc:  # surface to start() instead of dying silently
            self._bind_error = exc
            self._bind_ready.set()
            return
        self._bind_ready.set()
        poller = zmq.Poller()
        poller.register(sock, zmq.POLLIN)

        last_seen: dict[int, float] = {}
        started = time.monotonic()
        try:
            while not self._stop.is_set():
                cycle_started = time.monotonic()
                # Recompute every cycle: set_timeout_ms() is used after model
                if self._failure_marker_path:
                    failure = read_gpu_failure_marker(self._failure_marker_path)
                    if failure is not None:
                        rank, pid, source = failure
                        logger.warning(
                            "[GMS liveness] GPU crash interlock reported rank %d "
                            "pid %d (%s); writer fencing remains authoritative",
                            rank,
                            pid,
                            source,
                        )
                        if self._broadcast_fence_enabled:
                            self._broadcast_fence(
                                sock, poller, last_seen, lost_rank=rank
                            )
                        self._fire(rank, "gpu-crash-interlock")
                        return
                # startup and must change both the deadline and observation
                # cadence without recreating the bound ROUTER socket.
                poll_ms = self._poll_interval_ms(self._timeout)
                events = dict(poller.poll(poll_ms))
                now = time.monotonic()
                if sock in events:
                    for _ in range(_MAX_HEARTBEATS_PER_POLL):
                        try:
                            frames = sock.recv_multipart(flags=zmq.NOBLOCK)
                        except zmq.Again:
                            break
                        failure = self._gpu_failure_of(frames)
                        if failure is not None:
                            rank, pid, source = failure
                            logger.warning(
                                "[GMS liveness] direct GPU crash notification "
                                "rank %d pid %d (%s); writer fencing remains "
                                "authoritative",
                                rank,
                                pid,
                                source,
                            )
                            if self._broadcast_fence_enabled:
                                self._broadcast_fence(
                                    sock, poller, last_seen, lost_rank=rank
                                )
                            self._fire(rank, "gpu-crash-interlock-zmq")
                            return

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
                        if len(frames) == 3:
                            try:
                                marker = frames[-2].decode()
                            except UnicodeDecodeError:
                                marker = ""
                            if marker:
                                self._rank_failure_marker_paths.setdefault(
                                    rank, os.path.normpath(marker)
                                )
                        if rank not in last_seen:
                            logger.info("[GMS liveness] rank %d registered", rank)
                        last_seen[rank] = now
                        with self._seen_changed:
                            self._seen_ranks.add(rank)
                            self._seen_changed.notify_all()
                        try:
                            ack = b"ack" if self._runtime_armed else b"startup-ack"
                            sock.send_multipart([frames[0], ack], flags=zmq.NOBLOCK)
                        except zmq.ZMQError:
                            logger.debug(
                                "[GMS liveness] leader acknowledgement failed for rank %d",
                                rank,
                                exc_info=True,
                            )

                now = time.monotonic()
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
                            "[GMS liveness] rank %d silent for %.0fms (>%.0fms); "
                            "local poll cycle %.0fms; suspected failure, "
                            "writer fencing still required",
                            rank,
                            (now - seen) * 1000,
                            self._timeout * 1000,
                            (now - cycle_started) * 1000,
                        )
                        if self._broadcast_fence_enabled:
                            self._broadcast_fence(
                                sock, poller, last_seen, lost_rank=rank
                            )
                        self._fire(rank, "liveness-timeout")
                        return
        finally:
            sock.close(0)

    def _gpu_failure_of(self, frames: list[bytes]) -> tuple[int, int, str] | None:
        """Validate a direct GMS hint against this monitor's exact cohort."""

        if len(frames) != 6 or frames[1] != b"gpu-failed-v1":
            return None
        try:
            cohort = frames[2].decode()
            rank = int(frames[3])
            pid = int(frames[4])
            source = frames[5].decode()
        except (UnicodeDecodeError, ValueError):
            return None
        if rank < 0 or pid <= 0 or not source:
            return None
        notified_marker = os.path.normpath(str(gpu_failure_marker_path(cohort)))
        expected_marker = self._rank_failure_marker_paths.get(rank)
        if expected_marker is None and (self._expected_ranks is None or rank == 0):
            expected_marker = self._failure_marker_path
        if expected_marker is None or notified_marker != os.path.normpath(
            expected_marker
        ):
            logger.warning(
                "[GMS liveness] ignoring GPU crash notification for another cohort %s",
                cohort,
            )
            return None
        return rank, pid, source

    def _broadcast_fence(self, sock, poller, last_seen, *, lost_rank: int) -> None:
        """Prompt surviving ranks to fail-stop before the leader releases ownership.

        The writer-cohort lock remains the correctness fence: takeover still waits
        for every writer guard even when this best-effort latency hint is lost. An
        acknowledgement only keeps the ROUTER alive long enough to put the fence
        on each established connection; it is not treated as proof of process or
        CUDA termination.
        """

        import zmq

        pending = {rank for rank in last_seen if rank != lost_rank}
        if not pending:
            return
        for rank in pending:
            try:
                sock.send_multipart(
                    [f"rank-{rank}".encode(), b"fence"], flags=zmq.NOBLOCK
                )
            except zmq.ZMQError:
                logger.debug(
                    "[GMS liveness] failed to send cohort fence to rank %d",
                    rank,
                    exc_info=True,
                )

        # Bound this optimization well below the normal heartbeat deadline. Ranks
        # that do not acknowledge still fail-stop when the leader disappears, and
        # the successor cannot pass retire_writer_cohort until their guards close.
        deadline = time.monotonic() + min(0.1, self._timeout / 2)
        while pending:
            remaining_ms = int(max(0.0, deadline - time.monotonic()) * 1000)
            if remaining_ms <= 0:
                break
            events = dict(poller.poll(remaining_ms))
            if sock not in events:
                break
            for _ in range(_MAX_HEARTBEATS_PER_POLL):
                try:
                    frames = sock.recv_multipart(flags=zmq.NOBLOCK)
                except zmq.Again:
                    break
                if len(frames) < 2:
                    continue
                rank = self._rank_of(frames[0])
                if rank is not None and frames[-1] == b"fence-ack":
                    pending.discard(rank)
        if pending:
            logger.warning(
                "[GMS liveness] cohort fence unacknowledged by ranks %s; "
                "writer-cohort retirement remains authoritative",
                sorted(pending),
            )
        else:
            logger.info("[GMS liveness] surviving ranks acknowledged cohort fence")

    def _fire(self, rank: int, reason: str) -> bool:
        if self._fired:
            return False
        self._fired = True
        with self._seen_changed:
            self._seen_changed.notify_all()
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
