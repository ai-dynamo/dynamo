# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The controller: drives every DP rank one lockstep iteration at a time.

Per iteration it drains its inbox (adds, aborts, releases, close), picks a
rank for each new request, sends one ``Cmd`` to every rank, collects one
``Reply`` from each, checks that the ranks agree (same ``seq``, same ``u``,
all closed in the CLOSE iteration and none before), and hands all outputs to
the engine in a single callback. While no rank has work it blocks on the
inbox with a heartbeat instead of spinning collectives.

Every ``Cmd`` carries the iteration's global flags (``Sync``), so ranks skip
DeepInfer's per-step ``has_unfinished_requests`` all-reduce. The controller
derives them from the previous iteration's replies and this iteration's adds
and CLOSE:

- ``unfinished``: some rank reported unfinished requests after its last step,
  or this iteration adds requests. It errs only towards stepping: a rank with
  nothing to schedule runs a placeholder step, as DeepInfer's DP ranks do.
  Work a rank admits from outside its scheduler (a finished decode read, a
  prompt that waited for KV space) flushes in that iteration and steps in the
  next, which the controller runs at once (``Reply.unfinished``).
- ``close``: the CLOSE iteration; every rank runs DeepInfer's collective
  ``close()`` in it.
- ``new_prompts``: this iteration adds requests; NORMAL ranks make the next
  step an ANY prefill vote point.
- ``pending``: some rank has requests outside its scheduler; the controller
  polls every ``pending_poll_s`` instead of idling.

All ranks get the same flags in the same iteration, so they make the same
step/flush choice, advance ``scheduler.decode_step`` alike (DeepInfer resets
or increments it in every ``schedule()``), reach the same NORMAL prefill votes
and close together.

It runs on a thread of the Dynamo worker, or in a process of its own
(``controller_process``), where the outputs stay encoded as the ranks sent
them (``raw_outputs``).
"""

from __future__ import annotations

import logging
import os
import queue
import threading
import time
from collections.abc import Callable, Sequence
from typing import Any, Optional, Union

import msgspec

from dynamo.deepinfer.protocol import (
    AddReq,
    AnyReply,
    Cmd,
    EngineMeta,
    Op,
    Output,
    RankFailed,
    RankHandle,
    RankStats,
    Sync,
    num_outputs,
    trace_lockstep,
)

logger = logging.getLogger(__name__)

OutputSink = Callable[[Any], None]
StatsSink = Callable[[int, RankStats], None]


class Submit(msgspec.Struct, tag="submit", array_like=True):
    add: AddReq
    dp_rank: Optional[int] = None


class Abort(msgspec.Struct, tag="abort", array_like=True):
    request_id: str


class Release(msgspec.Struct, tag="release", array_like=True):
    handoff: dict[str, Any]
    rank: Optional[int] = None


class Close(msgspec.Struct, tag="close", array_like=True):
    timeout: float = 60.0


# What the controller's inbox holds; a controller process gets them from the
# head as msgpack.
InboxItem = Union[Submit, Abort, Release, Close]


def exit_process(reason: str) -> None:
    """Collectives cannot recover from a lost or diverged rank: exit hard.

    A normal unwind would run DeepInfer's ``close()`` barrier and hang; the
    children die with us through ``PR_SET_PDEATHSIG``.
    """
    logger.critical("DeepInfer lockstep failed, exiting: %s", reason)
    logging.shutdown()
    os._exit(1)


class Controller:
    def __init__(
        self,
        handles: Sequence[RankHandle],
        meta: EngineMeta,
        *,
        engine_type: str = "NORMAL",
        idle_heartbeat_s: float = 1.0,
        pending_poll_s: float = 0.002,
        snapshot_interval_s: float = 1.0,
        recv_timeout_s: Optional[float] = None,
        on_stats: Optional[StatsSink] = None,
        on_fatal: Callable[[str], None] = exit_process,
        inbox: Any = None,
        raw_outputs: bool = False,
    ) -> None:
        self._handles = sorted(handles, key=lambda h: h.rank)
        if [h.rank for h in self._handles] != list(range(len(self._handles))):
            raise ValueError("rank handles must cover ranks 0..N-1 exactly")
        if sum(h.inline for h in self._handles) > 1:
            raise ValueError("at most one inline rank handle")
        # Inline handles execute inside recv(); receive from them first so the
        # other ranks are already running the same iteration's collectives.
        self._recv_order = sorted(self._handles, key=lambda h: not h.inline)
        self.dp_size = len(self._handles)
        self.meta = meta
        self.engine_type = engine_type
        self.idle_heartbeat_s = idle_heartbeat_s
        self.pending_poll_s = pending_poll_s
        self.snapshot_interval_s = snapshot_interval_s
        self.recv_timeout_s = recv_timeout_s
        # Called ~once per snapshot interval per rank from the controller thread.
        self.on_stats = on_stats
        self._on_fatal = on_fatal
        # InboxItems. A controller process passes its head socket (get/get_nowait).
        self._inbox: Any = queue.SimpleQueue() if inbox is None else inbox
        # Hand emit() each rank's outputs as the rank encoded them (handles
        # that return RawReply) instead of one merged dict of Outputs.
        self.raw_outputs = raw_outputs
        self._owner: dict[str, int] = {}
        self._stats = [RankStats() for _ in self._handles]
        self._seq = 0
        # From the last replies: some rank has unfinished requests after its
        # step; some rank has requests outside its scheduler.
        self._unfinished = False
        self._pending = False
        self._trace = (
            LockstepTrace(self.dp_size * meta.max_num_seqs)
            if trace_lockstep()
            else None
        )
        self._last_snapshot = 0.0
        self._emit: Optional[OutputSink] = None
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._closed = False
        self._failed = False
        # The CLOSE iteration ran.
        self._ended = False
        # Bounds every rank's reply in the CLOSE iteration; from the Close item.
        self.close_timeout: Optional[float] = None

    # -- called from the asyncio side -------------------------------------

    def start(self, emit: OutputSink) -> None:
        self._emit = emit
        self._thread = threading.Thread(
            target=self.run, name="di-controller", daemon=True
        )
        self._thread.start()

    def submit(self, add: AddReq, dp_rank: Optional[int] = None) -> None:
        self._inbox.put(Submit(add, dp_rank))

    def abort(self, request_id: str) -> None:
        self._inbox.put(Abort(request_id))

    def release(self, handoff: dict[str, Any], rank: Optional[int] = None) -> None:
        """Free a KV handoff no decode peer will read, through ``rank``: the
        prefill rank holding it, or any decode rank (None: rank 0), which
        notifies the peer."""
        self._inbox.put(Release(handoff, rank))

    def stats(self) -> list[RankStats]:
        return list(self._stats)

    def owner(self, request_id: str) -> Optional[int]:
        return self._owner.get(request_id)

    def close(self, timeout: float = 60.0) -> None:
        """Run one CLOSE iteration on every rank, then release the handles.

        Idempotent. Works before :meth:`start` (the iteration then runs on the
        calling thread) and after :meth:`run` returned from the CLOSE
        iteration. A rank that does not reply to CLOSE within ``timeout``
        fails the engine: a remote rank that dies after the CLOSE went out
        drops its connection on purpose as far as the head can tell.
        """
        with self._lock:
            if self._closed:
                return
            self._closed = True
        try:
            if not self._ended:
                self._inbox.put(Close(timeout))
                if self._thread is not None:
                    self._thread.join(timeout)
                    if self._thread.is_alive():
                        self._fail(f"CLOSE iteration did not finish within {timeout}s")
                elif not self._failed:
                    try:
                        while not self._iterate(0.0):
                            pass
                    except RankFailed as e:
                        self._fail(str(e))
        finally:
            for handle in self._handles:
                handle.close(0.0 if self._failed else timeout)

    # -- controller thread --------------------------------------------------

    def run(self, emit: Optional[OutputSink] = None) -> None:
        """Iterate until the CLOSE iteration ran; a failure is fatal. On the
        thread :meth:`start` made, or on the caller's (controller process)."""
        if emit is not None:
            self._emit = emit
        try:
            while not self._iterate(self._next_timeout()):
                pass
        except RankFailed as e:
            self._fail(str(e))
        except Exception as e:
            logger.exception("DeepInfer controller failed")
            self._fail(repr(e))

    def _fail(self, reason: str) -> None:
        # Once: the controller thread and close() may both give up on CLOSE.
        with self._lock:
            first, self._failed = not self._failed, True
        if first:
            self._on_fatal(reason)

    def _next_timeout(self) -> float:
        if self._unfinished:
            return 0.0
        if self._pending:
            return self.pending_poll_s
        return self.idle_heartbeat_s

    def _drain(self, timeout: float) -> list[Any]:
        items: list[Any] = []
        try:
            items.append(
                self._inbox.get(timeout=timeout)
                if timeout > 0
                else self._inbox.get_nowait()
            )
            while True:
                items.append(self._inbox.get_nowait())
        except queue.Empty:
            pass
        return items

    def _plan(
        self, items: list[Any]
    ) -> tuple[bool, list[list[AddReq]], list[list[str]], list[list[dict[str, Any]]]]:
        n = self.dp_size
        adds: list[list[AddReq]] = [[] for _ in range(n)]
        aborts: list[list[str]] = [[] for _ in range(n)]
        releases: list[list[dict[str, Any]]] = [[] for _ in range(n)]
        load = [s.num_running + s.num_waiting + s.num_pending for s in self._stats]
        closing = False
        for item in items:
            if isinstance(item, Submit):
                rid = item.add.request_id
                rank = (
                    item.dp_rank
                    if item.dp_rank is not None
                    else min(range(n), key=load.__getitem__)
                )
                adds[rank].append(item.add)
                load[rank] += 1
                self._owner[rid] = rank
            elif isinstance(item, Abort):
                rid = item.request_id
                rank = self._owner.pop(rid, None)
                if rank is None:
                    continue
                # Ranks apply aborts before adds: drop an add from this same
                # batch, and free the prefill KV its handoff names.
                dropped = [a for a in adds[rank] if a.request_id == rid]
                if not dropped:
                    aborts[rank].append(rid)
                adds[rank] = [a for a in adds[rank] if a.request_id != rid]
                releases[rank] += [a.handoff for a in dropped if a.handoff]
            elif isinstance(item, Release):
                rank = 0 if item.rank is None else item.rank
                if 0 <= rank < n:
                    releases[rank].append(item.handoff)
                else:
                    logger.warning("Not releasing a KV handoff of rank %d", rank)
            elif isinstance(item, Close):
                closing = True
                self.close_timeout = item.timeout
            else:
                logger.error("Ignoring unknown DeepInfer controller item %r", item)
        if closing:
            for rank in range(n):
                releases[rank] += [a.handoff for a in adds[rank] if a.handoff]
            adds = [[] for _ in range(n)]
        return closing, adds, aborts, releases

    def _iterate(self, timeout: float) -> bool:
        """One lockstep iteration; returns True after the CLOSE iteration."""
        closing, adds, aborts, releases = self._plan(self._drain(timeout))
        added = any(adds)
        sync = Sync(
            unfinished=self._unfinished or added,
            close=closing,
            new_prompts=added,
            pending=self._pending,
        )
        op = Op.CLOSE if closing else Op.STEP
        started = time.monotonic()
        for handle in self._handles:
            handle.send(
                Cmd(
                    seq=self._seq,
                    op=op,
                    adds=adds[handle.rank],
                    aborts=aborts[handle.rank],
                    releases=releases[handle.rank],
                    sync=sync,
                )
            )
        sent = time.monotonic()
        timeout = self.close_timeout if closing else self.recv_timeout_s
        replies = {h.rank: h.recv(timeout) for h in self._recv_order}
        received = time.monotonic()
        self._check(replies, closing)

        ordered = [replies[rank] for rank in range(self.dp_size)]
        for rank, reply in enumerate(ordered):
            if not closing:
                self._stats[rank] = reply.stats
            for rid in reply.finished:
                self._owner.pop(rid, None)
        self._unfinished = any(reply.unfinished for reply in ordered)
        self._pending = any(reply.pending for reply in ordered)
        outputs = self._deliver(ordered)
        if self._trace is not None:
            running = sum(s.num_running for s in self._stats)
            self._trace.record(self._seq, started, sent, received, outputs, running)
        self._publish_stats()
        self._seq += 1
        self._ended = closing
        return closing

    def _deliver(self, replies: list[AnyReply]) -> int:
        """Hand this iteration's outputs to ``emit`` in one call; returns how
        many there were (counted when tracing only, for raw outputs)."""
        if self.raw_outputs:
            # An empty msgpack array is one byte.
            frames = [r.outputs for r in replies if len(r.outputs) > 1]
            if frames and self._emit is not None:
                self._emit(frames)
            return sum(map(num_outputs, frames)) if self._trace is not None else 0
        merged: dict[str, Output] = {}
        for reply in replies:
            for out in reply.outputs:
                merged[out.request_id] = out
        if merged and self._emit is not None:
            self._emit(merged)
        return len(merged)

    def _check(self, replies: dict[int, AnyReply], closing: bool) -> None:
        for rank, reply in replies.items():
            if reply.error is not None:
                raise RankFailed(f"rank {rank}: {reply.error}")
            if reply.seq != self._seq:
                raise RankFailed(
                    f"rank {rank} replied to seq {reply.seq}, expected {self._seq}"
                )
        us = {rank: reply.u for rank, reply in replies.items()}
        if len(set(us.values())) != 1:
            raise RankFailed(f"ranks disagree on has_unfinished_requests: {us}")
        closed = [rank for rank, reply in replies.items() if reply.closed]
        if len(closed) != (len(replies) if closing else 0):
            raise RankFailed(
                f"ranks {sorted(closed)} closed in iteration {self._seq}; every "
                "rank must close in the CLOSE iteration and none before"
            )

    def _publish_stats(self) -> None:
        now = time.monotonic()
        if (
            self.on_stats is None
            or now - self._last_snapshot < self.snapshot_interval_s
        ):
            return
        self._last_snapshot = now
        for rank, stats in enumerate(self._stats):
            self.on_stats(rank, stats)


class LockstepTrace:
    """``DYN_DEEPINFER_TRACE_LOCKSTEP=1`` on the controller: one line per
    iteration (period since the previous iteration's first send, the span of
    the sends, the wait for the last reply, outputs, running requests) and
    every ``interval_s`` a summary, also of the full-batch iterations alone
    (every rank at ``max_num_seqs``)."""

    def __init__(self, full_batch: int, interval_s: float = 10.0) -> None:
        self.full_batch = full_batch
        self.interval_s = interval_s
        self._to_wall = time.time() - time.monotonic()
        self._last: Optional[float] = None
        self._since = time.monotonic()
        # (period, send, recv) in ms, outputs, running
        self._rows: list[tuple[float, float, float, int, int]] = []

    def record(
        self,
        seq: int,
        started: float,
        sent: float,
        received: float,
        outputs: int,
        running: int,
    ) -> None:
        period = 0.0 if self._last is None else (started - self._last) * 1e3
        self._last = started
        send, recv = (sent - started) * 1e3, (received - sent) * 1e3
        logger.info(
            "lockstep head seq=%d t=%.6f period_ms=%.3f send_ms=%.3f "
            "recv_ms=%.3f outputs=%d running=%d",
            seq,
            started + self._to_wall,
            period,
            send,
            recv,
            outputs,
            running,
        )
        self._rows.append((period, send, recv, outputs, running))
        if started - self._since >= self.interval_s:
            self._summary()
            self._rows = []
            self._since = started

    def _summary(self) -> None:
        rows = self._rows[1:] if self._rows and self._rows[0][0] == 0.0 else self._rows
        full = [r[0] for r in rows if r[4] >= self.full_batch]
        logger.info(
            "lockstep summary iterations=%d period_ms p50=%.2f p90=%.2f p99=%.2f "
            "full_batch=%d full_period_ms p50=%.2f p90=%.2f p99=%.2f "
            "send_ms p50=%.3f p99=%.3f max=%.3f recv_ms p50=%.2f p99=%.2f "
            "outputs_per_iteration=%.1f",
            len(rows),
            *_percentiles([r[0] for r in rows], (50, 90, 99)),
            len(full),
            *_percentiles(full, (50, 90, 99)),
            *_percentiles([r[1] for r in rows], (50, 99, 100)),
            *_percentiles([r[2] for r in rows], (50, 99)),
            sum(r[3] for r in rows) / max(1, len(rows)),
        )


def _percentiles(values: list[float], qs: tuple[int, ...]) -> list[float]:
    if not values:
        return [0.0] * len(qs)
    ordered = sorted(values)
    return [ordered[min(len(ordered) - 1, len(ordered) * q // 100)] for q in qs]
