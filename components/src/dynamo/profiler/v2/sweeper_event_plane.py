# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Transport-agnostic Sweeper event emission, per DEP #15073.

This module owns the event schema (envelope + three event types), the
subject-naming convention, and the backpressure guarantee (emission must
never block ``Sweeper.run``'s synchronous, unguarded ``on_round`` callback).
It does not own how bytes actually leave the process -- that is delegated to
whatever ``EventEmitter`` implementation is injected, so this logic is
testable without a compiled ``dynamo._core`` binding and is not tied to
either transport proposal (event-plane vs. the earlier custom Unix socket).

Confirmed from real source, not assumed:
- ``Sweeper.run``'s ``on_round: Callable[[int, list[Candidate]], None]`` is
  called synchronously and unguarded inside the main search loop
  (``aisimulate/sweeper/search.py``) -- the reason emission must only
  enqueue and never perform I/O on the caller's thread.
- ``MaterializationResult``'s real shape (``.dgd``/``.experimental``) from
  ``materialize_dgd_from_candidate()`` is what ``search.resolved``'s
  ``candidate`` field carries -- no new shape invented here.

Not yet confirmed against a compiled binding: the exact Rust-side
constructor signature for the event-plane publisher this module will be
adapted to (see ``dynamo_event_plane_transport.py`` in this same directory
and its docstring for the precise, minimal contract this module needs from
it).
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Protocol

logger = logging.getLogger(__name__)

API_VERSION = "sweeper.dynamo.nvidia.com/v1alpha1"

# One subject suffix per event type (DEP #15073, "Subject naming"): a
# subscriber can subscribe to only the event types it cares about, rather
# than always receiving everything. Full subject is
# f"sweeper.{run_uid}.{SUBJECT_SUFFIX[event_type]}".
SUBJECT_SUFFIX = {
    "search.resolved": "search_resolved",
    "round.completed": "round_completed",
    "run.completed": "run_completed",
}


class EventEmitter(Protocol):
    """What any transport must provide. Implemented today by nothing real
    yet -- see ``dynamo_event_plane_transport.py`` for the event-plane
    adapter this is being built for, and the (already built, in a sibling
    proposal) custom-socket transport for the other implementation of this
    same protocol.
    """

    def publish(self, subject: str, payload: bytes) -> None:
        """Send one already-serialized envelope. May block -- callers of
        this protocol are responsible for keeping it off the search
        thread (see ``SweeperEventPublisher`` below)."""
        ...

    def close(self) -> None:
        """Release any underlying resources (connections, sockets, tasks)."""
        ...


class NonSerializableEventError(Exception):
    """A candidate event's ``data`` payload could not be JSON-serialized."""


@dataclass
class _PendingEvent:
    event_type: str
    data: dict[str, Any]
    sequence: int


class SweeperEventPublisher:
    """Emits Sweeper progress/outcome events without blocking the search.

    Usage:
        with SweeperEventPublisher(run_uid, emitter) as pub:
            pub.emit("round.completed", {"round_no": 1, "cumulative_candidates": 4})

    ``emit`` only enqueues onto a small, bounded, in-process queue; a
    background thread drains it and calls ``emitter.publish(...)``. If the
    queue fills (slow or absent consumer, or the underlying transport
    stalling), the oldest buffered event is dropped and a single warning is
    logged -- the search itself never waits on the transport.
    """

    def __init__(
        self,
        run_uid: str,
        emitter: EventEmitter,
        *,
        pending_queue_size: int = 100,
    ) -> None:
        self._run_uid = run_uid
        self._emitter = emitter
        self._queue: "queue.Queue[_PendingEvent]" = queue.Queue(maxsize=pending_queue_size)
        self._sequence = 0
        self._sequence_lock = threading.Lock()
        self._dropped_warning_logged = False
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._drain_loop, name=f"sweeper-event-publisher-{self._run_uid}", daemon=True
        )
        self._thread.start()

    def emit(self, event_type: str, data: dict[str, Any]) -> None:
        """Non-blocking. Never raises for a full queue (drops oldest instead)."""
        if event_type not in SUBJECT_SUFFIX:
            raise ValueError(f"unknown event type {event_type!r}")
        with self._sequence_lock:
            self._sequence += 1
            seq = self._sequence
        pending = _PendingEvent(event_type=event_type, data=data, sequence=seq)
        try:
            self._queue.put_nowait(pending)
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(pending)
            except queue.Full:
                pass
            if not self._dropped_warning_logged:
                logger.warning(
                    "SweeperEventPublisher: pending queue full for run %s; "
                    "dropping oldest buffered event(s). This warning is logged once per run.",
                    self._run_uid,
                )
                self._dropped_warning_logged = True

    def close(self, *, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None
        self._emitter.close()

    def __enter__(self) -> "SweeperEventPublisher":
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- internals --------------------------------------------------

    def _drain_loop(self) -> None:
        while not self._stop.is_set() or not self._queue.empty():
            try:
                pending = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            envelope = self._build_envelope(pending)
            try:
                payload = self._serialize(envelope)
            except NonSerializableEventError:
                logger.exception(
                    "SweeperEventPublisher: dropping non-serializable event "
                    "(run=%s, type=%s, sequence=%s)",
                    self._run_uid,
                    pending.event_type,
                    pending.sequence,
                )
                continue
            subject = f"sweeper.{self._run_uid}.{SUBJECT_SUFFIX[pending.event_type]}"
            try:
                self._emitter.publish(subject, payload)
            except Exception:
                logger.exception(
                    "SweeperEventPublisher: publish failed (run=%s, subject=%s, sequence=%s)",
                    self._run_uid,
                    subject,
                    pending.sequence,
                )

    def _build_envelope(self, pending: _PendingEvent) -> dict[str, Any]:
        return {
            "apiVersion": API_VERSION,
            "runUID": self._run_uid,
            "sequence": pending.sequence,
            "timestamp": _utc_now_iso(),
            "type": pending.event_type,
            "data": pending.data,
        }

    @staticmethod
    def _serialize(envelope: dict[str, Any]) -> bytes:
        try:
            return json.dumps(envelope).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise NonSerializableEventError(str(exc)) from exc


def _utc_now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def new_run_uid() -> str:
    return str(uuid.uuid4())
