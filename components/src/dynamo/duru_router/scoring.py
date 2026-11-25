# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Router event ingress and fresh score delivery, independent of accounting."""

import os
import time
from contextlib import contextmanager
from pathlib import Path

from dynamo.duru_router.events import open_feed, open_journal
from dynamo.duru_router.transport import (
    InProcessChannel,
    Receiver,
    ScoreSubscriber,
    channel_path,
)


def score_channel():
    value = os.environ.get("DURU_SCORES", "")
    if not value.strip():
        raise ValueError("DURU_SCORES must name the configured score IPC channel")
    return Path(value)


class Reader:
    """Read scores whose observation_time uses this host's monotonic clock."""

    def __init__(self, snapshot_source=None):
        self.snapshot_source = snapshot_source
        self.path = None if snapshot_source is not None else score_channel()
        self._subscriber = None
        self.snapshot = {}
        self.reason = "missing_snapshot"
        self.age = None
        self.pinned = None

    def pin(self, snapshot):
        @contextmanager
        def decision():
            previous = self.pinned
            self.pinned = snapshot
            try:
                yield
            finally:
                self.pinned = previous

        return decision()

    def refresh(self):
        if self.pinned is not None:
            return self.pinned
        if self.snapshot_source is not None:
            self.snapshot = self.snapshot_source()
        else:
            if self._subscriber is None:
                self._subscriber = ScoreSubscriber(channel_path(self.path))
            self.snapshot = self._subscriber.snapshot()
        self.age = (
            time.monotonic() - self.snapshot.get("observation_time", 0)
            if self.snapshot
            else None
        )
        if not self.snapshot:
            self.reason = "missing_snapshot"
        elif self.age < 0:
            self.reason = "future_snapshot"
        elif self.age > 2:
            self.reason = "stale_snapshot"
        else:
            self.reason = "fresh"
            return self.snapshot
        return {}

    def close(self):
        if self._subscriber is not None:
            self._subscriber.close()
        self.snapshot = {}


class EventStreamReader:
    """Read ordered local or IPC events and track producer progress.

    received_through is the latest received event/clock timestamp. Ordered
    delivery lets accounting use it as this source's safe observation cutoff.
    """

    def __init__(self, path, *, receiver=None):
        self.path = Path(path)
        self.sequence = 0
        self.received_through = 0.0
        self.closed = False
        self.receiver = receiver

    def read(self):
        if self.closed:
            return []
        if self.receiver is None:
            try:
                self.receiver = Receiver(channel_path(self.path))
            except FileNotFoundError:
                return []
        rows = self.receiver.read()
        for row in rows:
            if self.closed or row["sequence"] != self.sequence + 1:
                raise ValueError("Duru event stream sequence gap or event after footer")
            self.sequence += 1
            self.received_through = max(self.received_through, row["time"])
            self.closed = row["event"] == "footer"
        return rows

    def close(self):
        if self.receiver is not None:
            self.receiver.close()


def _program_id(fields):
    program_id = fields.get("program_id")
    if not isinstance(program_id, str) or not program_id:
        raise ValueError("Request arrival requires a nonempty program_id")
    return program_id


class Ingress:
    """Publish lifecycle events with program identity and monotonic timestamps."""

    def __init__(self, path, *, in_process=False):
        if in_process:
            recording = (
                open_journal(path, name="duru-ingress")
                if os.environ.get("DURU_RECORD_EVENTS") == "1"
                else None
            )
            self.journal = InProcessChannel(recording=recording)
        else:
            # Retain the external-producer path for standalone/replay harnesses.
            self.journal = open_feed(path, name="duru-ingress")
        self.sequence = 0

    def emit(self, event, timestamp, **fields):
        if event not in (
            "request_arrived",
            "request_dispatched",
            "request_completed",
            "request_cancelled",
            "request_failed",
            "agent_end",
            "agent_group_end",
            "u_clock",
            "footer",
        ):
            return
        if event == "request_arrived":
            _program_id(fields)
            keys = ("request_id", "program_id", "root", "agent", "prompt_tokens")
        elif event == "agent_end":
            keys = ("request_id", "program_id", "root", "agent")
        else:
            keys = ("request_id", "program_id", "root")
        self.sequence += 1
        self.journal.submit(
            "record",
            dict(
                sequence=self.sequence,
                event=event,
                time=timestamp,
                **{k: fields[k] for k in keys if k in fields},
            ),
        )

    def close(self):
        try:
            self.emit("footer", time.monotonic())
        finally:
            self.journal.close()
