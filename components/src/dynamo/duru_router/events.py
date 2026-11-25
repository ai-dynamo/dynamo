# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Capture accepted work and ordered clocks through an injected sink."""

import threading
import time

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


class CompactFeed:
    def __init__(self, sink):
        self.lock = threading.RLock()
        self.stop = threading.Event()
        self.error = None
        self.sequence = 0
        self.journal = sink
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
