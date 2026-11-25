# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Capture router lifecycle observations through an injected sink."""

import time


def _program_id(fields):
    program_id = fields.get("program_id")
    if not isinstance(program_id, str) or not program_id:
        raise ValueError("Request arrival requires a nonempty program_id")
    return program_id


class Ingress:
    """Publish request lifecycle events with explicit router program identity."""

    def __init__(self, sink):
        self.journal = sink
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
        keys = (
            ("request_id", "program_id", "root", "agent", "prompt_tokens")
            if event == "request_arrived"
            else ("request_id", "program_id", "root")
        )
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
            self.emit("footer", time.time())
        finally:
            self.journal.close()
