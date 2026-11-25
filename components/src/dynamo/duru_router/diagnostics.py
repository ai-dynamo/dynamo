# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in diagnostics; never used as input to scheduling or U accounting."""

import json
import logging
import math
import os
import time

logger = logging.getLogger(__name__)


def enabled():
    return os.environ.get("DURU_DIAGNOSTICS") == "1"


def emit(event, **fields):
    if logger.isEnabledFor(logging.INFO):
        # Invalid scores are diagnostic evidence, not a reason to fail admission.
        fields = {
            key: (
                None if isinstance(value, float) and not math.isfinite(value) else value
            )
            for key, value in fields.items()
        }
        logger.info("duru.%s %s", event, json.dumps(fields, allow_nan=False))


class PerformanceLogger:
    """Five-second summaries with constant-size counter state, no sample history."""

    def __init__(self):
        self.enabled = enabled()
        self.previous = None

    def observe(self, snapshot, pending_events):
        if not self.enabled or not logger.isEnabledFor(logging.INFO):
            return
        now = time.monotonic()
        current = (now, snapshot["output_tokens"], snapshot["accepted_steps"])
        if self.previous is None:
            self.previous = current
            return
        elapsed = now - self.previous[0]
        if elapsed < 5:
            return
        active = [r for r in snapshot["roots"].values() if r["active_agents"]]
        measured = [r["U"] for r in active if r["U"] is not None]
        emit(
            "performance",
            time=snapshot["time"],
            run_id=snapshot["run_id"],
            interval_s=elapsed,
            output_tokens_per_s=(current[1] - self.previous[1]) / elapsed,
            accepted_steps_per_s=(current[2] - self.previous[2]) / elapsed,
            output_tokens=current[1],
            accepted_steps=current[2],
            active_requests=len(snapshot["requests"]),
            active_roots=len(active),
            measured_roots=len(measured),
            u_min=min(measured, default=None),
            u_max=max(measured, default=None),
            u_mean=sum(measured) / len(measured) if measured else None,
            observation_time=snapshot["observation_time"],
            accounting_lag_s=snapshot["time"] - snapshot["observation_time"],
            pending_events=pending_events,
            accounting_state=snapshot.get("accounting_state"),
        )
        self.previous = current
