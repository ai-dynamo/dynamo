# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in, sampled scalar timing for asynchronous workflow paths."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Mapping

_TRACE_ENV = "DYN_WORKFLOW_PERF_TRACE"
_SAMPLE_EVERY_ENV = "DYN_WORKFLOW_PERF_SAMPLE_EVERY"
_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_FALSE_VALUES = frozenset({"0", "false", "no", "off"})


@dataclass(frozen=True)
class WorkflowPerfTracer:
    """Emit joinable JSON timing records without spanning ``await`` in NVTX."""

    enabled: bool
    sample_every: int

    @classmethod
    def from_environment(
        cls, environment: Mapping[str, str] | None = None
    ) -> "WorkflowPerfTracer":
        values = os.environ if environment is None else environment
        raw_enabled = values.get(_TRACE_ENV, "0").strip().lower()
        if raw_enabled in _TRUE_VALUES:
            enabled = True
        elif raw_enabled in _FALSE_VALUES:
            enabled = False
        else:
            raise ValueError(
                f"{_TRACE_ENV} must be one of "
                f"{sorted(_TRUE_VALUES | _FALSE_VALUES)}, got {raw_enabled!r}"
            )

        raw_sample_every = values.get(_SAMPLE_EVERY_ENV, "32")
        try:
            sample_every = int(raw_sample_every)
        except ValueError as error:
            raise ValueError(
                f"{_SAMPLE_EVERY_ENV} must be a positive integer"
            ) from error
        if sample_every < 1:
            raise ValueError(f"{_SAMPLE_EVERY_ENV} must be a positive integer")
        return cls(enabled=enabled, sample_every=sample_every)

    def samples(self, trace_id: str) -> bool:
        if not self.enabled:
            return False
        digest = hashlib.blake2s(trace_id.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, byteorder="big") % self.sample_every == 0

    def emit(
        self,
        logger: logging.Logger,
        event: str,
        trace_id: str,
        *,
        force: bool = False,
        **fields: Any,
    ) -> None:
        if not self.enabled or (not force and not self.samples(trace_id)):
            return
        payload = {
            "event": event,
            "trace_id": trace_id,
            **fields,
        }
        logger.info("workflow_perf %s", json.dumps(payload, sort_keys=True))


WORKFLOW_PERF_TRACE = WorkflowPerfTracer.from_environment()
