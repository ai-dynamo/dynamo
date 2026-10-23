#  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

"""Bounded same-host timing transport for the HTTP request's TTFT timeline."""

import sys
import time
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

TIMING_BATCH_VERSION = 1
MAX_TIMING_INTERVALS = 64
_MAX_NS = (1 << 64) - 1


@lru_cache(maxsize=1)
def stage_registry() -> dict[str, int]:
    """Read the canonical Rust registry; Python never maintains a second list."""
    from dynamo._core import ttft_stage_registry

    return {name: stage_id for stage_id, name in ttft_stage_registry()}


def clock_domain() -> str:
    """Only local ProcessPoolExecutor workers share this supported clock domain."""
    if sys.platform not in {"linux", "darwin"}:
        return "unsupported"
    return sys.platform + ":" + time.get_clock_info("monotonic").implementation


@dataclass(frozen=True)
class TimingInterval:
    stage_id: int
    start_ns: int
    end_ns: int


@dataclass(frozen=True)
class TimingBatch:
    version: int
    clock_domain: str
    intervals_ns: tuple[TimingInterval, ...]
    truncated: bool = False
    invalid: bool = False
    coarse_reasons: tuple[str, ...] = ()


class WorkerTiming:
    """Picklable fixed-stage intervals; allocated only for enabled requests."""

    ttft_enabled = True

    def __init__(self) -> None:
        self.intervals_ns: list[TimingInterval] = []
        self.truncated = False
        self.started = 0
        self.invalid = False
        self.coarse_reasons: set[str] = set()
        self.domain = clock_domain()

    @contextmanager
    def start_ttft_stage(self, stage: str):
        stage_id = stage_registry().get(stage)
        if stage_id is None:
            raise ValueError(f"Unsupported TTFT stage: {stage}")
        if self.domain == "unsupported":
            self.coarse_reasons.add("unsupported_clock")
            yield
            return
        if self.started >= MAX_TIMING_INTERVALS:
            self.truncated = True
            yield
            return
        self.started += 1
        start_ns = time.monotonic_ns()
        try:
            yield
        finally:
            end_ns = time.monotonic_ns()
            if len(self.intervals_ns) < MAX_TIMING_INTERVALS:
                self.intervals_ns.append(TimingInterval(stage_id, start_ns, end_ns))
            else:
                self.truncated = True

    def mark_ttft_quality(self, quality: str, reason: str) -> None:
        if quality == "coarse" and reason in {
            "opaque_template",
            "unsupported_clock",
            "missing_coverage",
        }:
            self.coarse_reasons.add(reason)
        else:
            self.invalid = True

    def batch(self) -> TimingBatch:
        return TimingBatch(
            TIMING_BATCH_VERSION,
            self.domain,
            tuple(self.intervals_ns),
            self.truncated,
            self.invalid,
            tuple(sorted(self.coarse_reasons)),
        )


class _TtftContext:
    """Cache native methods once per enabled request, outside the hot path."""

    ttft_enabled = True

    def __init__(self, context: Any) -> None:
        self.start_ttft_stage = context.start_ttft_stage
        self.record_ttft_batch_ago = context.record_ttft_batch_ago
        self.mark_ttft_quality = context.mark_ttft_quality


def ttft_context(context: Any | None) -> Any | None:
    if isinstance(context, _TtftContext):
        return context
    if not getattr(context, "ttft_enabled", False):
        return None
    return _TtftContext(context)


def ttft_stage(context: Any | None, stage: str) -> AbstractContextManager:
    if not getattr(context, "ttft_enabled", False):
        return nullcontext()
    return context.start_ttft_stage(stage)


def ttft_coarse(context: Any | None, reason: str) -> None:
    if getattr(context, "ttft_enabled", False):
        context.mark_ttft_quality("coarse", reason)


@contextmanager
def ttft_renderer_stage(context: Any | None):
    """Classify one renderer call after learning whether it also tokenized.

    This never executes the renderer/tokenizer twice or mutates backend objects.
    The result flag is set by the caller only when token IDs were returned.
    """
    result = [False]
    if not getattr(context, "ttft_enabled", False):
        yield result
        return
    start_ns = time.monotonic_ns()
    try:
        yield result
    finally:
        end_ns = time.monotonic_ns()
        stage = "template_tokenize" if result[0] else "template_render"
        context.record_ttft_batch_ago(
            [(stage_registry()[stage], end_ns - start_ns, 0)],
            ("opaque_template",) if result[0] else (),
        )


def import_worker_batch(context: Any, batch: Any, submitted_ns: int) -> None:
    """Validate local worker intervals and cross PyO3 once using integer ages.

    One receive clock read anchors every interval. Rust takes one corresponding
    Instant; the FFI scheduling delay remains a small, nonzero alignment error.
    Untrusted timing never changes the inference result.
    """
    now_ns = time.monotonic_ns()

    def degraded(reason: str) -> None:
        context.mark_ttft_quality("degraded", reason)

    if type(batch) is not TimingBatch:
        degraded("invalid_batch")
        return
    if type(batch.version) is not int or batch.version != TIMING_BATCH_VERSION:
        degraded("invalid_version")
        return
    if type(batch.clock_domain) is not str:
        degraded("invalid_batch")
        return
    if batch.clock_domain == "unsupported" or clock_domain() == "unsupported":
        context.mark_ttft_quality("coarse", "unsupported_clock")
        return
    if batch.clock_domain != clock_domain():
        degraded("clock_mismatch")
        return
    # Exact types make the size checks authoritative: tuple/dataclass subclasses
    # can override __len__, __iter__, or attribute access before validation.
    if (
        type(batch.truncated) is not bool
        or type(batch.invalid) is not bool
        or type(batch.intervals_ns) is not tuple
        or type(batch.coarse_reasons) is not tuple
        or len(batch.coarse_reasons) > 3
    ):
        degraded("invalid_batch")
        return
    if batch.truncated or len(batch.intervals_ns) > MAX_TIMING_INTERVALS:
        degraded("interval_limit")
        return
    if batch.invalid:
        degraded("invalid_batch")
        return
    allowed = set(stage_registry().values())
    ages = []
    for interval in batch.intervals_ns:
        if type(interval) is not TimingInterval:
            degraded("invalid_batch")
            return
        if type(interval.stage_id) is not int or interval.stage_id not in allowed:
            degraded("invalid_stage")
            return
        if any(
            type(value) is not int or not 0 <= value <= _MAX_NS
            for value in (interval.start_ns, interval.end_ns)
        ):
            degraded("timestamp_overflow")
            return
        if not submitted_ns <= interval.start_ns <= interval.end_ns <= now_ns:
            degraded("invalid_interval")
            return
        ages.append(
            (interval.stage_id, now_ns - interval.start_ns, now_ns - interval.end_ns)
        )
    if any(
        type(reason) is not str
        or reason not in {"opaque_template", "unsupported_clock", "missing_coverage"}
        for reason in batch.coarse_reasons
    ):
        degraded("invalid_batch")
        return
    context.record_ttft_batch_ago(ages, batch.coarse_reasons)
