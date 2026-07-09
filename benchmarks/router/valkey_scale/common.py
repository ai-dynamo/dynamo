#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: F401
"""Measure authoritative Valkey-router throughput while scaling frontends.

This driver deliberately delegates every sample to
``valkey_router_aiperf.py``.  Each child invocation creates and tears down a
new topology (four logical mock workers split across the configured number of
OS processes, two module-loaded Valkey servers, and the requested number of
frontends), so state from one sample cannot improve a
later one.

The default schedule is a balanced cyclic interleave.  With the default three
frontend counts and three repetitions it runs ``1,2,3``, then ``2,3,1``, then
``3,1,2``.  That preserves an initially increasing scale-up pass while making
each frontend count appear once in every position, reducing monotonic host
warm-up or thermal-drift bias.

Example:

    DYNAMO_GPU_PARALLEL_DOWNLOADS_READY=1 \\
      .venv/bin/python benchmarks/router/valkey_frontend_scale.py \\
      --output-dir /tmp/valkey-frontend-scale

The output directory contains all child harness artifacts, a machine-readable
summary, CSV tables, and PNG/SVG RPS plots.  A plot is generated only when all
samples pass the child harness's strict aiperf-validity checks.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shlex
import signal
import subprocess
import sys
import time
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Any


REPO = Path(__file__).resolve().parents[3]
DEFAULT_HARNESS = REPO / "benchmarks/router/valkey_router_aiperf.py"
DEFAULT_MODEL = "Qwen/Qwen3-0.6B"
SUMMARY_VERSION = 1
LOGICAL_MOCKER_WORKERS = 4

# The child harness owns these settings.  Allowing them in arbitrary forwarded
# arguments would make a supposedly authoritative Valkey HA scale point mean
# something different from every other point.
PROTECTED_HARNESS_ARGUMENTS = (
    "--arm",
    "--runs",
    "--frontend-count",
    "--mocker-processes",
    "--output-dir",
    "--valkey-authoritative-admission",
    "--valkey-admission-lease-ms",
    "--valkey-gc-interval-ms",
    "--valkey-gc-inspection-budget",
    "--frontend-cpus",
    "--mocker-cpus",
    "--valkey-cpus",
    "--aiperf-cpus",
)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def cpu_list(value: str) -> str:
    """Validate the common ``taskset --cpu-list`` comma/range syntax."""

    normalized: list[str] = []
    for raw_part in value.split(","):
        part = raw_part.strip()
        if not part:
            raise argparse.ArgumentTypeError("CPU list contains an empty segment")
        bounds = part.split("-")
        if len(bounds) == 1:
            try:
                cpu = int(bounds[0])
            except ValueError as error:
                raise argparse.ArgumentTypeError(
                    f"invalid CPU number {part!r}"
                ) from error
            if cpu < 0:
                raise argparse.ArgumentTypeError("CPU numbers must be non-negative")
        elif len(bounds) == 2:
            try:
                start, end = (int(bound) for bound in bounds)
            except ValueError as error:
                raise argparse.ArgumentTypeError(
                    f"invalid CPU range {part!r}"
                ) from error
            if start < 0 or end < start:
                raise argparse.ArgumentTypeError(
                    f"CPU range must satisfy 0 <= start <= end: {part!r}"
                )
        else:
            raise argparse.ArgumentTypeError(f"invalid CPU range {part!r}")
        normalized.append(part)
    return ",".join(normalized)


def frontend_counts(value: str) -> tuple[int, ...]:
    """Parse an ordered, duplicate-free comma-separated frontend-count list."""

    raw_counts = [part.strip() for part in value.split(",") if part.strip()]
    if not raw_counts:
        raise argparse.ArgumentTypeError("must contain at least one frontend count")
    try:
        counts = tuple(int(part) for part in raw_counts)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "frontend counts must be comma-separated integers"
        ) from error
    if any(count < 1 for count in counts):
        raise argparse.ArgumentTypeError("frontend counts must all be at least 1")
    if len(set(counts)) != len(counts):
        raise argparse.ArgumentTypeError("frontend counts must not contain duplicates")
    return counts


def finite_number(value: Any) -> float | None:
    """Return a finite numeric value, excluding booleans and NaNs."""

    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def nested_metric(
    metrics: Mapping[str, Any], metric_name: str, statistic: str
) -> float | None:
    metric = metrics.get(metric_name)
    if not isinstance(metric, Mapping):
        return None
    return finite_number(metric.get(statistic))


def commandstats_fields(value: Any) -> dict[str, str] | None:
    """Parse Valkey's comma-separated ``INFO commandstats`` value."""

    if not isinstance(value, str):
        return None
    fields: dict[str, str] = {}
    for part in value.split(","):
        key, separator, field_value = part.partition("=")
        if not separator or not key or not field_value:
            return None
        fields[key] = field_value
    return fields


def replication_offset(fields: Mapping[str, Any], *names: str) -> int | None:
    """Return the first valid replication offset from a Valkey INFO mapping."""

    for name in names:
        raw_value = fields.get(name)
        if raw_value is None:
            continue
        try:
            return int(raw_value)
        except (TypeError, ValueError):
            continue
    return None


def linear_percentile(values: Sequence[float], quantile: float) -> float | None:
    """Match the aiperf harness's stable, interpolation-based percentile."""

    if not values:
        return None
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be in [0, 1]")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def build_interleaved_schedule(
    counts: Sequence[int], repetitions: int
) -> list[tuple[int, int]]:
    """Return ``(repetition, frontend_count)`` samples in balanced cyclic order.

    Every full group of ``len(counts)`` repetitions puts each count in each
    ordinal position once.  The first pass follows the requested order, which
    makes a user-provided ``1,2,3`` list easy to inspect during a long run.
    """

    if not counts:
        raise ValueError("counts must not be empty")
    if repetitions < 1:
        raise ValueError("repetitions must be at least 1")
    schedule: list[tuple[int, int]] = []
    for repetition in range(1, repetitions + 1):
        offset = (repetition - 1) % len(counts)
        rotated = tuple(counts[offset:]) + tuple(counts[:offset])
        schedule.extend((repetition, count) for count in rotated)
    return schedule
