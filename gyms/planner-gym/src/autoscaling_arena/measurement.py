# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explicit arrival-cohort windows and provisioned GPU accounting."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from math import isfinite
from typing import Any


@dataclass(frozen=True)
class MeasurementWindow:
    """Half-open arrival interval in seconds on the replay clock.

    All eventual outcomes of arrivals in this interval are scored, including
    completions during drain. Rates and GPU-hours use this same interval.
    """

    start_s: float
    end_s: float

    def __post_init__(self) -> None:
        if (
            not isfinite(self.start_s)
            or not isfinite(self.end_s)
            or self.start_s < 0
            or self.end_s <= self.start_s
        ):
            raise ValueError("measurement window requires 0 <= start_s < end_s")

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


@dataclass(frozen=True)
class AllocationConfig:
    """Provisioned worker counts and GPUs per worker for native pool names.

    Pools are ``agg``, ``prefill``, or ``decode``. Initial counts apply at replay
    time zero; lifecycle snapshots replace them as workers start or stop.
    ``allow_idle_tail`` is valid only for a fixed allocation: it extends that
    allocation beyond replay completion to include an offered window's idle tail.
    """

    initial_workers: Mapping[str, int]
    gpus_per_worker: Mapping[str, int]
    allow_idle_tail: bool = False

    def __post_init__(self) -> None:
        pools = set(self.initial_workers)
        if not pools or pools != set(self.gpus_per_worker):
            raise ValueError("allocation requires matching nonempty pool mappings")
        if not pools <= {"agg", "prefill", "decode"} or (
            "agg" in pools and len(pools) != 1
        ):
            raise ValueError("allocation requires agg or prefill/decode pools")
        for pool in pools:
            workers = self.initial_workers[pool]
            gpus = self.gpus_per_worker[pool]
            if isinstance(workers, bool) or not isinstance(workers, int) or workers < 0:
                raise ValueError("initial_workers must contain nonnegative integers")
            if isinstance(gpus, bool) or not isinstance(gpus, int) or gpus <= 0:
                raise ValueError("gpus_per_worker must contain positive integers")


def window_gpu_hours(
    operations: Iterable[dict[str, Any]],
    allocation: AllocationConfig,
    window: MeasurementWindow,
    *,
    replay_duration_s: float,
) -> float:
    """Integrate exact provisioned pool sizes over a measurement window.

    Starting and draining workers consume GPUs until removed, just as in native
    replay. Controller decisions after replay termination are unknown; a dynamic
    window cannot extend past that point. Fixed allocation may explicitly retain
    its fleet through an idle tail. Lifecycle timestamps must use the replay clock.
    """
    if window.end_s > replay_duration_s and not allocation.allow_idle_tail:
        raise ValueError(
            "measurement window ends after replay; dynamic allocation "
            "cannot be extrapolated into an unobserved idle tail"
        )
    events = sorted(
        operations, key=lambda row: (row["at_ms"], row["operation_ordinal"])
    )
    counts = dict(allocation.initial_workers)
    previous_s = 0.0
    gpu_seconds = 0.0
    for operation in events:
        at_s = float(operation["at_ms"]) / 1000.0
        if not isfinite(at_s) or at_s < 0:
            raise ValueError("lifecycle timestamp must be finite and nonnegative")
        if at_s > window.end_s:
            break
        interval_s = max(0.0, at_s - max(previous_s, window.start_s))
        gpu_seconds += interval_s * sum(
            count * allocation.gpus_per_worker[pool] for pool, count in counts.items()
        )
        pool = operation["pool"]
        if pool not in counts:
            raise ValueError(f"lifecycle contains unconfigured pool: {pool}")
        state = operation["state_after_batch"]
        count = sum(len(state[key]) for key in ("active", "starting", "draining"))
        if allocation.allow_idle_tail and count != allocation.initial_workers[pool]:
            raise ValueError("allow_idle_tail requires a fixed worker allocation")
        counts[pool] = count
        previous_s = at_s
    interval_s = max(0.0, window.end_s - max(previous_s, window.start_s))
    gpu_seconds += interval_s * sum(
        count * allocation.gpus_per_worker[pool] for pool, count in counts.items()
    )
    return gpu_seconds / 3600.0
