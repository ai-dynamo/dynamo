# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Decode of the composite planner presets.

``planner_scaling_policy`` decodes (via :data:`SCALING_POLICIES`) to the four
planner fields it stands for (matching the Main Sweep Search Space table in the
design proposal): ``enable_throughput_scaling``, ``enable_load_scaling``,
``throughput_adjustment_interval_seconds``, ``load_adjustment_interval_seconds``.

Only policies with throughput scaling enabled drive the load-predictor sweep,
so :func:`throughput_intervals` extracts the distinct throughput intervals to
sweep over.

The two other named planner presets decode to numeric fields here too:
:data:`FPM_SAMPLING` (``planner_fpm_sampling`` -> ``max_num_fpm_samples`` +
``fpm_sample_bucket_size``) and :data:`LOAD_SENSITIVITY`
(``planner_load_sensitivity`` -> ``load_scaling_down_sensitivity`` +
``load_min_observations``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ScalingPolicy:
    enable_throughput: bool
    enable_load: bool
    throughput_interval_s: int | None
    load_interval_s: int | None


# preset id -> decoded policy (the {bool, bool, tput_interval, load_interval}
# tuples from the design proposal's planner_scaling_policy row).
SCALING_POLICIES: dict[str, ScalingPolicy] = {
    "disabled": ScalingPolicy(False, False, None, None),
    "throughput_180_5": ScalingPolicy(True, False, 180, 5),
    "throughput_600_5": ScalingPolicy(True, False, 600, 5),
    "load_180_5": ScalingPolicy(False, True, 180, 5),
    "load_180_10": ScalingPolicy(False, True, 180, 10),
    "hybrid_180_5": ScalingPolicy(True, True, 180, 5),
    "hybrid_600_5": ScalingPolicy(True, True, 600, 5),
}


# planner_fpm_sampling preset -> (max_num_fpm_samples, fpm_sample_bucket_size).
# Paired so the bucket size stays a perfect square and compatible with the
# sample count (design proposal's Main Sweep Search Space table; dynamo's
# planner validates fpm_sample_bucket_size is a perfect square).
FPM_SAMPLING: dict[str, dict[str, int]] = {
    "small": {"max_num_fpm_samples": 32, "fpm_sample_bucket_size": 4},
    "default": {"max_num_fpm_samples": 64, "fpm_sample_bucket_size": 16},
    "large": {"max_num_fpm_samples": 128, "fpm_sample_bucket_size": 16},
    "fine": {"max_num_fpm_samples": 128, "fpm_sample_bucket_size": 64},
}

# planner_load_sensitivity preset -> (load_scaling_down_sensitivity in 0..100,
# load_min_observations cold-start threshold): scale-down conservativeness and
# the regression cold-start point.
LOAD_SENSITIVITY: dict[str, dict[str, int]] = {
    "aggressive": {"load_scaling_down_sensitivity": 70, "load_min_observations": 3},
    "default": {"load_scaling_down_sensitivity": 80, "load_min_observations": 5},
    "conservative": {"load_scaling_down_sensitivity": 90, "load_min_observations": 8},
}


# A composite knob entry is either a preset id (str) or a dict pinning the
# unrolled fields directly. These decoders accept both and always return the flat
# field dict, so callers never branch on the entry type.


def scaling_fields(entry: str | dict[str, Any]) -> dict[str, Any]:
    """The four ``planner_scaling_policy`` fields for an entry (preset id or dict)."""
    if isinstance(entry, dict):
        return {
            "enable_throughput_scaling": bool(
                entry.get("enable_throughput_scaling", False)
            ),
            "enable_load_scaling": bool(entry.get("enable_load_scaling", False)),
            "throughput_adjustment_interval_seconds": entry.get(
                "throughput_adjustment_interval_seconds"
            ),
            "load_adjustment_interval_seconds": entry.get(
                "load_adjustment_interval_seconds"
            ),
        }
    p = SCALING_POLICIES[entry]
    return {
        "enable_throughput_scaling": p.enable_throughput,
        "enable_load_scaling": p.enable_load,
        "throughput_adjustment_interval_seconds": p.throughput_interval_s,
        "load_adjustment_interval_seconds": p.load_interval_s,
    }


def fpm_fields(entry: str | dict[str, Any]) -> dict[str, Any]:
    """The ``planner_fpm_sampling`` fields for an entry (preset id or dict)."""
    return dict(entry) if isinstance(entry, dict) else dict(FPM_SAMPLING[entry])


def load_sensitivity_fields(entry: str | dict[str, Any]) -> dict[str, Any]:
    """The ``planner_load_sensitivity`` fields for an entry (preset id or dict)."""
    return dict(entry) if isinstance(entry, dict) else dict(LOAD_SENSITIVITY[entry])


def filter_scaling_policies(
    policies: list[str | dict[str, Any]], *, allow_throughput: bool
) -> tuple[list[str | dict[str, Any]], list[str | dict[str, Any]]]:
    """Split ``planner_scaling_policy`` entries into ``(kept, dropped)``.

    Predictive throughput scaling only works under the planner's
    ``optimization_target="sla"`` (i.e. a goodput sweep). When the planner can't use
    SLA (``allow_throughput=False`` — a throughput/latency sweep), entries that enable
    throughput scaling are dropped. Handles preset ids and raw dicts uniformly.
    """
    if allow_throughput:
        return list(policies), []
    kept: list[str | dict[str, Any]] = []
    dropped: list[str | dict[str, Any]] = []
    for p in policies:
        (dropped if scaling_fields(p)["enable_throughput_scaling"] else kept).append(p)
    return kept, dropped


def throughput_intervals(policies: list[str | dict[str, Any]]) -> list[int]:
    """Distinct throughput-adjustment intervals (seconds), sorted, across the given
    ``planner_scaling_policy`` candidates (preset ids or dicts) that enable
    throughput scaling.

    Returns an empty list when no candidate enables throughput scaling — the
    load-predictor sweep is then unnecessary (it only matters for predictive
    throughput scaling).
    """
    intervals = set()
    for entry in policies:
        f = scaling_fields(entry)
        if f["enable_throughput_scaling"]:
            intervals.add(f["throughput_adjustment_interval_seconds"])
    return sorted(iv for iv in intervals if iv is not None)
