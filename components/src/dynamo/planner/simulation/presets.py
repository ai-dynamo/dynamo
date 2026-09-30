# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Planner composite search presets used by the Sweeper adapter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ScalingPolicy:
    """Decoded Planner scaling-policy preset."""

    enable_throughput: bool
    enable_load: bool
    throughput_interval_s: int | None
    load_interval_s: int | None


SCALING_POLICIES: dict[str, ScalingPolicy] = {
    "disabled": ScalingPolicy(False, False, None, None),
    "throughput_180_5": ScalingPolicy(True, False, 180, 5),
    "throughput_600_5": ScalingPolicy(True, False, 600, 5),
    "load_180_5": ScalingPolicy(False, True, 180, 5),
    "load_180_10": ScalingPolicy(False, True, 180, 10),
    "hybrid_180_5": ScalingPolicy(True, True, 180, 5),
    "hybrid_600_5": ScalingPolicy(True, True, 600, 5),
}

FPM_SAMPLING: dict[str, dict[str, int]] = {
    "small": {"max_num_fpm_samples": 32, "fpm_sample_bucket_size": 4},
    "default": {"max_num_fpm_samples": 64, "fpm_sample_bucket_size": 16},
    "large": {"max_num_fpm_samples": 128, "fpm_sample_bucket_size": 16},
    "fine": {"max_num_fpm_samples": 128, "fpm_sample_bucket_size": 64},
}

LOAD_SENSITIVITY: dict[str, dict[str, int]] = {
    "aggressive": {"load_scaling_down_sensitivity": 70, "load_min_observations": 3},
    "default": {"load_scaling_down_sensitivity": 80, "load_min_observations": 5},
    "conservative": {"load_scaling_down_sensitivity": 90, "load_min_observations": 8},
}


def scaling_fields(entry: str | dict[str, Any]) -> dict[str, Any]:
    """Expand one scaling-policy preset or self-contained custom value."""

    if isinstance(entry, dict):
        throughput_interval = entry.get("throughput_adjustment_interval_seconds")
        if throughput_interval is not None:
            throughput_interval = int(throughput_interval)
        return {
            "enable_throughput_scaling": bool(
                entry.get("enable_throughput_scaling", False)
            ),
            "enable_load_scaling": bool(entry.get("enable_load_scaling", False)),
            "throughput_adjustment_interval_seconds": throughput_interval,
            "load_adjustment_interval_seconds": entry.get(
                "load_adjustment_interval_seconds"
            ),
        }
    preset = SCALING_POLICIES[entry]
    return {
        "enable_throughput_scaling": preset.enable_throughput,
        "enable_load_scaling": preset.enable_load,
        "throughput_adjustment_interval_seconds": preset.throughput_interval_s,
        "load_adjustment_interval_seconds": preset.load_interval_s,
    }


def fpm_fields(entry: str | dict[str, Any]) -> dict[str, Any]:
    """Expand one FPM-sampling preset or custom value."""

    return dict(entry) if isinstance(entry, dict) else dict(FPM_SAMPLING[entry])


def load_sensitivity_fields(entry: str | dict[str, Any]) -> dict[str, Any]:
    """Expand one load-sensitivity preset or custom value."""

    return dict(entry) if isinstance(entry, dict) else dict(LOAD_SENSITIVITY[entry])


def throughput_intervals(policies: list[str | dict[str, Any]]) -> list[int]:
    """Return distinct intervals for policies using predictive throughput scaling."""

    intervals: set[int] = set()
    for entry in policies:
        fields = scaling_fields(entry)
        interval = fields["throughput_adjustment_interval_seconds"]
        if fields["enable_throughput_scaling"] and interval is not None:
            intervals.add(int(interval))
    return sorted(intervals)
