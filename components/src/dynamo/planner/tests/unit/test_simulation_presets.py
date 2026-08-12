# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Optional-dependency preflight must run before the simulation imports.
# ruff: noqa: E402

"""Unit tests for Planner simulation presets moved out of Sweeper core."""

import pytest

pytest.importorskip(
    "aisimulate.sweeper",
    reason="AI Simulate is an optional Dynamo simulation dependency",
)

from dynamo.planner.simulation.presets import (
    SCALING_POLICIES,
    fpm_fields,
    load_sensitivity_fields,
    scaling_fields,
    throughput_intervals,
)

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]


def test_scaling_policy_decode() -> None:
    policy = SCALING_POLICIES["hybrid_180_5"]
    assert policy.enable_throughput and policy.enable_load
    assert policy.throughput_interval_s == 180
    assert policy.load_interval_s == 5
    assert not SCALING_POLICIES["disabled"].enable_throughput
    assert SCALING_POLICIES["throughput_600_5"].throughput_interval_s == 600


def test_throughput_intervals_are_distinct_and_enabled_only() -> None:
    assert throughput_intervals(
        [
            "disabled",
            "load_180_5",
            "throughput_180_5",
            "hybrid_600_5",
            "throughput_600_5",
        ]
    ) == [180, 600]
    assert throughput_intervals(["disabled", "load_180_5", "load_180_10"]) == []


def test_scaling_fields_accept_presets_and_custom_search_values() -> None:
    assert scaling_fields("throughput_180_5") == {
        "enable_throughput_scaling": True,
        "enable_load_scaling": False,
        "throughput_adjustment_interval_seconds": 180,
        "load_adjustment_interval_seconds": 5,
    }
    custom = {
        "enable_throughput_scaling": True,
        "enable_load_scaling": False,
        "throughput_adjustment_interval_seconds": 240,
        "load_adjustment_interval_seconds": 5,
    }
    assert scaling_fields(custom) == custom
    custom["throughput_adjustment_interval_seconds"] = 180.0
    assert scaling_fields(custom)["throughput_adjustment_interval_seconds"] == 180


def test_fpm_and_sensitivity_accept_presets_and_custom_search_values() -> None:
    assert fpm_fields("default") == {
        "max_num_fpm_samples": 64,
        "fpm_sample_bucket_size": 16,
    }
    assert fpm_fields({"max_num_fpm_samples": 96, "fpm_sample_bucket_size": 16}) == {
        "max_num_fpm_samples": 96,
        "fpm_sample_bucket_size": 16,
    }
    assert load_sensitivity_fields("aggressive") == {
        "load_scaling_down_sensitivity": 70,
        "load_min_observations": 3,
    }
    assert load_sensitivity_fields(
        {"load_scaling_down_sensitivity": 75, "load_min_observations": 4}
    ) == {
        "load_scaling_down_sensitivity": 75,
        "load_min_observations": 4,
    }


def test_throughput_intervals_mix_presets_and_custom_search_values() -> None:
    assert throughput_intervals(
        [
            "disabled",
            "throughput_180_5",
            {
                "enable_throughput_scaling": True,
                "throughput_adjustment_interval_seconds": 240,
            },
            {
                "enable_load_scaling": True,
                "load_adjustment_interval_seconds": 5,
            },
        ]
    ) == [180, 240]
