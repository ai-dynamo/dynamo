# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dynamo.profiler.spica.planner import (
    SCALING_POLICIES,
    filter_scaling_policies,
    fpm_fields,
    load_sensitivity_fields,
    scaling_fields,
    throughput_intervals,
)


def test_scaling_policy_decode():
    p = SCALING_POLICIES["hybrid_180_5"]
    assert p.enable_throughput and p.enable_load
    assert p.throughput_interval_s == 180
    assert p.load_interval_s == 5
    assert SCALING_POLICIES["disabled"].enable_throughput is False
    assert SCALING_POLICIES["throughput_600_5"].throughput_interval_s == 600


def test_throughput_intervals_distinct_enabled_only():
    ids = [
        "disabled",
        "load_180_5",
        "throughput_180_5",
        "hybrid_600_5",
        "throughput_600_5",
    ]
    # enabled-throughput candidates: throughput_180_5 (180), hybrid_600_5 (600),
    # throughput_600_5 (600) -> {180, 600}
    assert throughput_intervals(ids) == [180, 600]


def test_throughput_intervals_empty_when_none_enabled():
    assert throughput_intervals(["disabled", "load_180_5", "load_180_10"]) == []


# --- composite knobs as raw dicts (the pin-the-unrolled-fields escape hatch) ---


def test_scaling_fields_from_preset_or_dict():
    assert scaling_fields("throughput_180_5") == {
        "enable_throughput_scaling": True,
        "enable_load_scaling": False,
        "throughput_adjustment_interval_seconds": 180,
        "load_adjustment_interval_seconds": 5,
    }
    # a raw dict is returned as the four normalized fields (custom interval=240)
    raw = {
        "enable_throughput_scaling": True,
        "enable_load_scaling": False,
        "throughput_adjustment_interval_seconds": 240,
        "load_adjustment_interval_seconds": 5,
    }
    assert scaling_fields(raw) == raw


def test_fpm_and_sensitivity_fields_from_preset_or_dict():
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


def test_throughput_intervals_mixes_presets_and_dicts():
    policies = [
        "disabled",
        "throughput_180_5",  # 180
        {
            "enable_throughput_scaling": True,
            "throughput_adjustment_interval_seconds": 240,
        },  # 240
        {
            "enable_load_scaling": True,
            "load_adjustment_interval_seconds": 5,
        },  # load-only -> ignored
    ]
    assert throughput_intervals(policies) == [180, 240]


def test_filter_scaling_policies():
    policies = [
        "disabled",
        "load_180_5",
        "throughput_180_5",
        "hybrid_600_5",
        {
            "enable_throughput_scaling": True,
            "throughput_adjustment_interval_seconds": 240,
        },  # dict, throughput
        {
            "enable_load_scaling": True,
            "load_adjustment_interval_seconds": 5,
        },  # dict, load-only
    ]
    # allow_throughput=True (goodput sweep / "sla") -> keep everything
    kept, dropped = filter_scaling_policies(policies, allow_throughput=True)
    assert kept == policies and dropped == []
    # allow_throughput=False (throughput/latency sweep) -> drop everything that enables
    # throughput scaling (presets AND dicts)
    kept, dropped = filter_scaling_policies(policies, allow_throughput=False)
    assert kept == [
        "disabled",
        "load_180_5",
        {"enable_load_scaling": True, "load_adjustment_interval_seconds": 5},
    ]
    assert dropped == [
        "throughput_180_5",
        "hybrid_600_5",
        {
            "enable_throughput_scaling": True,
            "throughput_adjustment_interval_seconds": 240,
        },
    ]
