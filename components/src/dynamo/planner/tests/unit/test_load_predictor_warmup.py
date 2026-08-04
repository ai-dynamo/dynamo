# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for load-predictor warmup failures."""

from types import SimpleNamespace

import pytest

from dynamo.planner.core.base import NativePlannerBase

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]


def test_invalid_warmup_trace_is_not_silently_ignored(tmp_path):
    trace = tmp_path / "invalid.jsonl"
    trace.write_text('{"timestamp": 0, "prompt_tokens": 100}\n')
    planner = object.__new__(NativePlannerBase)
    planner.config = SimpleNamespace(
        load_predictor_warmup_trace=str(trace),
        throughput_adjustment_interval_seconds=10,
    )

    with pytest.raises(ValueError, match="Unsupported warmup trace format"):
        planner._load_predictor_warmup_observations()
