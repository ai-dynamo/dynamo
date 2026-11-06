# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""XPU coverage for the native planner's FPM regression models."""

from pathlib import Path

import pytest

from dynamo.planner.config.defaults import SubComponentType
from dynamo.planner.core.perf_model import DecodeRegressionModel, PrefillRegressionModel
from dynamo.planner.monitoring.perf_metrics import _convert_profiling_data_to_fpms

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]

B60_PROFILE_DIR = Path(__file__).parent.parent / "data/profiling_results/B60_TP1P_TP1D"


def _prefill_model() -> PrefillRegressionModel:
    fpms = _convert_profiling_data_to_fpms(
        str(B60_PROFILE_DIR), SubComponentType.PREFILL
    )
    model = PrefillRegressionModel(max_num_fpm_samples=50, min_observations=5)
    model.load_benchmark_fpms(fpms)
    return model


def _decode_model() -> DecodeRegressionModel:
    fpms = _convert_profiling_data_to_fpms(
        str(B60_PROFILE_DIR), SubComponentType.DECODE
    )
    model = DecodeRegressionModel(max_num_fpm_samples=50, min_observations=5)
    model.load_benchmark_fpms(fpms)
    return model


def test_b60_prefill_profile_trains_ttft_model():
    model = _prefill_model()

    assert model.has_sufficient_data()
    estimate = model.estimate_next_ttft(
        queued_prefill_tokens=3_000, max_num_batched_tokens=2_048
    )
    assert estimate is not None
    assert estimate > 0


def test_b60_prefill_queue_increases_estimated_ttft():
    model = _prefill_model()

    idle = model.estimate_next_ttft(0, max_num_batched_tokens=2_048)
    queued = model.estimate_next_ttft(10_000, max_num_batched_tokens=2_048)

    assert idle is not None
    assert queued is not None
    assert queued > idle


def test_b60_decode_profile_trains_itl_model():
    model = _decode_model()

    assert model.has_sufficient_data()
    estimate = model.estimate_next_itl(
        scheduled_decode_kv=10_000, queued_decode_kv=2_000
    )
    assert estimate is not None
    assert estimate > 0


def test_b60_decode_queue_increases_estimated_itl():
    model = _decode_model()

    baseline = model.estimate_next_itl(10_000, 0)
    queued = model.estimate_next_itl(10_000, 10_000)

    assert baseline is not None
    assert queued is not None
    assert queued > baseline
