# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from dynamo import _core
from dynamo.common.forward_pass_metrics import (
    ForwardPassMetrics,
    QueuedRequestMetrics,
    ScheduledRequestMetrics,
)

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.parallel,
    pytest.mark.pre_merge,
    pytest.mark.unit,
]

RustEnginePerfModel = getattr(_core, "RustEnginePerfModel", None)
if RustEnginePerfModel is None:
    pytest.skip(
        "RustEnginePerfModel requires the aic-forward-pass Cargo feature",
        allow_module_level=True,
    )


def decode_fpm(
    num_decode_requests: int,
    sum_decode_kv_tokens: int,
    wall_time: float = 0.0,
    queued_decode: int = 0,
    queued_decode_kv_tokens: int = 0,
) -> ForwardPassMetrics:
    return ForwardPassMetrics(
        wall_time=wall_time,
        scheduled_requests=ScheduledRequestMetrics(
            num_decode_requests=num_decode_requests,
            sum_decode_kv_tokens=sum_decode_kv_tokens,
        ),
        queued_requests=QueuedRequestMetrics(
            num_decode_requests=queued_decode,
            sum_decode_kv_tokens=queued_decode_kv_tokens,
        ),
    )


def test_regression_decode_helpers_and_capacity() -> None:
    model = _core.RustEnginePerfModel.best_available(
        worker_type="decode",
        limits=_core.EnginePerfLimits(),
        options=_core.RustEnginePerfOptions(min_observations=2, max_observations=16),
        bootstrap_fpms=[
            [decode_fpm(1, 100, 0.010)],
            [decode_fpm(2, 200, 0.020)],
        ],
    )

    base = decode_fpm(1, 100)
    noisy_queue = decode_fpm(
        1,
        100,
        queued_decode=64,
        queued_decode_kv_tokens=1_000_000,
    )

    assert model.estimate_forward_pass_time([base]) is not None
    assert model.estimate_forward_pass_time({0: base}) is not None
    assert model.get_scheduled_decode_itl([base]) == pytest.approx(
        model.get_scheduled_decode_itl([noisy_queue])
    )
    assert model.get_queued_prefill_time([base]) is None

    diagnostics = json.loads(model.diagnostics())
    assert isinstance(diagnostics, dict)

    capacity = model.find_engine_capacity_rps(
        _core.EngineCapacityRequest(isl=100, osl=10, itl_sla_ms=1000.0)
    )
    assert capacity is not None
    assert capacity.rps > 0.0
    assert capacity.itl_ms is not None
    assert capacity.ttft_ms is None


def test_tune_with_fpms_accepts_rank_mapping() -> None:
    model = _core.RustEnginePerfModel.best_available(
        worker_type="decode",
        limits=_core.EnginePerfLimits(),
        options=_core.RustEnginePerfOptions(min_observations=2, max_observations=16),
    )

    model.tune_with_fpms({0: decode_fpm(1, 100, 0.010)})
    model.tune_with_fpms({0: decode_fpm(2, 200, 0.020)})

    assert model.estimate_forward_pass_time({0: decode_fpm(1, 100)}) is not None


def test_capacity_request_rejects_non_finite_sla() -> None:
    with pytest.raises(ValueError, match="finite"):
        _core.EngineCapacityRequest(isl=100, osl=10, ttft_sla_ms=float("nan"))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_num_batched_tokens": 0},
        {"max_num_seqs": 0},
        {"max_kv_tokens": 0},
    ],
)
def test_engine_perf_limits_reject_zero_values(kwargs: dict[str, int]) -> None:
    with pytest.raises(ValueError, match="invalid engine perf limits"):
        _core.EnginePerfLimits(**kwargs)
