# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from dynamo.planner.core.throughput_scaling import ThroughputScalingMixin
from dynamo.planner.core.types import EngineCapabilities, WorkerCapabilities

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]


class _PrefillRegression:
    def find_engine_capacity_rps(self, **kwargs):
        return SimpleNamespace(rps=1.0, ttft_ms=1002.0, eligible=False)


class _ThroughputScalingHarness(ThroughputScalingMixin):
    def __init__(self):
        self._config = SimpleNamespace(
            ttft_ms=200.0,
            min_endpoint=1,
            prefill_min_endpoint=None,
            decode_min_endpoint=None,
        )
        self._prefill_regression = _PrefillRegression()
        self._diag_throughput_reason = None
        self._diag_engine_rps_prefill = None


def test_unreachable_prefill_ttft_does_not_create_replica_floor():
    scaling = _ThroughputScalingHarness()

    replicas = scaling._compute_prefill_replicas(
        demand_rps=0.01,
        isl=1000,
        osl=150,
    )

    assert replicas == 1


def test_prefill_throughput_uses_component_minimum_override():
    scaling = _ThroughputScalingHarness()
    scaling._config.prefill_min_endpoint = 3

    replicas = scaling._compute_prefill_replicas(
        demand_rps=0.01,
        isl=1000,
        osl=150,
    )

    assert replicas == 3


class _ModelNotReadyHarness(ThroughputScalingMixin):
    def __init__(self, *, current_decode: int = 1):
        self._config = SimpleNamespace(
            enable_load_scaling=False,
            min_endpoint=1,
            prefill_min_endpoint=None,
            decode_min_endpoint=3,
            min_gpu_budget=-1,
            max_gpu_budget=100,
            ttft_ms=200.0,
            itl_ms=10.0,
        )
        self._capabilities = WorkerCapabilities(
            decode=EngineCapabilities(
                gpu_cost_per_replica=1,
                max_num_batched_tokens=4096,
            )
        )
        self._num_p_workers = 1
        self._num_d_workers = current_decode
        self._throughput_lower_bound_p = 1
        self._throughput_lower_bound_d = current_decode
        self._diag_throughput_reason = None
        self._diag_throughput_reason_prefill = None
        self._diag_throughput_reason_decode = None
        self._diag_engine_rps_prefill = None
        self._diag_engine_rps_decode = None
        self._agg_regression = SimpleNamespace(
            find_engine_capacity_rps=lambda **_kwargs: SimpleNamespace(
                rps=0.0,
                ttft_ms=1.0,
                itl_ms=1.0,
                eligible=True,
            )
        )

    def _compute_prefill_replicas(self, *_args, **_kwargs):
        self._diag_throughput_reason = "model_not_ready"
        return None

    def _compute_decode_replicas(self, *_args, **_kwargs):
        self._diag_throughput_reason = "model_not_ready"
        return None

    def _apply_single_budget(self, desired, _component):
        return desired

    def _apply_global_budget(self, num_p, num_d):
        return num_p, num_d

    def _current_decode_accept_length(self):
        return 1.0


def test_single_throughput_applies_floor_when_perf_model_not_ready():
    scaling = _ModelNotReadyHarness()

    decision = scaling._throughput_single(1.0, 1.0, 1.0, "decode")

    assert decision is not None
    assert decision.num_decode == 3
    assert scaling._diag_throughput_reason == "model_not_ready"


def test_disagg_throughput_applies_floor_when_perf_models_not_ready():
    scaling = _ModelNotReadyHarness()

    decision = scaling._throughput_disagg(1.0, 1.0, 1.0)

    assert decision is not None
    assert (decision.num_prefill, decision.num_decode) == (1, 3)
    assert scaling._diag_throughput_reason == "model_not_ready"


def test_disagg_throughput_holds_when_model_not_ready_and_floor_already_met():
    scaling = _ModelNotReadyHarness(current_decode=3)

    assert scaling._throughput_disagg(1.0, 1.0, 1.0) is None
    assert scaling._diag_throughput_reason == "model_not_ready"


def test_agg_throughput_applies_floor_when_perf_model_not_ready():
    scaling = _ModelNotReadyHarness()

    decision = scaling._throughput_agg(1.0, 1.0, 1.0)

    assert decision is not None
    assert decision.num_decode == 3
    assert scaling._diag_throughput_reason == "model_not_ready"


@pytest.mark.parametrize("gpu_cost_per_replica", [4, 5])
def test_engine_rps_recommendation_is_independent_of_sidecar_cost(
    gpu_cost_per_replica: int,
):
    scaling = _ThroughputScalingHarness()
    scaling._capabilities = WorkerCapabilities(
        prefill=EngineCapabilities(
            num_gpu=4,
            gpu_cost_per_replica=gpu_cost_per_replica,
        )
    )

    replicas = scaling._compute_prefill_replicas(
        demand_rps=2.1,
        isl=1000,
        osl=150,
    )

    assert replicas == 3
