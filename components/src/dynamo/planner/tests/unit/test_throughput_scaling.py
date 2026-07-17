# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from dynamo.planner.core.throughput_scaling import ThroughputScalingMixin

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
        self._config = SimpleNamespace(ttft_ms=200.0, min_endpoint=1)
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
