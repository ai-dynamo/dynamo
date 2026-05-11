# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""In-memory mock for PlannerPrometheusMetrics.

All Prometheus Counter/Gauge/Enum operations are forwarded to simple
in-memory accumulators.  The testbed reads these to assert on counter
increments and gauge values without starting a real Prometheus server.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any


class _FakeCounter:
    """Thread-safe (enough for sync testbed) counter mock."""

    def __init__(self) -> None:
        self._labels: dict[tuple, "_FakeLabeledCounter"] = {}
        self._value: float = 0.0

    def labels(self, **kwargs: Any) -> "_FakeLabeledCounter":
        key = tuple(sorted(kwargs.items()))
        if key not in self._labels:
            self._labels[key] = _FakeLabeledCounter()
        return self._labels[key]

    def inc(self, amount: float = 1.0) -> None:
        self._value += amount

    @property
    def value(self) -> float:
        return self._value

    def labeled_value(self, **kwargs: Any) -> float:
        key = tuple(sorted(kwargs.items()))
        return self._labels.get(key, _FakeLabeledCounter()).value


class _FakeLabeledCounter:
    def __init__(self) -> None:
        self._value: float = 0.0

    def inc(self, amount: float = 1.0) -> None:
        self._value += amount

    @property
    def value(self) -> float:
        return self._value


class _FakeGauge:
    """Gauge mock with optional labels."""

    def __init__(self) -> None:
        self._labels: dict[tuple, "_FakeLabeledGauge"] = {}
        self._value: float = 0.0

    def labels(self, **kwargs: Any) -> "_FakeLabeledGauge":
        key = tuple(sorted(kwargs.items()))
        if key not in self._labels:
            self._labels[key] = _FakeLabeledGauge()
        return self._labels[key]

    def set(self, value: float) -> None:
        self._value = value

    @property
    def value(self) -> float:
        return self._value

    def labeled_value(self, **kwargs: Any) -> float:
        key = tuple(sorted(kwargs.items()))
        return self._labels.get(key, _FakeLabeledGauge()).value


class _FakeLabeledGauge:
    def __init__(self) -> None:
        self._value: float = 0.0

    def set(self, value: float) -> None:
        self._value = value

    @property
    def value(self) -> float:
        return self._value


class _FakeEnum:
    """Enum-state mock."""

    def __init__(self) -> None:
        self._state: str = "unset"

    def state(self, state: str) -> None:
        self._state = state

    @property
    def value(self) -> str:
        return self._state


class FakePlannerMetrics:
    """Drop-in for PlannerPrometheusMetrics.

    Provides the same attribute names; all values are backed by simple
    in-memory accumulators.  Read ``metrics.aic_c_ttft.value`` etc. in
    assertions.
    """

    def __init__(self) -> None:
        # Worker counts
        self.num_prefill_replicas = _FakeGauge()
        self.num_decode_replicas = _FakeGauge()
        # Observed metrics
        self.observed_ttft_ms = _FakeGauge()
        self.observed_itl_ms = _FakeGauge()
        self.observed_requests_per_second = _FakeGauge()
        self.observed_request_duration_seconds = _FakeGauge()
        self.observed_input_sequence_tokens = _FakeGauge()
        self.observed_output_sequence_tokens = _FakeGauge()
        # Predicted metrics
        self.predicted_requests_per_second = _FakeGauge()
        self.predicted_input_sequence_tokens = _FakeGauge()
        self.predicted_output_sequence_tokens = _FakeGauge()
        self.predicted_num_prefill_replicas = _FakeGauge()
        self.predicted_num_decode_replicas = _FakeGauge()
        # GPU usage
        self.gpu_hours = _FakeGauge()
        # Diagnostics latency
        self.estimated_ttft_ms = _FakeGauge()
        self.estimated_itl_ms = _FakeGauge()
        # Engine capacity
        self.engine_prefill_capacity_requests_per_second = _FakeGauge()
        self.engine_decode_capacity_requests_per_second = _FakeGauge()
        # Scaling decision enums
        self.load_scaling_decision = _FakeEnum()
        self.throughput_scaling_decision = _FakeEnum()
        # FPM queue depths (labeled)
        self.engine_queued_prefill_tokens = _FakeGauge()
        self.engine_queued_decode_kv_tokens = _FakeGauge()
        self.engine_inflight_decode_kv_tokens = _FakeGauge()
        # Power-aware scaling
        self.power_budget_total_watts = _FakeGauge()
        self.power_projected_watts = _FakeGauge()
        self.power_budget_utilization = _FakeGauge()
        # AIC optimizer
        self.aic_c_ttft = _FakeGauge()
        self.aic_c_itl = _FakeGauge()
        self.aic_c_power = _FakeGauge()
        self.aic_correction_pegged_total = _FakeCounter()
        self.aic_consecutive_failures = _FakeGauge()
        self.aic_optimizer_exceptions_total = _FakeCounter()
        self.aic_optimizer_disabled_reason = _FakeGauge()
        self.aic_throughput_regression_total = _FakeCounter()
        # Admission control
        self.admission_implied_theta_decode = _FakeGauge()
        self.admission_implied_theta_prefill_frac = _FakeGauge()
        self.admission_set_theta_decode = _FakeGauge()
        self.admission_set_theta_prefill_frac = _FakeGauge()
        self.admission_set_theta_prefill_abs = _FakeGauge()
        self.admission_max_batched_tokens_unavailable_total = _FakeCounter()
        self.admission_partial_success_total = _FakeCounter()
        # Power-agent cap clamping
        self.power_agent_cap_clamped_total = _FakeCounter()

    # ------------------------------------------------------------------
    # Convenience read helpers for testbed assertions
    # ------------------------------------------------------------------

    def counter_value(self, name: str, **labels: Any) -> float:
        obj = getattr(self, name)
        if labels:
            return obj.labeled_value(**labels)
        return obj.value

    def gauge_value(self, name: str, **labels: Any) -> float:
        obj = getattr(self, name)
        if labels:
            return obj.labeled_value(**labels)
        return obj.value
