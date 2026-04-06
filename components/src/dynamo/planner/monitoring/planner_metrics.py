# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from prometheus_client import Counter, Gauge


class PlannerPrometheusMetrics:
    """Container for all Planner Prometheus metrics."""

    def __init__(self, prefix: str = "planner"):
        # Worker counts
        self.num_p_workers = Gauge(
            f"{prefix}:num_p_workers", "Number of prefill workers"
        )
        self.num_d_workers = Gauge(
            f"{prefix}:num_d_workers", "Number of decode workers"
        )

        # Observed metrics
        self.observed_ttft = Gauge(
            f"{prefix}:observed_ttft", "Observed time to first token (ms)"
        )
        self.observed_itl = Gauge(
            f"{prefix}:observed_itl", "Observed inter-token latency (ms)"
        )
        self.observed_request_rate = Gauge(
            f"{prefix}:observed_request_rate", "Observed request rate (req/s)"
        )
        self.observed_request_duration = Gauge(
            f"{prefix}:observed_request_duration", "Observed request duration (s)"
        )
        self.observed_isl = Gauge(
            f"{prefix}:observed_isl", "Observed input sequence length"
        )
        self.observed_osl = Gauge(
            f"{prefix}:observed_osl", "Observed output sequence length"
        )

        # Correction factors
        self.p_correction_factor = Gauge(
            f"{prefix}:p_correction_factor", "Prefill correction factor"
        )
        self.d_correction_factor = Gauge(
            f"{prefix}:d_correction_factor", "Decode correction factor"
        )

        # Predicted metrics
        self.predicted_request_rate = Gauge(
            f"{prefix}:predicted_request_rate", "Predicted request rate (req/s)"
        )
        self.predicted_isl = Gauge(
            f"{prefix}:predicted_isl", "Predicted input sequence length"
        )
        self.predicted_osl = Gauge(
            f"{prefix}:predicted_osl", "Predicted output sequence length"
        )
        self.predicted_num_p = Gauge(
            f"{prefix}:predicted_num_p", "Predicted number of prefill replicas"
        )
        self.predicted_num_d = Gauge(
            f"{prefix}:predicted_num_d", "Predicted number of decode replicas"
        )

        # Cumulative GPU usage
        self.gpu_hours = Gauge(f"{prefix}:gpu_hours", "Cumulative GPU hours used")

        # Advisory metrics — Gauges (12)
        # Follows Dynamo naming guideline: dynamo_planner_advisory_*
        # See lib/runtime/src/metrics/prometheus_names.rs
        _adv = "dynamo_planner_advisory"
        self.advisory_recommended_p = Gauge(
            f"{_adv}_recommended_p",
            "Recommended prefill replicas (after GPU budget)",
        )
        self.advisory_recommended_d = Gauge(
            f"{_adv}_recommended_d",
            "Recommended decode replicas (after GPU budget)",
        )
        self.advisory_current_p = Gauge(
            f"{_adv}_current_p",
            "Current actual prefill replicas from DGD",
        )
        self.advisory_current_d = Gauge(
            f"{_adv}_current_d",
            "Current actual decode replicas from DGD",
        )
        self.advisory_delta_p = Gauge(
            f"{_adv}_delta_p",
            "Prefill delta (recommended - current). Positive = scale up",
        )
        self.advisory_delta_d = Gauge(
            f"{_adv}_delta_d",
            "Decode delta (recommended - current). Positive = scale up",
        )
        self.advisory_scaling_action = Gauge(
            f"{_adv}_scaling_action",
            "Aggregate action: 1=scale up, 0=hold, -1=scale down",
        )
        self.advisory_action_reason = Gauge(
            f"{_adv}_action_reason",
            "Reason code for the advisory action",
        )
        self.advisory_est_ttft = Gauge(
            f"{_adv}_est_ttft",
            "Estimated TTFT after applying recommendation (ms). NaN if no profiling data",
        )
        self.advisory_est_itl = Gauge(
            f"{_adv}_est_itl",
            "Estimated ITL after applying recommendation (ms). NaN if no profiling data",
        )
        self.advisory_ttft_headroom = Gauge(
            f"{_adv}_ttft_headroom",
            "TTFT SLA target - estimated TTFT (ms). Positive = safe",
        )
        self.advisory_itl_headroom = Gauge(
            f"{_adv}_itl_headroom",
            "ITL SLA target - estimated ITL (ms). Positive = safe",
        )

        # Advisory metrics — Counters (3)
        self.advisory_scaleup_total = Counter(
            f"{_adv}_scaleup_total",
            "Cumulative scale-up recommendations",
        )
        self.advisory_scaledown_total = Counter(
            f"{_adv}_scaledown_total",
            "Cumulative scale-down recommendations",
        )
        self.advisory_hold_total = Counter(
            f"{_adv}_hold_total",
            "Cumulative hold recommendations",
        )
