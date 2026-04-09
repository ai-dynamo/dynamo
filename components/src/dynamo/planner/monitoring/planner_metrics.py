# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from prometheus_client import Gauge


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
