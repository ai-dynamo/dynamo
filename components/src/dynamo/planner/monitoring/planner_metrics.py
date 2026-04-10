# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from prometheus_client import Enum, Gauge

PREFIX = "dynamo_planner"

LOAD_DECISION_STATES = [
    "disabled",
    "no_fpm_data",
    "scaling_in_progress",
    "worker_count_mismatch",
    "insufficient_data",
    "no_change",
    "scale_up",
    "scale_down",
    "scale_down_capped_by_throughput",
]

THROUGHPUT_DECISION_STATES = [
    "disabled",
    "no_traffic_data",
    "predict_failed",
    "model_not_ready",
    "set_lower_bound",
    "scale",
]


class PlannerPrometheusMetrics:
    """Container for all Planner Prometheus metrics.

    All metric names follow the ``dynamo_planner_*`` convention, using
    underscores (not colons) and Prometheus-standard unit suffixes.
    """

    def __init__(self) -> None:
        # -- Worker counts ------------------------------------------------
        self.num_prefill_workers = Gauge(
            f"{PREFIX}_num_prefill_workers",
            "Current number of prefill workers",
        )
        self.num_decode_workers = Gauge(
            f"{PREFIX}_num_decode_workers",
            "Current number of decode workers",
        )

        # -- Observed metrics ---------------------------------------------
        self.observed_ttft_ms = Gauge(
            f"{PREFIX}_observed_ttft_ms",
            "Observed time to first token (ms)",
        )
        self.observed_itl_ms = Gauge(
            f"{PREFIX}_observed_itl_ms",
            "Observed inter-token latency (ms)",
        )
        self.observed_requests_per_second = Gauge(
            f"{PREFIX}_observed_requests_per_second",
            "Observed request rate (req/s)",
        )
        self.observed_request_duration_seconds = Gauge(
            f"{PREFIX}_observed_request_duration_seconds",
            "Observed average request duration (seconds)",
        )
        self.observed_isl = Gauge(
            f"{PREFIX}_observed_isl",
            "Observed average input sequence length (tokens)",
        )
        self.observed_osl = Gauge(
            f"{PREFIX}_observed_osl",
            "Observed average output sequence length (tokens)",
        )

        # -- Predicted metrics (throughput scaling) -----------------------
        self.predicted_requests_per_second = Gauge(
            f"{PREFIX}_predicted_requests_per_second",
            "Predicted request rate for next interval (req/s)",
        )
        self.predicted_isl = Gauge(
            f"{PREFIX}_predicted_isl",
            "Predicted input sequence length for next interval",
        )
        self.predicted_osl = Gauge(
            f"{PREFIX}_predicted_osl",
            "Predicted output sequence length for next interval",
        )

        # -- Predicted replica counts -------------------------------------
        self.predicted_num_prefill_replicas = Gauge(
            f"{PREFIX}_predicted_num_prefill_replicas",
            "Decided number of prefill replicas",
        )
        self.predicted_num_decode_replicas = Gauge(
            f"{PREFIX}_predicted_num_decode_replicas",
            "Decided number of decode replicas",
        )

        # -- Cumulative GPU usage -----------------------------------------
        self.gpu_hours = Gauge(
            f"{PREFIX}_gpu_hours",
            "Cumulative GPU hours consumed",
        )

        # -- Diagnostics: estimated latencies -----------------------------
        self.estimated_ttft_ms = Gauge(
            f"{PREFIX}_estimated_ttft_ms",
            "Max estimated TTFT from regression across engines (ms)",
        )
        self.estimated_itl_ms = Gauge(
            f"{PREFIX}_estimated_itl_ms",
            "Max estimated ITL from regression across engines (ms)",
        )

        # -- Diagnostics: engine capacity ---------------------------------
        self.engine_prefill_requests_per_second = Gauge(
            f"{PREFIX}_engine_prefill_requests_per_second",
            "Single prefill engine capacity under SLA (req/s)",
        )
        self.engine_decode_requests_per_second = Gauge(
            f"{PREFIX}_engine_decode_requests_per_second",
            "Single decode engine capacity under SLA (req/s)",
        )

        # -- Diagnostics: scaling decision enums --------------------------
        self.load_scaling_decision = Enum(
            f"{PREFIX}_load_scaling_decision",
            "Load-based scaling decision reason",
            states=LOAD_DECISION_STATES,
        )
        self.throughput_scaling_decision = Enum(
            f"{PREFIX}_throughput_scaling_decision",
            "Throughput-based scaling decision reason",
            states=THROUGHPUT_DECISION_STATES,
        )

        # -- Diagnostics: per-engine FPM queue depths ---------------------
        _engine_labels = ["worker_id", "dp_rank"]
        self.engine_queued_prefill_tokens = Gauge(
            f"{PREFIX}_engine_queued_prefill_tokens",
            "Queued prefill tokens per engine (from FPM)",
            labelnames=_engine_labels,
        )
        self.engine_queued_decode_kv_tokens = Gauge(
            f"{PREFIX}_engine_queued_decode_kv_tokens",
            "Queued decode KV tokens per engine (from FPM)",
            labelnames=_engine_labels,
        )
        self.engine_inflight_decode_kv_tokens = Gauge(
            f"{PREFIX}_engine_inflight_decode_kv_tokens",
            "Inflight (scheduled) decode KV tokens per engine (from FPM)",
            labelnames=_engine_labels,
        )
