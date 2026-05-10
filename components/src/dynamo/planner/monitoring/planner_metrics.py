# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from prometheus_client import Counter, Enum, Gauge

PREFIX = "dynamo_planner"

LOAD_DECISION_STATES = [
    "unset",
    "disabled",
    "no_fpm_data",
    "scaling_in_progress",
    "worker_count_mismatch",
    "insufficient_data",
    "no_change",
    "scale_up",
    "scale_down",
    "scale_down_capped_by_throughput",
    "scale_down_refused_consolidation",
]

THROUGHPUT_DECISION_STATES = [
    "unset",
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
        self.num_prefill_replicas = Gauge(
            f"{PREFIX}_num_prefill_replicas",
            "Current number of prefill replicas",
        )
        self.num_decode_replicas = Gauge(
            f"{PREFIX}_num_decode_replicas",
            "Current number of decode replicas",
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
        self.observed_input_sequence_tokens = Gauge(
            f"{PREFIX}_observed_input_sequence_tokens",
            "Observed average input sequence length (tokens)",
        )
        self.observed_output_sequence_tokens = Gauge(
            f"{PREFIX}_observed_output_sequence_tokens",
            "Observed average output sequence length (tokens)",
        )

        # -- Predicted metrics (throughput scaling) -----------------------
        self.predicted_requests_per_second = Gauge(
            f"{PREFIX}_predicted_requests_per_second",
            "Predicted request rate for next interval (req/s)",
        )
        self.predicted_input_sequence_tokens = Gauge(
            f"{PREFIX}_predicted_input_sequence_tokens",
            "Predicted input sequence length for next interval (tokens)",
        )
        self.predicted_output_sequence_tokens = Gauge(
            f"{PREFIX}_predicted_output_sequence_tokens",
            "Predicted output sequence length for next interval (tokens)",
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
        self.engine_prefill_capacity_requests_per_second = Gauge(
            f"{PREFIX}_engine_prefill_capacity_requests_per_second",
            "Single prefill engine capacity under SLA (req/s)",
        )
        self.engine_decode_capacity_requests_per_second = Gauge(
            f"{PREFIX}_engine_decode_capacity_requests_per_second",
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

        # -- Power-aware scaling (Phase 1) --------------------------------
        self.power_budget_total_watts = Gauge(
            f"{PREFIX}_power_budget_total_watts",
            "Configured total GPU power budget for this DGD (watts).",
        )
        self.power_projected_watts = Gauge(
            f"{PREFIX}_power_projected_watts",
            "Projected GPU power draw at current replica counts and caps (watts).",
        )
        self.power_budget_utilization = Gauge(
            f"{PREFIX}_power_budget_utilization",
            "Ratio of projected power to total budget (0.0–1.0+).",
        )

        # -- AIC closed-loop optimizer (Phase 3) --------------------------
        self.aic_c_ttft = Gauge(
            f"{PREFIX}_aic_c_ttft",
            "EMA-smoothed AIC TTFT correction coefficient (c_ttft).",
        )
        self.aic_c_itl = Gauge(
            f"{PREFIX}_aic_c_itl",
            "EMA-smoothed AIC ITL correction coefficient (c_itl).",
        )
        # Single labeled gauge for per-component power coefficients so queries
        # can filter by component={"prefill","decode","agg"} uniformly.
        self.aic_c_power = Gauge(
            f"{PREFIX}_aic_c_power",
            "EMA-smoothed AIC power correction coefficient per component.",
            labelnames=("component",),
        )
        # Counts how many times a coefficient saturated at its [0.5, 2.0] clamp.
        # A sustained increment is a CRITICAL calibration signal (§8 row 6).
        self.aic_correction_pegged_total = Counter(
            "dynamo_aic_correction_pegged_total",
            "Times an AIC correction coefficient pegged at its [0.5, 2.0] clamp.",
            labelnames=("coefficient",),
        )
        self.aic_consecutive_failures = Gauge(
            "dynamo_aic_consecutive_failures",
            "Current count of consecutive AIC sweep failures before auto-disable.",
        )
        self.aic_optimizer_exceptions_total = Counter(
            "dynamo_aic_optimizer_exceptions_total",
            "Total AIC sweep exceptions caught at runtime.",
        )
        # Info-style gauge: 1 when the optimizer is auto-disabled, 0 otherwise.
        # Label carries the reason (infeasible_at_startup | startup_exception).
        self.aic_optimizer_disabled_reason = Gauge(
            "dynamo_aic_optimizer_disabled_reason",
            "1 when the AIC optimizer is auto-disabled; reason in label.",
            labelnames=("reason",),
        )
        self.aic_throughput_regression_total = Counter(
            "dynamo_aic_throughput_regression_total",
            "Times re-optimization produced a lower predicted throughput than the "
            "previous config (informational — config is still applied).",
        )

        # -- Admission control / busy_threshold coupling (Phase 3) --------
        # Implied thresholds derived from the AIC operating point (always
        # updated in inherit + autoset modes).
        self.admission_implied_theta_decode = Gauge(
            f"{PREFIX}_admission_implied_theta_decode",
            "Implied decode KV utilization threshold for the current AIC config.",
        )
        self.admission_implied_theta_prefill_frac = Gauge(
            f"{PREFIX}_admission_implied_theta_prefill_frac",
            "Implied prefill fractional utilization threshold for the current AIC config.",
        )
        # Values actually POSTed to frontends (autoset mode only).
        self.admission_set_theta_decode = Gauge(
            f"{PREFIX}_admission_set_theta_decode",
            "Decode KV threshold value POSTed to frontend in autoset mode.",
        )
        self.admission_set_theta_prefill_frac = Gauge(
            f"{PREFIX}_admission_set_theta_prefill_frac",
            "Prefill fractional threshold value POSTed to frontend in autoset mode.",
        )
        self.admission_set_theta_prefill_abs = Gauge(
            f"{PREFIX}_admission_set_theta_prefill_abs",
            "Absolute prefill admission threshold (tokens) POSTed to the frontend.",
        )
        self.admission_max_batched_tokens_unavailable_total = Counter(
            f"{PREFIX}_admission_max_batched_tokens_unavailable_total",
            "Times the planner could not derive an absolute prefill threshold "
            "because no prefill worker reported max_num_batched_tokens via MDC.",
        )
        self.admission_partial_success_total = Counter(
            f"{PREFIX}_admission_partial_success_total",
            "Cumulative count of frontend POST failures across all fanout attempts.",
        )
