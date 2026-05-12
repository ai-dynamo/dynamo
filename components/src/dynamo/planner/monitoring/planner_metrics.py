# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Optional

from prometheus_client import Enum, Gauge

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


class _PreLabeledMetric:
    """Wrap a prometheus_client metric with a pre-bound static label.

    Used by ``PlannerPrometheusMetrics`` when a ``model_name`` is configured
    so that callers can keep using ``metric.set(...)`` /
    ``metric.labels(worker_id=..., dp_rank=...).set(...)`` without explicitly
    threading ``model_name`` through every callsite.
    """

    __slots__ = ("_metric", "_static")

    def __init__(self, metric: Any, **static_labels: str) -> None:
        self._metric = metric
        self._static = static_labels

    def labels(self, **kwargs: str) -> Any:
        # Merge so the caller cannot accidentally override the static label.
        merged = {**kwargs, **self._static}
        return self._metric.labels(**merged)

    def set(self, value: float) -> None:
        if self._static:
            self._metric.labels(**self._static).set(value)
        else:
            self._metric.set(value)

    def inc(self, amount: float = 1) -> None:
        if self._static:
            self._metric.labels(**self._static).inc(amount)
        else:
            self._metric.inc(amount)

    def dec(self, amount: float = 1) -> None:
        if self._static:
            self._metric.labels(**self._static).dec(amount)
        else:
            self._metric.dec(amount)

    def state(self, value: str) -> None:
        if self._static:
            self._metric.labels(**self._static).state(value)
        else:
            self._metric.state(value)

    def clear(self) -> None:
        # Clears all child series, including those from other planner instances
        # if they happen to share the same registry. Within a single planner
        # process there is at most one model_name, so this is safe.
        self._metric.clear()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._metric, name)


class PlannerPrometheusMetrics:
    """Container for all Planner Prometheus metrics.

    All metric names follow the ``dynamo_planner_*`` convention, using
    underscores (not colons) and Prometheus-standard unit suffixes.

    When ``model_name`` is provided, every metric carries an additional
    ``model_name`` label and value, allowing multiple planners (one per
    served model) to publish to the same Prometheus instance without
    collision.
    """

    def __init__(self, model_name: Optional[str] = None) -> None:
        self._model_name = model_name
        # Use ``is not None`` rather than truthiness so an explicit
        # ``model_name=""`` is treated as a real (empty-string) label rather
        # than as unset.  Empty-string is a valid Prometheus label value and
        # the caller's intent is unambiguous.
        labeled = model_name is not None
        extra: list[str] = ["model_name"] if labeled else []
        static: dict[str, str] = {"model_name": model_name} if labeled else {}

        def _gauge(
            name: str, doc: str, *, extra_labels: Optional[list[str]] = None
        ) -> Any:
            ln = (extra_labels or []) + extra
            g = Gauge(name, doc, labelnames=ln) if ln else Gauge(name, doc)
            return _PreLabeledMetric(g, **static)

        def _enum(name: str, doc: str, *, states: list[str]) -> Any:
            if extra:
                e = Enum(name, doc, labelnames=extra, states=states)
            else:
                e = Enum(name, doc, states=states)
            return _PreLabeledMetric(e, **static)

        # -- Worker counts ------------------------------------------------
        self.num_prefill_replicas = _gauge(
            f"{PREFIX}_num_prefill_replicas",
            "Current number of prefill replicas",
        )
        self.num_decode_replicas = _gauge(
            f"{PREFIX}_num_decode_replicas",
            "Current number of decode replicas",
        )

        # -- Observed metrics ---------------------------------------------
        self.observed_ttft_ms = _gauge(
            f"{PREFIX}_observed_ttft_ms",
            "Observed time to first token (ms)",
        )
        self.observed_itl_ms = _gauge(
            f"{PREFIX}_observed_itl_ms",
            "Observed inter-token latency (ms)",
        )
        self.observed_requests_per_second = _gauge(
            f"{PREFIX}_observed_requests_per_second",
            "Observed request rate (req/s)",
        )
        self.observed_request_duration_seconds = _gauge(
            f"{PREFIX}_observed_request_duration_seconds",
            "Observed average request duration (seconds)",
        )
        self.observed_input_sequence_tokens = _gauge(
            f"{PREFIX}_observed_input_sequence_tokens",
            "Observed average input sequence length (tokens)",
        )
        self.observed_output_sequence_tokens = _gauge(
            f"{PREFIX}_observed_output_sequence_tokens",
            "Observed average output sequence length (tokens)",
        )

        # -- Predicted metrics (throughput scaling) -----------------------
        self.predicted_requests_per_second = _gauge(
            f"{PREFIX}_predicted_requests_per_second",
            "Predicted request rate for next interval (req/s)",
        )
        self.predicted_input_sequence_tokens = _gauge(
            f"{PREFIX}_predicted_input_sequence_tokens",
            "Predicted input sequence length for next interval (tokens)",
        )
        self.predicted_output_sequence_tokens = _gauge(
            f"{PREFIX}_predicted_output_sequence_tokens",
            "Predicted output sequence length for next interval (tokens)",
        )

        # -- Predicted replica counts -------------------------------------
        self.predicted_num_prefill_replicas = _gauge(
            f"{PREFIX}_predicted_num_prefill_replicas",
            "Decided number of prefill replicas",
        )
        self.predicted_num_decode_replicas = _gauge(
            f"{PREFIX}_predicted_num_decode_replicas",
            "Decided number of decode replicas",
        )

        # -- Cumulative GPU usage -----------------------------------------
        self.gpu_hours = _gauge(
            f"{PREFIX}_gpu_hours",
            "Cumulative GPU hours consumed",
        )

        # -- Diagnostics: estimated latencies -----------------------------
        self.estimated_ttft_ms = _gauge(
            f"{PREFIX}_estimated_ttft_ms",
            "Max estimated TTFT from regression across engines (ms)",
        )
        self.estimated_itl_ms = _gauge(
            f"{PREFIX}_estimated_itl_ms",
            "Max estimated ITL from regression across engines (ms)",
        )

        # -- Diagnostics: engine capacity ---------------------------------
        self.engine_prefill_capacity_requests_per_second = _gauge(
            f"{PREFIX}_engine_prefill_capacity_requests_per_second",
            "Single prefill engine capacity under SLA (req/s)",
        )
        self.engine_decode_capacity_requests_per_second = _gauge(
            f"{PREFIX}_engine_decode_capacity_requests_per_second",
            "Single decode engine capacity under SLA (req/s)",
        )

        # -- Diagnostics: scaling decision enums --------------------------
        self.load_scaling_decision = _enum(
            f"{PREFIX}_load_scaling_decision",
            "Load-based scaling decision reason",
            states=LOAD_DECISION_STATES,
        )
        self.throughput_scaling_decision = _enum(
            f"{PREFIX}_throughput_scaling_decision",
            "Throughput-based scaling decision reason",
            states=THROUGHPUT_DECISION_STATES,
        )

        # -- Diagnostics: per-engine FPM queue depths ---------------------
        _engine_labels = ["worker_id", "dp_rank"]
        self.engine_queued_prefill_tokens = _gauge(
            f"{PREFIX}_engine_queued_prefill_tokens",
            "Queued prefill tokens per engine (from FPM)",
            extra_labels=_engine_labels,
        )
        self.engine_queued_decode_kv_tokens = _gauge(
            f"{PREFIX}_engine_queued_decode_kv_tokens",
            "Queued decode KV tokens per engine (from FPM)",
            extra_labels=_engine_labels,
        )
        self.engine_inflight_decode_kv_tokens = _gauge(
            f"{PREFIX}_engine_inflight_decode_kv_tokens",
            "Inflight (scheduled) decode KV tokens per engine (from FPM)",
            extra_labels=_engine_labels,
        )
