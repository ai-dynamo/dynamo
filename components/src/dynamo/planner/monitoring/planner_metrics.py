# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Optional

from prometheus_client import CollectorRegistry, Counter, Enum, Gauge, Histogram

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
    # Plugin-era decision reasons.
    # prometheus_client.Enum.states is construction-time fixed; new values
    # MUST be appended (never inserted or reordered) so existing scrapers
    # with older label sets keep parsing.
    "override_by_user_plugin",
    "reconcile_clamped_to_floor",
    "reconcile_clamped_to_ceiling",
    "held_over",
    "rejected_by_plugin",
]

THROUGHPUT_DECISION_STATES = [
    "unset",
    "disabled",
    "no_traffic_data",
    "predict_failed",
    "model_not_ready",
    "set_lower_bound",
    "scale",
    # Plugin-era decision reasons.
    # Same append-only rule as LOAD_DECISION_STATES.
    "override_by_user_plugin",
    "held_over",
    "circuit_open",
    "rejected_by_plugin",
]


class _PreLabeledMetric:
    """Wrap a prometheus_client metric with a pre-bound static label.

    Used by ``PlannerPrometheusMetrics`` when a ``model_name`` is configured
    so that callers can keep using ``metric.set(...)`` /
    ``metric.labels(worker_id=..., dp_rank=...).set(...)`` without explicitly
    threading ``model_name`` through every callsite.

    Mirrors prometheus_client's metric API (``labels``, ``set``, ``inc``,
    ``dec``, ``observe``, ``state``, ``clear``) so callers can use either
    positional or keyword forms.  If you add a new prometheus_client metric
    type (``Histogram``, ``Summary``, ``Counter`` with custom methods) and
    rely on a method not listed above, extend this wrapper so the static
    label still propagates -- otherwise ``__getattr__`` would forward the
    call straight to the underlying (unlabeled) metric and silently bypass
    the static label.
    """

    __slots__ = ("_metric", "_static")

    def __init__(self, metric: Any, **static_labels: str) -> None:
        self._metric = metric
        self._static = static_labels

    def _labeled(self) -> Any:
        return self._metric.labels(**self._static) if self._static else self._metric

    def labels(self, *labelvalues: str, **labelkwargs: str) -> Any:
        """Mirror prometheus_client's ``labels(*values, **kwargs)`` signature.

        Positional calls match the metric's declared label order minus the
        pre-bound static labels (which this wrapper injects).  Keyword
        calls are merged with the static dict.  Mixed positional+keyword
        calls are forwarded as-is and let prometheus_client raise.
        """
        if labelvalues and labelkwargs:
            # prometheus_client itself rejects this; preserve its error.
            return self._metric.labels(*labelvalues, **labelkwargs)
        if not self._static:
            return self._metric.labels(*labelvalues, **labelkwargs)
        if labelvalues:
            # Map positional values to the user-specified label names (i.e.
            # the metric's declared labelnames minus our static keys), then
            # forward everything by keyword so prometheus_client can splice
            # in the static labels in the correct slots.
            declared = list(self._metric._labelnames)
            user_names = [n for n in declared if n not in self._static]
            if len(labelvalues) != len(user_names):
                raise ValueError(
                    f"Expected {len(user_names)} positional label values "
                    f"({user_names!r}), got {len(labelvalues)}"
                )
            merged = dict(zip(user_names, labelvalues))
            merged.update(self._static)
            return self._metric.labels(**merged)
        return self._metric.labels(**labelkwargs, **self._static)

    def set(self, value: float) -> None:
        self._labeled().set(value)

    def inc(self, amount: float = 1) -> None:
        self._labeled().inc(amount)

    def dec(self, amount: float = 1) -> None:
        self._labeled().dec(amount)

    def observe(self, amount: float) -> None:
        # Forward-compat for Histogram / Summary.  No-op for Gauge/Enum since
        # they don't define ``observe``; callers shouldn't be calling it on
        # the wrong metric type.
        self._labeled().observe(amount)

    def state(self, value: str) -> None:
        self._labeled().state(value)

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

        # -- SLA targets (static: set once at planner startup) ------------
        self.sla_target_ttft_ms = Gauge(
            f"{PREFIX}_sla_target_ttft_ms",
            "Configured SLA target for time to first token (ms)",
        )
        self.sla_target_itl_ms = Gauge(
            f"{PREFIX}_sla_target_itl_ms",
            "Configured SLA target for inter-token latency (ms)",
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


# ---------------------------------------------------------------------------
# Plugin-framework metrics
# ---------------------------------------------------------------------------


# Circuit breaker state encoding — matches the ``CircuitBreaker`` enum in
# plugins/registry/circuit_breaker.py (CLOSED / HALF_OPEN / OPEN).  Exposed
# here so emitters import a single source of truth instead of picking
# floats by hand.
CIRCUIT_STATE_CLOSED = 0.0
CIRCUIT_STATE_HALF_OPEN = 0.5
CIRCUIT_STATE_OPEN = 1.0


class PluginFrameworkMetrics:
    """Plugin-layer Prometheus metrics.

    Separate from ``PlannerPrometheusMetrics`` because this set is
    about **plugin invocation mechanics** (eval count / latency / circuit
    state / HOLD_LAST cache / override contribution) rather than the
    planner's own decision outputs, and because callers construct it
    alongside ``LocalPlannerOrchestrator`` rather than at planner
    bootstrap.

    Registry hook
    -------------
    ``registry`` is threaded through to every metric constructor.  If
    ``None`` (default), metrics land on ``prometheus_client.REGISTRY``
    and scraping works as usual.  Unit tests pass an isolated
    ``CollectorRegistry()`` per instance to avoid the
    ``Duplicated timeseries`` error you get when you build the same
    metric twice against the global registry.
    """

    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        kw: dict = {} if registry is None else {"registry": registry}

        self.plugin_evaluations_total = Counter(
            f"{PREFIX}_plugin_evaluations_total",
            "Plugin evaluation calls, labelled by outcome.",
            labelnames=["plugin_id", "stage", "result"],
            **kw,
        )
        """``result`` values: ``accept`` / ``set`` / ``at_least`` /
        ``at_most`` / ``reject`` / ``timeout`` / ``error`` / ``held_over``.
        See ``plugins/merge/types.py`` for the result enum."""

        self.plugin_latency_seconds = Histogram(
            f"{PREFIX}_plugin_latency_seconds",
            "End-to-end plugin RPC latency in seconds.",
            labelnames=["plugin_id", "stage"],
            buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0),
            **kw,
        )
        """Histogram buckets cover the useful range from in-process
        plugin (~1ms) up to the ``request_timeout_seconds`` default
        (5s).  We do NOT record failed / timed-out calls here —
        they'd skew percentiles.  Those land on
        ``plugin_evaluations_total{result=timeout|error}`` only."""

        self.plugin_circuit_state = Gauge(
            f"{PREFIX}_plugin_circuit_state",
            "Per-plugin circuit breaker state "
            f"({CIRCUIT_STATE_CLOSED}=closed, "
            f"{CIRCUIT_STATE_HALF_OPEN}=half_open, "
            f"{CIRCUIT_STATE_OPEN}=open).",
            labelnames=["plugin_id"],
            **kw,
        )

        self.plugin_held_over_total = Counter(
            f"{PREFIX}_plugin_held_over_total",
            "HOLD_LAST cache replay events "
            "(scheduler returned cached result in lieu of calling the plugin).",
            labelnames=["plugin_id", "stage"],
            **kw,
        )

        self.plugin_cache_age_seconds = Gauge(
            f"{PREFIX}_plugin_cache_age_seconds",
            "Age of the HOLD_LAST cached result per plugin (seconds).",
            labelnames=["plugin_id"],
            **kw,
        )

        self.plugin_override_active = Gauge(
            f"{PREFIX}_plugin_override_active",
            "1 if the plugin contributed an override to the final "
            "merged proposal this tick, 0 otherwise.",
            labelnames=["plugin_id", "stage", "override_type"],
            **kw,
        )
        """``override_type`` values: ``SET`` / ``AT_LEAST`` / ``AT_MOST``
        / ``REJECT``.  A plugin that returned ``ACCEPT`` with no
        proposal emits 0 for every override_type (explicitly, via
        ``reset_overrides`` below).  This is a per-tick gauge — emitters
        MUST call ``reset_overrides(plugin_id, stage)`` at tick start
        (or the gauge will stay stuck at 1 from the previous tick)."""

        # ----- RECONCILE / CONSTRAIN behaviour metrics -----

        self.reconcile_clamped_total = Counter(
            f"{PREFIX}_reconcile_clamped_total",
            "RECONCILE stage clamped the recommendation by a floor/ceiling "
            "override (the final replica count differs from the lowest-priority "
            "SET because an AT_LEAST raised it or an AT_MOST lowered it).",
            labelnames=["sub_component_type", "source"],
            **kw,
        )
        """``source`` is the plugin_id of whichever AT_LEAST (for floor)
        or AT_MOST (for ceiling) actually won the clamp; ``"unknown"``
        when the merge could not back-reference the winning target (a
        degenerate case the merge helper logs)."""

        self.constrain_capped_total = Counter(
            f"{PREFIX}_constrain_capped_total",
            "CONSTRAIN stage capped the final replica count (same meaning "
            "as reconcile_clamped_total but fired by the CONSTRAIN pass).",
            labelnames=["sub_component_type", "source"],
            **kw,
        )

        self.reject_short_circuited_total = Counter(
            f"{PREFIX}_reject_short_circuited_total",
            "REJECT result triggered a stage short-circuit; the remaining "
            "pipeline was not invoked and EXECUTE was skipped.",
            labelnames=["plugin_id"],
            **kw,
        )

        # ----- Tick scheduling metrics -----
        #
        # These describe the plugin pipeline's tick-loop behaviour —
        # how often plugins get deferred, how much latency the cache
        # replay adds, and whether ticks meet their deadline.

        self.tick_skipped_total = Counter(
            f"{PREFIX}_tick_skipped_total",
            "Times a plugin was skipped in its stage because its "
            "execution_interval hadn't elapsed yet (cache replay or "
            "ACCEPT_WHEN_IDLE policy took over).",
            labelnames=["plugin_id"],
            **kw,
        )

        self.tick_requires_unsatisfied_total = Counter(
            f"{PREFIX}_tick_requires_unsatisfied_total",
            "Times a plugin was skipped this tick because one of its "
            "``requires_produced_fields`` dot-paths resolved to None "
            "in the current PipelineContext (i.e., the upstream "
            "stage that was expected to produce that field did not "
            "fire / did not produce). ``missing_field`` is the first "
            "dot-path that failed the check; useful for debugging "
            "dependency cascades.",
            labelnames=["plugin_id", "missing_field"],
            **kw,
        )

        self.tick_lag_seconds = Gauge(
            f"{PREFIX}_tick_lag_seconds",
            "Seconds between a plugin's scheduled 'due' moment and the "
            "tick that actually evaluated it; 0 for plugins evaluated "
            "right on schedule, positive when the planner lags the "
            "scheduled cadence.",
            labelnames=["plugin_id"],
            **kw,
        )

        self.tick_duration_seconds = Histogram(
            f"{PREFIX}_tick_duration_seconds",
            "Total time spent inside orchestrator.tick (PREDICT + "
            "PROPOSE + RECONCILE + CONSTRAIN), in seconds.",
            buckets=(0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0),
            **kw,
        )
        """Default ``tick_max_duration_seconds`` is 30s; buckets up to
        30s let operators see P99 tick duration approach the deadline
        before ``tick_timeout_total`` starts firing."""

        self.tick_timeout_total = Counter(
            f"{PREFIX}_tick_timeout_total",
            "Ticks that exceeded tick_max_duration_seconds and were "
            "aborted by the outer asyncio.wait_for.",
            **kw,
        )

    def reset_overrides(self, plugin_id: str, stage: str) -> None:
        """Zero every ``override_type`` for a (plugin_id, stage) pair.

        Called at tick start (or when a plugin is evaluated but
        produces no override) so the gauge doesn't stay stuck at 1
        from a previous tick when the plugin stops contributing.
        """
        for override_type in ("SET", "AT_LEAST", "AT_MOST", "REJECT"):
            self.plugin_override_active.labels(
                plugin_id=plugin_id,
                stage=stage,
                override_type=override_type,
            ).set(0)
