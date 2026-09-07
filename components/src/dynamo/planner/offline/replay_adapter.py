# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scaling-policy adapter for the unified offline replay loop.

The public replay entrypoint passes this adapter into the Rust runtime as
an optional scaling component. Rust owns the drive loop:

    run_replay(..., scaling_policy=adapter)    # Rust owns the loop
      adapter.initial_tick_ms()      -> first tick time
      per ScalingTick:
        adapter.on_tick(metrics)     -> _build_tick_input() -> TickInput
                                        EngineProtocol.tick() -> PlannerEffects
                                        -> {target_prefill, target_decode, next_tick_ms}
        # Rust applies the scaling decision and re-arms the next tick itself
      adapter.finalize() -> PlannerReplayDetails

The tick engine is the builtin orchestrator path:
``OrchestratorEngineAdapter`` wrapping ``LocalPlannerOrchestrator`` +
the builtin local-planner plugins. It preserves the planner's
``PlannerEffects.scale_to`` replay contract while using plugin-aware
observability (Prometheus metrics, audit events, diagnostics).

Async orchestrator calls (``bootstrap_from_fpms`` / ``tick``)
run inside a single replay-scoped event loop so callers don't change.

Supports both aggregated and disaggregated topologies. No I/O, no
runtime dependencies. Fully deterministic with offline replay.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from typing import Any, Literal, Optional

import msgspec

from dynamo.common.forward_pass_metrics import (
    ForwardPassMetrics,
    QueuedRequestMetrics,
    ScheduledRequestMetrics,
)
from dynamo.planner.config.planner_config import PlannerConfig
from dynamo.planner.core.engine_protocol import EngineProtocol
from dynamo.planner.core.types import (
    FpmObservations,
    PlannerEffects,
    ScheduledTick,
    TickInput,
    TrafficObservation,
    WorkerCapabilities,
    WorkerCounts,
)
from dynamo.planner.monitoring.diagnostics_recorder import DiagnosticsRecorder
from dynamo.planner.monitoring.traffic_metrics import Metrics
from dynamo.planner.plugins.clock import VirtualClock
from dynamo.planner.plugins.orchestrator.engine_adapter import OrchestratorEngineAdapter
from dynamo.replay.report import PlannerReplayDetails

logger = logging.getLogger(__name__)


@dataclass
class ScalingEvent:
    """Record of a single scaling decision."""

    at_s: float
    component: str  # "agg", "prefill", or "decode"
    from_count: int
    to_count: int
    reason: Optional[str] = None


def _build_fpm_from_dict(d: dict[str, Any]) -> ForwardPassMetrics:
    """Convert a replay FPM snapshot dict into a ForwardPassMetrics struct."""
    return ForwardPassMetrics(
        worker_id=str(d["worker_id"]),
        dp_rank=int(d.get("dp_rank", 0)),
        wall_time=d["wall_time"],
        scheduled_requests=ScheduledRequestMetrics(
            num_prefill_requests=d["num_prefill_requests"],
            sum_prefill_tokens=d["sum_prefill_tokens"],
            var_prefill_length=d["var_prefill_length"],
            sum_prefill_kv_tokens=d["sum_prefill_kv_tokens"],
            num_decode_requests=d["num_decode_requests"],
            sum_decode_kv_tokens=d["sum_decode_kv_tokens"],
            var_decode_kv_tokens=d["var_decode_kv_tokens"],
        ),
        queued_requests=QueuedRequestMetrics(
            num_prefill_requests=d["num_queued_prefill"],
            sum_prefill_tokens=d["sum_queued_prefill_tokens"],
            var_prefill_length=d["var_queued_prefill_length"],
            num_decode_requests=d["num_queued_decode"],
            sum_decode_kv_tokens=d["sum_queued_decode_kv_tokens"],
            var_decode_kv_tokens=d["var_queued_decode_kv_tokens"],
        ),
    )


def _update_fpm_cache(
    cache: dict[tuple[str, int], ForwardPassMetrics],
    snapshots: list[dict[str, Any]],
    active_worker_ids: list[int],
) -> None:
    """Update a last-seen FPM cache with new snapshots and prune removed workers."""
    for snap in snapshots:
        fpm = _build_fpm_from_dict(snap)
        cache[(fpm.worker_id, fpm.dp_rank)] = fpm

    active_worker_ids_as_str = {str(worker_id) for worker_id in active_worker_ids}
    for key in list(cache):
        if key[0] not in active_worker_ids_as_str:
            del cache[key]


def _merge_traffic(
    acc: Optional[dict[str, Any]], window: dict[str, Any]
) -> dict[str, Any]:
    """Merge two TrafficStats dicts into one window.

    Exact for every field the planner's scaling consumes:
      - ``duration_s``/``num_req``: summed.
      - ``avg_isl``/``avg_osl``: weighted by ``shape_count`` (completed,
        non-rejected requests). ``num_req`` is offered load and intentionally
        has different timing under queueing.
      - ``avg_ttft_ms``/``avg_itl_ms``: weighted by their native sample counts.
      - ``avg_kv_hit_rate``: weighted by ``hit_rate_count`` (its true
        denominator: router admissions with ``isl_blocks > 0``), so the merge
        reconstructs the exact sample mean rather than approximating it.
      - ``avg_accept_length``: weighted by ``accept_length_forward_count``
        (decode request-forwards, its true denominator), exact across windows.

    Older bridge payloads without native counts fall back to ``num_req`` for
    compatibility."""
    if acc is None:
        return dict(window)
    na = float(acc.get("num_req", 0.0))
    nw = float(window.get("num_req", 0.0))
    n = na + nw

    def _weighted(key: str, wa: float, ww: float) -> float:
        w = wa + ww
        if w <= 0:
            return 0.0
        return (acc.get(key, 0.0) * wa + window.get(key, 0.0) * ww) / w

    hit_a = float(acc.get("hit_rate_count", 0.0))
    hit_w = float(window.get("hit_rate_count", 0.0))
    fwd_a = float(acc.get("accept_length_forward_count", 0.0))
    fwd_w = float(window.get("accept_length_forward_count", 0.0))
    shape_a = float(acc.get("shape_count", na))
    shape_w = float(window.get("shape_count", nw))
    ttft_a = float(acc.get("ttft_count", na))
    ttft_w = float(window.get("ttft_count", nw))
    itl_a = float(acc.get("itl_count", na))
    itl_w = float(window.get("itl_count", nw))

    merged: dict[str, Any] = {
        "duration_s": acc.get("duration_s", 0.0) + window.get("duration_s", 0.0),
        "num_req": n,
        # Carry the native denominators so chained multi-window merges stay exact.
        "hit_rate_count": hit_a + hit_w,
        "accept_length_forward_count": fwd_a + fwd_w,
        "shape_count": shape_a + shape_w,
        "ttft_count": ttft_a + ttft_w,
        "itl_count": itl_a + itl_w,
        "avg_isl": _weighted("avg_isl", shape_a, shape_w),
        "avg_osl": _weighted("avg_osl", shape_a, shape_w),
        "avg_ttft_ms": _weighted("avg_ttft_ms", ttft_a, ttft_w),
        "avg_itl_ms": _weighted("avg_itl_ms", itl_a, itl_w),
        # Count-weighted by the true denominator -> exact across windows.
        "avg_kv_hit_rate": _weighted("avg_kv_hit_rate", hit_a, hit_w),
    }
    a_acc = acc.get("avg_accept_length")
    a_win = window.get("avg_accept_length")
    if a_acc is None and a_win is None:
        merged["avg_accept_length"] = None
    elif a_acc is None:
        merged["avg_accept_length"] = a_win
    elif a_win is None:
        merged["avg_accept_length"] = a_acc
    else:
        fwd = fwd_a + fwd_w
        merged["avg_accept_length"] = (
            (a_acc * fwd_a + a_win * fwd_w) / fwd if fwd > 0 else None
        )
    return merged


class ReplayPlannerAdapter:
    """Context-managed planner scaling policy for offline replay."""

    def __init__(
        self,
        planner_config: PlannerConfig,
        engine: EngineProtocol,
        capabilities: Optional[WorkerCapabilities] = None,
        warmup_observations: Optional[list[TrafficObservation]] = None,
        benchmark_granularity: Optional[int] = None,
        capture_details: bool = True,
    ) -> None:
        self._config = planner_config
        self._capabilities = capabilities
        self._is_disagg = planner_config.mode == "disagg"

        self._engine = engine
        self._warmup_observations = list(warmup_observations or [])
        self._benchmark_granularity = benchmark_granularity
        self._capture_details = capture_details
        self._bootstrap_metadata: dict[str, Any] = {"status": "not_attempted"}
        self._orchestrator_bootstrapped = False
        # Replay's ``run()`` is synchronous; we own a scoped event loop to
        # drive the async engine calls without forcing callers to use
        # ``asyncio.run``.
        self._loop: Optional[asyncio.AbstractEventLoop] = asyncio.new_event_loop()

        # Last-seen FPM caches (separate for prefill/decode)
        self._prefill_fpm_cache: dict[tuple[str, int], ForwardPassMetrics] = {}
        self._decode_fpm_cache: dict[tuple[str, int], ForwardPassMetrics] = {}
        # Partial traffic window accumulated across ticks until a throughput tick
        # consumes it (``None`` = nothing pending).
        self._pending_traffic: Optional[dict[str, Any]] = None

        # Scaling targets -- used as `expected` in WorkerCounts
        self._scaling_target_prefill: Optional[int] = None
        self._scaling_target_decode: Optional[int] = None

        # Diagnostics recorder for HTML report generation
        decode_max_kv = (
            capabilities.decode.max_kv_tokens
            if capabilities and capabilities.decode
            else None
        )
        self._recorder = DiagnosticsRecorder(
            config=planner_config, max_kv_tokens=decode_max_kv
        )
        self._cumulative_gpu_hours: float = 0.0
        self._last_tick_s: float = 0.0
        self._last_traffic: Metrics = Metrics()

        # Orchestrator Bootstrap is deferred until ``run()`` because replay
        # installs benchmark FPMs after adapter construction and before the
        # first tick.

    def __enter__(self) -> ReplayPlannerAdapter:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> Literal[False]:
        try:
            self.close()
        except BaseException:
            if exc_value is None:
                raise
            logger.exception("planner replay cleanup failed")
        return False

    # ------------------------------------------------------------------
    # Sync/async bridging
    # ------------------------------------------------------------------

    def _run_sync(self, coro):
        """Run a coroutine on the replay-owned event loop. Used to call
        the orchestrator path's async APIs from replay's sync surface."""
        assert self._loop is not None, "sync execution requires an active replay scope"
        return self._loop.run_until_complete(coro)

    def _bootstrap_orchestrator_if_needed(self) -> None:
        if self._orchestrator_bootstrapped:
            return
        bootstrap_plugins = getattr(self._engine, "bootstrap_plugins", None)
        if bootstrap_plugins is not None:
            self._run_sync(
                bootstrap_plugins(historical_traffic=self._warmup_observations or None)
            )
        self._orchestrator_bootstrapped = True

    def install_benchmark_fpms(
        self,
        *,
        prefill_fpms: Optional[list[ForwardPassMetrics]] = None,
        decode_fpms: Optional[list[ForwardPassMetrics]] = None,
        agg_fpms: Optional[list[ForwardPassMetrics]] = None,
    ) -> None:
        """Install AIC benchmark FPMs into the regression model(s).

        Normal replay uses ``OrchestratorEngineAdapter
        .install_regressions_from_fpms``.

        Without this, replay's throughput regression stays empty and
        planner-in-the-loop scaling decisions diverge from live planner
        behavior."""
        self._engine.install_regressions_from_fpms(  # type: ignore[attr-defined]
            prefill_fpms=prefill_fpms,
            decode_fpms=decode_fpms,
            agg_fpms=agg_fpms,
        )

    def set_bootstrap_metadata(self, metadata: dict[str, Any]) -> None:
        self._bootstrap_metadata = dict(metadata)

    # ------------------------------------------------------------------
    # Rust calls ``initial_tick_ms`` once and ``on_tick`` for each ``ScalingTick``.
    # The entrypoint wraps the returned trace report via ``finalize``.
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Bootstrap the orchestrator and compute the first tick. Idempotent."""
        self._bootstrap_orchestrator_if_needed()
        self._pending_tick: ScheduledTick = self._engine.initial_tick(0.0)
        self._scaling_events: list[ScalingEvent] = []
        self._ticks: list[dict[str, Any]] = []
        self._total_ticks = 0

    def initial_tick_ms(self) -> float:
        """First tick time in milliseconds."""
        if not self._orchestrator_bootstrapped or not hasattr(self, "_pending_tick"):
            self.start()
        return self._pending_tick.at_s * 1000.0

    def on_tick(self, result: dict[str, Any]) -> dict[str, Any]:
        """Drive one planner tick from the runtime snapshot. Returns the scaling
        decision (absolute targets, ``None`` = unchanged) + the next tick time in ms
        (``None`` = stop) for the Rust loop to apply and re-arm."""
        tick = self._pending_tick
        tick_input = self._build_tick_input(tick, result)
        effects: PlannerEffects = self._run_sync(self._engine.tick(tick, tick_input))
        emit_diagnostics = self._should_emit_tick_diagnostics(tick, effects)
        self._total_ticks += 1
        if self._capture_details:
            self._record_diagnostics(tick_input, effects, result, emit_diagnostics)

        current_p = result.get(
            "non_draining_prefill_count", result["active_prefill_count"]
        )
        current_d = result.get(
            "non_draining_decode_count", result["active_decode_count"]
        )
        if (
            self._scaling_target_prefill is not None
            and current_p == self._scaling_target_prefill
        ):
            self._scaling_target_prefill = None
        if (
            self._scaling_target_decode is not None
            and current_d == self._scaling_target_decode
        ):
            self._scaling_target_decode = None

        target_prefill: Optional[int] = None
        target_decode: Optional[int] = None
        if effects.scale_to is not None:
            target_prefill, target_decode = self._compute_scale_decision(
                effects, result, tick_input.now_s
            )

        next_tick_ms: Optional[float] = None
        if effects.next_tick is not None:
            self._pending_tick = effects.next_tick
            next_tick_ms = effects.next_tick.at_s * 1000.0

        decision = {
            "target_prefill": target_prefill,
            "target_decode": target_decode,
            "next_tick_ms": next_tick_ms,
        }
        tick_ordinal = int(result["tick_ordinal"])
        if self._capture_details:
            self._ticks.append(
                {
                    "ordinal": tick_ordinal,
                    "at_ms": tick.at_s * 1000.0,
                    "scheduled_tick": asdict(tick),
                    "input": self._tick_input_record(tick_input),
                    "topology": self._topology_record(result),
                    "effects": asdict(effects),
                    "runtime_decision": decision,
                }
            )
        return decision

    def finalize(
        self, lifecycle_operations: list[dict[str, Any]] | None = None
    ) -> PlannerReplayDetails:
        """Finalize planner-owned replay details after successful execution."""
        html_report_path = self._recorder.finalize() if self._capture_details else None
        return PlannerReplayDetails(
            metadata=self._planner_metadata(),
            ticks=self._ticks if self._capture_details else [],
            scaling_events=self._scaling_events,
            lifecycle_operations=(
                list(lifecycle_operations or []) if self._capture_details else []
            ),
            total_ticks=self._total_ticks,
            html_report_path=html_report_path,
        )

    @staticmethod
    def _topology_record(result: dict[str, Any]) -> dict[str, dict[str, Any]]:
        active_prefill = list(result["active_prefill_ids"])
        starting_prefill = list(result.get("starting_prefill_ids", []))
        draining_prefill = list(result.get("draining_prefill_ids", []))
        active_decode = list(result["active_decode_ids"])
        starting_decode = list(result.get("starting_decode_ids", []))
        draining_decode = list(result.get("draining_decode_ids", []))
        return {
            "prefill": {
                "active": active_prefill,
                "active_count": len(active_prefill),
                "starting": starting_prefill,
                "starting_count": len(starting_prefill),
                "draining": draining_prefill,
                "draining_count": len(draining_prefill),
            },
            "decode": {
                "active": active_decode,
                "active_count": len(active_decode),
                "starting": starting_decode,
                "starting_count": len(starting_decode),
                "draining": draining_decode,
                "draining_count": len(draining_decode),
            },
        }

    @staticmethod
    def _tick_input_record(tick_input: TickInput) -> dict[str, Any]:
        record: dict[str, Any] = {
            "now_s": tick_input.now_s,
            "traffic": (
                None if tick_input.traffic is None else asdict(tick_input.traffic)
            ),
            "worker_counts": (
                None
                if tick_input.worker_counts is None
                else asdict(tick_input.worker_counts)
            ),
            "fpm_observations": None,
        }
        observations = tick_input.fpm_observations
        if observations is None:
            return record

        def ordered(values):
            if not values:
                return None
            return [
                {
                    "worker_id": worker_id,
                    "dp_rank": dp_rank,
                    "metrics": msgspec.to_builtins(metrics),
                }
                for (worker_id, dp_rank), metrics in sorted(values.items())
            ]

        record["fpm_observations"] = {
            "prefill": ordered(observations.prefill),
            "decode": ordered(observations.decode),
        }
        return record

    def _planner_metadata(self) -> dict[str, Any]:
        excluded_fields = [
            "report_output_dir",
            "report_filename",
            "report_interval_hours",
            "report_write_gzip_log",
            "live_dashboard_port",
            "control_api_port",
            "plugin_registration.auth",
            "plugin_registration.transport",
            "plugin_registration.protocol_version_min",
            "plugin_registration.protocol_version_max",
            "plugin_registration.heartbeat_timeout_seconds",
            "plugin_registration.heartbeat_missed_threshold",
            "plugin_registration.admin",
            "plugin_registration.in_process_plugins.kwargs",
            "scheduling.external_plugins.endpoint",
            "scheduling.external_plugins.auth_token",
            "scheduling.gateway",
        ]
        config = self._config.model_dump(mode="json", by_alias=True)
        decision_config = dict(config)
        for field in (name for name in excluded_fields if "." not in name):
            decision_config.pop(field, None)
        registration = dict(decision_config.get("plugin_registration", {}))
        in_process = []
        for plugin in registration.get("in_process_plugins", []):
            identity = dict(plugin)
            kwargs = identity.pop("kwargs", {})
            identity["kwargs_digest"] = hashlib.sha256(
                json.dumps(kwargs, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            identity["source"] = "in_process"
            in_process.append(identity)
        in_process.sort(
            key=lambda item: (
                item["plugin_type"],
                item["priority"],
                item["plugin_id"],
                item["module"],
                item["class"],
            )
        )
        registration = {"in_process_plugins": in_process}
        decision_config["plugin_registration"] = registration

        scheduling = dict(decision_config.get("scheduling", {}))
        external = []
        for plugin in scheduling.get("external_plugins", []):
            identity = {
                key: value
                for key, value in plugin.items()
                if key not in {"endpoint", "auth_token"}
            }
            identity["source"] = "external"
            external.append(identity)
        external.sort(
            key=lambda item: (
                item["plugin_type"],
                item["priority"],
                item["plugin_id"],
                item["version"],
            )
        )
        scheduling["external_plugins"] = external
        scheduling.pop("gateway", None)
        decision_config["scheduling"] = scheduling

        builtin_plugin_ids = [
            "builtin_load_predict",
            "builtin_load_propose",
            "builtin_throughput_propose",
        ]
        plugin_pipeline = [
            {"plugin_id": plugin_id, "source": "builtin"}
            for plugin_id in builtin_plugin_ids
        ]
        plugin_pipeline.extend(in_process)
        plugin_pipeline.extend(external)
        config_bytes = json.dumps(
            decision_config, sort_keys=True, separators=(",", ":")
        ).encode()
        return {
            "planner_config_digest": hashlib.sha256(config_bytes).hexdigest(),
            "planner_config_identity_exclusions": excluded_fields,
            "builtin_plugin_ids": builtin_plugin_ids,
            "configured_plugin_identities": [*in_process, *external],
            "plugin_pipeline": plugin_pipeline,
            "pipeline_schema_version": "planner-plugin-pipeline.v1",
            "mode": self._config.mode,
            "benchmark_granularity": self._benchmark_granularity,
            "bootstrap": self._bootstrap_metadata,
            "details_captured": self._capture_details,
        }

    def close(self) -> None:
        """Shut down the engine and replay-scoped event loop. Idempotent."""
        loop = self._loop
        if loop is None:
            return
        self._loop = None
        try:
            loop.run_until_complete(self._engine.shutdown())
        finally:
            loop.close()

    def _record_diagnostics(
        self,
        tick_input: TickInput,
        effects: PlannerEffects,
        result: dict[str, Any],
        emit_diagnostics: bool,
    ) -> None:
        """Update GPU-hours tracking and feed the diagnostics recorder."""
        if not self._recorder.enabled:
            return

        now_s = tick_input.now_s
        if self._last_tick_s > 0.0:
            dt_h = (now_s - self._last_tick_s) / 3600.0
            num_p = result["active_prefill_count"]
            num_d = result["active_decode_count"]
            gpu_p = self._config.prefill_engine_num_gpu or 0
            gpu_d = self._config.decode_engine_num_gpu or 0
            self._cumulative_gpu_hours += (num_p * gpu_p + num_d * gpu_d) * dt_h
        self._last_tick_s = now_s

        if not emit_diagnostics:
            return

        self._recorder.record(
            tick_input,
            effects,
            self._last_traffic,
            self._cumulative_gpu_hours,
        )

    @staticmethod
    def _should_emit_tick_diagnostics(
        tick: ScheduledTick, effects: PlannerEffects
    ) -> bool:
        diag = effects.diagnostics
        return (
            tick.run_load_scaling
            or tick.run_throughput_scaling
            or effects.scale_to is not None
            or bool(diag.audit_events)
            or bool(diag.short_circuit_reason)
        )

    def _compute_scale_decision(
        self,
        effects: PlannerEffects,
        result: dict[str, Any],
        now_s: float,
    ) -> tuple[Optional[int], Optional[int]]:
        """Compute the (prefill, decode) absolute scale targets and record the scaling
        event. Returns ``(None, None)`` for a no-op. The Rust loop applies the targets,
        so this method only records the requested transition."""
        scale = effects.scale_to
        if scale is None:
            raise ValueError(
                "_compute_scale_decision requires effects.scale_to to be set"
            )
        current_p = result.get(
            "non_draining_prefill_count", result["active_prefill_count"]
        )
        current_d = result.get(
            "non_draining_decode_count", result["active_decode_count"]
        )
        target_p = scale.num_prefill if scale.num_prefill is not None else current_p
        target_d = scale.num_decode if scale.num_decode is not None else current_d

        if target_p == current_p and target_d == current_d:
            return (None, None)

        if self._is_disagg:
            if scale.num_prefill is not None and target_p != current_p:
                direction = "scale_up" if target_p > current_p else "scale_down"
                logger.info(
                    "Planner scaling prefill: %d -> %d at t=%.1fs (%s)",
                    current_p,
                    target_p,
                    now_s,
                    direction,
                )
                self._scaling_target_prefill = target_p
                self._scaling_events.append(
                    ScalingEvent(
                        at_s=now_s,
                        component="prefill",
                        from_count=current_p,
                        to_count=target_p,
                        reason=direction,
                    )
                )
            if scale.num_decode is not None and target_d != current_d:
                direction = "scale_up" if target_d > current_d else "scale_down"
                logger.info(
                    "Planner scaling decode: %d -> %d at t=%.1fs (%s)",
                    current_d,
                    target_d,
                    now_s,
                    direction,
                )
                self._scaling_target_decode = target_d
                self._scaling_events.append(
                    ScalingEvent(
                        at_s=now_s,
                        component="decode",
                        from_count=current_d,
                        to_count=target_d,
                        reason=direction,
                    )
                )
        else:
            direction = "scale_up" if target_d > current_d else "scale_down"
            logger.info(
                "Planner scaling: %d -> %d workers at t=%.1fs (%s)",
                current_d,
                target_d,
                now_s,
                direction,
            )
            self._scaling_target_decode = target_d
            self._scaling_events.append(
                ScalingEvent(
                    at_s=now_s,
                    component="agg",
                    from_count=current_d,
                    to_count=target_d,
                    reason=direction,
                )
            )
        return (target_p, target_d)

    def _is_easy_mode(self) -> bool:
        """Easy-mode check routed via config — both paths honour this
        the same way (no regression in non-SLA modes)."""
        return self._config.optimization_target != "sla"

    def _build_tick_input(
        self, tick: ScheduledTick, result: dict[str, Any]
    ) -> TickInput:
        """Convert the Rust scaling snapshot to planner ``TickInput``."""
        # Keep planner cadence on the scheduled replay clock. Rust also
        # advances idle gaps to this timestamp so traffic windows drain
        # with the same duration the planner sees.
        now_s = tick.at_s

        worker_counts = None
        if tick.need_worker_states:
            active_p = result["active_prefill_count"]
            active_d = result["active_decode_count"]
            current_p = result.get("non_draining_prefill_count", active_p)
            current_d = result.get("non_draining_decode_count", active_d)
            expected_p = (
                self._scaling_target_prefill
                if self._scaling_target_prefill is not None
                else current_p
            )
            expected_d = (
                self._scaling_target_decode
                if self._scaling_target_decode is not None
                else current_d
            )
            worker_counts = WorkerCounts(
                ready_num_prefill=active_p if self._is_disagg else None,
                ready_num_decode=active_d,
                expected_num_prefill=expected_p if self._is_disagg else None,
                expected_num_decode=expected_d,
                prefill_scaling_in_progress=(
                    self._is_disagg
                    and self._scaling_target_prefill is not None
                    and self._scaling_target_prefill != current_p
                ),
                decode_scaling_in_progress=(
                    self._scaling_target_decode is not None
                    and self._scaling_target_decode != current_d
                ),
            )

        fpm_observations = None
        # Merge each callback's latest worker/rank snapshots into the last-seen
        # cache, then expose the cache only on FPM ticks. This matches the live
        # subscriber's latest-snapshot semantics.
        _update_fpm_cache(
            self._prefill_fpm_cache,
            result.get("prefill_fpm_snapshots", []),
            result["active_prefill_ids"],
        )
        _update_fpm_cache(
            self._decode_fpm_cache,
            result.get("decode_fpm_snapshots", []),
            result["active_decode_ids"],
        )
        if tick.need_worker_fpm:
            prefill_dict = (
                dict(self._prefill_fpm_cache) if self._prefill_fpm_cache else None
            )
            decode_dict = (
                dict(self._decode_fpm_cache) if self._decode_fpm_cache else None
            )
            fpm_observations = FpmObservations(
                prefill=prefill_dict,
                decode=decode_dict,
            )

        # Rust drains the per-tick traffic window into ``result["traffic"]``;
        # accumulate it so a need_traffic_metrics tick sees the full window since the
        # last consumed one (the planner consumes traffic only on throughput ticks).
        tick_traffic = result.get("traffic")
        if tick_traffic is not None:
            self._pending_traffic = _merge_traffic(
                getattr(self, "_pending_traffic", None), tick_traffic
            )

        traffic = None
        if tick.need_traffic_metrics:
            t = getattr(self, "_pending_traffic", None) or {}
            self._pending_traffic = None
            duration_s = t.get("duration_s", 0.0)
            if duration_s > 0:
                num_req = float(t.get("num_req", 0))
                # The mocker publishes avg_kv_hit_rate as 0.0 when the
                # window had no admissions with non-zero ISL blocks;
                # pass it through as-is so the planner can distinguish
                # "no datapoint" from an explicit zero hit rate.
                traffic = TrafficObservation(
                    duration_s=duration_s,
                    num_req=num_req,
                    isl=t.get("avg_isl", 0.0),
                    osl=t.get("avg_osl", 0.0),
                    kv_hit_rate=t.get("avg_kv_hit_rate"),
                    accept_length=t.get("avg_accept_length"),
                )
                # Stash observed TTFT/ITL for the diagnostics recorder.
                ttft_count = float(t.get("ttft_count", num_req))
                itl_count = float(t.get("itl_count", num_req))
                self._last_traffic = Metrics(
                    ttft=t.get("avg_ttft_ms") if ttft_count > 0 else None,
                    itl=t.get("avg_itl_ms") if itl_count > 0 else None,
                    num_req=traffic.num_req,
                    isl=traffic.isl,
                    osl=traffic.osl,
                    kv_hit_rate=traffic.kv_hit_rate,
                    accept_length=traffic.accept_length,
                )

        return TickInput(
            now_s=now_s,
            traffic=traffic,
            worker_counts=worker_counts,
            fpm_observations=fpm_observations,
        )


def create_replay_planner_adapter(
    planner_config: PlannerConfig,
    capabilities: Optional[WorkerCapabilities] = None,
    warmup_observations: Optional[list[TrafficObservation]] = None,
    benchmark_granularity: Optional[int] = None,
    capture_details: bool = True,
) -> ReplayPlannerAdapter:
    """Create a replay adapter backed by the builtin planner orchestrator."""
    engine = OrchestratorEngineAdapter(
        planner_config,
        capabilities or WorkerCapabilities(),
        clock=VirtualClock(),
    )
    adapter = ReplayPlannerAdapter(
        planner_config=planner_config,
        engine=engine,
        capabilities=capabilities,
        warmup_observations=warmup_observations,
        benchmark_granularity=benchmark_granularity,
        capture_details=capture_details,
    )
    return adapter
